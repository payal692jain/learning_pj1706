"""Global market context — major indices and GIFT Nifty pre-market data."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests
import yfinance as yf

from nifty_ai_agent.config import get_settings

from nifty_ai_agent.data.nse_provider import _NSE_HEADERS, _retry

logger = logging.getLogger(__name__)

# ── Global indices to track ────────────────────────────────────────────────────
_GLOBAL_INDICES: dict[str, str] = {
    "S&P 500":    "^GSPC",
    "Dow Jones":  "^DJI",
    "NASDAQ":     "^IXIC",
    "Nikkei 225": "^N225",
    "Hang Seng":  "^HSI",
    "FTSE 100":   "^FTSE",
    "DAX":        "^GDAXI",
    "India VIX":  "^INDIAVIX",
}


# Crude is a first-order macro input for India, which imports ~85% of what it
# burns. Brent leads because the Indian crude basket tracks it far more closely
# than WTI; WTI is carried as a cross-check on whether a move is global or a
# US-specific inventory story.
#
# Kept OUT of _GLOBAL_INDICES deliberately. The sign is inverted for India — a
# crude rally is a headwind, not a risk-on signal — so folding it into a bias
# routine that counts "how many are green" would read a rising oil price as
# bullish for NIFTY, which is backwards.
_COMMODITIES: dict[str, str] = {
    "Brent": "BZ=F",
    "WTI":   "CL=F",
}

# NSE commodity underlyings, quoted in rupees per barrel/unit. Brent is listed
# as BRCRUDEOIL but does not actually trade, so it is deliberately absent —
# see _fetch_commodities_upstox.
_UPSTOX_COMMODITIES: dict[str, str] = {
    "Crude (INR)": "CRUDEOIL",
    "Nat Gas (INR)": "NATURALGAS",
}

# Move needed before crude is called a genuine tailwind or headwind rather than
# noise. Crude routinely moves ±1% on nothing.
_CRUDE_MATERIAL_PCT = 1.5


@dataclass
class IndexSnapshot:
    name: str
    symbol: str
    price: float
    change_pct: float       # percentage change from previous close
    direction: str          # "↑" or "↓" or "→"


@dataclass
class GiftNiftySnapshot:
    price: float
    change: float           # absolute points change
    change_pct: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    source: str = "NSE IFSC"


@dataclass
class MarketContext:
    indices: list[IndexSnapshot]
    gift_nifty: GiftNiftySnapshot | None
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    global_bias: str = "NEUTRAL"  # "BULLISH", "BEARISH", "NEUTRAL"
    # Defaulted so existing callers that build a context by hand keep working.
    commodities: list[IndexSnapshot] = field(default_factory=list)

    @property
    def crude(self) -> IndexSnapshot | None:
        """Whichever crude reading is available — rupee first, then Brent."""
        for wanted in ("Crude (INR)", "Brent", "WTI"):
            match = next((c for c in self.commodities if c.name == wanted), None)
            if match is not None:
                return match
        return None


def fetch_global_indices() -> list[IndexSnapshot]:
    """Fetch current prices and % change for major global indices via yfinance."""
    snapshots: list[IndexSnapshot] = []

    for name, symbol in _GLOBAL_INDICES.items():
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            price = float(info.last_price)
            prev_close = float(info.previous_close)
            change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0.0
            direction = "↑" if change_pct > 0.05 else ("↓" if change_pct < -0.05 else "→")
            snapshots.append(
                IndexSnapshot(
                    name=name,
                    symbol=symbol,
                    price=round(price, 2),
                    change_pct=round(change_pct, 2),
                    direction=direction,
                )
            )
            logger.debug("%s: %.2f (%.2f%%)", name, price, change_pct)
        except Exception as exc:
            logger.warning("Could not fetch %s (%s): %s", name, symbol, exc)

    return snapshots


def fetch_gift_nifty() -> GiftNiftySnapshot | None:
    """Fetch GIFT Nifty from the live NSE IX feed.

    Delegates to data/gift_nifty.py. The two sources this used to try are both dead:
    NSE's `/api/liveanalysis-giftnifty` now 404s, and yfinance's ^NSEIFSC is delisted
    — so this function had been returning None on every single call, which meant GIFT
    silently never reached the morning report OR the intraday confidence adjuster.
    """
    from nifty_ai_agent.data.gift_nifty import fetch_gift_nifty as _fetch

    quote = _fetch()
    if quote is None:
        return None
    return GiftNiftySnapshot(
        price=round(quote.price, 2),
        change=round(quote.change, 2),
        change_pct=round(quote.change_pct, 2),
        source="NSE IX",
    )


def _snapshot(name: str, symbol: str, price: float, change_pct: float) -> IndexSnapshot:
    return IndexSnapshot(
        name=name,
        symbol=symbol,
        price=round(price, 2),
        change_pct=round(change_pct, 2),
        direction="↑" if change_pct > 0.05 else ("↓" if change_pct < -0.05 else "→"),
    )


def _fetch_commodities_upstox(token: str) -> list[IndexSnapshot]:
    """Near-month NSE commodity futures, priced in rupees, via Upstox.

    Rupee-denominated crude is the more honest number for an Indian trader than a
    dollar benchmark: it already contains the rupee move, which is half of why
    crude matters here in the first place.

    Contracts with no volume are skipped rather than reported. NSE lists Brent
    (BRCRUDEOIL) but it does not trade — quoting its stale last price as today's
    crude move would be inventing a data point.
    """
    from datetime import date

    from nifty_ai_agent.data.instrument_master import _epoch_ms_to_date, get_instrument_master

    rows = get_instrument_master()._load()  # noqa: SLF001 — no public row accessor
    today = date.today()
    snapshots: list[IndexSnapshot] = []

    for name, asset in _UPSTOX_COMMODITIES.items():
        futures = sorted(
            (e, r) for r in rows
            if r.get("segment") == "NSE_COM"
            and r.get("asset_symbol") == asset
            and r.get("instrument_type") == "FUT"
            and (e := _epoch_ms_to_date(r.get("expiry"))) is not None
            and e >= today
        )
        for _, row in futures[:3]:      # walk past an untraded front month
            try:
                quote = _quote_one(row["instrument_key"], token)
                if quote is None:
                    continue
                last, net, volume = quote
                if volume <= 0 or last <= 0:
                    continue
                previous = last - net
                if previous <= 0:
                    continue
                snapshots.append(_snapshot(
                    name, row.get("trading_symbol", asset), last, net / previous * 100,
                ))
                break
            except Exception as exc:
                logger.debug("Upstox commodity %s failed: %s", asset, exc)
    return snapshots


def _quote_one(instrument_key: str, token: str) -> tuple[float, float, float] | None:
    """(last_price, net_change, volume) for one instrument, or None."""
    resp = requests.get(
        "https://api.upstox.com/v2/market-quote/quotes",
        params={"instrument_key": instrument_key},
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json().get("data") or {}
    if not data:
        return None
    payload = next(iter(data.values()))
    return (
        float(payload.get("last_price") or 0.0),
        float(payload.get("net_change") or 0.0),
        float(payload.get("volume") or 0.0),
    )


def _fetch_commodities_yfinance() -> list[IndexSnapshot]:
    """Dollar benchmarks — the fallback, and the global reference for Brent."""
    snapshots: list[IndexSnapshot] = []
    for name, symbol in _COMMODITIES.items():
        try:
            info = yf.Ticker(symbol).fast_info
            price = float(info.last_price)
            prev_close = float(info.previous_close)
            change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0.0
            snapshots.append(_snapshot(name, symbol, price, change_pct))
        except Exception as exc:
            logger.warning("Commodity %s (%s) failed: %s", name, symbol, exc)
    return snapshots


def fetch_commodities(upstox_token: str = "") -> list[IndexSnapshot]:
    """Commodities that move Indian equities — Upstox first, yfinance as fallback."""
    if upstox_token:
        try:
            snapshots = _fetch_commodities_upstox(upstox_token)
            if snapshots:
                logger.info(
                    "Commodities via Upstox: %s",
                    ", ".join(f"{s.name} {s.change_pct:+.2f}%" for s in snapshots),
                )
                return snapshots
            logger.info("Upstox served no traded commodity future — using yfinance")
        except Exception as exc:
            logger.warning("Upstox commodities failed (%s) — using yfinance", exc)
    return _fetch_commodities_yfinance()


def crude_impact(crude: IndexSnapshot | None) -> tuple[str, str]:
    """Return *(label, explanation)* for what crude is doing to Indian equities.

    The sign is inverted against the usual risk-on reading: India imports roughly
    85% of its crude, so a rally widens the import bill and the current-account
    deficit, pressures the rupee and feeds inflation. Cheap oil is the tailwind.

    The effect is not uniform across the market, which is why the explanation
    names sectors rather than just a direction — upstream producers (ONGC, Oil
    India) gain from exactly the move that hurts refiners, paints, tyres and
    aviation.
    """
    if crude is None:
        return "UNKNOWN", "crude data unavailable"

    move = crude.change_pct
    if move >= _CRUDE_MATERIAL_PCT:
        return "HEADWIND", (
            f"{crude.name} {move:+.1f}% — costlier imports pressure the rupee and OMCs, "
            f"paints, tyres, aviation; upstream (ONGC) benefits"
        )
    if move <= -_CRUDE_MATERIAL_PCT:
        return "TAILWIND", (
            f"{crude.name} {move:+.1f}% — cheaper imports ease the deficit and inflation; "
            f"good for OMCs, paints, tyres, aviation; upstream suffers"
        )
    return "NEUTRAL", f"{crude.name} {move:+.1f}% — no material push either way"


def compute_global_bias(indices: list[IndexSnapshot]) -> str:
    """Determine overall global market bias from major indices."""
    if not indices:
        return "NEUTRAL"
    # Weight S&P 500, Dow, NASDAQ more heavily (they drive Indian pre-market)
    key_indices = {"S&P 500", "Dow Jones", "NASDAQ", "Nikkei 225"}
    key = [s for s in indices if s.name in key_indices]
    if not key:
        key = indices
    positive = sum(1 for s in key if s.change_pct > 0.2)
    negative = sum(1 for s in key if s.change_pct < -0.2)
    if positive > negative + 1:
        return "BULLISH"
    if negative > positive + 1:
        return "BEARISH"
    return "NEUTRAL"


def fetch_market_context() -> MarketContext:
    """Fetch all global context: indices + GIFT Nifty + bias."""
    logger.info("Fetching global market context")
    indices = fetch_global_indices()
    commodities = fetch_commodities(get_settings().upstox_access_token)
    gift = fetch_gift_nifty()
    bias = compute_global_bias(indices)

    # Override bias with GIFT Nifty if strongly directional
    if gift:
        if gift.change_pct > 0.5:
            bias = "BULLISH"
        elif gift.change_pct < -0.5:
            bias = "BEARISH"

    return MarketContext(
        indices=indices, gift_nifty=gift, global_bias=bias, commodities=commodities,
    )


def format_context_for_notification(ctx: MarketContext) -> str:
    """Compact format for Pushover notification."""
    lines = [f"🌍 Global Bias: {ctx.global_bias}"]

    if ctx.gift_nifty:
        g = ctx.gift_nifty
        arrow = "↑" if g.change > 0 else "↓"
        lines.append(
            f"GIFT Nifty: {g.price:,.0f}  {arrow}{abs(g.change):.0f} ({g.change_pct:+.2f}%)"
        )

    lines.append("")
    for idx in ctx.indices:
        lines.append(f"{idx.direction} {idx.name}: {idx.change_pct:+.2f}%")

    if ctx.commodities:
        lines.append("")
        for c in ctx.commodities:
            lines.append(f"{c.direction} {c.name}: {c.change_pct:+.2f}%")
        label, explanation = crude_impact(ctx.crude)
        if label != "UNKNOWN":
            lines.append(f"🛢 Crude → {label}")
            lines.append(f"   {explanation}")

    return "\n".join(lines)
