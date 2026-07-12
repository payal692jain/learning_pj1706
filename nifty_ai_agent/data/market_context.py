"""Global market context — major indices and GIFT Nifty pre-market data."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import yfinance as yf

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
    gift = fetch_gift_nifty()
    bias = compute_global_bias(indices)

    # Override bias with GIFT Nifty if strongly directional
    if gift:
        if gift.change_pct > 0.5:
            bias = "BULLISH"
        elif gift.change_pct < -0.5:
            bias = "BEARISH"

    return MarketContext(indices=indices, gift_nifty=gift, global_bias=bias)


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

    return "\n".join(lines)
