"""Top gainers and losers across the NSE F&O universe.

Universe choice matters more than anything else here. A gainers list over *all*
NSE equities is dominated by illiquid microcaps up 15% on a few hundred shares —
technically the top of the market, useless to act on, and impossible to trade
options in. The ~208 F&O underlyings are the names that actually carry
derivatives, so a mover on this list is a mover you could do something about.

Prices come from Upstox's batched quotes endpoint, which returns net_change
against the previous close directly. Deriving the change from daily bars instead
would be a session behind during market hours, which is exactly when a movers
list is worth reading.
"""

import logging
from dataclasses import dataclass, field

import requests

from nifty_ai_agent.data.instrument_master import InstrumentMaster

logger = logging.getLogger(__name__)

_QUOTES_URL = "https://api.upstox.com/v2/market-quote/quotes"
_TIMEOUT = 20
# Upstox caps how many instruments one quotes call accepts; 100 stays well inside
# it and keeps a single failure from costing the whole universe.
_CHUNK = 100

# A name that has not traded today has a stale price, and a stale price produces
# a fake mover. Anything below this is treated as untraded rather than flat.
_MIN_VOLUME = 1


@dataclass
class Mover:
    symbol: str
    last_price: float
    change_pct: float
    net_change: float
    volume: float

    @property
    def turnover_cr(self) -> float:
        """Traded value in crore — the practical liquidity check."""
        return self.last_price * self.volume / 1e7


@dataclass
class MoversSnapshot:
    gainers: list[Mover] = field(default_factory=list)
    losers: list[Mover] = field(default_factory=list)
    scanned: int = 0
    advances: int = 0
    declines: int = 0

    @property
    def advance_decline_ratio(self) -> float:
        return round(self.advances / self.declines, 2) if self.declines else 0.0


def fo_universe(master: InstrumentMaster) -> list[str]:
    """Bare symbols of every F&O underlying that also has a cash listing.

    The handful without one are index products, which have no meaningful
    "gainer" reading against a stock list.
    """
    rows = master._load()  # noqa: SLF001 — same package, no public row accessor
    fo = {
        r.get("asset_symbol") for r in rows
        if r.get("segment") == "NSE_FO" and r.get("instrument_type") in ("CE", "PE")
    }
    equities = {
        r.get("trading_symbol") for r in rows if r.get("segment") == "NSE_EQ"
    }
    return sorted(s for s in fo if s and s in equities)


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i: i + size]


def fetch_market_movers(
    upstox_token: str,
    master: InstrumentMaster,
    *,
    symbols: list[str] | None = None,
    top_n: int = 10,
    min_turnover_cr: float = 1.0,
) -> MoversSnapshot:
    """Return the top *top_n* gainers and losers by percent change.

    *min_turnover_cr* drops names that moved on negligible value — the whole
    point of the F&O universe filter is defeated if a name that traded 40 lakh
    rupees tops the list. Requires a token: quotes are an authenticated endpoint.
    """
    if not upstox_token:
        logger.info("Movers: no Upstox token — skipping")
        return MoversSnapshot()

    symbols = symbols or fo_universe(master)
    keys: dict[str, str] = {}
    for sym in symbols:
        key = master.equity_key(sym)
        if key:
            keys[key] = sym

    movers: list[Mover] = []
    headers = {"Authorization": f"Bearer {upstox_token}", "Accept": "application/json"}

    for chunk in _chunks(list(keys), _CHUNK):
        try:
            resp = requests.get(
                _QUOTES_URL, params={"instrument_key": ",".join(chunk)},
                headers=headers, timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json().get("data") or {}
        except Exception as exc:
            # One bad chunk should cost 100 names, not the whole scan.
            logger.warning("Movers: quote chunk failed — %s", exc)
            continue

        for payload in data.values():
            mover = _to_mover(payload)
            if mover is not None:
                movers.append(mover)

    if not movers:
        logger.warning("Movers: no usable quotes returned")
        return MoversSnapshot(scanned=0)

    liquid = [m for m in movers if m.turnover_cr >= min_turnover_cr]
    ranked = sorted(liquid, key=lambda m: m.change_pct, reverse=True)

    snapshot = MoversSnapshot(
        gainers=ranked[:top_n],
        losers=ranked[-top_n:][::-1],
        scanned=len(movers),
        advances=sum(1 for m in movers if m.change_pct > 0),
        declines=sum(1 for m in movers if m.change_pct < 0),
    )
    logger.info(
        "Movers: %d scanned, %d liquid, %d↑ / %d↓",
        snapshot.scanned, len(liquid), snapshot.advances, snapshot.declines,
    )
    return snapshot


def _to_mover(payload: dict) -> Mover | None:
    """Build a Mover from one quote payload, or None when it is not usable.

    Change percent is derived from net_change rather than from ohlc.close: that
    field carries the CURRENT session's close, which during market hours simply
    equals the last price and would make every stock look unchanged.
    """
    try:
        symbol = str(payload.get("symbol") or "").strip()
        last = float(payload.get("last_price") or 0.0)
        net = float(payload.get("net_change") or 0.0)
        volume = float(payload.get("volume") or 0.0)
    except (TypeError, ValueError):
        return None

    if not symbol or last <= 0 or volume < _MIN_VOLUME:
        return None
    previous = last - net
    if previous <= 0:
        return None

    return Mover(
        symbol=symbol,
        last_price=round(last, 2),
        change_pct=round(net / previous * 100, 2),
        net_change=round(net, 2),
        volume=volume,
    )
