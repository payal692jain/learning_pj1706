"""GIFT Nifty — the overnight read on where NIFTY opens tomorrow.

GIFT Nifty (NSE International Exchange, formerly SGX Nifty) is a NIFTY futures
contract that trades roughly 21 hours a day, in two sessions:

    Session 1   06:30 – 15:40 IST   overlaps the Indian cash market
    Session 2   16:35 – 02:45 IST   runs overnight, through the US session

Session 2 is the one that matters for tomorrow: it prices in Wall Street's whole
day and the early Asian tape while NSE is shut. Session 1 from 06:30 is the final
pre-open read, ~2h45m before the cash market opens at 09:15.

The old NSE endpoint this project used (`/api/liveanalysis-giftnifty`) has been
retired and now 404s, and the yfinance fallback symbol (^NSEIFSC) is delisted —
so `fetch_gift_nifty()` had been silently returning None on every call, meaning
GIFT never actually reached the morning report or the intraday adjuster. This
module replaces it with the live NSE IX feed that the exchange's own site uses.

IMPLIED OPEN — the one calculation worth being careful about:

GIFT is a FUTURE, so it trades at a basis (carry premium) to NIFTY spot. Taking
its level as the implied open — "GIFT at 24,250 vs Nifty's 24,200 close means a
50-point gap up!" — double-counts that basis and systematically overstates the
gap. What actually carries information is GIFT's *percentage move*, which is
basis-neutral to first order, applied to NIFTY's previous close.
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time

import requests

logger = logging.getLogger(__name__)

_BASE = "https://www.nseix.com"
_TIMEOUT = 20
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": f"{_BASE}/markets/derivatives-watch",
}

# NSE IX trading sessions, IST.
_S1_OPEN, _S1_CLOSE = dt_time(6, 30), dt_time(15, 40)
_S2_OPEN, _S2_CLOSE = dt_time(16, 35), dt_time(2, 45)

# Gap size buckets, as a percentage of the previous close. Below FLAT the "gap" is
# noise that the first minute of trading erases.
_FLAT_PCT = 0.15
_LARGE_PCT = 0.75


@dataclass
class GiftNiftyQuote:
    price: float
    change: float          # points, vs GIFT's own previous close
    change_pct: float
    expiry: str            # front-month contract, e.g. "28-Jul-2026"
    timestamp: str         # exchange-stamped, e.g. "11-Jul-2026 02:44:59"
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    prev_close: float = 0.0
    session: str = "UNKNOWN"     # SESSION_1 / SESSION_2 / CLOSED
    status_text: str = ""        # the exchange's own words


@dataclass
class OpenOutlook:
    """What GIFT implies for the next NIFTY cash open."""
    gift: GiftNiftyQuote
    nifty_prev_close: float
    implied_open: float
    gap_points: float
    gap_pct: float

    @property
    def direction(self) -> str:
        if self.gap_pct > _FLAT_PCT:
            return "GAP_UP"
        if self.gap_pct < -_FLAT_PCT:
            return "GAP_DOWN"
        return "FLAT"

    @property
    def magnitude(self) -> str:
        size = abs(self.gap_pct)
        if size < _FLAT_PCT:
            return "FLAT"
        return "LARGE" if size >= _LARGE_PCT else "SMALL"

    @property
    def bucket(self) -> str:
        """The label used to look up this gap's historical base rate."""
        if self.direction == "FLAT":
            return "FLAT"
        return f"{self.magnitude}_{'UP' if self.gap_points > 0 else 'DOWN'}"


def current_session(now: dt_time) -> str:
    """Which NSE IX session *now* falls in (IST wall clock).

    Session 2 wraps past midnight, so it is the union of two clock ranges — a naive
    `open <= now <= close` test would call 20:00 "closed" and 02:00 "closed" both.
    """
    if _S1_OPEN <= now <= _S1_CLOSE:
        return "SESSION_1"
    if now >= _S2_OPEN or now <= _S2_CLOSE:
        return "SESSION_2"
    return "CLOSED"


def fetch_session_status() -> str:
    """The exchange's own market-status string, or '' if unreachable."""
    try:
        resp = requests.get(
            f"{_BASE}/api/derivatives-market-status", headers=_HEADERS, timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return str(resp.json().get("marketstatus", "")).strip()
    except Exception as exc:
        logger.debug("GIFT session status unavailable: %s", exc)
        return ""


def fetch_gift_nifty(now: dt_time | None = None) -> GiftNiftyQuote | None:
    """Front-month GIFT Nifty futures quote from NSE IX, or None if unreachable.

    The feed returns every index contract (and repeats rows), so the front month is
    selected by nearest expiry rather than by taking the first row.
    """
    try:
        resp = requests.get(
            f"{_BASE}/api/derivatives-watch",
            params={"inst_type1": "IDX", "type": "live"},
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        rows = resp.json().get("data", [])
    except Exception as exc:
        logger.warning("GIFT Nifty fetch failed: %s", exc)
        return None

    futures = [
        row for row in rows
        if row.get("INSTRUMENTTYPE") == "FUTIDX"
        and row.get("SYMBOL") == "NIFTY"
        and row.get("LASTPRICE")
    ]
    if not futures:
        logger.warning("GIFT Nifty: no NIFTY index futures in the feed")
        return None

    front = min(futures, key=lambda r: _parse_expiry(r.get("EXPIRYDATE")))
    price = float(front["LASTPRICE"])
    change = float(front.get("CHANGE") or 0.0)

    quote = GiftNiftyQuote(
        price=price,
        change=change,
        change_pct=float(front.get("PERCHANGE") or 0.0),
        expiry=str(front.get("EXPIRYDATE", "")),
        timestamp=str(front.get("TIMESTMP", "")),
        open=float(front.get("OPEN") or 0.0),
        high=float(front.get("HIGH") or 0.0),
        low=float(front.get("LOW") or 0.0),
        prev_close=price - change,
        session=current_session(now or datetime.now().time()),
        status_text=fetch_session_status(),
    )
    logger.info(
        "GIFT Nifty: %.1f (%+.2f%%) expiry=%s session=%s [%s]",
        quote.price, quote.change_pct, quote.expiry, quote.session, quote.timestamp,
    )
    return quote


def build_outlook(gift: GiftNiftyQuote, nifty_prev_close: float) -> OpenOutlook:
    """Imply tomorrow's NIFTY open from GIFT's percentage move.

    Uses GIFT's % change rather than its level: the level carries a futures basis
    that NIFTY spot does not, so differencing the two would book the carry premium
    as a phantom gap-up every single day.
    """
    implied = nifty_prev_close * (1 + gift.change_pct / 100)
    gap_points = implied - nifty_prev_close
    return OpenOutlook(
        gift=gift,
        nifty_prev_close=nifty_prev_close,
        implied_open=round(implied, 2),
        gap_points=round(gap_points, 2),
        gap_pct=round(gift.change_pct, 2),
    )


def _parse_expiry(value) -> date:
    """'28-Jul-2026' → date. Unparseable expiries sort last, never first."""
    try:
        return datetime.strptime(str(value), "%d-%b-%Y").date()
    except (TypeError, ValueError):
        return date.max
