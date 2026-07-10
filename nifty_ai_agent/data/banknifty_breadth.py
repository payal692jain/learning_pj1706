"""Real-time breadth check — top 10 NIFTY BANK heavyweight stocks.

Used to confirm or contradict the intraday signal for BANKNIFTY, mirroring the
same logic used for NIFTY (breadth.py) and SENSEX (sensex_breadth.py). Unlike
those two broad-market indices, BANKNIFTY is a single-sector index, so its
breadth is dominated by a handful of large private/PSU banks.
"""

import logging

import yfinance as yf

from nifty_ai_agent.data.breadth import BreadthSnapshot

logger = logging.getLogger(__name__)

# Top 10 NIFTY BANK constituents by approximate index weight (2025)
BANKNIFTY_HEAVYWEIGHT_SYMBOLS: list[str] = [
    "HDFCBANK.NS",     # ~28%
    "ICICIBANK.NS",    # ~24%
    "KOTAKBANK.NS",    # ~11%
    "AXISBANK.NS",     # ~10%
    "SBIN.NS",         # ~10%
    "INDUSINDBK.NS",   # ~4%
    "BANKBARODA.NS",   # ~3%
    "PNB.NS",          # ~3%
    "AUBANK.NS",       # ~2.5%
    "FEDERALBNK.NS",   # ~2.5%
]

_ADVANCE_THRESHOLD = 0.15
_DECLINE_THRESHOLD = -0.15


def fetch_banknifty_breadth() -> BreadthSnapshot:
    """Return live advance/decline for the top 10 BANKNIFTY heavyweight stocks."""
    leaders: list[str] = []
    laggards: list[str] = []
    flat: list[str] = []

    for sym in BANKNIFTY_HEAVYWEIGHT_SYMBOLS:
        try:
            info = yf.Ticker(sym).fast_info
            price = float(info.last_price)
            prev = float(info.previous_close)
            if prev <= 0:
                continue
            chg_pct = (price - prev) / prev * 100
            name = sym.replace(".NS", "")
            if chg_pct > _ADVANCE_THRESHOLD:
                leaders.append(name)
            elif chg_pct < _DECLINE_THRESHOLD:
                laggards.append(name)
            else:
                flat.append(name)
        except Exception as exc:
            logger.debug("BANKNIFTY breadth: skipping %s — %s", sym, exc)

    total = len(leaders) + len(laggards) + len(flat)
    if total == 0:
        logger.warning("BANKNIFTY breadth: no data returned — using neutral fallback")
        return BreadthSnapshot(
            advancing=0, declining=0, unchanged=0,
            total=0, score=0.0, bias="NEUTRAL",
            leaders=[], laggards=[],
        )

    score = round((len(leaders) - len(laggards)) / total, 2)
    bias = "BULLISH" if score > 0.2 else ("BEARISH" if score < -0.2 else "NEUTRAL")
    logger.info(
        "BANKNIFTY breadth: %d↑ %d↓ %d→  score=%.2f  bias=%s  leaders=%s  laggards=%s",
        len(leaders), len(laggards), len(flat), score, bias, leaders, laggards,
    )
    return BreadthSnapshot(
        advancing=len(leaders),
        declining=len(laggards),
        unchanged=len(flat),
        total=total,
        score=score,
        bias=bias,
        leaders=leaders,
        laggards=laggards,
    )
