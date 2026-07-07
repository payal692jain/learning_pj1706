"""Real-time breadth check — top 10 BSE SENSEX heavyweight stocks.

Used to confirm or contradict the intraday EMA-crossover signal for SENSEX,
mirroring the same logic used for NIFTY in nifty_ai_agent/data/breadth.py.
The BSE SENSEX (BSE 30) heavyweights are a subset of Nifty 50, so yfinance
NSE symbols (.NS suffix) work for all of them.
"""

import logging

import yfinance as yf

from nifty_ai_agent.data.breadth import BreadthSnapshot

logger = logging.getLogger(__name__)

# Top 10 BSE SENSEX stocks by approximate index weight (2025)
SENSEX_HEAVYWEIGHT_SYMBOLS: list[str] = [
    "RELIANCE.NS",     # ~15%
    "TCS.NS",          # ~11%
    "HDFCBANK.NS",     # ~10%
    "INFY.NS",         # ~8%
    "ICICIBANK.NS",    # ~7%
    "BHARTIARTL.NS",   # ~5%
    "ITC.NS",          # ~4%
    "LT.NS",           # ~4%
    "SBIN.NS",         # ~4%
    "AXISBANK.NS",     # ~3%
]

_ADVANCE_THRESHOLD = 0.15
_DECLINE_THRESHOLD = -0.15


def fetch_sensex_breadth() -> BreadthSnapshot:
    """Return live advance/decline for the top 10 SENSEX heavyweight stocks."""
    leaders: list[str] = []
    laggards: list[str] = []
    flat: list[str] = []

    for sym in SENSEX_HEAVYWEIGHT_SYMBOLS:
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
            logger.debug("SENSEX breadth: skipping %s — %s", sym, exc)

    total = len(leaders) + len(laggards) + len(flat)
    if total == 0:
        logger.warning("SENSEX breadth: no data returned — using neutral fallback")
        return BreadthSnapshot(
            advancing=0, declining=0, unchanged=0,
            total=0, score=0.0, bias="NEUTRAL",
            leaders=[], laggards=[],
        )

    score = round((len(leaders) - len(laggards)) / total, 2)
    bias = "BULLISH" if score > 0.2 else ("BEARISH" if score < -0.2 else "NEUTRAL")
    logger.info(
        "SENSEX breadth: %d↑ %d↓ %d→  score=%.2f  bias=%s  leaders=%s  laggards=%s",
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
