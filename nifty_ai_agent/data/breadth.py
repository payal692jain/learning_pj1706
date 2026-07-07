"""Real-time breadth check — top 10 NIFTY 50 heavyweight stocks.

Advances/declines among the highest-weight index constituents are used
to confirm or contradict the intraday EMA-crossover signal before it
fires.  Bullish signal + majority heavyweights declining = lower
confidence.  Bearish signal + majority advancing = lower confidence.
"""

import logging
from dataclasses import dataclass

import yfinance as yf

logger = logging.getLogger(__name__)

# Top 10 NIFTY 50 stocks by approximate index weight (2025)
HEAVYWEIGHT_SYMBOLS: list[str] = [
    "HDFCBANK.NS",    # ~13%
    "RELIANCE.NS",    # ~10%
    "ICICIBANK.NS",   # ~8%
    "INFY.NS",        # ~6%
    "TCS.NS",         # ~4%
    "LT.NS",          # ~4%
    "KOTAKBANK.NS",   # ~3.5%
    "AXISBANK.NS",    # ~3%
    "SBIN.NS",        # ~3%
    "BHARTIARTL.NS",  # ~2.5%
]

_ADVANCE_THRESHOLD = 0.15   # % above prev close → advancing
_DECLINE_THRESHOLD = -0.15  # % below prev close → declining


@dataclass
class BreadthSnapshot:
    advancing: int
    declining: int
    unchanged: int
    total: int
    score: float        # (advancing − declining) / total; range −1 to +1
    bias: str           # "BULLISH", "BEARISH", or "NEUTRAL"
    leaders: list[str]  # short names of advancing heavyweights
    laggards: list[str] # short names of declining heavyweights


def fetch_realtime_breadth() -> BreadthSnapshot:
    """Return live advance/decline for the top 10 NIFTY 50 heavyweights.

    Uses yfinance fast_info (last_price + previous_close) — one lightweight
    request per ticker.  Falls back to a neutral snapshot on total failure.
    """
    leaders: list[str] = []
    laggards: list[str] = []
    flat: list[str] = []

    for sym in HEAVYWEIGHT_SYMBOLS:
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
            logger.debug("Breadth: skipping %s — %s", sym, exc)

    total = len(leaders) + len(laggards) + len(flat)
    if total == 0:
        logger.warning("Breadth: no data returned — using neutral fallback")
        return BreadthSnapshot(
            advancing=0, declining=0, unchanged=0,
            total=0, score=0.0, bias="NEUTRAL",
            leaders=[], laggards=[],
        )

    score = round((len(leaders) - len(laggards)) / total, 2)
    bias = "BULLISH" if score > 0.2 else ("BEARISH" if score < -0.2 else "NEUTRAL")

    logger.info(
        "Breadth: %d↑ %d↓ %d→  score=%.2f  bias=%s  leaders=%s  laggards=%s",
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
