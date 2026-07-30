"""Shared indicator + strategy stack.

Both the index signal loop (main.py) and the stock scanner run the same six
strategies over the same indicator set. Keeping that definition in one place
means a strategy added here is picked up by both — and that a stock idea and an
index call are produced by identical machinery, not two drifting copies of it.
"""

import pandas as pd

from nifty_ai_agent.indicators.atr import compute_atr
from nifty_ai_agent.indicators.bollinger import compute_bollinger
from nifty_ai_agent.indicators.ema import compute_ema
from nifty_ai_agent.indicators.macd import compute_macd
from nifty_ai_agent.indicators.rsi import compute_rsi
from nifty_ai_agent.indicators.supertrend import compute_supertrend
from nifty_ai_agent.indicators.vwap import compute_vwap
from nifty_ai_agent.strategies.base import BaseStrategy
from nifty_ai_agent.strategies.bollinger_squeeze import BollingerSqueezeStrategy
from nifty_ai_agent.strategies.ema_crossover import EMACrossoverStrategy
from nifty_ai_agent.strategies.macd_momentum import MACDMomentumStrategy
from nifty_ai_agent.strategies.orb import OpeningRangeBreakoutStrategy
from nifty_ai_agent.strategies.supertrend import SupertrendStrategy
from nifty_ai_agent.strategies.vwap_breakout import VWAPBreakoutStrategy


def compute_all_indicators(hist: pd.DataFrame) -> pd.DataFrame:
    """Attach every indicator column the strategy engine reads to *hist*."""
    df = compute_ema(hist, periods=[20, 50])
    df = compute_rsi(df)
    df = compute_macd(df)
    df = compute_atr(df)
    df = compute_vwap(df)
    df = compute_supertrend(df)
    df = compute_bollinger(df)
    return df


def default_strategies() -> list[BaseStrategy]:
    """A fresh instance of every strategy in the standard book.

    Every strategy runs independently each cycle; the consensus engine folds
    their votes into one verdict.
    """
    return [
        EMACrossoverStrategy(),
        VWAPBreakoutStrategy(),
        SupertrendStrategy(),
        MACDMomentumStrategy(),
        OpeningRangeBreakoutStrategy(),
        BollingerSqueezeStrategy(),
    ]


# Shared, stateless instances — the strategies hold no per-symbol state, so one
# set can serve every index and every stock in a scan.
DEFAULT_STRATEGIES: list[BaseStrategy] = default_strategies()
