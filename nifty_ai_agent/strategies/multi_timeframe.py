"""Read the same instrument on several timeframes at once.

One timeframe is one opinion. An index that looks bullish on 5-minute bars and
bearish on 60-minute ones is not a trade, it is noise arguing with trend — and
the backtest says so plainly: the strategy stack has no measurable edge on
5-minute bars (expectancy within ±0.02% of zero) and a positive one on 60-minute
bars (+0.090%). Reading several timeframes together is how that gets used
without throwing away the responsiveness of the fast ones.

Bars are resampled locally from a single fetch rather than pulled per timeframe.
Four API calls per symbol per cycle would be wasteful, and worse, they would not
be guaranteed to describe the same instant.
"""

import logging
from dataclasses import dataclass
from datetime import time as dt_time

import pandas as pd

from nifty_ai_agent.strategies.base import BaseStrategy, SignalType
from nifty_ai_agent.strategies.consensus import Consensus, build_consensus
from nifty_ai_agent.strategies.pipeline import DEFAULT_STRATEGIES, compute_all_indicators

logger = logging.getLogger(__name__)

# Indices move fast enough that the quick timeframes carry information; the slow
# ones say whether it is worth acting on. Stocks are scanned on 30m only — the
# scan covers 50 names every cycle, and the faster reads did not survive testing.
INDEX_TIMEFRAMES: list[str] = ["5m", "15m", "30m", "60m"]
STOCK_TIMEFRAMES: list[str] = ["30m"]

_RULES = {"5m": "5min", "15m": "15min", "30m": "30min", "60m": "60min"}
_AGG = {"open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"}

# Enough bars for EMA50 plus a usable warm-up. Below this an indicator is mostly
# seeded from its own initial value and the read means little.
_MIN_BARS = 60


@dataclass
class TimeframeRead:
    """One timeframe's verdict on the instrument."""
    timeframe: str
    consensus: Consensus | None      # None when there were too few bars
    bars: int

    @property
    def signal(self) -> SignalType:
        return self.consensus.signal if self.consensus else SignalType.HOLD

    @property
    def confidence(self) -> int:
        return self.consensus.confidence if self.consensus else 0

    @property
    def is_usable(self) -> bool:
        return self.consensus is not None


@dataclass
class MultiTimeframeRead:
    """Every timeframe's verdict, plus what they agree on."""
    reads: list[TimeframeRead]

    @property
    def usable(self) -> list[TimeframeRead]:
        return [r for r in self.reads if r.is_usable]

    @property
    def votes(self) -> dict[SignalType, int]:
        tally: dict[SignalType, int] = {}
        for r in self.usable:
            tally[r.signal] = tally.get(r.signal, 0) + 1
        return tally

    def agreement(self) -> tuple[SignalType, int]:
        """The most-supported direction and how many timeframes back it.

        HOLD is not a direction: a timeframe with no opinion neither supports nor
        opposes a trade, so it is excluded from the winner rather than allowed to
        win by abstention.
        """
        directional = {
            s: n for s, n in self.votes.items()
            if s in (SignalType.BUY_CE, SignalType.BUY_PE)
        }
        if not directional:
            return SignalType.HOLD, 0
        signal = max(directional, key=lambda s: directional[s])
        return signal, directional[signal]

    def is_conflicted(self) -> bool:
        """True when timeframes point in opposite directions at the same time."""
        v = self.votes
        return bool(v.get(SignalType.BUY_CE, 0) and v.get(SignalType.BUY_PE, 0))

    def summary(self) -> str:
        """Compact per-timeframe line, e.g. '5m PE62 · 15m -- · 30m CE58'."""
        parts = []
        for r in self.reads:
            if not r.is_usable:
                parts.append(f"{r.timeframe} n/a")
                continue
            mark = {"BUY_CE": "CE", "BUY_PE": "PE"}.get(r.signal.value, "--")
            parts.append(
                f"{r.timeframe} {mark}" + (f"{r.confidence}" if mark != "--" else "")
            )
        return " · ".join(parts)


def gated_signal(
    read: MultiTimeframeRead, entry_tf: str, trend_tf: str = "60m"
) -> tuple[SignalType, str]:
    """Return *(signal, reason)* for an entry on *entry_tf* confirmed by *trend_tf*.

    The entry timeframe decides direction; the trend timeframe decides whether to
    act on it at all. Measured over 51,244 five-minute bars across 12 symbols, the
    filter roughly halves trade count while raising both hit rate and expectancy:

        5m alone            2915 trades   33% hit   +0.003%
        5m + 60m agreeing    875 trades   43% hit   +0.072%

    Note the average win and loss barely move (+0.61/-0.30 → +0.60/-0.32). The
    filter is not trading payoff for frequency the way a target change does — it
    removes entries that were losing, which is what a real filter looks like.

    Falls back to the entry timeframe alone when the trend read is unavailable:
    a missing slow read is missing information, not permission to block every
    trade, and blocking everything would be indistinguishable from a dead scan.
    """
    by_tf = {r.timeframe: r for r in read.reads}
    entry = by_tf.get(entry_tf)
    if entry is None or not entry.is_usable or entry.signal == SignalType.HOLD:
        return SignalType.HOLD, "no entry signal"

    trend = by_tf.get(trend_tf)
    if trend is None or not trend.is_usable:
        return entry.signal, f"{entry_tf} only — no {trend_tf} read available"

    if trend.signal == entry.signal:
        return entry.signal, f"{entry_tf} confirmed by {trend_tf}"
    if trend.signal == SignalType.HOLD:
        return SignalType.HOLD, f"{trend_tf} is flat — not confirming {entry_tf}"
    return SignalType.HOLD, f"{trend_tf} opposes {entry_tf} — conflicted"


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Aggregate *df* up to *timeframe*. Returns it unchanged for an unknown rule.

    Bins align to the wall clock (09:00, 09:30, …), matching how the Upstox
    provider already resamples so indicators stay consistent across the codebase.
    One consequence worth knowing: NSE opens at 09:15, so the first bin of each
    session is a partial bar — three 5-minute bars in a 30-minute slot — and its
    range and volume read correspondingly low.
    """
    rule = _RULES.get(timeframe)
    if rule is None:
        logger.warning("Unknown timeframe %r — leaving bars unchanged", timeframe)
        return df
    out = df.resample(rule).agg(_AGG).dropna(how="any")
    out.index.name = "datetime"
    return out


def read_timeframes(
    base: pd.DataFrame,
    timeframes: list[str],
    *,
    now: dt_time,
    intraday: bool = True,
    strategies: list[BaseStrategy] | None = None,
) -> MultiTimeframeRead:
    """Run the strategy stack over *base* resampled to each of *timeframes*.

    *base* must be at or finer than the shortest timeframe requested — bars can
    be aggregated up but not split down.
    """
    strategies = strategies or DEFAULT_STRATEGIES
    reads: list[TimeframeRead] = []

    for tf in timeframes:
        try:
            bars = resample_ohlcv(base, tf)
            if len(bars) < _MIN_BARS:
                reads.append(TimeframeRead(tf, None, len(bars)))
                continue
            df = compute_all_indicators(bars)
            signals = [s.generate_signal(df) for s in strategies]
            reads.append(TimeframeRead(
                tf, build_consensus(signals, now=now, intraday=intraday), len(bars),
            ))
        except Exception as exc:
            logger.warning("Timeframe %s failed: %s", tf, exc)
            reads.append(TimeframeRead(tf, None, 0))

    return MultiTimeframeRead(reads=reads)
