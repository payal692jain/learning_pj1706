"""Walk-forward backtest of the consensus stack over historical bars.

Every tuning decision in this project has so far been an assertion. This makes
them testable: replay real bars, generate the same signals the live agent would
have generated, simulate the entry against its own stop and target, and count
what happened.

Two correctness rules this engine is built around:

1. **No look-ahead.** Indicators are causal (EMA/RSI/MACD/ATR/VWAP/Supertrend
   all depend only on bars at or before the current one), so they are computed
   once over the full series and then *sliced* — the value at bar i is identical
   to recomputing over bars[0:i], but O(n) instead of O(n²). The strategies only
   ever see `df.iloc[:i+1]`, so they cannot read a bar that had not printed.

2. **The stop is checked before the target within a bar.** A single OHLC bar that
   spans both levels does not say which came first. Assuming the target would be
   a backtest that flatters itself; assuming the stop is the conservative reading
   and the one that survives contact with a real fill.
"""

import logging
from dataclasses import dataclass, field
from datetime import time as dt_time

import pandas as pd

from nifty_ai_agent.risk.calculator import RiskCalculator
from nifty_ai_agent.strategies.base import BaseStrategy, SignalType
from nifty_ai_agent.strategies.consensus import build_consensus
from nifty_ai_agent.strategies.pipeline import DEFAULT_STRATEGIES, compute_all_indicators

logger = logging.getLogger(__name__)

_REQUIRED = ["ema_20", "ema_50", "rsi", "atr"]

# Conviction ranking, so a gate can be expressed as "STRONG only" / "MODERATE+".
_CONVICTION_RANK = {"WEAK": 1, "MODERATE": 2, "STRONG": 3}

WIN, LOSS, TIMEOUT = "WIN", "LOSS", "TIMEOUT"


@dataclass
class Trade:
    symbol: str
    entry_index: int
    entry_price: float
    signal: str
    conviction: str
    confidence: int
    stop_loss: float
    target: float
    exit_price: float = 0.0
    exit_index: int = 0
    outcome: str = ""
    bars_held: int = 0

    @property
    def pnl_pct(self) -> float:
        """Underlying move in the position's favour, as a percent of entry."""
        if not self.entry_price:
            return 0.0
        raw = (self.exit_price - self.entry_price) / self.entry_price * 100
        return raw if self.signal == SignalType.BUY_CE.value else -raw


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    bars_tested: int = 0
    signals_seen: int = 0        # actionable consensus readings, pre-gating

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.outcome == WIN)

    @property
    def losses(self) -> int:
        return sum(1 for t in self.trades if t.outcome == LOSS)

    @property
    def timeouts(self) -> int:
        return sum(1 for t in self.trades if t.outcome == TIMEOUT)

    @property
    def hit_rate(self) -> float | None:
        decided = self.wins + self.losses
        return None if not decided else self.wins / decided

    @property
    def expectancy_pct(self) -> float | None:
        """Average per-trade return on the underlying, in percent.

        This is the number that decides whether a change helped. A hit rate can
        rise while expectancy falls (smaller wins, bigger losses), so tuning
        against hit rate alone is how a strategy is made worse with confidence.
        Timeouts count: an exit at the close is a real outcome, not a non-event.
        """
        if not self.trades:
            return None
        return sum(t.pnl_pct for t in self.trades) / len(self.trades)

    @property
    def avg_win_pct(self) -> float:
        wins = [t.pnl_pct for t in self.trades if t.outcome == WIN]
        return sum(wins) / len(wins) if wins else 0.0

    @property
    def avg_loss_pct(self) -> float:
        losses = [t.pnl_pct for t in self.trades if t.outcome == LOSS]
        return sum(losses) / len(losses) if losses else 0.0


def _simulate_exit(
    df: pd.DataFrame, entry_index: int, trade: Trade, max_bars: int,
) -> Trade:
    """Walk forward from the entry until a level is hit or the horizon expires."""
    bullish = trade.signal == SignalType.BUY_CE.value
    last = min(entry_index + max_bars, len(df) - 1)

    for i in range(entry_index + 1, last + 1):
        high = float(df["high"].iloc[i])
        low = float(df["low"].iloc[i])

        hit_stop = low <= trade.stop_loss if bullish else high >= trade.stop_loss
        hit_target = high >= trade.target if bullish else low <= trade.target

        # Stop first: a bar spanning both is ambiguous, and the pessimistic read
        # is the only one that does not quietly inflate the result.
        if hit_stop:
            trade.exit_price, trade.outcome = trade.stop_loss, LOSS
        elif hit_target:
            trade.exit_price, trade.outcome = trade.target, WIN
        else:
            continue
        trade.exit_index, trade.bars_held = i, i - entry_index
        return trade

    trade.exit_price = float(df["close"].iloc[last])
    trade.outcome = TIMEOUT
    trade.exit_index, trade.bars_held = last, last - entry_index
    return trade


def backtest_symbol(
    symbol: str,
    hist: pd.DataFrame,
    *,
    atr_sl_multiplier: float = 1.5,
    min_rr: float = 2.0,
    min_conviction: str = "WEAK",
    min_confidence: int = 0,
    max_bars_held: int = 75,
    warmup: int = 60,
    strategies: list[BaseStrategy] | None = None,
    now: dt_time = dt_time(11, 0),
    one_trade_at_a_time: bool = True,
    lookback: int = 200,
    htf_rule: str | None = None,
) -> BacktestResult:
    """Replay *hist* bar by bar and score every entry the stack would have taken.

    *max_bars_held* is the horizon before an unresolved trade is closed at market
    — 75 five-minute bars is roughly one session. *one_trade_at_a_time* mirrors the
    live position tracker, which refuses a second position in the same underlying.

    *lookback* caps how many trailing bars each strategy sees. Indicators are
    already computed over the full series, so trimming the window changes no
    indicator value — but it turns the per-bar cost from O(n) into O(1). Without
    it, OpeningRangeBreakout's per-call `df.index.date` scan over an ever-growing
    slice makes the whole run quadratic. 200 bars comfortably covers a full
    5-minute session (75 bars) plus every strategy's lookback.

    *htf_rule* ("30min", "60min") adds a higher-timeframe filter: an entry is
    only taken when the slower timeframe agrees with it. The higher-timeframe
    consensus is computed once per slow bar and then matched to each fast bar by
    the *previous* slow close — using the slow bar the fast bar sits inside would
    read a candle that has not finished forming, which is look-ahead.
    """
    strategies = strategies or DEFAULT_STRATEGIES
    result = BacktestResult()

    df = compute_all_indicators(hist)
    if len(df) <= warmup + 2:
        return result

    risk = RiskCalculator(
        max_risk_pct=100.0,       # sizing is not what is under test here
        min_rr=min_rr,
        atr_sl_multiplier=atr_sl_multiplier,
    )
    floor = _CONVICTION_RANK.get(min_conviction.upper(), 1)
    busy_until = -1

    htf_signal = _htf_series(df, htf_rule, strategies, now) if htf_rule else None

    for i in range(warmup, len(df) - 1):
        result.bars_tested += 1
        if one_trade_at_a_time and i <= busy_until:
            continue

        window = df.iloc[max(0, i + 1 - lookback) : i + 1]
        if window[_REQUIRED].iloc[-1].isna().any():
            continue

        try:
            signals = [s.generate_signal(window) for s in strategies]
            consensus = build_consensus(signals, now=now, intraday=False)
        except Exception as exc:
            logger.debug("%s bar %d: signal generation failed — %s", symbol, i, exc)
            continue

        if not consensus.is_actionable:
            continue
        result.signals_seen += 1

        if _CONVICTION_RANK.get(consensus.conviction.upper(), 0) < floor:
            continue
        if consensus.confidence < min_confidence:
            continue
        if htf_signal is not None:
            slow = htf_signal.get(df.index[i])
            if slow is None or slow is not consensus.signal:
                continue

        entry = float(df["close"].iloc[i])
        params = risk.calculate(consensus.signal, entry, float(df["atr"].iloc[i]))
        if not params.stop_loss or not params.target:
            continue

        trade = _simulate_exit(
            df, i,
            Trade(
                symbol=symbol, entry_index=i, entry_price=entry,
                signal=consensus.signal.value, conviction=consensus.conviction,
                confidence=consensus.confidence,
                stop_loss=params.stop_loss, target=params.target,
            ),
            max_bars_held,
        )
        result.trades.append(trade)
        busy_until = trade.exit_index

    return result


def _htf_series(
    df: pd.DataFrame, rule: str, strategies: list[BaseStrategy], now: dt_time,
) -> dict:
    """Map every fast-bar timestamp to the higher timeframe's signal at that moment.

    The slow bar a fast bar sits *inside* has not closed yet, so using it would
    let the backtest see the rest of that candle. Each fast bar is therefore
    matched to the last slow bar that had already closed — the same information a
    live run would have had.
    """
    from nifty_ai_agent.strategies.multi_timeframe import resample_ohlcv

    slow = compute_all_indicators(
        resample_ohlcv(df[["open", "high", "low", "close", "volume"]], rule.replace("min", "m"))
    )
    if len(slow) < 60:
        return {}

    decided: list[tuple] = []
    for j in range(50, len(slow)):
        window = slow.iloc[max(0, j - 200): j + 1]
        if window[_REQUIRED].iloc[-1].isna().any():
            continue
        try:
            signals = [s.generate_signal(window) for s in strategies]
            decided.append((slow.index[j], build_consensus(
                signals, now=now, intraday=False).signal))
        except Exception:
            continue

    if not decided:
        return {}

    stamps = [t for t, _ in decided]
    lookup: dict = {}
    cursor = -1
    for ts in df.index:
        # Advance only past slow bars that CLOSED strictly before this fast bar.
        while cursor + 1 < len(stamps) and stamps[cursor + 1] < ts:
            cursor += 1
        if cursor >= 0:
            lookup[ts] = decided[cursor][1]
    return lookup


def backtest_universe(
    histories: dict[str, pd.DataFrame], **kwargs
) -> BacktestResult:
    """Run backtest_symbol over every symbol and merge the results."""
    merged = BacktestResult()
    for symbol, hist in histories.items():
        try:
            one = backtest_symbol(symbol, hist, **kwargs)
        except Exception as exc:
            logger.warning("Backtest failed for %s: %s", symbol, exc)
            continue
        merged.trades.extend(one.trades)
        merged.bars_tested += one.bars_tested
        merged.signals_seen += one.signals_seen
    return merged


def format_result(label: str, result: BacktestResult) -> str:
    """One-line summary of a backtest run, for comparing configurations."""
    rate = result.hit_rate
    exp = result.expectancy_pct
    return (
        f"{label:<28} trades {len(result.trades):>4}  "
        f"hit {('n/a' if rate is None else f'{rate * 100:.0f}%'):>4}  "
        f"exp {('n/a' if exp is None else f'{exp:+.3f}%'):>8}  "
        f"W {result.avg_win_pct:+.2f}%  L {result.avg_loss_pct:+.2f}%  "
        f"timeouts {result.timeouts:>3}"
    )
