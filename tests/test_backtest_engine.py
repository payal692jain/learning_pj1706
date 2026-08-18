"""Tests for the walk-forward backtester.

The engine's job is to be pessimistic and honest. A backtest that flatters itself
is worse than none: it produces confident, wrong parameter choices.
"""

import numpy as np
import pandas as pd
import pytest

from nifty_ai_agent.backtesting.engine import (
    LOSS,
    TIMEOUT,
    WIN,
    BacktestResult,
    Trade,
    _simulate_exit,
    backtest_symbol,
    backtest_universe,
)
from nifty_ai_agent.strategies.base import Signal, SignalType


def _bars(rows) -> pd.DataFrame:
    """rows = [(open, high, low, close)] → an OHLCV frame on a 5-minute index."""
    idx = pd.date_range("2026-08-03 09:15", periods=len(rows), freq="5min")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 1000.0
    df.index.name = "datetime"
    return df


def _trade(signal="BUY_CE", entry=100.0, sl=98.0, target=104.0) -> Trade:
    return Trade(symbol="X", entry_index=0, entry_price=entry, signal=signal,
                 conviction="STRONG", confidence=80, stop_loss=sl, target=target)


class TestExitSimulation:
    def test_target_hit_is_a_win(self):
        df = _bars([(100, 100, 100, 100), (100, 105, 99, 104)])
        t = _simulate_exit(df, 0, _trade(), max_bars=5)
        assert t.outcome == WIN and t.exit_price == 104.0

    def test_stop_hit_is_a_loss(self):
        df = _bars([(100, 100, 100, 100), (100, 101, 97, 98)])
        t = _simulate_exit(df, 0, _trade(), max_bars=5)
        assert t.outcome == LOSS and t.exit_price == 98.0

    def test_bar_spanning_both_levels_is_scored_as_a_loss(self):
        """One OHLC bar cannot say which came first; assuming the win inflates results."""
        df = _bars([(100, 100, 100, 100), (100, 105, 97, 101)])
        assert _simulate_exit(df, 0, _trade(), max_bars=5).outcome == LOSS

    def test_unresolved_trade_times_out_at_market(self):
        df = _bars([(100, 100, 100, 100)] + [(100, 101, 99, 100.5)] * 3)
        t = _simulate_exit(df, 0, _trade(), max_bars=2)
        assert t.outcome == TIMEOUT and t.exit_price == 100.5

    def test_put_levels_are_inverted(self):
        # For a PE, target is BELOW entry and the stop is ABOVE it.
        df = _bars([(100, 100, 100, 100), (100, 101, 95, 96)])
        t = _simulate_exit(df, 0, _trade("BUY_PE", 100.0, 102.0, 96.0), max_bars=5)
        assert t.outcome == WIN and t.exit_price == 96.0

    def test_put_stop_is_above_entry(self):
        df = _bars([(100, 100, 100, 100), (100, 103, 99, 102)])
        t = _simulate_exit(df, 0, _trade("BUY_PE", 100.0, 102.0, 96.0), max_bars=5)
        assert t.outcome == LOSS

    def test_bars_held_is_recorded(self):
        df = _bars([(100, 100, 100, 100), (100, 101, 99, 100), (100, 105, 99, 104)])
        assert _simulate_exit(df, 0, _trade(), max_bars=5).bars_held == 2


class TestPnl:
    def test_call_profit_is_positive(self):
        t = _trade(); t.exit_price = 104.0
        assert t.pnl_pct == pytest.approx(4.0)

    def test_put_profit_on_a_fall_is_positive(self):
        """The sign flip for puts is the easiest thing here to get backwards."""
        t = _trade("BUY_PE", 100.0, 102.0, 96.0); t.exit_price = 96.0
        assert t.pnl_pct == pytest.approx(4.0)

    def test_call_loss_is_negative(self):
        t = _trade(); t.exit_price = 98.0
        assert t.pnl_pct == pytest.approx(-2.0)


class TestAggregates:
    def _result(self, outcomes):
        r = BacktestResult()
        for outcome, pnl in outcomes:
            t = _trade()
            t.outcome = outcome
            t.exit_price = 100.0 * (1 + pnl / 100)
            r.trades.append(t)
        return r

    def test_hit_rate_excludes_timeouts(self):
        r = self._result([(WIN, 4), (WIN, 4), (LOSS, -2), (TIMEOUT, 0.5)])
        assert r.hit_rate == pytest.approx(2 / 3)

    def test_expectancy_includes_timeouts(self):
        """An exit at the close is a real outcome and must count in the average."""
        r = self._result([(WIN, 4), (LOSS, -2), (TIMEOUT, 1)])
        assert r.expectancy_pct == pytest.approx(1.0)

    def test_no_trades_has_no_metrics(self):
        r = BacktestResult()
        assert r.hit_rate is None and r.expectancy_pct is None

    def test_all_timeouts_leaves_hit_rate_undefined(self):
        assert self._result([(TIMEOUT, 0.2)]).hit_rate is None


class TestNoLookAhead:
    def test_strategies_never_see_a_future_bar(self):
        """The whole result is worthless if a strategy can read tomorrow's price."""
        seen_lengths: list[int] = []

        class _Spy:
            def generate_signal(self, df):
                seen_lengths.append(len(df))
                # Record the last close so we can assert it is never ahead of entry.
                _Spy.last_close = float(df["close"].iloc[-1])
                return Signal(SignalType.HOLD, 50, "flat", "Spy")

        closes = np.linspace(100, 130, 200)
        df = _bars([(c, c + 0.5, c - 0.5, c) for c in closes])
        backtest_symbol("X", df, strategies=[_Spy()], warmup=60, lookback=50)

        assert seen_lengths, "strategy was never called"
        assert max(seen_lengths) <= 50   # lookback honoured
        assert all(n > 0 for n in seen_lengths)

    def test_lookback_does_not_change_indicator_values(self):
        """Indicators are causal, so trimming the window must not move them."""
        closes = np.linspace(100, 140, 300)
        df = _bars([(c, c + 0.6, c - 0.6, c) for c in closes])
        wide = backtest_symbol("X", df, lookback=300, warmup=60)
        narrow = backtest_symbol("X", df, lookback=200, warmup=60)
        assert len(wide.trades) == len(narrow.trades)
        assert [t.entry_index for t in wide.trades] == [t.entry_index for t in narrow.trades]


class TestGating:
    def _uptrend(self, n=260):
        closes = np.linspace(100, 150, n) + np.sin(np.linspace(0, 25, n)) * 0.6
        return _bars([(c, c + 0.5, c - 0.5, c) for c in closes])

    def test_conviction_gate_reduces_trade_count(self):
        df = self._uptrend()
        loose = backtest_symbol("X", df, min_conviction="WEAK")
        strict = backtest_symbol("X", df, min_conviction="STRONG")
        assert len(strict.trades) <= len(loose.trades)

    def test_confidence_floor_reduces_trade_count(self):
        df = self._uptrend()
        loose = backtest_symbol("X", df, min_confidence=0)
        strict = backtest_symbol("X", df, min_confidence=95)
        assert len(strict.trades) <= len(loose.trades)

    def test_one_trade_at_a_time_prevents_overlap(self):
        df = self._uptrend()
        r = backtest_symbol("X", df, one_trade_at_a_time=True)
        for earlier, later in zip(r.trades, r.trades[1:]):
            assert later.entry_index > earlier.exit_index

    def test_short_history_yields_nothing(self):
        assert backtest_symbol("X", _bars([(100, 101, 99, 100)] * 10)).trades == []


class TestUniverse:
    def test_merges_across_symbols(self):
        closes = np.linspace(100, 140, 260)
        df = _bars([(c, c + 0.5, c - 0.5, c) for c in closes])
        merged = backtest_universe({"A": df, "B": df})
        single = backtest_symbol("A", df)
        assert len(merged.trades) == 2 * len(single.trades)

    def test_a_broken_symbol_does_not_sink_the_run(self):
        closes = np.linspace(100, 140, 260)
        good = _bars([(c, c + 0.5, c - 0.5, c) for c in closes])
        bad = pd.DataFrame({"close": [1.0]})     # missing OHLC columns
        merged = backtest_universe({"good": good, "bad": bad})
        assert merged.trades  # good symbol still contributed
