"""Tests for multi-timeframe reads and their agreement logic."""

import numpy as np
import pandas as pd
import pytest

from nifty_ai_agent.strategies.base import Signal, SignalType
from nifty_ai_agent.strategies.consensus import build_consensus
from nifty_ai_agent.strategies.multi_timeframe import (
    INDEX_TIMEFRAMES,
    STOCK_TIMEFRAMES,
    MultiTimeframeRead,
    TimeframeRead,
    read_timeframes,
    resample_ohlcv,
)


def _bars(n=400, start=100.0, end=140.0):
    idx = pd.date_range("2026-08-03 09:15", periods=n, freq="5min")
    close = np.linspace(start, end, n)
    df = pd.DataFrame(
        {"open": close - 0.1, "high": close + 0.4, "low": close - 0.4,
         "close": close, "volume": np.full(n, 1000.0)},
        index=idx,
    )
    df.index.name = "datetime"
    return df


def _read(tf, signal, confidence=60):
    if signal is None:
        return TimeframeRead(tf, None, 0)
    c = build_consensus(
        [Signal(signal, confidence, "r", "S")] * 3, now=pd.Timestamp("2026-08-03 11:00").time(),
        intraday=False,
    )
    return TimeframeRead(tf, c, 100)


class TestResample:
    def test_aggregates_ohlcv_correctly(self):
        """Each output bar is first-open / max-high / min-low / last-close / sum-volume."""
        df = _bars(n=12)
        out = resample_ohlcv(df, "30m")
        for stamp, row in out.iterrows():
            members = df[(df.index >= stamp) & (df.index < stamp + pd.Timedelta("30min"))]
            assert row["open"] == pytest.approx(members["open"].iloc[0])
            assert row["close"] == pytest.approx(members["close"].iloc[-1])
            assert row["high"] == pytest.approx(members["high"].max())
            assert row["low"] == pytest.approx(members["low"].min())
            assert row["volume"] == pytest.approx(members["volume"].sum())

    def test_bins_align_to_the_clock_not_the_first_bar(self):
        """NSE opens 09:15, so the day's first 30m bin holds only 3 five-minute bars.

        This matches how the Upstox provider already resamples, so indicators stay
        consistent across the codebase — but it does mean the opening bar of each
        session is short, and its range and volume read low as a result.
        """
        out = resample_ohlcv(_bars(n=12), "30m")
        assert out.index[0] == pd.Timestamp("2026-08-03 09:00")
        assert out["volume"].iloc[0] == pytest.approx(3000.0)   # 3 bars, not 6

    def test_bar_count_shrinks_as_timeframe_grows(self):
        df = _bars(n=480)
        counts = [len(resample_ohlcv(df, tf)) for tf in ("5m", "15m", "30m", "60m")]
        assert counts == sorted(counts, reverse=True)

    def test_unknown_timeframe_returns_input_unchanged(self):
        df = _bars(n=20)
        assert len(resample_ohlcv(df, "7m")) == len(df)


class TestAgreement:
    def test_unanimous_direction(self):
        r = MultiTimeframeRead([_read(tf, SignalType.BUY_CE) for tf in INDEX_TIMEFRAMES])
        assert r.agreement() == (SignalType.BUY_CE, 4)
        assert not r.is_conflicted()

    def test_majority_wins(self):
        r = MultiTimeframeRead([
            _read("5m", SignalType.BUY_PE), _read("15m", SignalType.BUY_CE),
            _read("30m", SignalType.BUY_CE), _read("60m", SignalType.BUY_CE),
        ])
        assert r.agreement() == (SignalType.BUY_CE, 3)

    def test_opposing_timeframes_are_flagged_as_conflicted(self):
        """Bullish fast bars against bearish slow ones is noise arguing with trend."""
        r = MultiTimeframeRead([
            _read("5m", SignalType.BUY_CE), _read("60m", SignalType.BUY_PE),
        ])
        assert r.is_conflicted()

    def test_hold_cannot_win_by_abstention(self):
        """A timeframe with no opinion neither supports nor opposes a trade."""
        r = MultiTimeframeRead([
            _read("5m", SignalType.HOLD), _read("15m", SignalType.HOLD),
            _read("30m", SignalType.HOLD), _read("60m", SignalType.BUY_CE),
        ])
        assert r.agreement() == (SignalType.BUY_CE, 1)

    def test_all_flat_has_no_direction(self):
        r = MultiTimeframeRead([_read(tf, SignalType.HOLD) for tf in INDEX_TIMEFRAMES])
        assert r.agreement() == (SignalType.HOLD, 0)

    def test_unusable_timeframes_are_excluded(self):
        r = MultiTimeframeRead([
            _read("5m", SignalType.BUY_CE), _read("60m", None),
        ])
        assert len(r.usable) == 1
        assert r.agreement() == (SignalType.BUY_CE, 1)

    def test_summary_names_every_timeframe(self):
        r = MultiTimeframeRead([
            _read("5m", SignalType.BUY_CE), _read("15m", SignalType.HOLD),
            _read("30m", None),
        ])
        s = r.summary()
        assert "5m CE" in s and "15m --" in s and "30m n/a" in s


class TestReadTimeframes:
    def test_reads_every_requested_timeframe(self):
        r = read_timeframes(_bars(n=1200), INDEX_TIMEFRAMES,
                            now=pd.Timestamp("2026-08-03 11:00").time())
        assert [x.timeframe for x in r.reads] == INDEX_TIMEFRAMES

    def test_thin_history_is_marked_unusable_not_guessed(self):
        """Too few bars means the indicators are mostly seeded from themselves."""
        r = read_timeframes(_bars(n=120), ["5m", "60m"],
                            now=pd.Timestamp("2026-08-03 11:00").time())
        by_tf = {x.timeframe: x for x in r.reads}
        assert by_tf["5m"].is_usable
        assert not by_tf["60m"].is_usable      # 120 5m bars = 10 hourly bars

    def test_a_sustained_uptrend_is_read_as_bullish_somewhere(self):
        r = read_timeframes(_bars(n=1200, start=100, end=180), INDEX_TIMEFRAMES,
                            now=pd.Timestamp("2026-08-03 11:00").time())
        signal, votes = r.agreement()
        assert signal == SignalType.BUY_CE and votes >= 1

    def test_stock_timeframes_are_the_slower_ones(self):
        assert STOCK_TIMEFRAMES == ["30m"]
        assert "5m" not in STOCK_TIMEFRAMES
