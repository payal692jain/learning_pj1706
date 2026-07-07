"""Tests for the VWAP Breakout strategy."""

import numpy as np
import pandas as pd
import pytest

from nifty_ai_agent.strategies.base import SignalType
from nifty_ai_agent.strategies.vwap_breakout import VWAPBreakoutStrategy


def _make_df(n: int = 10, closes: list[float] | None = None, vwap: float = 24000.0) -> pd.DataFrame:
    """Build a minimal DataFrame with a close series and a flat vwap column."""
    if closes is None:
        closes = [24000.0] * n
    else:
        n = len(closes)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    close = np.array(closes, dtype=float)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 10,
            "low": close - 10,
            "close": close,
            "volume": np.ones(n, dtype=int) * 100_000,
        },
        index=idx,
    )
    df["vwap"] = vwap
    return df


class TestVWAPBreakoutStrategy:
    def setup_method(self):
        self.strategy = VWAPBreakoutStrategy()

    def test_buy_ce_on_bullish_breakout(self):
        # Rising closes finishing well above VWAP
        closes = [23950, 23970, 24010, 24050, 24100]
        df = _make_df(closes=closes, vwap=24000.0)
        signal = self.strategy.generate_signal(df)
        assert signal.signal == SignalType.BUY_CE

    def test_buy_pe_on_bearish_breakdown(self):
        closes = [24050, 24030, 23990, 23950, 23900]
        df = _make_df(closes=closes, vwap=24000.0)
        signal = self.strategy.generate_signal(df)
        assert signal.signal == SignalType.BUY_PE

    def test_hold_when_price_hugs_vwap(self):
        closes = [24000, 24002, 23999, 24001, 24000]
        df = _make_df(closes=closes, vwap=24000.0)
        signal = self.strategy.generate_signal(df)
        assert signal.signal == SignalType.HOLD

    def test_hold_when_above_vwap_but_falling(self):
        # Above VWAP but momentum has turned down — no confirmed breakout
        closes = [24150, 24120, 24100, 24080, 24070]
        df = _make_df(closes=closes, vwap=24000.0)
        signal = self.strategy.generate_signal(df)
        assert signal.signal == SignalType.HOLD

    def test_hold_when_insufficient_bars(self):
        df = _make_df(closes=[24100, 24150], vwap=24000.0)
        signal = self.strategy.generate_signal(df)
        assert signal.signal == SignalType.HOLD

    def test_confidence_in_range(self):
        closes = [23950, 23970, 24010, 24050, 24100]
        df = _make_df(closes=closes, vwap=24000.0)
        signal = self.strategy.generate_signal(df)
        assert 0 <= signal.confidence <= 100

    def test_reason_is_string(self):
        closes = [23950, 23970, 24010, 24050, 24100]
        df = _make_df(closes=closes, vwap=24000.0)
        signal = self.strategy.generate_signal(df)
        assert isinstance(signal.reason, str)
        assert len(signal.reason) > 0

    def test_strategy_name(self):
        closes = [23950, 23970, 24010, 24050, 24100]
        df = _make_df(closes=closes, vwap=24000.0)
        signal = self.strategy.generate_signal(df)
        assert signal.strategy == "VWAP_Breakout"

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"close": [24000.0]})
        with pytest.raises(ValueError, match="missing columns"):
            self.strategy.generate_signal(df)

    def test_stronger_breakout_boosts_confidence(self):
        mild = _make_df(closes=[23980, 23990, 24000, 24010, 24020], vwap=24000.0)
        strong = _make_df(closes=[23900, 23950, 24050, 24150, 24300], vwap=24000.0)
        s_mild = self.strategy.generate_signal(mild)
        s_strong = self.strategy.generate_signal(strong)
        assert s_strong.confidence >= s_mild.confidence
