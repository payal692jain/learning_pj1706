"""Tests for the strategy engine."""

import numpy as np
import pandas as pd
import pytest

from nifty_ai_agent.strategies.base import SignalType
from nifty_ai_agent.strategies.ema_crossover import EMACrossoverStrategy


def _make_df_with_indicators(
    n: int = 60,
    ema20_above_ema50: bool = True,
    rsi_value: float = 65.0,
) -> pd.DataFrame:
    """Build a minimal DataFrame with pre-set indicator values."""
    base = 24000.0
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = np.full(n, base)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 50,
            "low": close - 50,
            "close": close,
            "volume": np.ones(n, dtype=int) * 1_000_000,
        },
        index=idx,
    )
    if ema20_above_ema50:
        df["ema_20"] = base + 50
        df["ema_50"] = base - 50
    else:
        df["ema_20"] = base - 50
        df["ema_50"] = base + 50
    df["rsi"] = rsi_value
    return df


class TestEMACrossoverStrategy:
    def setup_method(self):
        self.strategy = EMACrossoverStrategy()

    def test_buy_ce_when_bullish(self):
        df = _make_df_with_indicators(ema20_above_ema50=True, rsi_value=65)
        signal = self.strategy.generate_signal(df)
        assert signal.signal == SignalType.BUY_CE

    def test_buy_pe_when_bearish(self):
        df = _make_df_with_indicators(ema20_above_ema50=False, rsi_value=35)
        signal = self.strategy.generate_signal(df)
        assert signal.signal == SignalType.BUY_PE

    def test_hold_when_rsi_neutral(self):
        df = _make_df_with_indicators(ema20_above_ema50=True, rsi_value=50)
        signal = self.strategy.generate_signal(df)
        assert signal.signal == SignalType.HOLD

    def test_hold_when_ema_neutral_rsi_overbought(self):
        df = _make_df_with_indicators(ema20_above_ema50=False, rsi_value=70)
        signal = self.strategy.generate_signal(df)
        assert signal.signal == SignalType.HOLD

    def test_confidence_in_range(self):
        df = _make_df_with_indicators(ema20_above_ema50=True, rsi_value=65)
        signal = self.strategy.generate_signal(df)
        assert 0 <= signal.confidence <= 100

    def test_reason_is_string(self):
        df = _make_df_with_indicators(ema20_above_ema50=True, rsi_value=65)
        signal = self.strategy.generate_signal(df)
        assert isinstance(signal.reason, str)
        assert len(signal.reason) > 0

    def test_strategy_name(self):
        df = _make_df_with_indicators(ema20_above_ema50=True, rsi_value=65)
        signal = self.strategy.generate_signal(df)
        assert signal.strategy == "EMA_Crossover"

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"close": [24000.0]})
        with pytest.raises(ValueError, match="missing columns"):
            self.strategy.generate_signal(df)

    def test_high_rsi_boosts_confidence(self):
        low_rsi = _make_df_with_indicators(ema20_above_ema50=True, rsi_value=61)
        high_rsi = _make_df_with_indicators(ema20_above_ema50=True, rsi_value=90)
        s_low = self.strategy.generate_signal(low_rsi)
        s_high = self.strategy.generate_signal(high_rsi)
        assert s_high.confidence >= s_low.confidence
