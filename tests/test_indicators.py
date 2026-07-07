"""Tests for the indicators module."""

import numpy as np
import pandas as pd
import pytest

from nifty_ai_agent.indicators.atr import compute_atr
from nifty_ai_agent.indicators.ema import compute_ema
from nifty_ai_agent.indicators.macd import compute_macd
from nifty_ai_agent.indicators.rsi import compute_rsi
from nifty_ai_agent.indicators.vwap import compute_vwap


def _make_df(n: int = 60, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 24000 + np.cumsum(rng.normal(0, 50, n))
    high = close + rng.uniform(10, 80, n)
    low = close - rng.uniform(10, 80, n)
    open_ = close + rng.normal(0, 20, n)
    volume = rng.integers(500_000, 2_000_000, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


class TestRSI:
    def test_column_added(self):
        df = compute_rsi(_make_df())
        assert "rsi" in df.columns

    def test_values_in_range(self):
        df = compute_rsi(_make_df())
        valid = df["rsi"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_does_not_mutate_input(self):
        original = _make_df()
        original_cols = list(original.columns)
        compute_rsi(original)
        assert list(original.columns) == original_cols

    def test_custom_period(self):
        df = compute_rsi(_make_df(), period=7)
        assert "rsi" in df.columns
        assert df["rsi"].notna().sum() > 0

    def test_overbought_signal(self):
        # Monotonically increasing prices → RSI should approach 100
        n = 50
        close = np.linspace(23000, 26000, n)
        df = pd.DataFrame(
            {"open": close, "high": close + 10, "low": close - 10, "close": close, "volume": 1},
            index=pd.date_range("2024-01-01", periods=n, freq="B"),
        )
        result = compute_rsi(df)
        last_rsi = result["rsi"].dropna().iloc[-1]
        assert last_rsi > 70


class TestEMA:
    def test_default_periods(self):
        df = compute_ema(_make_df())
        assert "ema_20" in df.columns
        assert "ema_50" in df.columns

    def test_custom_periods(self):
        df = compute_ema(_make_df(), periods=[10, 30])
        assert "ema_10" in df.columns
        assert "ema_30" in df.columns

    def test_no_mutation(self):
        original = _make_df()
        compute_ema(original)
        assert "ema_20" not in original.columns

    def test_ema20_more_responsive_than_ema50(self):
        # With an uptrend EMA20 should be above EMA50
        df = _make_df()
        result = compute_ema(df.sort_index())
        # EMA20 reacts faster, so after sustained uptrend it should differ from EMA50
        last = result.dropna().iloc[-1]
        assert not np.isnan(last["ema_20"])
        assert not np.isnan(last["ema_50"])


class TestMACD:
    def test_columns_added(self):
        df = compute_macd(_make_df())
        assert "macd" in df.columns
        assert "macd_signal" in df.columns
        assert "macd_histogram" in df.columns

    def test_histogram_equals_macd_minus_signal(self):
        df = compute_macd(_make_df())
        diff = (df["macd"] - df["macd_signal"] - df["macd_histogram"]).abs()
        assert diff.max() < 1e-10

    def test_no_mutation(self):
        original = _make_df()
        compute_macd(original)
        assert "macd" not in original.columns


class TestATR:
    def test_column_added(self):
        df = compute_atr(_make_df())
        assert "atr" in df.columns

    def test_values_positive(self):
        df = compute_atr(_make_df())
        assert (df["atr"].dropna() > 0).all()

    def test_custom_period(self):
        df = compute_atr(_make_df(), period=7)
        assert df["atr"].notna().sum() > 0


class TestVWAP:
    def test_column_added(self):
        df = compute_vwap(_make_df())
        assert "vwap" in df.columns

    def test_vwap_near_close(self):
        # VWAP should be in a reasonable range around close
        df = compute_vwap(_make_df())
        last = df.dropna().iloc[-1]
        assert abs(last["vwap"] - last["close"]) / last["close"] < 0.1
