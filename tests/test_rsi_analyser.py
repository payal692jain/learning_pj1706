"""Tests for the RSI analyser — zone, trend, divergence, confidence adjustment."""

import numpy as np
import pandas as pd
import pytest

from nifty_ai_agent.strategies.rsi_analyser import (
    RSIAnalysis,
    analyse_rsi,
    rsi_confidence_adjustment,
    _zone,
    _trend,
    _divergence,
)


def _df(rsi_values: list[float], close_values: list[float] | None = None) -> pd.DataFrame:
    """Build a minimal DataFrame with rsi and close columns."""
    n = len(rsi_values)
    if close_values is None:
        close_values = [100.0] * n
    return pd.DataFrame({"rsi": rsi_values, "close": close_values})


class TestZoneClassification:
    def test_below_30_deeply_oversold(self):
        assert _zone(25.0) == "DEEPLY_OVERSOLD"

    def test_30_to_40_oversold(self):
        assert _zone(35.0) == "OVERSOLD"

    def test_40_to_60_neutral(self):
        assert _zone(50.0) == "NEUTRAL"

    def test_60_to_70_overbought(self):
        assert _zone(65.0) == "OVERBOUGHT"

    def test_above_70_deeply_overbought(self):
        assert _zone(75.0) == "DEEPLY_OVERBOUGHT"

    def test_exact_boundaries(self):
        assert _zone(30.0) == "OVERSOLD"
        assert _zone(40.0) == "NEUTRAL"
        assert _zone(60.0) == "OVERBOUGHT"
        assert _zone(70.0) == "DEEPLY_OVERBOUGHT"


class TestRSITrend:
    def _make(self, start: float, end: float, n: int = 10) -> pd.DataFrame:
        rsi = list(np.linspace(start, end, n))
        return _df(rsi)

    def test_rising_trend(self):
        df = self._make(55.0, 68.0)
        assert _trend(df) == "RISING"

    def test_falling_trend(self):
        df = self._make(65.0, 50.0)
        assert _trend(df) == "FALLING"

    def test_flat_trend(self):
        df = _df([62.0] * 10)
        assert _trend(df) == "FLAT"


class TestRSIDivergence:
    def test_bearish_divergence(self):
        # Price higher high, RSI lower high
        prices = [100.0] * 10 + [103.0]   # price went up
        rsi_vals = [65.0] * 10 + [58.0]   # RSI went down
        df = _df(rsi_vals, prices)
        assert _divergence(df) == "BEARISH_DIV"

    def test_bullish_divergence(self):
        # Price lower low, RSI higher low
        prices = [100.0] * 10 + [97.0]    # price went down
        rsi_vals = [35.0] * 10 + [42.0]   # RSI went up
        df = _df(rsi_vals, prices)
        assert _divergence(df) == "BULLISH_DIV"

    def test_no_divergence_flat(self):
        df = _df([50.0] * 15, [100.0] * 15)
        assert _divergence(df) == "NONE"

    def test_no_divergence_aligned(self):
        # Price up, RSI also up → no bearish divergence
        prices = [100.0] * 10 + [103.0]
        rsi_vals = [60.0] * 10 + [67.0]
        df = _df(rsi_vals, prices)
        assert _divergence(df) == "NONE"


class TestAnalyseRSI:
    def test_returns_rsi_analysis(self):
        df = _df([65.0] * 20)
        result = analyse_rsi(df)
        assert isinstance(result, RSIAnalysis)

    def test_neutral_on_missing_column(self):
        df = pd.DataFrame({"close": [100.0] * 10})
        result = analyse_rsi(df)
        assert result.zone == "NEUTRAL"

    def test_neutral_on_insufficient_data(self):
        df = _df([65.0] * 3)
        result = analyse_rsi(df)
        assert result.zone == "NEUTRAL"

    def test_overbought_zone_detected(self):
        df = _df([65.0] * 20)
        result = analyse_rsi(df)
        assert result.zone == "OVERBOUGHT"
        assert result.value == 65.0

    def test_oversold_zone_detected(self):
        df = _df([35.0] * 20)
        result = analyse_rsi(df)
        assert result.zone == "OVERSOLD"

    def test_note_is_string(self):
        df = _df([72.0] * 20)
        result = analyse_rsi(df)
        assert isinstance(result.note, str)
        assert len(result.note) > 0


class TestRSIConfidenceAdjustment:
    def _analysis(
        self,
        value: float = 50.0,
        zone: str = "NEUTRAL",
        trend: str = "FLAT",
        divergence: str = "NONE",
    ) -> RSIAnalysis:
        return RSIAnalysis(value=value, zone=zone, trend=trend,
                           divergence=divergence, note="test")

    def test_hold_returns_zero_delta(self):
        a = self._analysis()
        delta, detail = rsi_confidence_adjustment(a, "HOLD")
        assert delta == 0
        assert detail == ""

    def test_buy_ce_deeply_overbought_boosts(self):
        a = self._analysis(value=75.0, zone="DEEPLY_OVERBOUGHT")
        delta, detail = rsi_confidence_adjustment(a, "BUY_CE")
        assert delta > 0
        assert "deeply overbought" in detail.lower()

    def test_buy_pe_deeply_oversold_boosts(self):
        a = self._analysis(value=25.0, zone="DEEPLY_OVERSOLD")
        delta, detail = rsi_confidence_adjustment(a, "BUY_PE")
        assert delta > 0
        assert "deeply oversold" in detail.lower()

    def test_buy_ce_rising_rsi_boosts(self):
        a = self._analysis(value=65.0, zone="OVERBOUGHT", trend="RISING")
        delta, detail = rsi_confidence_adjustment(a, "BUY_CE")
        assert delta > 0

    def test_buy_ce_falling_rsi_penalises(self):
        a = self._analysis(value=65.0, zone="OVERBOUGHT", trend="FALLING")
        delta, detail = rsi_confidence_adjustment(a, "BUY_CE")
        # Rising zone bonus (+4) but falling trend (-5) → net negative
        assert delta < 0 or "fading" in detail

    def test_bearish_divergence_penalises_buy_ce(self):
        a = self._analysis(value=65.0, zone="OVERBOUGHT", divergence="BEARISH_DIV")
        delta, detail = rsi_confidence_adjustment(a, "BUY_CE")
        assert delta < 0
        assert "divergence" in detail.lower()

    def test_bullish_divergence_penalises_buy_pe(self):
        a = self._analysis(value=35.0, zone="OVERSOLD", divergence="BULLISH_DIV")
        delta, detail = rsi_confidence_adjustment(a, "BUY_PE")
        assert delta < 0
        assert "divergence" in detail.lower()

    def test_counter_trend_zone_penalises(self):
        # Oversold RSI on a BUY_CE signal → counter-trend risk
        a = self._analysis(value=35.0, zone="OVERSOLD")
        delta, detail = rsi_confidence_adjustment(a, "BUY_CE")
        assert delta < 0

    def test_detail_is_empty_when_neutral_zone_flat_no_div(self):
        a = self._analysis(value=50.0, zone="NEUTRAL", trend="FLAT", divergence="NONE")
        delta, detail = rsi_confidence_adjustment(a, "BUY_CE")
        assert delta == 0
        assert detail == ""
