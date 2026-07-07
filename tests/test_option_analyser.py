"""Tests for the option chain analyser."""

import pandas as pd
import pytest

from nifty_ai_agent.strategies.option_analyser import (
    ExpiryAnalysis,
    _bs_price,
    _compute_max_pain,
    _days_to_expiry,
    _nearest_atm,
    _norm_cdf,
    analyse_option_chain,
    format_analysis_for_notification,
    option_chain_confidence_adjustment,
)


def _make_chain(spot: float = 24050.0, n_strikes: int = 10) -> pd.DataFrame:
    atm = round(spot / 50) * 50
    strikes = [atm + (i - n_strikes // 2) * 50 for i in range(n_strikes)]
    rows = []
    for k in strikes:
        dist = abs(k - spot)
        ce_oi = max(1000, int(500_000 / (1 + dist / 100)))
        pe_oi = max(1000, int(400_000 / (1 + dist / 100)))
        rows.append({
            "strike": k,
            "ce_oi": ce_oi, "pe_oi": pe_oi,
            "ce_ltp": max(1, round(100 - dist * 0.1, 1)),
            "pe_ltp": max(1, round(90 - dist * 0.1, 1)),
            "ce_iv": 15.0, "pe_iv": 14.0,
        })
    return pd.DataFrame(rows)


class TestNearestATM:
    def test_exact_multiple(self):
        assert _nearest_atm(24000.0) == 24000

    def test_rounds_to_nearest_50(self):
        assert _nearest_atm(24025.0) == 24000
        assert _nearest_atm(24026.0) == 24050

    def test_high_value(self):
        assert _nearest_atm(24380.0) == 24400


class TestNormCDF:
    def test_at_zero(self):
        assert abs(_norm_cdf(0) - 0.5) < 1e-6

    def test_large_positive(self):
        assert _norm_cdf(5) > 0.999

    def test_large_negative(self):
        assert _norm_cdf(-5) < 0.001


class TestBlackScholes:
    def test_ce_positive(self):
        price = _bs_price(S=24000, K=24000, T=1/365, r=0.068, sigma=0.15, option_type="CE")
        assert price > 0

    def test_pe_positive(self):
        price = _bs_price(S=24000, K=24000, T=1/365, r=0.068, sigma=0.15, option_type="PE")
        assert price > 0

    def test_deep_itm_ce_high(self):
        price = _bs_price(S=25000, K=22000, T=7/365, r=0.068, sigma=0.15, option_type="CE")
        assert price > 2900  # deeply in the money

    def test_zero_dte_ce(self):
        price = _bs_price(S=24100, K=24000, T=0, r=0.068, sigma=0.15, option_type="CE")
        assert price == pytest.approx(100.0, abs=1)

    def test_zero_dte_otm_ce(self):
        price = _bs_price(S=23900, K=24000, T=0, r=0.068, sigma=0.15, option_type="CE")
        assert price == 0.0


class TestMaxPain:
    def test_empty(self):
        assert _compute_max_pain(pd.DataFrame()) == 0.0

    def test_balanced_oi_pins_at_atm(self):
        df = pd.DataFrame({
            "strike": [23900, 24000, 24100],
            "ce_oi":  [10000,  5000,  1000],
            "pe_oi":  [ 1000,  5000, 10000],
        })
        assert _compute_max_pain(df) == 24000.0


class TestDaysToExpiry:
    def test_future_date(self):
        from datetime import date, timedelta
        future = date.today() + timedelta(days=3)
        expiry_str = future.strftime("%d-%b-%Y")
        assert _days_to_expiry(expiry_str) == 3

    def test_past_date_returns_zero(self):
        assert _days_to_expiry("01-Jan-2020") == 0

    def test_invalid_format_returns_one(self):
        assert _days_to_expiry("invalid") == 1


class TestAnalyseOptionChain:
    def test_returns_expiry_analysis(self):
        chain = _make_chain(24050.0)
        result = analyse_option_chain(chain, 24050.0, "27-Jun-2024")
        assert isinstance(result, ExpiryAnalysis)

    def test_atm_strike_correct(self):
        result = analyse_option_chain(_make_chain(24050.0), 24050.0, "27-Jun-2024")
        assert result.atm_strike == 24050

    def test_atm_leg_flagged(self):
        result = analyse_option_chain(_make_chain(24000.0), 24000.0, "27-Jun-2024")
        atm_legs = [l for l in result.legs if l.is_atm]
        assert len(atm_legs) == 1

    def test_legs_count(self):
        result = analyse_option_chain(_make_chain(24000.0), 24000.0, "27-Jun-2024", strikes_each_side=3)
        assert len(result.legs) <= 7  # ATM ± 3

    def test_bias_returned(self):
        result = analyse_option_chain(_make_chain(24000.0), 24000.0, "27-Jun-2024")
        assert result.bias in ("BULLISH", "BEARISH", "NEUTRAL")

    def test_empty_chain_returns_stub(self):
        result = analyse_option_chain(pd.DataFrame(), 24000.0, "27-Jun-2024")
        assert result.atm_strike == 24000

    def test_theoretical_prices_positive(self):
        result = analyse_option_chain(_make_chain(24000.0), 24000.0, "27-Jun-2024")
        if result.theoretical_ce_atm:
            assert result.theoretical_ce_atm > 0
            assert result.theoretical_pe_atm > 0


class TestOptionChainConfidenceAdjustment:
    def _analysis(
        self,
        spot: float = 24000.0,
        pcr: float = 1.0,
        ce_resistance: int = 24400,
        pe_support: int = 23600,
        max_pain: float = 24000.0,
        dte: int = 3,
    ) -> ExpiryAnalysis:
        return ExpiryAnalysis(
            expiry="27-Jun-2024",
            spot=spot,
            atm_strike=int(round(spot / 50) * 50),
            max_pain=max_pain,
            pcr=pcr,
            legs=[],
            call_oi_resistance=ce_resistance,
            put_oi_support=pe_support,
            bias="NEUTRAL",
            days_to_expiry=dte,
        )

    def test_hold_returns_zero(self):
        delta, detail = option_chain_confidence_adjustment(self._analysis(), "HOLD")
        assert delta == 0
        assert detail == ""

    def test_bullish_pcr_boosts_buy_ce(self):
        a = self._analysis(pcr=1.3)
        delta, _ = option_chain_confidence_adjustment(a, "BUY_CE")
        assert delta > 0

    def test_bullish_pcr_penalises_buy_pe(self):
        a = self._analysis(pcr=1.3)
        delta, _ = option_chain_confidence_adjustment(a, "BUY_PE")
        assert delta < 0

    def test_bearish_pcr_boosts_buy_pe(self):
        a = self._analysis(pcr=0.7)
        delta, _ = option_chain_confidence_adjustment(a, "BUY_PE")
        assert delta > 0

    def test_ce_resistance_very_close_penalises_buy_ce(self):
        # Spot 24000, resistance 24100 → 0.42% away → strong penalty
        a = self._analysis(spot=24000.0, ce_resistance=24100)
        delta, detail = option_chain_confidence_adjustment(a, "BUY_CE")
        assert delta <= -12
        assert "ceiling" in detail.lower()

    def test_ce_resistance_far_boosts_buy_ce(self):
        # Spot 24000, resistance 24400 → 1.67% away → boost
        a = self._analysis(spot=24000.0, ce_resistance=24400)
        delta, detail = option_chain_confidence_adjustment(a, "BUY_CE")
        assert delta > 0

    def test_pe_support_very_close_penalises_buy_pe(self):
        # Spot 24000, support 23900 → 0.42% below → penalty
        a = self._analysis(spot=24000.0, pe_support=23900)
        delta, detail = option_chain_confidence_adjustment(a, "BUY_PE")
        assert delta <= -10
        assert "floor" in detail.lower()

    def test_max_pain_pinning_on_expiry_day(self):
        # Spot within 0.3% of max pain with DTE=0
        a = self._analysis(spot=24000.0, max_pain=24050.0, dte=0)
        delta, detail = option_chain_confidence_adjustment(a, "BUY_CE")
        assert delta < 0
        assert "pinning" in detail.lower()

    def test_max_pain_far_no_pin_penalty(self):
        a = self._analysis(spot=24000.0, max_pain=23000.0, dte=0)
        delta, _ = option_chain_confidence_adjustment(a, "BUY_CE")
        # No pinning penalty (4.2% away), might still have other adjustments
        # Just check no huge negative from pinning alone
        assert delta > -15

    def test_zero_spot_returns_zero(self):
        a = self._analysis(spot=0.0)
        delta, detail = option_chain_confidence_adjustment(a, "BUY_CE")
        assert delta == 0

    def test_detail_contains_expiry_and_pcr(self):
        a = self._analysis(pcr=1.3)
        _, detail = option_chain_confidence_adjustment(a, "BUY_CE")
        assert "27-Jun-2024" in detail
        assert "1.3" in detail


class TestFormatAnalysis:
    def test_returns_string(self):
        result = analyse_option_chain(_make_chain(24000.0), 24000.0, "27-Jun-2024")
        text = format_analysis_for_notification(result)
        assert isinstance(text, str)

    def test_contains_expiry(self):
        result = analyse_option_chain(_make_chain(24000.0), 24000.0, "27-Jun-2024")
        text = format_analysis_for_notification(result)
        assert "27-Jun-2024" in text

    def test_contains_atm_marker(self):
        result = analyse_option_chain(_make_chain(24000.0), 24000.0, "27-Jun-2024")
        text = format_analysis_for_notification(result)
        assert "ATM" in text
