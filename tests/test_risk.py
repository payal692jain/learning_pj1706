"""Tests for the risk management module."""

import pytest

from nifty_ai_agent.risk.calculator import RiskCalculator
from nifty_ai_agent.strategies.base import SignalType


@pytest.fixture
def calc():
    return RiskCalculator(max_risk_pct=1.0, daily_loss_limit_pct=3.0, min_rr=2.0)


class TestRiskCalculator:
    def test_buy_ce_sl_below_entry(self, calc):
        params = calc.calculate(SignalType.BUY_CE, entry_price=24000.0, atr=100.0)
        assert params.stop_loss < params.entry_price

    def test_buy_ce_target_above_entry(self, calc):
        params = calc.calculate(SignalType.BUY_CE, entry_price=24000.0, atr=100.0)
        assert params.target > params.entry_price

    def test_buy_pe_sl_above_entry(self, calc):
        params = calc.calculate(SignalType.BUY_PE, entry_price=24000.0, atr=100.0)
        assert params.stop_loss > params.entry_price

    def test_buy_pe_target_below_entry(self, calc):
        params = calc.calculate(SignalType.BUY_PE, entry_price=24000.0, atr=100.0)
        assert params.target < params.entry_price

    def test_rr_meets_minimum(self, calc):
        params = calc.calculate(SignalType.BUY_CE, entry_price=24000.0, atr=100.0)
        assert params.risk_reward_ratio >= 2.0

    def test_hold_returns_invalid(self, calc):
        params = calc.calculate(SignalType.HOLD, entry_price=24000.0, atr=100.0)
        assert not params.is_valid
        assert params.stop_loss == 0.0
        assert params.target == 0.0

    def test_risk_pct_within_limit(self, calc):
        # ATR of 100 on entry of 24000 → SL distance = 150 → 0.625% risk
        params = calc.calculate(SignalType.BUY_CE, entry_price=24000.0, atr=100.0)
        assert params.risk_pct <= 1.0
        assert params.is_valid

    def test_large_atr_may_exceed_risk_limit(self):
        strict_calc = RiskCalculator(max_risk_pct=0.1, min_rr=2.0)
        # With tiny risk limit and large ATR, trade should be rejected
        params = strict_calc.calculate(SignalType.BUY_CE, entry_price=24000.0, atr=500.0)
        assert not params.is_valid
        assert params.rejection_reason != ""

    def test_risk_reward_ratio_positive(self, calc):
        params = calc.calculate(SignalType.BUY_CE, entry_price=24000.0, atr=100.0)
        assert params.risk_reward_ratio > 0

    def test_risk_amount_correct(self, calc):
        params = calc.calculate(SignalType.BUY_CE, entry_price=24000.0, atr=100.0)
        expected_risk = 100.0 * 1.5  # atr × multiplier
        assert abs(params.risk_amount - expected_risk) < 0.01
