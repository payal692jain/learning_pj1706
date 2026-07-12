"""Tests for the margin engine — per-lot cost and lot sizing."""

import pytest

from nifty_ai_agent.risk.margin import (
    MarginCalculator,
    futures_margin,
    margin_rates,
    option_buy_margin,
    option_sell_margin,
)


class TestMarginRequirements:
    def test_futures_margin_is_span_plus_exposure_of_notional(self):
        req = futures_margin("NIFTY", spot=24_000.0, lot_size=65)
        rates = margin_rates("NIFTY")
        notional = 24_000.0 * 65
        assert req.notional_per_lot == pytest.approx(notional)
        assert req.total_per_lot == pytest.approx(notional * rates.total_pct / 100)

    def test_unknown_index_falls_back_to_conservative_rates(self):
        assert margin_rates("MIDCPNIFTY").total_pct >= margin_rates("NIFTY").total_pct

    def test_option_buy_margin_is_premium_only(self):
        req = option_buy_margin("NIFTY", 24_000, "CE", premium=104.0, lot_size=65, spot=24_000.0)
        assert req.span == 0.0
        assert req.exposure == 0.0
        assert req.total_per_lot == pytest.approx(104.0 * 65)

    def test_futures_leverage_exceeds_option_buy_leverage_is_false(self):
        # A long option is the more levered instrument — same notional, far less cash.
        fut = futures_margin("NIFTY", 24_000.0, 65)
        opt = option_buy_margin("NIFTY", 24_000, "CE", 104.0, 65, 24_000.0)
        assert opt.leverage > fut.leverage

    def test_option_sell_margin_is_far_above_buy_margin(self):
        buy = option_buy_margin("NIFTY", 24_000, "CE", 104.0, 65, 24_000.0)
        sell = option_sell_margin("NIFTY", 24_000, "CE", 104.0, 65, 24_000.0)
        assert sell.total_per_lot > buy.total_per_lot * 5

    def test_deep_otm_short_is_cheaper_than_atm_short(self):
        atm = option_sell_margin("NIFTY", 24_000, "CE", 104.0, 65, 24_000.0)
        otm = option_sell_margin("NIFTY", 25_000, "CE", 10.0, 65, 24_000.0)
        assert otm.total_per_lot < atm.total_per_lot

    def test_short_margin_never_falls_below_the_floor(self):
        # A strike absurdly far OTM would otherwise compute a negative margin.
        req = option_sell_margin("NIFTY", 40_000, "CE", 0.5, 65, 24_000.0)
        assert req.total_per_lot > 0
        assert req.total_per_lot >= req.notional_per_lot * 0.05 * 0.999

    def test_premium_credit_on_a_short_does_not_inflate_blocked_cash(self):
        req = option_sell_margin("NIFTY", 24_000, "PE", 120.0, 65, 24_000.0)
        assert req.premium < 0  # credit, not debit
        assert req.total_per_lot == pytest.approx(req.span + req.exposure)


class TestPositionSizing:
    def test_lots_capped_by_margin_when_capital_is_the_constraint(self):
        calc = MarginCalculator(capital=50_000, max_risk_per_trade_pct=100.0)
        req = option_buy_margin("NIFTY", 24_000, "CE", 100.0, 65, 24_000.0)  # ₹6,500/lot
        sizing = calc.size(req, loss_per_lot_at_sl=1.0)
        assert sizing.lots == 7
        assert sizing.binding_constraint == "MARGIN"

    def test_lots_capped_by_risk_rule_when_the_stop_is_wide(self):
        calc = MarginCalculator(capital=500_000, max_risk_per_trade_pct=1.0)  # ₹5,000 budget
        req = option_buy_margin("NIFTY", 24_000, "CE", 100.0, 65, 24_000.0)
        sizing = calc.size(req, loss_per_lot_at_sl=2_000.0)
        assert sizing.lots == 2  # 5,000 // 2,000 — not the 76 the capital could fund
        assert sizing.lots_by_margin > sizing.lots_by_risk
        assert sizing.binding_constraint == "RISK"

    def test_one_percent_rule_blocks_a_small_account(self):
        # The realistic case: ₹50k capital, 1% = ₹500 budget, one lot loses ₹3,770 at SL.
        calc = MarginCalculator(capital=50_000, max_risk_per_trade_pct=1.0)
        req = option_buy_margin("NIFTY", 24_000, "CE", 100.0, 65, 24_000.0)
        sizing = calc.size(req, loss_per_lot_at_sl=3_770.0)
        assert sizing.lots == 0
        assert not sizing.is_tradeable
        assert "risk cap" in sizing.blocked_reason
        assert sizing.binding_constraint == "BLOCKED"

    def test_blocked_reason_names_the_capital_needed_for_the_risk_rule(self):
        calc = MarginCalculator(capital=50_000, max_risk_per_trade_pct=1.0)
        assert calc.min_capital_for_risk_rule(3_770.0) == pytest.approx(377_000.0)

    def test_unaffordable_lot_blocks_on_margin_not_risk(self):
        calc = MarginCalculator(capital=50_000, max_risk_per_trade_pct=100.0)
        req = futures_margin("NIFTY", 24_000.0, 65)  # ~₹1.95L per lot
        sizing = calc.size(req, loss_per_lot_at_sl=100.0)
        assert sizing.lots == 0
        assert "margin" in sizing.blocked_reason

    def test_daily_loss_limit_shrinks_size_as_losses_accumulate(self):
        calc = MarginCalculator(
            capital=500_000, max_risk_per_trade_pct=100.0, daily_loss_limit_pct=3.0,
        )  # ₹15,000 day budget
        req = option_buy_margin("NIFTY", 24_000, "CE", 100.0, 65, 24_000.0)
        fresh = calc.size(req, loss_per_lot_at_sl=5_000.0)
        bruised = calc.size(req, loss_per_lot_at_sl=5_000.0, realised_loss_today=11_000.0)
        assert fresh.lots == 3          # 15,000 // 5,000
        assert bruised.lots == 0        # only 4,000 of the day budget left

    def test_daily_limit_breach_stops_trading_entirely(self):
        calc = MarginCalculator(capital=500_000, daily_loss_limit_pct=3.0)
        req = option_buy_margin("NIFTY", 24_000, "CE", 100.0, 65, 24_000.0)
        sizing = calc.size(req, loss_per_lot_at_sl=1_000.0, realised_loss_today=15_000.0)
        assert sizing.lots == 0
        assert "daily loss limit" in sizing.blocked_reason

    def test_utilisation_cap_keeps_a_capital_buffer(self):
        full = MarginCalculator(capital=50_000, max_risk_per_trade_pct=100.0)
        capped = MarginCalculator(
            capital=50_000, max_risk_per_trade_pct=100.0, max_margin_utilisation_pct=50.0,
        )
        req = option_buy_margin("NIFTY", 24_000, "CE", 100.0, 65, 24_000.0)
        assert capped.size(req, 1.0).lots < full.size(req, 1.0).lots

    def test_margin_used_and_free_reconcile_with_capital(self):
        calc = MarginCalculator(capital=50_000, max_risk_per_trade_pct=100.0)
        req = option_buy_margin("NIFTY", 24_000, "CE", 100.0, 65, 24_000.0)
        sizing = calc.size(req, loss_per_lot_at_sl=1.0)
        assert sizing.margin_used + sizing.margin_free == pytest.approx(50_000)
        assert sizing.margin_utilisation_pct <= 100.0

    def test_risk_at_sl_scales_with_recommended_lots(self):
        calc = MarginCalculator(capital=500_000, max_risk_per_trade_pct=1.0)
        req = option_buy_margin("NIFTY", 24_000, "CE", 100.0, 65, 24_000.0)
        sizing = calc.size(req, loss_per_lot_at_sl=2_000.0)
        assert sizing.risk_at_sl == pytest.approx(sizing.lots * 2_000.0)
        assert sizing.risk_pct_of_capital <= calc.max_risk_per_trade_pct
