"""Tests for the standalone risk & margin notification."""

import pytest

from nifty_ai_agent.reports.margin_report import (
    build_index_margin_view,
    format_margin_report,
)
from nifty_ai_agent.risk.margin import MarginCalculator
from nifty_ai_agent.strategies.option_analyser import ExpiryAnalysis, OptionLeg


def _analysis(
    spot: float = 24_000.0,
    strike: int = 24_000,
    ce: float = 104.0,
    pe: float = 96.0,
    is_live: bool = True,
) -> ExpiryAnalysis:
    return ExpiryAnalysis(
        expiry="14-Jul-2026",
        spot=spot,
        atm_strike=strike,
        max_pain=float(strike),
        pcr=1.0,
        legs=[OptionLeg(strike, ce, pe, 1000, 1000, 12.0, 12.0, is_atm=True)],
        call_oi_resistance=strike + 200,
        put_oi_support=strike - 200,
        bias="NEUTRAL",
        days_to_expiry=4,
        atm_ce_ltp=ce,
        atm_pe_ltp=pe,
        is_live=is_live,
    )


@pytest.fixture
def calc():
    return MarginCalculator(
        capital=50_000, max_risk_per_trade_pct=1.0, daily_loss_limit_pct=3.0,
    )


class TestBuildIndexMarginView:
    def test_prices_both_option_legs_and_the_future(self, calc):
        view = build_index_margin_view("NIFTY", _analysis(), atr=100.0, lot_size=65, calculator=calc)
        assert view.ce is not None and view.pe is not None
        assert view.futures.total_per_lot > view.ce.buy.total_per_lot
        assert view.sl_points == pytest.approx(150.0)  # 1.5 × ATR

    def test_a_call_loses_money_when_the_index_falls_to_its_stop(self, calc):
        view = build_index_margin_view("NIFTY", _analysis(), atr=100.0, lot_size=65, calculator=calc)
        assert view.ce.loss_per_lot > 0
        assert view.ce.loss_per_lot <= view.ce.buy.total_per_lot  # can't lose more than the premium

    def test_small_account_cannot_size_an_index_lot_within_the_risk_rule(self, calc):
        view = build_index_margin_view("NIFTY", _analysis(), atr=100.0, lot_size=65, calculator=calc)
        # ₹500 risk budget vs a lot that loses thousands at the stop.
        assert view.ce.sizing.lots == 0
        assert "risk cap" in view.ce.sizing.blocked_reason

    def test_large_account_can_size_within_the_risk_rule(self):
        rich = MarginCalculator(capital=2_000_000, max_risk_per_trade_pct=1.0)
        view = build_index_margin_view("NIFTY", _analysis(), atr=100.0, lot_size=65, calculator=rich)
        assert view.ce.sizing.lots >= 1
        assert view.ce.sizing.risk_pct_of_capital <= 1.0

    def test_missing_premiums_leave_the_leg_unpriced(self, calc):
        blank = _analysis(ce=0.0, pe=0.0)
        view = build_index_margin_view("NIFTY", blank, atr=100.0, lot_size=65, calculator=calc)
        assert view.ce is None and view.pe is None
        assert view.futures.total_per_lot > 0  # the future is still priceable


class TestFormatMarginReport:
    def test_report_states_capital_risk_and_day_budgets(self, calc):
        view = build_index_margin_view("NIFTY", _analysis(), atr=100.0, lot_size=65, calculator=calc)
        title, body = format_margin_report([view], calc)
        assert "Risk & Margin" in title
        assert "₹50,000" in body
        assert "₹500" in body      # 1% per-trade budget
        assert "₹1,500" in body    # 3% day stop

    def test_report_shows_futures_and_both_option_legs(self, calc):
        view = build_index_margin_view("NIFTY", _analysis(), atr=100.0, lot_size=65, calculator=calc)
        _, body = format_margin_report([view], calc)
        assert "── FUTURES (1 lot) ──" in body
        assert "── ATM OPTIONS (BUY) ──" in body
        assert "CE prm/lot" in body
        assert "PE prm/lot" in body
        assert "Short mrg" in body

    def test_verdict_explains_a_zero_lot_outcome(self, calc):
        view = build_index_margin_view("NIFTY", _analysis(), atr=100.0, lot_size=65, calculator=calc)
        _, body = format_margin_report([view], calc)
        assert "0 lots" in body
        assert "risk cap" in body

    def test_verdict_sizes_a_tradeable_account(self):
        rich = MarginCalculator(capital=2_000_000, max_risk_per_trade_pct=1.0)
        view = build_index_margin_view("NIFTY", _analysis(), atr=100.0, lot_size=65, calculator=rich)
        _, body = format_margin_report([view], rich)
        assert "✓ NIFTY:" in body
        assert "CE" in body
        assert "capped by" in body

    def test_estimated_premiums_are_flagged(self, calc):
        view = build_index_margin_view(
            "NIFTY", _analysis(is_live=False), atr=100.0, lot_size=65, calculator=calc,
        )
        _, body = format_margin_report([view], calc)
        assert "NIFTY*" in body
        assert "premiums estimated" in body

    def test_report_always_carries_the_estimate_disclaimer(self, calc):
        view = build_index_margin_view("NIFTY", _analysis(), atr=100.0, lot_size=65, calculator=calc)
        _, body = format_margin_report([view], calc)
        assert "SPAN estimates" in body

    def test_multi_index_report_lines_up_every_index(self, calc):
        views = [
            build_index_margin_view("NIFTY", _analysis(), 100.0, 65, calc),
            build_index_margin_view(
                "BANKNIFTY", _analysis(spot=52_000, strike=52_000, ce=310, pe=290),
                220.0, 30, calc,
            ),
            build_index_margin_view(
                "SENSEX", _analysis(spot=79_000, strike=79_000, ce=250, pe=240),
                300.0, 20, calc,
            ),
        ]
        title, body = format_margin_report(views, calc)
        assert "BANKNIF" in title and "SENSEX" in title
        # Three data columns on the spot row.
        spot_row = next(line for line in body.splitlines() if line.startswith("Spot"))
        assert len(spot_row.split()) == 4

    def test_empty_report_does_not_crash(self, calc):
        title, body = format_margin_report([], calc)
        assert "margins unavailable" in body

    def test_remaining_day_budget_reflects_realised_losses(self, calc):
        view = build_index_margin_view("NIFTY", _analysis(), 100.0, 65, calc)
        _, body = format_margin_report([view], calc, realised_loss_today=1_000.0)
        assert "₹500 left" in body

    @pytest.mark.parametrize("capital", [50_000, 200_000, 2_000_000, 50_000_000])
    def test_body_never_breaches_the_pushover_size_limit(self, capital):
        """Pushover REJECTS a message over 1024 chars — it does not truncate it. An
        over-long body means no notification arrives at all, so this is a hard invariant."""
        calculator = MarginCalculator(capital=capital, max_risk_per_trade_pct=1.0)
        _, body = format_margin_report(_three_indices(calculator), calculator)
        assert len(body) <= 1024

    def test_both_option_legs_survive_a_blocked_small_account(self, calc):
        """CE and PE live or die together — a risk report that shows the call leg and
        silently omits the put leg (because it ran out of room) is worse than neither."""
        _, body = format_margin_report(_three_indices(calc), calc)
        assert ("CE prm/lot" in body) == ("PE prm/lot" in body)
        assert "CE prm/lot" in body  # both fit even at ₹50k, where nothing is tradeable


def _three_indices(calculator: MarginCalculator) -> list:
    return [
        build_index_margin_view("NIFTY", _analysis(), 100.0, 65, calculator),
        build_index_margin_view(
            "BANKNIFTY", _analysis(spot=52_000, strike=52_000, ce=310, pe=290),
            220.0, 30, calculator,
        ),
        build_index_margin_view(
            "SENSEX", _analysis(spot=79_000, strike=79_000, ce=250, pe=240),
            300.0, 20, calculator,
        ),
    ]
