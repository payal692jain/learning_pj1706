"""Tests for the capital-aware trade plan builder and formatter."""

import pytest

from nifty_ai_agent.reports.trade_plan import (
    FALLBACK_LOT_SIZES,
    TradeIdea,
    build_trade_idea,
    format_trade_plan,
)
from nifty_ai_agent.risk.calculator import RiskCalculator
from nifty_ai_agent.strategies.base import SignalType
from nifty_ai_agent.strategies.option_analyser import ExpiryAnalysis, OptionLeg


def _analysis(spot: float = 24200.0, atm: int = 24200,
              ce_ltp: float = 104.0, pe_ltp: float = 95.0) -> ExpiryAnalysis:
    leg = OptionLeg(strike=atm, ce_ltp=ce_ltp, pe_ltp=pe_ltp,
                    ce_oi=1000, pe_oi=1000, ce_iv=12.0, pe_iv=11.5, is_atm=True)
    return ExpiryAnalysis(
        expiry="14-Jul-2026", spot=spot, atm_strike=atm, max_pain=spot,
        pcr=1.0, legs=[leg], call_oi_resistance=atm + 200,
        put_oi_support=atm - 200, bias="NEUTRAL", days_to_expiry=3,
        atm_ce_ltp=ce_ltp, atm_pe_ltp=pe_ltp,
    )


def _risk(sig: SignalType, spot: float = 24200.0):
    return RiskCalculator().calculate(sig, spot, spot * 0.004)


def _idea(entry: float = 100.0, target: float = 180.0, sl: float = 60.0,
          lot: int = 65) -> TradeIdea:
    return TradeIdea(
        index_name="NIFTY", signal="BUY_CE", confidence=74, strike=24200,
        opt_type="CE", expiry="14-Jul-2026", entry_premium=entry,
        target_sell=target, sl_sell=sl, lot_size=lot, is_live=True,
    )


class TestBuildTradeIdea:
    def test_hold_returns_none(self):
        assert build_trade_idea(
            "NIFTY", SignalType.HOLD, 50, _analysis(), _risk(SignalType.BUY_CE), 65,
        ) is None

    def test_zero_premium_returns_none(self):
        analysis = _analysis(ce_ltp=0.0)
        assert build_trade_idea(
            "NIFTY", SignalType.BUY_CE, 74, analysis, _risk(SignalType.BUY_CE), 65,
        ) is None

    def test_buy_ce_sell_prices_bracket_entry(self):
        idea = build_trade_idea(
            "NIFTY", SignalType.BUY_CE, 74, _analysis(), _risk(SignalType.BUY_CE), 65,
        )
        # Index target is above entry for CE → premium gains; SL below → loses.
        assert idea.target_sell > idea.entry_premium
        assert idea.sl_sell < idea.entry_premium
        assert idea.opt_type == "CE"

    def test_buy_pe_sell_prices_bracket_entry(self):
        idea = build_trade_idea(
            "NIFTY", SignalType.BUY_PE, 60, _analysis(), _risk(SignalType.BUY_PE), 65,
        )
        assert idea.target_sell > idea.entry_premium
        assert idea.sl_sell < idea.entry_premium
        assert idea.opt_type == "PE"

    def test_lot_economics_properties(self):
        idea = _idea(entry=100.0, target=180.0, sl=60.0, lot=65)
        assert idea.cost_per_lot == 6500.0
        assert idea.pnl_target_per_lot == pytest.approx(80.0 * 65)
        assert idea.pnl_sl_per_lot == pytest.approx(-40.0 * 65)


class TestFormatTradePlan:
    def test_reachable_target_flagged_with_check(self):
        # +₹5,200/lot at target, 7 lots affordable → ₹10k needs 2 lots → reachable
        _, body = format_trade_plan([_idea()], [], 50000, 10000)
        assert "✓ NIFTY: ₹10,000 needs 2 lot(s)" in body

    def test_unreachable_target_flagged_honestly(self):
        # Tiny move: +₹5/lot at target → needs 2000 lots — flag NOT reachable.
        idea = _idea(entry=100.0, target=100.1, sl=60.0, lot=50)
        _, body = format_trade_plan([idea], [], 50000, 10000)
        assert "NOT reachable" in body

    def test_unaffordable_lot_flagged(self):
        idea = _idea(entry=1000.0, lot=65)  # 1 lot = ₹65,000 > ₹50,000
        _, body = format_trade_plan([idea], [], 50000, 10000)
        assert "1 lot ₹65,000 > capital ₹50,000" in body

    def test_holds_listed(self):
        title, body = format_trade_plan([], ["NIFTY", "SENSEX"], 50000, 10000)
        assert "NIFTY: HOLD" in body
        assert "SENSEX: HOLD" in body
        assert "NIFTY —" in title

    def test_sell_and_exit_prices_shown(self):
        _, body = format_trade_plan([_idea()], [], 50000, 10000)
        sell_row = next(l for l in body.splitlines() if l.startswith("Sell ₹"))
        exit_row = next(l for l in body.splitlines() if l.startswith("Exit ₹"))
        assert "180" in sell_row
        assert "60" in exit_row

    def test_disclaimer_always_present(self):
        _, body = format_trade_plan([], ["NIFTY"], 50000, 10000)
        assert "not guarantees" in body
        assert "20%/day" in body

    def test_title_summarises_all_indices(self):
        pe = _idea()
        pe.index_name, pe.opt_type = "SENSEX", "PE"
        title, _ = format_trade_plan([_idea(), pe], ["BANKNIFTY"], 50000, 10000)
        assert "NIFTY CE" in title
        assert "SENSEX PE" in title
        assert "BANKNIFTY —" in title


class TestFallbackLotSizes:
    def test_all_three_indices_covered(self):
        assert set(FALLBACK_LOT_SIZES) == {"NIFTY", "SENSEX", "BANKNIFTY"}
        assert all(v > 0 for v in FALLBACK_LOT_SIZES.values())
