"""Tests for the trade plan builder and formatter (prices only — no sizing)."""

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


def _idea(entry: float = 100.0, target: float = 180.0, sl: float = 60.0) -> TradeIdea:
    return TradeIdea(
        index_name="NIFTY", signal="BUY_CE", confidence=74, strike=24200,
        opt_type="CE", expiry="14-Jul-2026", entry_premium=entry,
        target_sell=target, sl_sell=sl, is_live=True,
    )


class TestBuildTradeIdea:
    def test_hold_returns_none(self):
        assert build_trade_idea(
            "NIFTY", SignalType.HOLD, 50, _analysis(), _risk(SignalType.BUY_CE),
        ) is None

    def test_zero_premium_returns_none(self):
        analysis = _analysis(ce_ltp=0.0)
        assert build_trade_idea(
            "NIFTY", SignalType.BUY_CE, 74, analysis, _risk(SignalType.BUY_CE),
        ) is None

    def test_buy_ce_sell_prices_bracket_entry(self):
        idea = build_trade_idea(
            "NIFTY", SignalType.BUY_CE, 74, _analysis(), _risk(SignalType.BUY_CE),
        )
        # Index target is above entry for CE → premium gains; SL below → loses.
        assert idea.target_sell > idea.entry_premium
        assert idea.sl_sell < idea.entry_premium
        assert idea.opt_type == "CE"

    def test_buy_pe_sell_prices_bracket_entry(self):
        idea = build_trade_idea(
            "NIFTY", SignalType.BUY_PE, 60, _analysis(), _risk(SignalType.BUY_PE),
        )
        assert idea.target_sell > idea.entry_premium
        assert idea.sl_sell < idea.entry_premium
        assert idea.opt_type == "PE"


class TestFormatTradePlan:
    def test_buy_sell_exit_prices_shown(self):
        _, body = format_trade_plan([_idea()], [])
        buy_row = next(l for l in body.splitlines() if l.startswith("Buy ₹"))
        sell_row = next(l for l in body.splitlines() if l.startswith("Sell ₹"))
        exit_row = next(l for l in body.splitlines() if l.startswith("Exit ₹"))
        assert "100" in buy_row
        assert "180" in sell_row
        assert "60" in exit_row

    def test_strike_and_expiry_shown(self):
        _, body = format_trade_plan([_idea()], [])
        assert "24200CE" in body
        assert "14-Jul" in body

    def test_no_capital_or_sizing_anywhere(self):
        """The whole point of the prices-only plan: no rupee capital, no lots."""
        _, body = format_trade_plan([_idea()], ["SENSEX"])
        for banned in ("Capital", "Lot qty", "Lots/cap", "1lot", "P/L", "50,000", "lot(s)"):
            assert banned not in body

    def test_holds_listed(self):
        title, body = format_trade_plan([], ["NIFTY", "SENSEX"])
        assert "NIFTY: HOLD" in body
        assert "SENSEX: HOLD" in body
        assert "NIFTY —" in title

    def test_disclaimer_always_present(self):
        _, body = format_trade_plan([], ["NIFTY"])
        assert "not guarantees" in body

    def test_estimated_premiums_flagged_with_star(self):
        idea = _idea()
        idea.is_live = False
        _, body = format_trade_plan([idea], [])
        assert "NIFTY*" in body

    def test_title_summarises_all_indices(self):
        pe = _idea()
        pe.index_name, pe.opt_type = "SENSEX", "PE"
        title, _ = format_trade_plan([_idea(), pe], ["BANKNIFTY"])
        assert "NIFTY CE" in title
        assert "SENSEX PE" in title
        assert "BANKNIFTY —" in title


class TestFallbackLotSizes:
    def test_all_three_indices_covered(self):
        assert set(FALLBACK_LOT_SIZES) == {"NIFTY", "SENSEX", "BANKNIFTY"}
        assert all(v > 0 for v in FALLBACK_LOT_SIZES.values())
