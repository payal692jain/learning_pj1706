"""Tests for the Pushover notifier (mocked HTTP)."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from nifty_ai_agent.notifier.pushover import (
    PushoverNotifier,
    _PRIORITY_LOW,
    _PRIORITY_NORMAL,
    _find_itm_legs,
)
from nifty_ai_agent.risk.calculator import RiskCalculator
from nifty_ai_agent.strategies.base import Signal, SignalType
from nifty_ai_agent.strategies.option_analyser import ExpiryAnalysis, OptionLeg


def _dummy_signal(sig_type: SignalType = SignalType.BUY_CE) -> Signal:
    return Signal(
        signal=sig_type,
        confidence=78,
        reason="EMA20 > EMA50, RSI = 66",
        strategy="EMA_Crossover",
    )


def _dummy_risk(sig_type: SignalType = SignalType.BUY_CE):
    return RiskCalculator().calculate(sig_type, 24000.0, 100.0)


def _dummy_expiry_analysis(
    expiry: str, atm_ce_ltp: float, atm_pe_ltp: float, is_live: bool = True,
) -> ExpiryAnalysis:
    return ExpiryAnalysis(
        expiry=expiry,
        spot=24000.0,
        atm_strike=24000,
        max_pain=24000.0,
        pcr=1.0,
        legs=[],
        call_oi_resistance=24200,
        put_oi_support=23800,
        bias="NEUTRAL",
        atm_ce_ltp=atm_ce_ltp,
        atm_pe_ltp=atm_pe_ltp,
        is_live=is_live,
    )


def _leg(strike: int, ce_ltp: float, pe_ltp: float, is_atm: bool = False) -> OptionLeg:
    return OptionLeg(
        strike=strike, ce_ltp=ce_ltp, pe_ltp=pe_ltp,
        ce_oi=1000, pe_oi=1000, ce_iv=15.0, pe_iv=14.0, is_atm=is_atm,
    )


def _expiry_analysis_with_legs(atm_strike: int, legs: list[OptionLeg]) -> ExpiryAnalysis:
    return ExpiryAnalysis(
        expiry="14-Jul-2026", spot=float(atm_strike), atm_strike=atm_strike,
        max_pain=float(atm_strike), pcr=1.0, legs=legs,
        call_oi_resistance=atm_strike + 200, put_oi_support=atm_strike - 200,
        bias="NEUTRAL",
        atm_ce_ltp=next((l.ce_ltp for l in legs if l.is_atm), 0.0),
        atm_pe_ltp=next((l.pe_ltp for l in legs if l.is_atm), 0.0),
    )


@pytest.fixture
def notifier():
    return PushoverNotifier(user_key="test_user_key", api_token="test_app_token")


class TestPushoverNotifier:
    def test_send_signal_success(self, notifier):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"status": 1}
        with patch("requests.post", return_value=mock_resp) as mock_post:
            result = notifier.send_signal(_dummy_signal(), _dummy_risk(), "Bullish momentum.")
        assert result is True
        mock_post.assert_called_once()

    def test_send_text_success(self, notifier):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"status": 1}
        with patch("requests.post", return_value=mock_resp):
            result = notifier.send_text("Test", "Hello from tests")
        assert result is True

    def test_retry_on_failure_then_success(self, notifier):
        fail = MagicMock()
        fail.raise_for_status.side_effect = requests.RequestException("error")
        success = MagicMock()
        success.raise_for_status.return_value = None
        success.json.return_value = {"status": 1}
        with patch("requests.post", side_effect=[fail, success]):
            with patch("time.sleep"):
                result = notifier.send_text("T", "retry test")
        assert result is True

    def test_all_retries_fail_returns_false(self, notifier):
        fail = MagicMock()
        fail.raise_for_status.side_effect = requests.RequestException("error")
        with patch("requests.post", return_value=fail):
            with patch("time.sleep"):
                result = notifier.send_text("T", "fail test")
        assert result is False

    def test_pushover_api_error_returns_false(self, notifier):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"status": 0, "errors": ["invalid token"]}
        with patch("requests.post", return_value=mock_resp):
            with patch("time.sleep"):
                result = notifier.send_text("T", "bad token test")
        assert result is False

    def test_buy_ce_normal_priority(self, notifier):
        _, _, priority = notifier._format_signal(_dummy_signal(SignalType.BUY_CE), _dummy_risk(), "")
        assert priority == _PRIORITY_NORMAL

    def test_hold_low_priority(self, notifier):
        _, _, priority = notifier._format_signal(
            _dummy_signal(SignalType.HOLD),
            _dummy_risk(SignalType.HOLD),
            "",
        )
        assert priority == _PRIORITY_LOW

    def test_buy_pe_message_contains_signal(self, notifier):
        title, body, _ = notifier._format_signal(
            _dummy_signal(SignalType.BUY_PE),
            _dummy_risk(SignalType.BUY_PE),
            "",
        )
        assert "BUY_PE" in title

    def test_ai_explanation_in_body(self, notifier):
        _, body, _ = notifier._format_signal(
            _dummy_signal(), _dummy_risk(), "Strong EMA crossover detected."
        )
        assert "Strong EMA crossover detected." in body

    def test_risk_levels_in_body_when_valid(self, notifier):
        risk = _dummy_risk(SignalType.BUY_CE)
        _, body, _ = notifier._format_signal(_dummy_signal(), risk, "")
        assert "SL:" in body
        assert "Target:" in body

    def test_no_risk_levels_for_hold(self, notifier):
        risk = _dummy_risk(SignalType.HOLD)
        _, body, _ = notifier._format_signal(
            _dummy_signal(SignalType.HOLD), risk, ""
        )
        assert "SL:" not in body

    def test_weekly_and_monthly_columns_both_shown(self, notifier):
        weekly = _dummy_expiry_analysis("10-Jul-2026", atm_ce_ltp=142.0, atm_pe_ltp=110.0)
        monthly = _dummy_expiry_analysis("31-Jul-2026", atm_ce_ltp=310.0, atm_pe_ltp=280.0)
        _, body, _ = notifier._format_signal(
            _dummy_signal(SignalType.BUY_CE), _dummy_risk(SignalType.BUY_CE), "",
            option_analysis=weekly, monthly_option_analysis=monthly,
        )
        assert "📌 BUY NIFTY 24000 CE" in body
        assert "Weekly" in body and "Monthly" in body
        assert "10-Jul" in body and "31-Jul" in body
        # Buy row carries both premiums in one aligned line
        buy_row = next(l for l in body.splitlines() if l.startswith("Buy ₹"))
        assert "142" in buy_row and "310" in buy_row

    def test_only_weekly_shown_when_monthly_unavailable(self, notifier):
        weekly = _dummy_expiry_analysis("10-Jul-2026", atm_ce_ltp=142.0, atm_pe_ltp=110.0)
        _, body, _ = notifier._format_signal(
            _dummy_signal(SignalType.BUY_CE), _dummy_risk(SignalType.BUY_CE), "",
            option_analysis=weekly, monthly_option_analysis=None,
        )
        assert "Weekly" in body
        assert "Monthly" not in body

    def test_synthetic_prices_labelled_estimate(self, notifier):
        weekly = _dummy_expiry_analysis("10-Jul-2026", atm_ce_ltp=96.0, atm_pe_ltp=75.0, is_live=False)
        monthly = _dummy_expiry_analysis("30-Jul-2026", atm_ce_ltp=350.0, atm_pe_ltp=233.0, is_live=False)
        _, body, _ = notifier._format_signal(
            _dummy_signal(SignalType.BUY_CE), _dummy_risk(SignalType.BUY_CE), "",
            option_analysis=weekly, monthly_option_analysis=monthly,
        )
        assert "Weekly*" in body
        assert "Monthly*" in body
        assert "* estimated" in body

    def test_sub_rupee_premium_shows_decimals_not_zero(self, notifier):
        # Near-expiry weekly premiums can decay to paise (e.g. 0.10) — must not
        # round to a misleading "0".
        weekly = _dummy_expiry_analysis("07-Jul-2026", atm_ce_ltp=0.10, atm_pe_ltp=1.2)
        _, body, _ = notifier._format_signal(
            _dummy_signal(SignalType.BUY_CE), _dummy_risk(SignalType.BUY_CE), "",
            option_analysis=weekly,
        )
        buy_row = next(l for l in body.splitlines() if l.startswith("Buy ₹"))
        assert "0.10" in buy_row

    def test_live_prices_not_labelled_estimate(self, notifier):
        weekly = _dummy_expiry_analysis("10-Jul-2026", atm_ce_ltp=142.0, atm_pe_ltp=110.0, is_live=True)
        _, body, _ = notifier._format_signal(
            _dummy_signal(SignalType.BUY_CE), _dummy_risk(SignalType.BUY_CE), "",
            option_analysis=weekly,
        )
        assert "Weekly" in body
        assert "Weekly*" not in body
        assert "* estimated" not in body


class TestSellTargetInSignalNotification:
    def _weekly_with_legs(self):
        legs = [
            _leg(23950, ce_ltp=178.0, pe_ltp=55.0),
            _leg(24000, ce_ltp=142.0, pe_ltp=110.0, is_atm=True),
            _leg(24050, ce_ltp=110.0, pe_ltp=140.0),
        ]
        return _expiry_analysis_with_legs(24000, legs)

    @staticmethod
    def _row_values(body: str, label: str) -> list[float]:
        import re
        row = next(l for l in body.splitlines() if l.startswith(label))
        return [float(v.replace(",", "")) for v in re.findall(r"[\d,]+\.?\d*", row)]

    def test_buy_ce_shows_sell_target_and_sl_exit_rows(self, notifier):
        _, body, _ = notifier._format_signal(
            _dummy_signal(SignalType.BUY_CE), _dummy_risk(SignalType.BUY_CE), "",
            option_analysis=self._weekly_with_legs(),
        )
        assert "Sell ₹" in body
        assert "Exit ₹" in body
        assert "(Sell=at target, Exit=at stop-loss)" in body

    def test_sell_target_above_entry_for_buy_ce(self, notifier):
        # Risk target is above spot for CE, so the sell premium must exceed entry.
        _, body, _ = notifier._format_signal(
            _dummy_signal(SignalType.BUY_CE), _dummy_risk(SignalType.BUY_CE), "",
            option_analysis=self._weekly_with_legs(),
        )
        assert self._row_values(body, "Sell ₹")[0] > 142.0   # entry premium
        assert self._row_values(body, "Exit ₹")[0] < 142.0

    def test_no_sell_row_when_risk_invalid(self, notifier):
        _, body, _ = notifier._format_signal(
            _dummy_signal(SignalType.BUY_CE), _dummy_risk(SignalType.HOLD), "",
            option_analysis=self._weekly_with_legs(),
        )
        assert "Sell ₹" not in body

    def test_sell_row_carries_weekly_and_monthly_cells(self, notifier):
        weekly = self._weekly_with_legs()
        monthly = self._weekly_with_legs()
        monthly.expiry = "28-Jul-2026"
        monthly.days_to_expiry = 17
        _, body, _ = notifier._format_signal(
            _dummy_signal(SignalType.BUY_CE), _dummy_risk(SignalType.BUY_CE), "",
            option_analysis=weekly, monthly_option_analysis=monthly,
        )
        assert len(self._row_values(body, "Sell ₹")) == 2
        assert len(self._row_values(body, "Exit ₹")) == 2


class TestFindItmLegs:
    def test_finds_strike_below_and_above_atm(self):
        legs = [
            _leg(24300, ce_ltp=210.0, pe_ltp=40.0),
            _leg(24350, ce_ltp=170.0, pe_ltp=55.0),
            _leg(24400, ce_ltp=135.0, pe_ltp=75.0, is_atm=True),
            _leg(24450, ce_ltp=100.0, pe_ltp=100.0),
            _leg(24500, ce_ltp=70.0, pe_ltp=130.0),
        ]
        analysis = _expiry_analysis_with_legs(24400, legs)
        itm_call, itm_put = _find_itm_legs(analysis)
        assert itm_call.strike == 24350
        assert itm_put.strike == 24450

    def test_never_returns_the_atm_strike_itself(self):
        # Regression: anchoring on spot (rather than the ATM strike) could
        # return the same strike already shown on the ATM line, making the
        # "ITM" line a confusing repeat. Spot sits just above the ATM strike
        # here, which would trigger that bug if not anchored on ATM.
        legs = [
            _leg(24300, ce_ltp=210.0, pe_ltp=40.0),
            _leg(24400, ce_ltp=135.0, pe_ltp=75.0, is_atm=True),
            _leg(24500, ce_ltp=70.0, pe_ltp=130.0),
        ]
        analysis = _expiry_analysis_with_legs(24400, legs)
        analysis.spot = 24417.0  # just above the ATM strike
        itm_call, itm_put = _find_itm_legs(analysis)
        assert itm_call.strike != analysis.atm_strike
        assert itm_put.strike != analysis.atm_strike
        assert itm_call.strike == 24300
        assert itm_put.strike == 24500

    def test_empty_legs_returns_none_none(self):
        analysis = _expiry_analysis_with_legs(24400, [])
        assert _find_itm_legs(analysis) == (None, None)

    def test_no_strikes_below_atm(self):
        legs = [_leg(24400, ce_ltp=135.0, pe_ltp=75.0, is_atm=True), _leg(24500, ce_ltp=70.0, pe_ltp=130.0)]
        analysis = _expiry_analysis_with_legs(24400, legs)
        itm_call, itm_put = _find_itm_legs(analysis)
        assert itm_call is None
        assert itm_put.strike == 24500


class TestItmLinesInNotification:
    def test_itm_ce_and_pe_shown_for_buy_ce_signal(self, notifier):
        legs = [
            _leg(24350, ce_ltp=178.0, pe_ltp=55.0),
            _leg(24400, ce_ltp=142.0, pe_ltp=75.0, is_atm=True),
            _leg(24450, ce_ltp=110.0, pe_ltp=95.0),
        ]
        weekly = _expiry_analysis_with_legs(24400, legs)
        _, body, _ = notifier._format_signal(
            _dummy_signal(SignalType.BUY_CE), _dummy_risk(SignalType.BUY_CE), "",
            option_analysis=weekly,
        )
        assert "ITM CE" in body and "24350@178" in body
        assert "ITM PE" in body and "24450@95" in body

    def test_itm_lines_omitted_when_no_legs(self, notifier):
        weekly = _dummy_expiry_analysis("10-Jul-2026", atm_ce_ltp=142.0, atm_pe_ltp=110.0)
        _, body, _ = notifier._format_signal(
            _dummy_signal(SignalType.BUY_CE), _dummy_risk(SignalType.BUY_CE), "",
            option_analysis=weekly,
        )
        assert "ITM" not in body

    def test_itm_lines_shown_for_both_weekly_and_monthly(self, notifier):
        weekly_legs = [
            _leg(24350, ce_ltp=178.0, pe_ltp=55.0),
            _leg(24400, ce_ltp=142.0, pe_ltp=75.0, is_atm=True),
            _leg(24450, ce_ltp=110.0, pe_ltp=95.0),
        ]
        monthly_legs = [
            _leg(24350, ce_ltp=410.0, pe_ltp=180.0),
            _leg(24400, ce_ltp=351.0, pe_ltp=210.0, is_atm=True),
            _leg(24450, ce_ltp=300.0, pe_ltp=250.0),
        ]
        weekly = _expiry_analysis_with_legs(24400, weekly_legs)
        monthly = _expiry_analysis_with_legs(24400, monthly_legs)
        monthly.expiry = "28-Jul-2026"
        _, body, _ = notifier._format_signal(
            _dummy_signal(SignalType.BUY_CE), _dummy_risk(SignalType.BUY_CE), "",
            option_analysis=weekly, monthly_option_analysis=monthly,
        )
        # One ITM CE row with a cell per expiry: weekly 24350@178, monthly 24350@410
        itm_ce_row = next(l for l in body.splitlines() if l.startswith("ITM CE"))
        assert "24350@178" in itm_ce_row and "24350@410" in itm_ce_row
        itm_pe_row = next(l for l in body.splitlines() if l.startswith("ITM PE"))
        assert "24450@95" in itm_pe_row and "24450@250" in itm_pe_row


class TestPushoverNotifierMultiSignal:
    def _results(self):
        ema_signal = _dummy_signal(SignalType.BUY_CE)
        vwap_signal = Signal(
            signal=SignalType.HOLD,
            confidence=50,
            reason="No confirmed VWAP breakout.",
            strategy="VWAP_Breakout",
        )
        return [
            (ema_signal, _dummy_risk(SignalType.BUY_CE), "Bullish momentum."),
            (vwap_signal, _dummy_risk(SignalType.HOLD), ""),
        ]

    def test_title_lists_every_strategy(self, notifier):
        title, _, _ = notifier._format_multi_signal(self._results())
        assert "EMA_Crossover" in title
        assert "VWAP_Breakout" in title

    def test_body_contains_each_strategy_section(self, notifier):
        _, body, _ = notifier._format_multi_signal(self._results())
        assert "EMA_Crossover" in body
        assert "VWAP_Breakout" in body
        assert "BUY_CE" in body
        assert "HOLD" in body

    def test_send_multi_signal_success(self, notifier):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"status": 1}
        with patch("requests.post", return_value=mock_resp) as mock_post:
            result = notifier.send_multi_signal(self._results())
        assert result is True
        mock_post.assert_called_once()

    def test_weekly_and_monthly_shown_for_each_strategy(self, notifier):
        weekly = _dummy_expiry_analysis("10-Jul-2026", atm_ce_ltp=142.0, atm_pe_ltp=110.0)
        monthly = _dummy_expiry_analysis("31-Jul-2026", atm_ce_ltp=310.0, atm_pe_ltp=280.0)
        _, body, _ = notifier._format_multi_signal(
            self._results(), option_analysis=weekly, monthly_option_analysis=monthly,
        )
        assert "📌 BUY NIFTY 24000 CE" in body
        assert "10-Jul" in body and "31-Jul" in body
        buy_row = next(l for l in body.splitlines() if l.startswith("Buy ₹"))
        assert "142" in buy_row and "310" in buy_row

    def test_send_multi_signal_silent_when_all_hold(self, notifier):
        hold_signal = Signal(
            signal=SignalType.HOLD, confidence=50, reason="flat", strategy="EMA_Crossover",
        )
        results = [(hold_signal, _dummy_risk(SignalType.HOLD), "")]
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"status": 1}
        with patch("requests.post", return_value=mock_resp) as mock_post:
            notifier.send_multi_signal(results)
        assert mock_post.call_args.kwargs["data"]["sound"] == "none"
