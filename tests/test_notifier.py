"""Tests for the Pushover notifier (mocked HTTP)."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from nifty_ai_agent.notifier.pushover import PushoverNotifier, _PRIORITY_LOW, _PRIORITY_NORMAL
from nifty_ai_agent.risk.calculator import RiskCalculator
from nifty_ai_agent.strategies.base import Signal, SignalType
from nifty_ai_agent.strategies.option_analyser import ExpiryAnalysis


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

    def test_weekly_and_monthly_contract_both_shown(self, notifier):
        weekly = _dummy_expiry_analysis("10-Jul-2026", atm_ce_ltp=142.0, atm_pe_ltp=110.0)
        monthly = _dummy_expiry_analysis("31-Jul-2026", atm_ce_ltp=310.0, atm_pe_ltp=280.0)
        _, body, _ = notifier._format_signal(
            _dummy_signal(SignalType.BUY_CE), _dummy_risk(SignalType.BUY_CE), "",
            option_analysis=weekly, monthly_option_analysis=monthly,
        )
        assert "Buy (Weekly): NIFTY 24000 CE  10-Jul-2026  @ ₹142" in body
        assert "Buy (Monthly): NIFTY 24000 CE  31-Jul-2026  @ ₹310" in body

    def test_only_weekly_shown_when_monthly_unavailable(self, notifier):
        weekly = _dummy_expiry_analysis("10-Jul-2026", atm_ce_ltp=142.0, atm_pe_ltp=110.0)
        _, body, _ = notifier._format_signal(
            _dummy_signal(SignalType.BUY_CE), _dummy_risk(SignalType.BUY_CE), "",
            option_analysis=weekly, monthly_option_analysis=None,
        )
        assert "Buy (Weekly)" in body
        assert "Buy (Monthly)" not in body

    def test_synthetic_prices_labelled_estimate(self, notifier):
        weekly = _dummy_expiry_analysis("10-Jul-2026", atm_ce_ltp=96.0, atm_pe_ltp=75.0, is_live=False)
        monthly = _dummy_expiry_analysis("30-Jul-2026", atm_ce_ltp=350.0, atm_pe_ltp=233.0, is_live=False)
        _, body, _ = notifier._format_signal(
            _dummy_signal(SignalType.BUY_CE), _dummy_risk(SignalType.BUY_CE), "",
            option_analysis=weekly, monthly_option_analysis=monthly,
        )
        assert "Buy (Weekly, Est.)" in body
        assert "Buy (Monthly, Est.)" in body

    def test_sub_rupee_premium_shows_decimals_not_zero(self, notifier):
        # Near-expiry weekly premiums can decay to paise (e.g. 0.10) — must not
        # round to a misleading "@ ₹0".
        weekly = _dummy_expiry_analysis("07-Jul-2026", atm_ce_ltp=0.10, atm_pe_ltp=1.2)
        _, body, _ = notifier._format_signal(
            _dummy_signal(SignalType.BUY_CE), _dummy_risk(SignalType.BUY_CE), "",
            option_analysis=weekly,
        )
        assert "@ ₹0.10" in body
        assert "@ ₹0 " not in body and not body.rstrip().endswith("@ ₹0")

    def test_live_prices_not_labelled_estimate(self, notifier):
        weekly = _dummy_expiry_analysis("10-Jul-2026", atm_ce_ltp=142.0, atm_pe_ltp=110.0, is_live=True)
        _, body, _ = notifier._format_signal(
            _dummy_signal(SignalType.BUY_CE), _dummy_risk(SignalType.BUY_CE), "",
            option_analysis=weekly,
        )
        assert "Buy (Weekly):" in body
        assert "Est." not in body


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
        assert "Buy (Weekly): NIFTY 24000 CE  10-Jul-2026  @ ₹142" in body
        assert "Buy (Monthly): NIFTY 24000 CE  31-Jul-2026  @ ₹310" in body

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
