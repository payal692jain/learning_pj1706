"""Tests for Upstox token health monitoring and auto-resume."""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from nifty_ai_agent.data.token_health import TokenMonitor
from nifty_ai_agent.data.upstox_provider import UpstoxAuthError


@pytest.fixture
def notifier():
    return MagicMock()


def _monitor(notifier):
    return TokenMonitor(notifier=notifier)


class TestTokenCheck:
    def test_missing_token_is_invalid(self, notifier):
        status = _monitor(notifier).check("")
        assert not status.is_valid
        assert status.is_missing

    def test_rejected_token_is_invalid(self, notifier):
        with patch(
            "nifty_ai_agent.data.token_health.UpstoxClient.get_expiries",
            side_effect=UpstoxAuthError("Upstox rejected the access token (HTTP 401)"),
        ):
            status = _monitor(notifier).check("dead-token")
        assert not status.is_valid
        assert "401" in status.detail

    def test_working_token_is_valid(self, notifier):
        with patch(
            "nifty_ai_agent.data.token_health.UpstoxClient.get_expiries",
            return_value=["2026-07-14"],
        ):
            assert _monitor(notifier).check("good-token").is_valid

    def test_a_network_blip_is_not_reported_as_an_expired_token(self, notifier):
        """Crying wolf on every timeout would train the user to ignore the alert that
        matters. Only an explicit auth rejection counts as expiry."""
        with patch(
            "nifty_ai_agent.data.token_health.UpstoxClient.get_expiries",
            side_effect=ConnectionError("network unreachable"),
        ):
            assert _monitor(notifier).check("good-token").is_valid


class TestAlerting:
    def test_expiry_alerts_once_not_every_cycle(self, notifier):
        """A dead token would otherwise fire ~75 identical pushes in one session."""
        monitor = _monitor(notifier)
        with patch(
            "nifty_ai_agent.data.token_health.UpstoxClient.get_expiries",
            side_effect=UpstoxAuthError("expired"),
        ):
            for _ in range(10):
                monitor.check_and_alert("dead-token")

        assert notifier.send_text.call_count == 1
        assert "expired" in notifier.send_text.call_args.kwargs["title"].lower()

    def test_the_alert_names_the_fix(self, notifier):
        monitor = _monitor(notifier)
        with patch(
            "nifty_ai_agent.data.token_health.UpstoxClient.get_expiries",
            side_effect=UpstoxAuthError("expired"),
        ):
            monitor.check_and_alert("dead-token")
        message = notifier.send_text.call_args.kwargs["message"]
        assert "upstox_login.py" in message
        assert "Analytics Access Token" in message

    def test_recovery_is_announced_and_needs_no_restart(self, notifier):
        monitor = _monitor(notifier)
        with patch(
            "nifty_ai_agent.data.token_health.UpstoxClient.get_expiries",
            side_effect=UpstoxAuthError("expired"),
        ):
            monitor.check_and_alert("dead-token")
        with patch(
            "nifty_ai_agent.data.token_health.UpstoxClient.get_expiries",
            return_value=["2026-07-14"],
        ):
            status = monitor.check_and_alert("fresh-token")

        assert status.is_valid
        assert notifier.send_text.call_count == 2
        assert "restored" in notifier.send_text.call_args.kwargs["title"].lower()

    def test_a_healthy_token_never_alerts(self, notifier):
        monitor = _monitor(notifier)
        with patch(
            "nifty_ai_agent.data.token_health.UpstoxClient.get_expiries",
            return_value=["2026-07-14"],
        ):
            for _ in range(5):
                monitor.check_and_alert("good-token")
        notifier.send_text.assert_not_called()

    def test_a_token_still_dead_the_next_day_alerts_again(self, notifier):
        """Overnight expiry is the common case — it must be flagged each morning,
        not suppressed forever because it was already broken yesterday."""
        monitor = _monitor(notifier)
        with patch(
            "nifty_ai_agent.data.token_health.UpstoxClient.get_expiries",
            side_effect=UpstoxAuthError("expired"),
        ):
            monitor.check_and_alert("dead-token")
            monitor._alerted_on = date.today() - timedelta(days=1)  # simulate a new day
            monitor.check_and_alert("dead-token")

        assert notifier.send_text.call_count == 2

    def test_a_failing_notifier_does_not_crash_the_agent(self, notifier):
        notifier.send_text.side_effect = RuntimeError("pushover down")
        monitor = _monitor(notifier)
        with patch(
            "nifty_ai_agent.data.token_health.UpstoxClient.get_expiries",
            side_effect=UpstoxAuthError("expired"),
        ):
            status = monitor.check_and_alert("dead-token")
        assert not status.is_valid  # survived
