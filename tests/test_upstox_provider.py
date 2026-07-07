"""Tests for the Upstox option chain client (mocked HTTP)."""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from nifty_ai_agent.data.upstox_provider import (
    UpstoxAuthError,
    UpstoxOptionChainClient,
    drop_expiring_today,
)


@pytest.fixture
def client():
    return UpstoxOptionChainClient(access_token="fake-token")


class TestDropExpiringToday:
    def test_drops_todays_date(self):
        today = date.today().isoformat()
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        assert drop_expiring_today([today, tomorrow]) == [tomorrow]

    def test_keeps_list_unchanged_when_today_absent(self):
        future = [(date.today() + timedelta(days=i)).isoformat() for i in (7, 14)]
        assert drop_expiring_today(future) == future

    def test_falls_back_to_original_if_only_today_present(self):
        today = date.today().isoformat()
        assert drop_expiring_today([today]) == [today]


class TestGetExpiries:
    def test_returns_sorted_unique_expiries(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "data": [{"expiry": "2026-07-30"}, {"expiry": "2026-07-09"}, {"expiry": "2026-07-09"}]
        }
        with patch("requests.get", return_value=mock_resp):
            expiries = client.get_expiries("NIFTY")
        assert expiries == ["2026-07-09", "2026-07-30"]

    def test_drops_expiry_dated_today(self, client):
        today = date.today().isoformat()
        future = (date.today() + timedelta(days=7)).isoformat()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"data": [{"expiry": today}, {"expiry": future}]}
        with patch("requests.get", return_value=mock_resp):
            expiries = client.get_expiries("NIFTY")
        assert expiries == [future]

    def test_raises_without_token(self):
        client = UpstoxOptionChainClient(access_token="")
        with pytest.raises(UpstoxAuthError):
            client.get_expiries("NIFTY")

    def test_unknown_index_raises(self, client):
        with pytest.raises(ValueError):
            client.get_expiries("BANKNIFTY")

    def test_401_raises_auth_error(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        with patch("requests.get", return_value=mock_resp):
            with pytest.raises(UpstoxAuthError):
                client.get_expiries("NIFTY")


class TestGetOptionChain:
    def test_parses_calls_and_puts_into_dataframe(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "status": "success",
            "data": [
                {
                    "strike_price": 24500,
                    "call_options": {
                        "market_data": {"ltp": 96.4, "oi": 123000},
                        "option_greeks": {"iv": 12.5},
                    },
                    "put_options": {
                        "market_data": {"ltp": 74.9, "oi": 145000},
                        "option_greeks": {"iv": 11.8},
                    },
                },
            ],
        }
        with patch("requests.get", return_value=mock_resp):
            df = client.get_option_chain("NIFTY", "2026-07-09")

        assert len(df) == 1
        row = df.iloc[0]
        assert row["strike"] == 24500
        assert row["ce_ltp"] == 96.4
        assert row["pe_ltp"] == 74.9
        assert row["ce_oi"] == 123000
        assert row["pe_oi"] == 145000
        assert row["ce_iv"] == 12.5
        assert row["pe_iv"] == 11.8

    def test_status_not_success_raises(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"status": "error", "errors": ["bad request"]}
        with patch("requests.get", return_value=mock_resp):
            with pytest.raises(RuntimeError):
                client.get_option_chain("NIFTY", "2026-07-09")

    def test_raises_without_token(self):
        client = UpstoxOptionChainClient(access_token="")
        with pytest.raises(UpstoxAuthError):
            client.get_option_chain("NIFTY", "2026-07-09")

    def test_403_raises_auth_error(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        with patch("requests.get", return_value=mock_resp):
            with pytest.raises(UpstoxAuthError):
                client.get_option_chain("NIFTY", "2026-07-09")
