"""Tests for the Upstox option chain client (mocked HTTP)."""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from nifty_ai_agent.data.upstox_provider import (
    UpstoxAuthError,
    UpstoxClient,
    drop_expiring_today,
)


@pytest.fixture
def client():
    return UpstoxClient(access_token="fake-token")


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
        # Use future-relative dates — a hardcoded date can collide with the
        # real "today" as the test suite runs across different calendar days,
        # which would make drop_expiring_today() filter it out unexpectedly.
        near = (date.today() + timedelta(days=7)).isoformat()
        far = (date.today() + timedelta(days=28)).isoformat()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "data": [{"expiry": far}, {"expiry": near}, {"expiry": near}]
        }
        with patch("requests.get", return_value=mock_resp):
            expiries = client.get_expiries("NIFTY")
        assert expiries == [near, far]

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
        client = UpstoxClient(access_token="")
        with pytest.raises(UpstoxAuthError):
            client.get_expiries("NIFTY")

    def test_unknown_index_raises(self, client):
        with pytest.raises(ValueError):
            client.get_expiries("FINNIFTY")

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
        client = UpstoxClient(access_token="")
        with pytest.raises(UpstoxAuthError):
            client.get_option_chain("NIFTY", "2026-07-09")

    def test_403_raises_auth_error(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        with patch("requests.get", return_value=mock_resp):
            with pytest.raises(UpstoxAuthError):
                client.get_option_chain("NIFTY", "2026-07-09")


class TestGetLotSize:
    def test_reads_lot_size_from_contract_data(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "data": [{"expiry": "2026-07-28", "lot_size": 65}]
        }
        with patch("requests.get", return_value=mock_resp):
            assert client.get_lot_size("NIFTY") == 65

    def test_raises_when_no_lot_size_present(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"data": [{"expiry": "2026-07-28"}]}
        with patch("requests.get", return_value=mock_resp):
            with pytest.raises(RuntimeError):
                client.get_lot_size("NIFTY")

    def test_raises_without_token(self):
        client = UpstoxClient(access_token="")
        with pytest.raises(UpstoxAuthError):
            client.get_lot_size("NIFTY")


class TestGetQuote:
    def test_parses_price_and_ohlc(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "status": "success",
            "data": {
                "NSE_INDEX:Nifty 50": {
                    "last_price": 24171.1,
                    "ohlc": {"open": 24124.7, "high": 24187.9, "low": 24120.35, "close": 24162.7},
                    "volume": None,
                }
            },
        }
        with patch("requests.get", return_value=mock_resp):
            quote = client.get_quote("NIFTY")
        assert quote == {
            "price": 24171.1, "open": 24124.7, "high": 24187.9, "low": 24120.35, "volume": 0.0,
        }

    def test_empty_data_raises(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"status": "error", "data": {}}
        with patch("requests.get", return_value=mock_resp):
            with pytest.raises(RuntimeError):
                client.get_quote("NIFTY")

    def test_raises_without_token(self):
        client = UpstoxClient(access_token="")
        with pytest.raises(UpstoxAuthError):
            client.get_quote("NIFTY")


class TestGetHistoricalOhlcv:
    def _candle(self, ts, o, h, l, c, v=0):
        return [ts, o, h, l, c, v, 0]

    def test_resamples_1minute_to_5minute(self, client):
        # 10 one-minute candles starting 09:15 -> two complete 5-minute bars.
        historical = [
            self._candle(f"2026-07-09T09:{15+i:02d}:00+05:30", 100 + i, 101 + i, 99 + i, 100.5 + i)
            for i in range(5)
        ]
        intraday = [
            self._candle(f"2026-07-10T09:{15+i:02d}:00+05:30", 200 + i, 201 + i, 199 + i, 200.5 + i)
            for i in range(5)
        ]
        with patch.object(client, "_fetch_candles", return_value=historical):
            with patch.object(client, "_fetch_intraday_candles", return_value=intraday):
                df = client.get_historical_ohlcv("NIFTY", days=10, interval="5m")

        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert len(df) == 2
        first = df.iloc[0]
        assert first["open"] == 100      # first candle's open
        assert first["high"] == 105      # max high across the 5 bars
        assert first["low"] == 99        # min low
        assert first["close"] == 104.5   # last candle's close

    def test_native_day_interval_no_resample(self, client):
        candles = [self._candle("2026-07-08T00:00:00+05:30", 100, 105, 98, 103)]
        with patch.object(client, "_fetch_candles", return_value=candles) as mock_fetch:
            df = client.get_historical_ohlcv("NIFTY", days=10, interval="1d")
        mock_fetch.assert_called_once()
        assert mock_fetch.call_args[0][1] == "day"
        assert len(df) == 1

    def test_raises_when_no_candles(self, client):
        with patch.object(client, "_fetch_candles", return_value=[]):
            with patch.object(client, "_fetch_intraday_candles", return_value=[]):
                with pytest.raises(ValueError):
                    client.get_historical_ohlcv("NIFTY", days=10, interval="5m")

    def test_raises_without_token(self):
        client = UpstoxClient(access_token="")
        with pytest.raises(UpstoxAuthError):
            client.get_historical_ohlcv("NIFTY", days=10)
