"""Tests for NSE data provider (mocked yfinance and requests)."""

from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from nifty_ai_agent.data.nse_provider import (
    NSEDataProvider,
    _compute_max_pain,
    _compute_pcr,
    _drop_expiring_today,
    _identify_expiries,
    _iso_to_nse_date,
    _nse_date_to_iso,
)


@pytest.fixture
def provider():
    return NSEDataProvider(symbol="^NSEI")


def _mock_hist_df(n: int = 60) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": [24000.0] * n,
            "High": [24100.0] * n,
            "Low": [23900.0] * n,
            "Close": [24050.0] * n,
            "Volume": [1_000_000] * n,
        },
        index=idx,
    )


class TestNSEDataProvider:
    def test_get_spot_data_success(self, provider):
        mock_ticker = MagicMock()
        mock_ticker.fast_info.last_price = 24000.0
        hist = _mock_hist_df(5)
        mock_ticker.history.return_value = hist

        with patch("yfinance.Ticker", return_value=mock_ticker):
            spot = provider.get_spot_data()

        assert spot.price == 24000.0
        assert spot.symbol == "^NSEI"
        assert isinstance(spot.timestamp, datetime)

    def test_get_spot_data_empty_history_retries(self, provider):
        mock_ticker = MagicMock()
        mock_ticker.fast_info.last_price = 24000.0
        mock_ticker.history.return_value = pd.DataFrame()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            with patch("time.sleep"):
                with pytest.raises(Exception):
                    provider.get_spot_data()

    def test_get_historical_data(self, provider):
        hist = _mock_hist_df(60)
        with patch("yfinance.download", return_value=hist):
            df = provider.get_historical_data(days=60)

        assert isinstance(df, pd.DataFrame)
        assert "close" in df.columns
        assert len(df) <= 60

    def test_get_historical_data_empty_raises(self, provider):
        with patch("yfinance.download", return_value=pd.DataFrame()):
            with patch("time.sleep"):
                with pytest.raises(Exception):
                    provider.get_historical_data()

    def test_get_option_chain_success(self, provider):
        mock_json = {
            "records": {
                "expiryDates": ["27-Jun-2024", "04-Jul-2024", "11-Jul-2024", "25-Jul-2024"],
                "data": [
                    {
                        "strikePrice": 24000,
                        "expiryDate": "27-Jun-2024",
                        "CE": {"openInterest": 100000, "lastPrice": 50.0, "impliedVolatility": 14.5},
                        "PE": {"openInterest": 120000, "lastPrice": 45.0, "impliedVolatility": 13.0},
                    },
                    {
                        "strikePrice": 24100,
                        "expiryDate": "27-Jun-2024",
                        "CE": {"openInterest": 80000, "lastPrice": 30.0, "impliedVolatility": 15.0},
                        "PE": {"openInterest": 90000, "lastPrice": 60.0, "impliedVolatility": 12.0},
                    },
                    {
                        "strikePrice": 24000,
                        "expiryDate": "25-Jul-2024",
                        "CE": {"openInterest": 200000, "lastPrice": 80.0, "impliedVolatility": 16.0},
                        "PE": {"openInterest": 180000, "lastPrice": 75.0, "impliedVolatility": 15.5},
                    },
                ],
            }
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_json
        mock_resp.raise_for_status.return_value = None

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch.object(provider, "_get_nse_session", return_value=mock_session):
            chain = provider.get_option_chain()

        assert chain.symbol == "NIFTY"
        assert chain.expiry == "27-Jun-2024"
        assert chain.monthly_expiry == "25-Jul-2024"
        assert chain.pcr > 0
        assert chain.max_pain > 0
        assert not chain.strikes.empty
        assert not chain.monthly_strikes.empty
        assert len(chain.strikes) == 2       # only weekly rows
        assert len(chain.monthly_strikes) == 1  # only monthly rows

    def test_get_option_chain_skips_expiry_dated_today(self, provider):
        today = date.today().strftime("%d-%b-%Y")
        next_week = (date.today() + timedelta(days=7)).strftime("%d-%b-%Y")
        mock_json = {
            "records": {
                "expiryDates": [today, next_week],
                "data": [
                    {
                        "strikePrice": 24000,
                        "expiryDate": today,
                        "CE": {"openInterest": 1, "lastPrice": 0.05, "impliedVolatility": 0},
                        "PE": {"openInterest": 1, "lastPrice": 0.05, "impliedVolatility": 0},
                    },
                    {
                        "strikePrice": 24000,
                        "expiryDate": next_week,
                        "CE": {"openInterest": 100000, "lastPrice": 120.0, "impliedVolatility": 14.0},
                        "PE": {"openInterest": 90000, "lastPrice": 95.0, "impliedVolatility": 13.5},
                    },
                ],
            }
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_json
        mock_resp.raise_for_status.return_value = None
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch.object(provider, "_get_nse_session", return_value=mock_session):
            chain = provider.get_option_chain()

        # Today's expiry should be skipped entirely — "weekly" rolls to next week.
        assert chain.expiry == next_week
        assert len(chain.strikes) == 1
        assert chain.strikes.iloc[0]["ce_ltp"] == 120.0

    def test_get_option_chain_falls_back_to_browser_when_http_blocked(self, provider):
        mock_json = {
            "records": {
                "expiryDates": ["27-Jun-2024", "25-Jul-2024"],
                "data": [
                    {
                        "strikePrice": 24000,
                        "expiryDate": "27-Jun-2024",
                        "CE": {"openInterest": 100000, "lastPrice": 50.0, "impliedVolatility": 14.5},
                        "PE": {"openInterest": 120000, "lastPrice": 45.0, "impliedVolatility": 13.0},
                    },
                ],
            }
        }
        with patch.object(provider, "_fetch_option_chain_json_http", side_effect=ValueError("blocked")):
            with patch.object(provider, "_fetch_option_chain_json_browser", return_value=mock_json) as mock_browser:
                chain = provider.get_option_chain()

        mock_browser.assert_called_once()
        assert chain.symbol == "NIFTY"
        assert chain.expiry == "27-Jun-2024"
        assert not chain.strikes.empty

    def test_get_option_chain_falls_back_to_synthetic_when_both_blocked(self, provider):
        with patch.object(provider, "_fetch_option_chain_json_http", side_effect=ValueError("blocked")):
            with patch.object(
                provider, "_fetch_option_chain_json_browser", side_effect=RuntimeError("browser blocked too")
            ):
                with patch.object(NSEDataProvider, "_synthetic_option_chain") as mock_synth:
                    mock_synth.return_value = "SYNTHETIC_SENTINEL"
                    chain = provider.get_option_chain()

        assert chain == "SYNTHETIC_SENTINEL"

    def test_get_option_chain_uses_upstox_when_token_configured(self):
        provider = NSEDataProvider(symbol="^NSEI", upstox_access_token="fake-token")
        with patch.object(provider, "_fetch_option_chain_via_upstox") as mock_upstox:
            mock_upstox.return_value = "UPSTOX_SENTINEL"
            with patch.object(provider, "_fetch_option_chain_json_http") as mock_http:
                chain = provider.get_option_chain()

        mock_upstox.assert_called_once()
        mock_http.assert_not_called()
        assert chain == "UPSTOX_SENTINEL"

    def test_get_option_chain_falls_back_to_nse_when_upstox_fails(self):
        provider = NSEDataProvider(symbol="^NSEI", upstox_access_token="fake-token")
        with patch.object(provider, "_fetch_option_chain_via_upstox", side_effect=RuntimeError("token expired")):
            with patch.object(provider, "_fetch_option_chain_json_http") as mock_http:
                mock_http.return_value = {"records": {"expiryDates": [], "data": []}}
                chain = provider.get_option_chain()

        mock_http.assert_called_once()
        assert chain.symbol == "NIFTY"

    def test_no_upstox_attempt_when_token_blank(self, provider):
        with patch.object(provider, "_fetch_option_chain_via_upstox") as mock_upstox:
            with patch.object(provider, "_fetch_option_chain_json_http") as mock_http:
                mock_http.return_value = {"records": {"expiryDates": [], "data": []}}
                provider.get_option_chain()

        mock_upstox.assert_not_called()

    def test_fetch_option_chain_via_upstox_builds_weekly_and_monthly(self, provider):
        provider = NSEDataProvider(symbol="^NSEI", upstox_access_token="fake-token")
        mock_client = MagicMock()
        mock_client.get_expiries.return_value = ["2026-07-09", "2026-07-30"]
        mock_client.get_option_chain.side_effect = [
            pd.DataFrame([{"strike": 24500, "ce_oi": 1000, "pe_oi": 1200, "ce_ltp": 96.0,
                            "pe_ltp": 75.0, "ce_iv": 12.0, "pe_iv": 11.0}]),
            pd.DataFrame([{"strike": 24500, "ce_oi": 2000, "pe_oi": 2200, "ce_ltp": 350.0,
                            "pe_ltp": 233.0, "ce_iv": 13.0, "pe_iv": 12.5}]),
        ]

        with patch("nifty_ai_agent.data.upstox_provider.UpstoxOptionChainClient", return_value=mock_client):
            chain = provider._fetch_option_chain_via_upstox()

        assert chain.expiry == "09-Jul-2026"
        assert chain.monthly_expiry == "30-Jul-2026"
        assert not chain.strikes.empty
        assert not chain.monthly_strikes.empty
        mock_client.get_option_chain.assert_any_call("NIFTY", "2026-07-09")
        mock_client.get_option_chain.assert_any_call("NIFTY", "2026-07-30")


class TestIsoDateConversion:
    def test_iso_to_nse_date(self):
        assert _iso_to_nse_date("2026-07-09") == "09-Jul-2026"

    def test_nse_date_to_iso(self):
        assert _nse_date_to_iso("09-Jul-2026") == "2026-07-09"

    def test_round_trip(self):
        original = "2026-12-31"
        assert _nse_date_to_iso(_iso_to_nse_date(original)) == original


class TestDropExpiringTodayNseFormat:
    def test_drops_todays_date(self):
        today = date.today().strftime("%d-%b-%Y")
        future = (date.today() + timedelta(days=7)).strftime("%d-%b-%Y")
        assert _drop_expiring_today([today, future]) == [future]

    def test_keeps_list_unchanged_when_today_absent(self):
        dates = [
            (date.today() + timedelta(days=i)).strftime("%d-%b-%Y") for i in (7, 14)
        ]
        assert _drop_expiring_today(dates) == dates

    def test_falls_back_to_original_if_only_today_present(self):
        today = date.today().strftime("%d-%b-%Y")
        assert _drop_expiring_today([today]) == [today]


class TestIdentifyExpiries:
    def test_weekly_and_monthly_different(self):
        # Weekly = 03-Jul, Monthly = 31-Jul
        dates = ["03-Jul-2025", "10-Jul-2025", "17-Jul-2025", "31-Jul-2025"]
        weekly, monthly = _identify_expiries(dates)
        assert weekly == "03-Jul-2025"
        assert monthly == "31-Jul-2025"

    def test_weekly_is_monthly_goes_to_next_month(self):
        # Scenario: only ONE date in July, so that date IS both weekly and monthly.
        # The function should then return the last Thursday of August as the monthly.
        from datetime import datetime as dt
        # One date in July, two dates in August (last one = monthly)
        dates = ["03-Jul-2025", "07-Aug-2025", "28-Aug-2025"]
        weekly, monthly = _identify_expiries(dates)
        assert weekly == "03-Jul-2025"
        # July has only one entry → weekly IS monthly → go to August's last
        assert monthly == "28-Aug-2025"
        w_date = dt.strptime(weekly, "%d-%b-%Y").date()
        m_date = dt.strptime(monthly, "%d-%b-%Y").date()
        assert (m_date.year, m_date.month) > (w_date.year, w_date.month)

    def test_empty_list_returns_empty_strings(self):
        assert _identify_expiries([]) == ("", "")

    def test_single_date_returns_same_for_both(self):
        from datetime import date, timedelta
        future = (date.today() + timedelta(days=10)).strftime("%d-%b-%Y")
        weekly, monthly = _identify_expiries([future])
        assert weekly == monthly == future

    def test_invalid_format_falls_back_to_first(self):
        # Unparseable strings → empty parsed list → fallback to index 0
        dates = ["not-a-date", "also-bad"]
        weekly, monthly = _identify_expiries(dates)
        assert weekly == monthly == dates[0]


class TestComputePCR:
    def test_normal_pcr(self):
        import pandas as pd
        df = pd.DataFrame({"ce_oi": [100, 200], "pe_oi": [150, 250]})
        pcr = _compute_pcr(df)
        assert abs(pcr - 400 / 300) < 0.001

    def test_zero_ce_oi_returns_zero(self):
        import pandas as pd
        df = pd.DataFrame({"ce_oi": [0, 0], "pe_oi": [100, 200]})
        assert _compute_pcr(df) == 0.0

    def test_empty_df(self):
        import pandas as pd
        assert _compute_pcr(pd.DataFrame()) == 0.0


class TestMaxPain:
    def test_empty_df(self):
        assert _compute_max_pain(pd.DataFrame()) == 0.0

    def test_single_strike(self):
        df = pd.DataFrame({"strike": [24000], "ce_oi": [1000], "pe_oi": [1000]})
        result = _compute_max_pain(df)
        assert result == 24000.0

    def test_returns_minimum_pain_strike(self):
        df = pd.DataFrame({
            "strike": [23900, 24000, 24100],
            "ce_oi": [10000, 5000, 1000],
            "pe_oi": [1000, 5000, 10000],
        })
        result = _compute_max_pain(df)
        # Pain is minimized at the balanced strike (24000)
        assert result == 24000.0
