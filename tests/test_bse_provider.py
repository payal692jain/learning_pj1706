"""Tests for the BSE SENSEX data provider (mocked yfinance and Upstox)."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from nifty_ai_agent.data.bse_provider import BSEDataProvider, _next_thursday


@pytest.fixture
def provider():
    return BSEDataProvider(symbol="^BSESN")


class TestGetOptionChain:
    def test_uses_upstox_when_token_configured(self):
        provider = BSEDataProvider(symbol="^BSESN", upstox_access_token="fake-token")
        with patch.object(provider, "_fetch_option_chain_via_upstox") as mock_upstox:
            mock_upstox.return_value = "UPSTOX_SENTINEL"
            chain = provider.get_option_chain()

        mock_upstox.assert_called_once()
        assert chain == "UPSTOX_SENTINEL"

    def test_falls_back_to_synthetic_when_upstox_fails(self):
        provider = BSEDataProvider(symbol="^BSESN", upstox_access_token="fake-token")
        with patch.object(provider, "_fetch_option_chain_via_upstox", side_effect=RuntimeError("boom")):
            with patch.object(provider, "_synthetic_option_chain", return_value="SYNTHETIC_SENTINEL") as mock_synth:
                chain = provider.get_option_chain()

        mock_synth.assert_called_once()
        assert chain == "SYNTHETIC_SENTINEL"

    def test_uses_synthetic_directly_when_no_token(self, provider):
        with patch.object(provider, "_fetch_option_chain_via_upstox") as mock_upstox:
            with patch.object(provider, "_synthetic_option_chain", return_value="SYNTHETIC_SENTINEL") as mock_synth:
                chain = provider.get_option_chain()

        mock_upstox.assert_not_called()
        mock_synth.assert_called_once()
        assert chain == "SYNTHETIC_SENTINEL"

    def test_fetch_option_chain_via_upstox_builds_weekly_and_monthly(self):
        provider = BSEDataProvider(symbol="^BSESN", upstox_access_token="fake-token")
        mock_client = MagicMock()
        mock_client.get_expiries.return_value = ["2026-07-09", "2026-07-30"]
        mock_client.get_option_chain.side_effect = [
            pd.DataFrame([{"strike": 82000, "ce_oi": 1000, "pe_oi": 1200, "ce_ltp": 210.0,
                            "pe_ltp": 180.0, "ce_iv": 12.0, "pe_iv": 11.0}]),
            pd.DataFrame([{"strike": 82000, "ce_oi": 2000, "pe_oi": 2200, "ce_ltp": 610.0,
                            "pe_ltp": 540.0, "ce_iv": 13.0, "pe_iv": 12.5}]),
        ]

        with patch("nifty_ai_agent.data.upstox_provider.UpstoxOptionChainClient", return_value=mock_client):
            chain = provider._fetch_option_chain_via_upstox()

        assert chain.symbol == "SENSEX"
        assert chain.expiry == "09-Jul-2026"
        assert chain.monthly_expiry == "30-Jul-2026"
        assert not chain.strikes.empty
        assert not chain.monthly_strikes.empty
        mock_client.get_option_chain.assert_any_call("SENSEX", "2026-07-09")
        mock_client.get_option_chain.assert_any_call("SENSEX", "2026-07-30")

    def test_fetch_option_chain_via_upstox_raises_on_no_expiries(self):
        provider = BSEDataProvider(symbol="^BSESN", upstox_access_token="fake-token")
        mock_client = MagicMock()
        mock_client.get_expiries.return_value = []
        with patch("nifty_ai_agent.data.upstox_provider.UpstoxOptionChainClient", return_value=mock_client):
            with pytest.raises(RuntimeError):
                provider._fetch_option_chain_via_upstox()


class TestNextThursday:
    def test_returns_a_thursday(self):
        from datetime import datetime as dt
        result = _next_thursday()
        parsed = dt.strptime(result, "%d-%b-%Y").date()
        assert parsed.weekday() == 3  # Thursday

    def test_is_strictly_in_the_future(self):
        from datetime import date, datetime as dt
        result = _next_thursday()
        parsed = dt.strptime(result, "%d-%b-%Y").date()
        assert parsed > date.today()


class TestSyntheticOptionChain:
    def test_returns_thursday_expiry_and_empty_strikes(self, provider):
        mock_spot_ticker = MagicMock()
        mock_spot_ticker.fast_info.last_price = 82000.0
        mock_vix_ticker = MagicMock()
        mock_vix_ticker.fast_info.last_price = 14.0

        def _ticker(symbol):
            return mock_vix_ticker if symbol == "^INDIAVIX" else mock_spot_ticker

        with patch("yfinance.Ticker", side_effect=_ticker):
            chain = provider._synthetic_option_chain()

        assert chain.symbol == "SENSEX"
        assert chain.strikes.empty
        assert chain.pcr > 0
