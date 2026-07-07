"""Tests for the morning report orchestrator (all external calls mocked)."""

from unittest.mock import MagicMock, patch
import os

import pytest

from nifty_ai_agent.data.market_context import MarketContext, GiftNiftySnapshot, IndexSnapshot
from nifty_ai_agent.data.news_fetcher import NewsItem
from nifty_ai_agent.data.nifty50_stocks import Nifty50Summary, StockMover
from nifty_ai_agent.strategies.option_analyser import ExpiryAnalysis


def _mock_settings():
    s = MagicMock()
    s.pushover_user_key = "key"
    s.pushover_api_token = "token"
    s.nifty_symbol = "^NSEI"
    s.historical_days = 10
    s.data_interval = "5m"
    return s


def _mock_context():
    return MarketContext(
        indices=[IndexSnapshot("S&P 500", "^GSPC", 5000, 0.3, "↑")],
        gift_nifty=GiftNiftySnapshot(24100, 50, 0.21),
        global_bias="BULLISH",
    )


def _mock_analysis():
    return ExpiryAnalysis(
        expiry="27-Jun-2024", spot=24050, atm_strike=24050,
        max_pain=24000, pcr=1.1, legs=[],
        call_oi_resistance=24200, put_oi_support=23800,
        bias="BULLISH",
    )


class TestRunMorningReport:
    @patch("nifty_ai_agent.reports.morning_report.fetch_nifty50_movers")
    @patch("nifty_ai_agent.reports.morning_report.fetch_news")
    @patch("nifty_ai_agent.reports.morning_report.fetch_market_context")
    @patch("nifty_ai_agent.reports.morning_report.NSEDataProvider")
    @patch("nifty_ai_agent.reports.morning_report.PushoverNotifier")
    def test_runs_without_error(
        self, mock_notifier_cls, mock_nse_cls, mock_ctx, mock_news, mock_movers
    ):
        mock_ctx.return_value = _mock_context()
        mock_news.return_value = [NewsItem("Headline 1", "ET", "")]
        mock_movers.return_value = Nifty50Summary(
            advances=30, declines=15, unchanged=5,
            top_gainers=[StockMover("TCS.NS", "TCS", 1.5, 3900)],
            top_losers=[StockMover("ADANIENT.NS", "ADANIENT", -1.2, 2500)],
            advance_decline_ratio=2.0,
        )

        mock_provider = MagicMock()
        mock_provider.get_spot_data.return_value = MagicMock(price=24050.0)
        chain = MagicMock()
        chain.strikes = __import__("pandas").DataFrame()
        chain.expiry = "27-Jun-2024"
        mock_provider.get_option_chain.return_value = chain
        mock_nse_cls.return_value = mock_provider

        mock_notifier = MagicMock()
        mock_notifier.send_text.return_value = True
        mock_notifier_cls.return_value = mock_notifier

        from nifty_ai_agent.reports.morning_report import run_morning_report
        run_morning_report(_mock_settings())

        assert mock_notifier.send_text.call_count >= 2  # at least global + news

    @patch("nifty_ai_agent.reports.morning_report.fetch_market_context", side_effect=Exception("fail"))
    @patch("nifty_ai_agent.reports.morning_report.fetch_news", return_value=[])
    @patch("nifty_ai_agent.reports.morning_report.fetch_nifty50_movers", side_effect=Exception("fail"))
    @patch("nifty_ai_agent.reports.morning_report.NSEDataProvider")
    @patch("nifty_ai_agent.reports.morning_report.PushoverNotifier")
    def test_survives_all_failures(
        self, mock_notifier_cls, mock_nse_cls, *_
    ):
        mock_nse_cls.return_value.get_spot_data.side_effect = Exception("nse fail")
        mock_notifier_cls.return_value = MagicMock()

        from nifty_ai_agent.reports.morning_report import run_morning_report
        run_morning_report(_mock_settings())  # must not raise
