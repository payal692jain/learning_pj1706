"""Tests for the BANKNIFTY heavyweight breadth checker."""

from unittest.mock import MagicMock, patch

from nifty_ai_agent.data.banknifty_breadth import (
    BANKNIFTY_HEAVYWEIGHT_SYMBOLS,
    fetch_banknifty_breadth,
)


def _make_ticker(price: float, prev: float) -> MagicMock:
    ticker = MagicMock()
    ticker.fast_info.last_price = price
    ticker.fast_info.previous_close = prev
    return ticker


class TestFetchBankniftyBreadth:
    @patch("nifty_ai_agent.data.banknifty_breadth.yf.Ticker")
    def test_all_advancing_returns_bullish(self, mock_ticker):
        mock_ticker.side_effect = lambda sym: _make_ticker(101.0, 100.0)

        result = fetch_banknifty_breadth()

        assert result.bias == "BULLISH"
        assert result.advancing == len(BANKNIFTY_HEAVYWEIGHT_SYMBOLS)
        assert result.declining == 0

    @patch("nifty_ai_agent.data.banknifty_breadth.yf.Ticker")
    def test_all_declining_returns_bearish(self, mock_ticker):
        mock_ticker.side_effect = lambda sym: _make_ticker(99.0, 100.0)

        result = fetch_banknifty_breadth()

        assert result.bias == "BEARISH"
        assert result.declining == len(BANKNIFTY_HEAVYWEIGHT_SYMBOLS)

    @patch("nifty_ai_agent.data.banknifty_breadth.yf.Ticker", side_effect=Exception("network error"))
    def test_total_failure_returns_neutral_snapshot(self, _):
        result = fetch_banknifty_breadth()

        assert result.bias == "NEUTRAL"
        assert result.total == 0
        assert result.score == 0.0

    @patch("nifty_ai_agent.data.banknifty_breadth.yf.Ticker")
    def test_leaders_and_laggards_populated(self, mock_ticker):
        symbols = BANKNIFTY_HEAVYWEIGHT_SYMBOLS

        def side(sym):
            if sym == symbols[0]:
                return _make_ticker(102.0, 100.0)  # +2% -> leader
            if sym == symbols[1]:
                return _make_ticker(98.0, 100.0)   # -2% -> laggard
            return _make_ticker(100.0, 100.0)       # flat

        mock_ticker.side_effect = side

        result = fetch_banknifty_breadth()

        assert symbols[0].replace(".NS", "") in result.leaders
        assert symbols[1].replace(".NS", "") in result.laggards
