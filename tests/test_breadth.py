"""Tests for the real-time heavyweight breadth checker."""

from unittest.mock import MagicMock, patch

import pytest

from nifty_ai_agent.data.breadth import (
    HEAVYWEIGHT_SYMBOLS,
    BreadthSnapshot,
    fetch_realtime_breadth,
)


def _make_ticker(price: float, prev: float) -> MagicMock:
    ticker = MagicMock()
    ticker.fast_info.last_price = price
    ticker.fast_info.previous_close = prev
    return ticker


class TestFetchRealtimeBreadth:
    @patch("nifty_ai_agent.data.breadth.yf.Ticker")
    def test_all_advancing_returns_bullish(self, mock_ticker):
        mock_ticker.side_effect = lambda sym: _make_ticker(101.0, 100.0)

        result = fetch_realtime_breadth()

        assert result.bias == "BULLISH"
        assert result.advancing == len(HEAVYWEIGHT_SYMBOLS)
        assert result.declining == 0
        assert result.score > 0.2

    @patch("nifty_ai_agent.data.breadth.yf.Ticker")
    def test_all_declining_returns_bearish(self, mock_ticker):
        mock_ticker.side_effect = lambda sym: _make_ticker(99.0, 100.0)

        result = fetch_realtime_breadth()

        assert result.bias == "BEARISH"
        assert result.declining == len(HEAVYWEIGHT_SYMBOLS)
        assert result.advancing == 0
        assert result.score < -0.2

    @patch("nifty_ai_agent.data.breadth.yf.Ticker")
    def test_equal_split_returns_neutral(self, mock_ticker):
        symbols = HEAVYWEIGHT_SYMBOLS
        half = len(symbols) // 2

        def side(sym):
            idx = symbols.index(sym) if sym in symbols else 0
            return _make_ticker(101.0, 100.0) if idx < half else _make_ticker(99.0, 100.0)

        mock_ticker.side_effect = side

        result = fetch_realtime_breadth()

        assert result.bias == "NEUTRAL"
        assert result.advancing == half
        assert result.declining == half

    @patch("nifty_ai_agent.data.breadth.yf.Ticker")
    def test_flat_moves_counted_as_unchanged(self, mock_ticker):
        mock_ticker.side_effect = lambda sym: _make_ticker(100.0, 100.0)

        result = fetch_realtime_breadth()

        assert result.unchanged == len(HEAVYWEIGHT_SYMBOLS)
        assert result.advancing == 0
        assert result.declining == 0
        assert result.bias == "NEUTRAL"

    @patch("nifty_ai_agent.data.breadth.yf.Ticker", side_effect=Exception("network error"))
    def test_total_failure_returns_neutral_snapshot(self, _):
        result = fetch_realtime_breadth()

        assert result.bias == "NEUTRAL"
        assert result.total == 0
        assert result.score == 0.0
        assert result.leaders == []
        assert result.laggards == []

    @patch("nifty_ai_agent.data.breadth.yf.Ticker")
    def test_leaders_and_laggards_populated(self, mock_ticker):
        symbols = HEAVYWEIGHT_SYMBOLS

        def side(sym):
            if sym == symbols[0]:
                return _make_ticker(102.0, 100.0)  # +2% → leader
            if sym == symbols[1]:
                return _make_ticker(98.0, 100.0)   # -2% → laggard
            return _make_ticker(100.0, 100.0)       # flat

        mock_ticker.side_effect = side

        result = fetch_realtime_breadth()

        assert symbols[0].replace(".NS", "") in result.leaders
        assert symbols[1].replace(".NS", "") in result.laggards

    @patch("nifty_ai_agent.data.breadth.yf.Ticker")
    def test_partial_failures_skipped_gracefully(self, mock_ticker):
        def side(sym):
            if sym == "RELIANCE.NS":
                raise RuntimeError("timeout")
            return _make_ticker(101.0, 100.0)

        mock_ticker.side_effect = side

        result = fetch_realtime_breadth()

        assert result.total == len(HEAVYWEIGHT_SYMBOLS) - 1
        assert result.bias == "BULLISH"

    @patch("nifty_ai_agent.data.breadth.yf.Ticker")
    def test_score_range(self, mock_ticker):
        mock_ticker.side_effect = lambda sym: _make_ticker(101.0, 100.0)

        result = fetch_realtime_breadth()

        assert -1.0 <= result.score <= 1.0
