"""Tests for news fetcher, market context, and NIFTY 50 movers."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from nifty_ai_agent.data.news_fetcher import (
    NewsItem,
    _strip_html,
    fetch_news,
    format_news_for_notification,
    format_news_for_prompt,
)
from nifty_ai_agent.data.market_context import (
    IndexSnapshot,
    GiftNiftySnapshot,
    MarketContext,
    compute_global_bias,
    format_context_for_notification,
)
from nifty_ai_agent.data.nifty50_stocks import (
    StockMover,
    Nifty50Summary,
    format_movers_for_notification,
)


# ── News fetcher ──────────────────────────────────────────────────────────────

class TestStripHtml:
    def test_removes_tags(self):
        assert _strip_html("<b>Hello</b> <i>World</i>") == "Hello World"

    def test_plain_text_unchanged(self):
        assert _strip_html("Plain text") == "Plain text"

    def test_empty(self):
        assert _strip_html("") == ""


class TestFetchNews:
    def _make_feed(self, titles):
        feed = MagicMock()
        entries = []
        for t in titles:
            e = MagicMock()
            e.get = lambda k, d="", title=t: title if k == "title" else d
            entries.append(e)
        feed.entries = entries
        return feed

    def test_returns_list_of_news_items(self):
        mock_feed = MagicMock()
        mock_feed.entries = []
        with patch("feedparser.parse", return_value=mock_feed):
            items = fetch_news()
        assert isinstance(items, list)

    def test_bad_feed_silently_skipped(self):
        with patch("feedparser.parse", side_effect=Exception("network error")):
            items = fetch_news()
        assert items == []


class TestFormatNews:
    def _items(self, n=3):
        return [NewsItem(title=f"Headline {i}", source="ET", published="") for i in range(n)]

    def test_format_for_notification(self):
        text = format_news_for_notification(self._items())
        assert "Headline 0" in text

    def test_format_for_prompt(self):
        text = format_news_for_prompt(self._items())
        assert "LATEST MARKET HEADLINES" in text

    def test_empty_returns_no_news(self):
        assert "No news" in format_news_for_notification([])


# ── Market context ────────────────────────────────────────────────────────────

class TestComputeGlobalBias:
    def _snapshots(self, changes):
        names = ["S&P 500", "Dow Jones", "NASDAQ", "Nikkei 225"]
        return [
            IndexSnapshot(name=names[i], symbol="X", price=100, change_pct=c, direction="↑")
            for i, c in enumerate(changes)
        ]

    def test_all_positive_is_bullish(self):
        assert compute_global_bias(self._snapshots([0.5, 0.6, 0.7, 0.4])) == "BULLISH"

    def test_all_negative_is_bearish(self):
        assert compute_global_bias(self._snapshots([-0.5, -0.6, -0.7, -0.4])) == "BEARISH"

    def test_mixed_is_neutral(self):
        assert compute_global_bias(self._snapshots([0.5, -0.5, 0.1, -0.1])) == "NEUTRAL"

    def test_empty_is_neutral(self):
        assert compute_global_bias([]) == "NEUTRAL"


class TestFormatContext:
    def test_returns_string(self):
        ctx = MarketContext(
            indices=[IndexSnapshot("S&P 500", "^GSPC", 5000, 0.3, "↑")],
            gift_nifty=GiftNiftySnapshot(price=24100, change=50, change_pct=0.21),
            global_bias="BULLISH",
        )
        text = format_context_for_notification(ctx)
        assert "BULLISH" in text
        assert "GIFT Nifty" in text
        assert "S&P 500" in text

    def test_no_gift_nifty(self):
        ctx = MarketContext(indices=[], gift_nifty=None, global_bias="NEUTRAL")
        text = format_context_for_notification(ctx)
        assert "NEUTRAL" in text


# ── NIFTY 50 movers ───────────────────────────────────────────────────────────

class TestFormatMovers:
    def _summary(self):
        gainers = [StockMover("TCS.NS", "TCS", 2.1, 3900), StockMover("INFY.NS", "INFY", 1.8, 1800)]
        losers  = [StockMover("ADANIENT.NS", "ADANIENT", -2.3, 2500)]
        return Nifty50Summary(
            advances=30, declines=18, unchanged=2,
            top_gainers=gainers, top_losers=losers,
            advance_decline_ratio=1.67,
        )

    def test_returns_string(self):
        text = format_movers_for_notification(self._summary())
        assert isinstance(text, str)

    def test_contains_advance_decline(self):
        text = format_movers_for_notification(self._summary())
        assert "30" in text
        assert "18" in text

    def test_contains_gainer(self):
        text = format_movers_for_notification(self._summary())
        assert "TCS" in text

    def test_contains_loser(self):
        text = format_movers_for_notification(self._summary())
        assert "ADANIENT" in text
