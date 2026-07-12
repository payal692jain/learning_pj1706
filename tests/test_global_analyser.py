"""Tests for the global context analyser (indices, GIFT Nifty, VIX, news)."""

import pytest

from nifty_ai_agent.data.market_context import GiftNiftySnapshot, IndexSnapshot, MarketContext
from nifty_ai_agent.data.news_fetcher import NewsItem
from nifty_ai_agent.strategies.global_analyser import (
    GlobalSnapshot,
    NewsSentiment,
    analyse_news,
    build_snapshot,
    global_confidence_adjustment,
)


def _news(*titles: str) -> list[NewsItem]:
    return [NewsItem(title=t, source="test", published="") for t in titles]


def _snapshot(bias="NEUTRAL", gift=0.0, vix=13.0, news_score=0.0) -> GlobalSnapshot:
    return GlobalSnapshot(
        global_bias=bias,
        gift_nifty_pct=gift,
        vix=vix,
        news=NewsSentiment(news_score, 0, 0, 5),
        is_available=True,
    )


class TestNewsSentiment:
    def test_bullish_headlines_score_positive(self):
        s = analyse_news(_news("Sensex rallies to record high", "Banks surge on strong inflows"))
        assert s.score > 0
        assert s.label == "BULLISH"

    def test_bearish_headlines_score_negative(self):
        s = analyse_news(_news("Markets plunge on recession fears", "Nifty tumbles in selloff"))
        assert s.score < 0
        assert s.label == "BEARISH"

    def test_mixed_headlines_are_neutral(self):
        s = analyse_news(_news("Nifty rallies on inflows", "Sensex tumbles on fears"))
        assert s.label == "NEUTRAL"

    def test_no_headlines_is_neutral_not_bearish(self):
        """An absent feed must not read as bad news — silence is not a bear signal."""
        s = analyse_news([])
        assert s.score == 0.0
        assert s.label == "NEUTRAL"

    def test_top_headline_is_the_most_lopsided_one(self):
        s = analyse_news(_news("Market flat today", "Stocks crash amid selloff and recession fears"))
        assert "crash" in s.top_headline


class TestBuildSnapshot:
    def test_vix_is_lifted_out_of_the_index_list(self):
        ctx = MarketContext(
            indices=[
                IndexSnapshot("S&P 500", "^GSPC", 5000, 0.8, "↑"),
                IndexSnapshot("India VIX", "^INDIAVIX", 21.5, 6.0, "↑"),
            ],
            gift_nifty=GiftNiftySnapshot(price=24100, change=90, change_pct=0.4),
            global_bias="BULLISH",
        )
        snap = build_snapshot(ctx, _news("Markets rally"))
        assert snap.vix == 21.5
        assert snap.vix_regime == "ELEVATED"
        assert snap.gift_nifty_pct == 0.4
        assert snap.is_available

    def test_vix_regimes(self):
        assert _snapshot(vix=12.0).vix_regime == "CALM"
        assert _snapshot(vix=19.0).vix_regime == "ELEVATED"
        assert _snapshot(vix=25.0).vix_regime == "HIGH"


class TestConfidenceAdjustment:
    def test_agreeing_global_tape_adds_confidence(self):
        delta, detail = global_confidence_adjustment(_snapshot(bias="BULLISH"), "BUY_CE")
        assert delta > 0
        assert "confirms" in detail

    def test_opposing_global_tape_costs_more_than_agreement_pays(self):
        agree, _ = global_confidence_adjustment(_snapshot(bias="BULLISH"), "BUY_CE")
        oppose, _ = global_confidence_adjustment(_snapshot(bias="BEARISH"), "BUY_CE")
        assert oppose < 0
        assert abs(oppose) > abs(agree)

    def test_gift_nifty_confirms_a_matching_signal(self):
        delta, detail = global_confidence_adjustment(_snapshot(gift=0.6), "BUY_CE")
        assert delta > 0
        assert "GIFT" in detail

    def test_gift_nifty_below_the_threshold_is_ignored(self):
        delta, _ = global_confidence_adjustment(_snapshot(gift=0.1), "BUY_CE")
        assert delta == 0

    def test_high_vix_penalises_both_directions(self):
        """A rich VIX makes premiums expensive, and this system BUYS options — the
        buyer overpays for the same move whichever way it goes."""
        ce, _ = global_confidence_adjustment(_snapshot(vix=25.0), "BUY_CE")
        pe, _ = global_confidence_adjustment(_snapshot(vix=25.0), "BUY_PE")
        assert ce < 0 and pe < 0
        assert ce == pe

    def test_bearish_news_contradicts_a_bullish_signal(self):
        delta, detail = global_confidence_adjustment(_snapshot(news_score=-0.8), "BUY_CE")
        assert delta < 0
        assert "contradicts" in detail

    def test_hold_signals_are_never_adjusted(self):
        delta, detail = global_confidence_adjustment(_snapshot(bias="BEARISH", vix=30), "HOLD")
        assert delta == 0
        assert detail == ""

    def test_unavailable_context_has_no_opinion(self):
        """A failed fetch must be neutral, not bearish — otherwise a broken RSS feed
        quietly biases every signal downward."""
        delta, detail = global_confidence_adjustment(GlobalSnapshot(), "BUY_CE")
        assert delta == 0
        assert detail == ""

    def test_everything_aligned_beats_everything_opposed(self):
        aligned, _ = global_confidence_adjustment(
            _snapshot(bias="BULLISH", gift=0.8, news_score=0.9, vix=12.0), "BUY_CE",
        )
        opposed, _ = global_confidence_adjustment(
            _snapshot(bias="BEARISH", gift=-0.8, news_score=-0.9, vix=25.0), "BUY_CE",
        )
        assert aligned > 0
        assert opposed < 0
        assert aligned - opposed > 20
