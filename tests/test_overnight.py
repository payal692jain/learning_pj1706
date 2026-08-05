"""Tests for the overnight market analysis report."""

from nifty_ai_agent.data.gift_nifty import GiftNiftyQuote, OpenOutlook
from nifty_ai_agent.data.market_context import IndexSnapshot
from nifty_ai_agent.data.news_fetcher import NewsItem
from nifty_ai_agent.reports.overnight import blend_market_mood, format_overnight_analysis
from nifty_ai_agent.strategies.gap_analyser import GapStats
from nifty_ai_agent.strategies.global_analyser import NewsSentiment


def _indices() -> list[IndexSnapshot]:
    names = [
        ("S&P 500", 0.80), ("Dow Jones", 0.55), ("NASDAQ", 1.10),
        ("Nikkei 225", 0.40), ("Hang Seng", -0.20),
        ("FTSE 100", 0.15), ("DAX", 0.30), ("India VIX", -3.5),
    ]
    out = []
    for name, chg in names:
        d = "↑" if chg > 0.05 else ("↓" if chg < -0.05 else "→")
        out.append(IndexSnapshot(name=name, symbol="X", price=100.0, change_pct=chg, direction=d))
    return out


def _outlook(gap_pct: float = 0.55) -> OpenOutlook:
    gift = GiftNiftyQuote(
        price=24350.0, change=130.0, change_pct=0.54, expiry="28-Aug-2026",
        timestamp="03-Aug-2026 06:40:00", session="SESSION_1",
    )
    prev = 24200.0
    implied = prev * (1 + gap_pct / 100)
    return OpenOutlook(
        gift=gift, nifty_prev_close=prev, implied_open=implied,
        gap_points=implied - prev, gap_pct=gap_pct,
    )


def _stats() -> GapStats:
    return GapStats(
        bucket="SMALL_UP", sample=81, continued=39, faded=42,
        continuation_pct=48.0, median_day_range_pct=0.9, avg_close_vs_open_pct=-0.05,
    )


def _news() -> list[NewsItem]:
    return [
        NewsItem(title="RBI holds rates", source="ET", published="today"),
        NewsItem(title="US tech rallies", source="Reuters", published="today"),
    ]


class TestBlendMarketMood:
    _BULL = NewsSentiment(score=0.5, bullish_hits=3, bearish_hits=1, headlines=4)
    _BEAR = NewsSentiment(score=-0.5, bullish_hits=1, bearish_hits=3, headlines=4)
    _EMPTY = NewsSentiment(score=0.0, bullish_hits=0, bearish_hits=0, headlines=0)

    def test_news_confirms_index_bias(self):
        assert blend_market_mood("BULLISH", self._BULL, self._BULL) == "BULLISH"

    def test_two_opposing_news_regions_neutralise_index_bias(self):
        # index +2, two bearish regions -2 → net 0 → NEUTRAL
        assert blend_market_mood("BULLISH", self._BEAR, self._BEAR) == "NEUTRAL"

    def test_neutral_index_tilts_with_news(self):
        assert blend_market_mood("NEUTRAL", self._BULL, self._EMPTY) == "BULLISH"

    def test_empty_news_keeps_index_bias(self):
        assert blend_market_mood("BEARISH", self._EMPTY, self._EMPTY) == "BEARISH"


class TestFormatOvernightAnalysis:
    def test_full_digest_renders_all_sections(self):
        india = NewsSentiment(score=0.6, bullish_hits=5, bearish_hits=1, headlines=6)
        world = NewsSentiment(score=-0.3, bullish_hits=1, bearish_hits=3, headlines=4)
        title, body = format_overnight_analysis(
            _indices(), "BULLISH", _outlook(), _stats(), _news(),
            india_sentiment=india, world_sentiment=world,
        )
        assert "Overnight" in title and "BULLISH" in title
        assert "NIFTY +" in title                       # gap points in title
        assert "S&P +0.80%" in body and "Nasdaq +1.10%" in body
        assert "India VIX" in body
        assert "India news: BULLISH (5↑/1↓)" in body     # sentiment integrated
        assert "World news: BEARISH (1↑/3↓)" in body
        assert "Implied open" in body
        assert "Usually:" in body                        # base-rate verdict
        assert "HEADLINES" in body and "RBI holds rates" in body
        assert "not a signal" in body                    # disclaimer

    def test_sentiment_omitted_when_no_headlines(self):
        empty = NewsSentiment(score=0.0, bullish_hits=0, bearish_hits=0, headlines=0)
        _, body = format_overnight_analysis(
            _indices(), "NEUTRAL", None, None, [],
            india_sentiment=empty, world_sentiment=empty,
        )
        assert "news:" not in body                        # nothing to show

    def test_no_gift_falls_back_to_bias_only_title(self):
        title, body = format_overnight_analysis(_indices(), "NEUTRAL", None, None, [])
        assert title == "🌍 Overnight: NEUTRAL"
        assert "Implied open" not in body
        assert "S&P" in body                             # global block still present

    def test_missing_index_shows_na(self):
        # Only S&P available — the others render "n/a" rather than crashing.
        one = [IndexSnapshot(name="S&P 500", symbol="X", price=100.0, change_pct=0.5, direction="↑")]
        _, body = format_overnight_analysis(one, "NEUTRAL", None, None, [])
        assert "S&P +0.50%" in body
        assert "n/a" in body                             # Dow/Nasdaq/etc. absent

    def test_gap_down_uses_red_icon(self):
        title, _ = format_overnight_analysis(_indices(), "BEARISH", _outlook(gap_pct=-0.9), _stats(), [])
        assert title.startswith("🔴")

    def test_body_respects_pushover_limit(self):
        title, body = format_overnight_analysis(_indices(), "BULLISH", _outlook(), _stats(), _news())
        assert len(body) <= 1024
