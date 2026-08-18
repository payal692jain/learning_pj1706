"""Tests for stock fundamentals, event detection and the earnings blackout."""

from datetime import date, time, timedelta

import pytest

from nifty_ai_agent.data.fundamentals import (
    Headline,
    StockFundamentals,
    _tag_headline,
    volume_pace_ratio,
)


def _fund(**kw) -> StockFundamentals:
    base = dict(symbol="RELIANCE", pe=23.8, week52_high=1600.0, week52_low=1200.0,
                avg_volume=14_000_000.0)
    base.update(kw)
    return StockFundamentals(**base)


class TestDaysToEarnings:
    def test_future_date(self):
        target = date.today() + timedelta(days=5)
        assert _fund(next_earnings=target.isoformat()).days_to_earnings == 5

    def test_today_is_zero(self):
        assert _fund(next_earnings=date.today().isoformat()).days_to_earnings == 0

    def test_past_date_reads_as_unknown(self):
        """yfinance carries stale dates; a past one must not read as 'far away'."""
        stale = (date.today() - timedelta(days=200)).isoformat()
        assert _fund(next_earnings=stale).days_to_earnings is None

    def test_missing_and_malformed_are_unknown(self):
        assert _fund(next_earnings="").days_to_earnings is None
        assert _fund(next_earnings="not-a-date").days_to_earnings is None


class TestPositionInRange:
    def test_midpoint(self):
        assert _fund().position_in_range(1400.0) == pytest.approx(0.5)

    def test_at_high_and_low(self):
        assert _fund().position_in_range(1600.0) == pytest.approx(1.0)
        assert _fund().position_in_range(1200.0) == pytest.approx(0.0)

    def test_new_high_is_clamped(self):
        """A stock above its published 52w high must not render as '104%'."""
        assert _fund().position_in_range(1650.0) == pytest.approx(1.0)

    def test_unusable_range_returns_none(self):
        assert _fund(week52_high=0.0, week52_low=0.0).position_in_range(1400.0) is None
        assert _fund().position_in_range(0.0) is None


class TestVolumePace:
    def test_half_session_doubles_the_projection(self):
        """Part-day volume must be paced, or every morning reads as quiet."""
        # 09:15 + 187.5 min ≈ 12:22 is half the 375-minute session.
        ratio = volume_pace_ratio(500_000, 1_000_000, time(12, 22))
        assert ratio == pytest.approx(1.0, abs=0.02)

    def test_full_session_is_unscaled(self):
        assert volume_pace_ratio(1_000_000, 1_000_000, time(15, 30)) == pytest.approx(1.0)

    def test_after_close_does_not_over_project(self):
        """Past 15:30 the elapsed clock is capped — otherwise the ratio shrinks."""
        assert volume_pace_ratio(2_000_000, 1_000_000, time(17, 0)) == pytest.approx(2.0)

    def test_pre_open_and_first_bars_return_none(self):
        assert volume_pace_ratio(1000, 1_000_000, time(9, 0)) is None
        assert volume_pace_ratio(1000, 1_000_000, time(9, 17)) is None

    def test_missing_inputs_return_none(self):
        assert volume_pace_ratio(0, 1_000_000, time(12, 0)) is None
        assert volume_pace_ratio(500_000, 0, time(12, 0)) is None


class TestHeadlineTagging:
    @pytest.mark.parametrize("title,tag", [
        ("Acme announces merger with Beta Corp", "MERGER"),
        ("Firm completes acquisition of rival", "M&A"),
        ("Temasek in line-up to sell stake", "STAKE"),
        ("Promoter stake sale hits the counter", "STAKE"),
        ("Fund offloads 5% stake in the company", "STAKE"),
        ("Large block deal crosses on the exchange", "STAKE"),
        ("NSE files for IPO", "IPO"),
        ("Q1 2027 Earnings Call Highlights", "RESULTS"),
        ("Broker upgrades the stock", "RATING"),
    ])
    def test_tags_are_detected(self, title, tag):
        assert tag in _tag_headline(title)

    def test_plain_headline_is_untagged(self):
        assert _tag_headline("Company opens a new office in Pune") == []

    def test_stake_needs_a_verb_to_count(self):
        """Bare 'stakes' is ordinary English, not a corporate action."""
        assert "STAKE" not in _tag_headline("Price war raises the stakes for telcos")
        assert "STAKE" not in _tag_headline("Stakeholders meet on Tuesday")

    def test_event_tags_are_deduplicated_across_headlines(self):
        f = _fund(headlines=[
            Headline("A merger is announced", "2026-08-01", ["MERGER"]),
            Headline("Merger cleared by CCI", "2026-08-02", ["MERGER"]),
            Headline("Q1 results beat", "2026-08-03", ["RESULTS"]),
        ])
        assert f.event_tags == ["MERGER", "RESULTS"]
