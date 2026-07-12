"""Tests for GIFT Nifty, gap base rates, and the next-session outlook."""

from datetime import time as dt_time
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from nifty_ai_agent.data.gift_nifty import (
    GiftNiftyQuote,
    build_outlook,
    current_session,
    fetch_gift_nifty,
)
from nifty_ai_agent.reports.next_session import format_next_session
from nifty_ai_agent.strategies.gap_analyser import (
    analyse_gap_history,
    classify_gap,
    compute_pivots,
)

_FEED = {
    "data": [
        # Front month, and the feed genuinely repeats rows — dedupe must not break.
        {"INSTRUMENTTYPE": "FUTIDX", "SYMBOL": "NIFTY", "EXPIRYDATE": "28-Jul-2026",
         "LASTPRICE": 24200.5, "CHANGE": -34.0, "PERCHANGE": -0.14,
         "OPEN": 24099.5, "HIGH": 24282.0, "LOW": 24081.5, "TIMESTMP": "11-Jul-2026 02:44:59"},
        {"INSTRUMENTTYPE": "FUTIDX", "SYMBOL": "NIFTY", "EXPIRYDATE": "28-Jul-2026",
         "LASTPRICE": 24200.5, "CHANGE": -34.0, "PERCHANGE": -0.14,
         "OPEN": 24099.5, "HIGH": 24282.0, "LOW": 24081.5, "TIMESTMP": "11-Jul-2026 02:44:59"},
        # A far month that must never be mistaken for the front month.
        {"INSTRUMENTTYPE": "FUTIDX", "SYMBOL": "NIFTY", "EXPIRYDATE": "25-Aug-2026",
         "LASTPRICE": 24306.0, "CHANGE": 0.0, "PERCHANGE": 0.0, "TIMESTMP": "10-Jul-2026 15:00:09"},
        # Options rows share the feed and must be filtered out.
        {"INSTRUMENTTYPE": "Index Options", "SYMBOL": "NIFTY", "EXPIRYDATE": "28-Jul-2026",
         "OPTIONTYPE": "CE", "STRIKEPRICE": 24200, "LASTPRICE": 0},
    ]
}


def _resp(payload, status=200):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload
    r.raise_for_status.return_value = None
    return r


class TestSessions:
    def test_morning_session(self):
        assert current_session(dt_time(7, 0)) == "SESSION_1"

    def test_overnight_session_before_midnight(self):
        assert current_session(dt_time(20, 0)) == "SESSION_2"

    def test_overnight_session_after_midnight(self):
        """Session 2 wraps past midnight — a naive open<=now<=close test calls both
        20:00 and 02:00 'closed', which would blank the outlook exactly when it runs."""
        assert current_session(dt_time(2, 0)) == "SESSION_2"

    def test_closed_between_sessions(self):
        assert current_session(dt_time(16, 0)) == "CLOSED"
        assert current_session(dt_time(4, 0)) == "CLOSED"


class TestFetch:
    def test_picks_the_front_month_not_the_first_row(self):
        with patch("nifty_ai_agent.data.gift_nifty.requests.get", return_value=_resp(_FEED)):
            q = fetch_gift_nifty(now=dt_time(20, 0))
        assert q.expiry == "28-Jul-2026"
        assert q.price == 24200.5
        assert q.change_pct == -0.14
        assert q.session == "SESSION_2"

    def test_prev_close_is_derived_from_the_change(self):
        with patch("nifty_ai_agent.data.gift_nifty.requests.get", return_value=_resp(_FEED)):
            q = fetch_gift_nifty(now=dt_time(20, 0))
        assert q.prev_close == pytest.approx(24234.5)

    def test_option_rows_are_ignored(self):
        with patch("nifty_ai_agent.data.gift_nifty.requests.get", return_value=_resp(_FEED)):
            q = fetch_gift_nifty(now=dt_time(20, 0))
        assert q.price != 0  # would be the CE row's LASTPRICE of 0

    def test_a_dead_feed_returns_none_rather_than_raising(self):
        with patch("nifty_ai_agent.data.gift_nifty.requests.get",
                   side_effect=ConnectionError("nseix down")):
            assert fetch_gift_nifty(now=dt_time(20, 0)) is None

    def test_an_empty_feed_returns_none(self):
        with patch("nifty_ai_agent.data.gift_nifty.requests.get",
                   return_value=_resp({"data": []})):
            assert fetch_gift_nifty(now=dt_time(20, 0)) is None


class TestImpliedOpen:
    def _quote(self, change_pct: float) -> GiftNiftyQuote:
        return GiftNiftyQuote(
            price=24200.0, change=change_pct * 242, change_pct=change_pct,
            expiry="28-Jul-2026", timestamp="", session="SESSION_2",
        )

    def test_implied_open_uses_percent_move_not_the_futures_level(self):
        """GIFT is a FUTURE and carries a basis over spot. Differencing its level
        against NIFTY's close would book that carry premium as a phantom gap-up
        every single day — the % move is basis-neutral."""
        outlook = build_outlook(self._quote(0.0), nifty_prev_close=24000.0)
        # GIFT sits 200 pts above spot, yet a 0% GIFT move implies a FLAT open.
        assert outlook.implied_open == pytest.approx(24000.0)
        assert outlook.gap_points == pytest.approx(0.0)
        assert outlook.direction == "FLAT"

    def test_gap_up_is_detected(self):
        outlook = build_outlook(self._quote(0.8), 24000.0)
        assert outlook.direction == "GAP_UP"
        assert outlook.bucket == "LARGE_UP"
        assert outlook.gap_points > 0

    def test_gap_down_is_detected(self):
        outlook = build_outlook(self._quote(-0.4), 24000.0)
        assert outlook.direction == "GAP_DOWN"
        assert outlook.bucket == "SMALL_DOWN"

    def test_noise_sized_moves_are_flat(self):
        assert build_outlook(self._quote(0.05), 24000.0).direction == "FLAT"


class TestClassifyGap:
    @pytest.mark.parametrize("pct,bucket", [
        (0.0, "FLAT"), (0.1, "FLAT"), (-0.1, "FLAT"),
        (0.3, "SMALL_UP"), (-0.3, "SMALL_DOWN"),
        (1.2, "LARGE_UP"), (-1.2, "LARGE_DOWN"),
    ])
    def test_buckets(self, pct, bucket):
        assert classify_gap(pct) == bucket

    def test_forecast_and_lookup_use_the_same_boundaries(self):
        """If gift_nifty and gap_analyser disagreed on a boundary, a gap would be
        forecast in one bucket and its base rate looked up in another."""
        quote = GiftNiftyQuote(price=1, change=1, change_pct=0.8, expiry="", timestamp="")
        outlook = build_outlook(quote, 24000.0)
        assert outlook.bucket == classify_gap(outlook.gap_pct)


def _daily(gaps_and_outcomes: list[tuple[float, float]]) -> pd.DataFrame:
    """Build daily bars where each entry is (gap_pct, close_vs_open_pct)."""
    rows = [{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}]
    for gap_pct, cvo_pct in gaps_and_outcomes:
        prev_close = rows[-1]["close"]
        o = prev_close * (1 + gap_pct / 100)
        c = o * (1 + cvo_pct / 100)
        rows.append({"open": o, "high": max(o, c) * 1.005, "low": min(o, c) * 0.995, "close": c})
    return pd.DataFrame(rows, index=pd.date_range("2025-01-01", periods=len(rows), freq="D"))


class TestGapHistory:
    def test_continuation_is_measured_from_the_open_not_the_prior_close(self):
        """A gap trader gets filled at the OPEN. Measuring from yesterday's close
        would score a gap-up that bled all day as a 'win' merely because it stayed
        above the prior close."""
        # Gapped up 1%, then fell 0.5% from the open — that is a FADE, not a win.
        stats = analyse_gap_history(_daily([(1.0, -0.5)] * 10), "LARGE_UP")
        assert stats.continuation_pct == 0.0
        assert stats.faded == 10

    def test_a_continuing_gap_is_scored_as_continuation(self):
        stats = analyse_gap_history(_daily([(1.0, 0.6)] * 10), "LARGE_UP")
        assert stats.continuation_pct == 100.0

    def test_gap_down_continuation_means_closing_lower_still(self):
        stats = analyse_gap_history(_daily([(-1.0, -0.6)] * 10), "LARGE_DOWN")
        assert stats.continuation_pct == 100.0
        stats_bounce = analyse_gap_history(_daily([(-1.0, +0.6)] * 10), "LARGE_DOWN")
        assert stats_bounce.continuation_pct == 0.0

    def test_a_thin_sample_reports_itself_as_unreliable(self):
        """Three data points must not be dressed up as a percentage."""
        stats = analyse_gap_history(_daily([(1.0, 0.5)] * 3), "LARGE_UP")
        assert not stats.is_reliable
        assert "no usable base rate" in stats.verdict

    def test_no_matches_yields_an_empty_sample(self):
        stats = analyse_gap_history(_daily([(0.0, 0.1)] * 10), "LARGE_UP")
        assert stats.sample == 0
        assert not stats.is_reliable

    def test_a_fading_bucket_warns_against_chasing_the_open(self):
        stats = analyse_gap_history(_daily([(1.0, -0.5)] * 12), "LARGE_UP")
        assert stats.is_reliable
        assert "losing side" in stats.verdict

    def test_a_following_bucket_endorses_the_gap(self):
        stats = analyse_gap_history(_daily([(1.0, 0.5)] * 12), "LARGE_UP")
        assert "worth following" in stats.verdict

    def test_missing_columns_raise(self):
        with pytest.raises(ValueError, match="missing columns"):
            analyse_gap_history(pd.DataFrame({"open": [1.0]}), "FLAT")


class TestPivots:
    def test_pivot_ladder_is_ordered(self):
        p = compute_pivots(high=24300, low=24100, close=24200)
        assert p.s2 < p.s1 < p.pivot < p.r1 < p.r2

    def test_pivot_is_the_average_of_hlc(self):
        p = compute_pivots(24300, 24100, 24200)
        assert p.pivot == pytest.approx(24200.0)

    def test_context_locates_a_price_in_the_ladder(self):
        p = compute_pivots(24300, 24100, 24200)
        assert "above R2" in p.context_for(p.r2 + 10)
        assert "below S2" in p.context_for(p.s2 - 10)


class TestNextSessionNotification:
    def _parts(self, change_pct: float, outcomes: list[tuple[float, float]]):
        quote = GiftNiftyQuote(
            price=24200.0, change=change_pct * 242, change_pct=change_pct,
            expiry="28-Jul-2026", timestamp="11-Jul-2026 02:44:59", session="SESSION_2",
        )
        outlook = build_outlook(quote, 24000.0)
        stats = analyse_gap_history(_daily(outcomes), outlook.bucket)
        pivots = compute_pivots(24300, 24100, 24200)
        return outlook, stats, pivots

    def test_body_shows_gift_implied_open_and_base_rate(self):
        outlook, stats, pivots = self._parts(0.9, [(1.0, -0.5)] * 12)
        title, body = format_next_session(outlook, stats, pivots)
        assert "GIFT" in body
        assert "Implied open" in body
        assert "WHAT THIS GAP USUALLY DOES" in body
        assert "LEVELS FOR TOMORROW" in body
        assert "Gap Up" in title

    def test_body_never_breaches_the_pushover_limit(self):
        for pct in (-1.5, -0.4, 0.0, 0.4, 1.5):
            outlook, stats, pivots = self._parts(pct, [(pct or 0.01, 0.3)] * 20)
            _, body = format_next_session(outlook, stats, pivots)
            assert len(body) <= 1024

    def test_the_session_is_named_so_you_know_how_stale_the_read_is(self):
        outlook, stats, pivots = self._parts(0.5, [(0.5, 0.2)] * 12)
        _, body = format_next_session(outlook, stats, pivots)
        assert "Session 2" in body

    def test_it_never_presents_the_gap_as_a_trading_signal(self):
        outlook, stats, pivots = self._parts(0.9, [(1.0, 0.5)] * 12)
        _, body = format_next_session(outlook, stats, pivots)
        assert "not a signal" in body

    def test_it_works_without_pivots(self):
        outlook, stats, _ = self._parts(0.5, [(0.5, 0.2)] * 12)
        _, body = format_next_session(outlook, stats, None)
        assert "LEVELS FOR TOMORROW" not in body
