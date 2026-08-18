"""Tests for signal scoring — the feedback loop that makes accuracy measurable."""

import pytest

from nifty_ai_agent.backtesting.scorecard import (
    LOSS,
    MIN_SAMPLE,
    SCRATCH,
    WIN,
    build_scorecard,
    classify,
    format_scorecard,
)


class _P:
    def __init__(self, reason, conviction="STRONG", underlying="NIFTY"):
        self.exit_reason = reason
        self.conviction = conviction
        self.underlying = underlying


def _win(**kw):    return _P("target hit — NIFTY at 25,210", **kw)
def _loss(**kw):   return _P("stop-loss hit — NIFTY at 24,890", **kw)
def _scratch(**kw): return _P("signal reversed to BUY_PE", **kw)


class TestClassify:
    @pytest.mark.parametrize("reason,expected", [
        ("target hit — NIFTY at 25,210", WIN),
        ("stop-loss hit — NIFTY at 24,890", LOSS),
        ("signal reversed to BUY_PE — the CE thesis is gone", SCRATCH),
    ])
    def test_known_reasons(self, reason, expected):
        assert classify(reason) == expected

    def test_unrecognised_reason_is_not_silently_a_win(self):
        assert classify("price looks wrong") not in (WIN, LOSS)
        assert classify("") not in (WIN, LOSS)


class TestScorecard:
    def test_hit_rate_over_decided_trades(self):
        card = build_scorecard([_win(), _win(), _win(), _loss()])
        assert card.wins == 3 and card.losses == 1
        assert card.hit_rate == pytest.approx(0.75)

    def test_scratches_are_excluded_from_the_hit_rate(self):
        """A reversal abandoned the thesis; scoring it either way distorts the number."""
        card = build_scorecard([_win(), _loss(), _scratch(), _scratch()])
        assert card.scratches == 2
        assert card.decided == 2
        assert card.hit_rate == pytest.approx(0.5)

    def test_no_decided_trades_has_no_hit_rate(self):
        assert build_scorecard([_scratch()]).hit_rate is None
        assert build_scorecard([]).hit_rate is None

    def test_significance_threshold(self):
        assert not build_scorecard([_win()] * 5).is_significant
        assert build_scorecard([_win()] * MIN_SAMPLE).is_significant

    def test_buckets_split_by_conviction_and_symbol(self):
        card = build_scorecard([
            _win(conviction="STRONG", underlying="NIFTY"),
            _loss(conviction="WEAK", underlying="NIFTY"),
            _loss(conviction="WEAK", underlying="TCS"),
        ])
        assert card.by_conviction["STRONG"] == [1, 0]
        assert card.by_conviction["WEAK"] == [0, 2]
        assert card.by_underlying["NIFTY"] == [1, 1]


class TestFormatting:
    def test_small_sample_is_called_out(self):
        """4 wins from 5 renders as 80% — the exact number a user must not trust."""
        out = format_scorecard(build_scorecard([_win()] * 4 + [_loss()]))
        assert "80%" in out
        assert "too few to mean anything" in out

    def test_large_sample_has_no_warning(self):
        out = format_scorecard(build_scorecard([_win()] * MIN_SAMPLE))
        assert "too few" not in out

    def test_empty_explains_itself(self):
        assert "No closed positions yet" in format_scorecard(build_scorecard([]))

    def test_scratches_reported_separately(self):
        out = format_scorecard(build_scorecard([_win(), _loss(), _scratch()]))
        assert "Scratched on reversal: 1" in out
