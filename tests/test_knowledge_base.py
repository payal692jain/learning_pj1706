"""Tests for the book-grounded knowledge base."""

import pytest

from nifty_ai_agent.ai.knowledge_base import (
    ALL_PATTERNS,
    BROOKS_PATTERNS,
    BULKOWSKI_PATTERNS,
    COULLING_PATTERNS,
    GRIMES_PATTERNS,
    PRING_PATTERNS,
    VOLMAN_PATTERNS,
    TradingPattern,
    format_patterns_for_prompt,
    get_relevant_patterns,
)
from nifty_ai_agent.strategies.base import SignalType


class TestAllPatterns:
    def test_all_books_represented(self):
        sources = {p.source for p in ALL_PATTERNS}
        assert any("Brooks" in s for s in sources)
        assert any("Coulling" in s for s in sources)
        assert any("Bulkowski" in s for s in sources)
        assert any("Grimes" in s for s in sources)
        assert any("Pring" in s for s in sources)
        assert any("Volman" in s for s in sources)

    def test_all_patterns_have_required_fields(self):
        for p in ALL_PATTERNS:
            assert p.name, f"Pattern missing name: {p}"
            assert p.source, f"Pattern missing source: {p}"
            assert p.description, f"Pattern missing description: {p}"
            assert p.entry_note, f"Pattern missing entry_note: {p}"
            assert p.risk_note, f"Pattern missing risk_note: {p}"
            assert p.signal_bias in ("BUY_CE", "BUY_PE", "HOLD", "ANY")
            assert 0 <= p.success_rate <= 100

    def test_total_pattern_count(self):
        assert len(ALL_PATTERNS) >= 15  # at least 15 patterns across 6 books

    def test_each_book_has_patterns(self):
        assert len(BROOKS_PATTERNS) >= 4
        assert len(COULLING_PATTERNS) >= 3
        assert len(BULKOWSKI_PATTERNS) >= 3
        assert len(GRIMES_PATTERNS) >= 2
        assert len(PRING_PATTERNS) >= 2
        assert len(VOLMAN_PATTERNS) >= 2


class TestGetRelevantPatterns:
    def test_returns_patterns_for_buy_ce(self):
        patterns = get_relevant_patterns(
            signal=SignalType.BUY_CE, rsi=65, ema20=24100, ema50=24000
        )
        assert len(patterns) > 0
        for p in patterns:
            assert p.signal_bias in ("BUY_CE", "ANY")

    def test_returns_patterns_for_buy_pe(self):
        patterns = get_relevant_patterns(
            signal=SignalType.BUY_PE, rsi=35, ema20=23900, ema50=24000
        )
        assert len(patterns) > 0
        for p in patterns:
            assert p.signal_bias in ("BUY_PE", "ANY")

    def test_returns_patterns_for_hold(self):
        patterns = get_relevant_patterns(
            signal=SignalType.HOLD, rsi=50, ema20=24000, ema50=24000
        )
        assert len(patterns) > 0

    def test_respects_max_patterns_limit(self):
        for limit in (1, 2, 3, 4, 5):
            patterns = get_relevant_patterns(
                signal=SignalType.BUY_CE, rsi=65, ema20=24100, ema50=24000,
                max_patterns=limit,
            )
            assert len(patterns) <= limit

    def test_variety_across_books(self):
        patterns = get_relevant_patterns(
            signal=SignalType.BUY_CE, rsi=65, ema20=24100, ema50=24000,
            max_patterns=4,
        )
        books = {p.source.split("—")[0].strip() for p in patterns}
        assert len(books) >= 2  # at least 2 different books

    def test_no_duplicates(self):
        patterns = get_relevant_patterns(
            signal=SignalType.BUY_CE, rsi=65, ema20=24100, ema50=24000,
            max_patterns=6,
        )
        names = [p.name for p in patterns]
        assert len(names) == len(set(names))


class TestFormatPatternsForPrompt:
    def test_returns_string(self):
        patterns = get_relevant_patterns(SignalType.BUY_CE, 65, 24100, 24000)
        result = format_patterns_for_prompt(patterns)
        assert isinstance(result, str)

    def test_contains_pattern_names(self):
        patterns = get_relevant_patterns(SignalType.BUY_CE, 65, 24100, 24000)
        result = format_patterns_for_prompt(patterns)
        for p in patterns:
            assert p.name in result

    def test_contains_sources(self):
        patterns = get_relevant_patterns(SignalType.BUY_CE, 65, 24100, 24000)
        result = format_patterns_for_prompt(patterns)
        for p in patterns:
            assert p.source in result

    def test_contains_success_rates_when_nonzero(self):
        patterns = [p for p in ALL_PATTERNS if p.success_rate > 0][:2]
        result = format_patterns_for_prompt(patterns)
        assert "%" in result

    def test_empty_patterns_returns_header(self):
        result = format_patterns_for_prompt([])
        assert "RELEVANT PATTERNS" in result
