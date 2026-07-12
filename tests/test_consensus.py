"""Tests for the strategy consensus engine."""

from datetime import time as dt_time

import pytest

from nifty_ai_agent.strategies.base import Signal, SignalType
from nifty_ai_agent.strategies.consensus import build_consensus, effective_weight

_MORNING = dt_time(10, 0)
_AFTERNOON = dt_time(14, 0)
_LATE = dt_time(15, 10)


def _sig(strategy: str, signal: SignalType, confidence: int = 70) -> Signal:
    return Signal(signal=signal, confidence=confidence, reason=f"{strategy} says so",
                  strategy=strategy)


class TestEffectiveWeight:
    def test_orb_is_strongest_in_the_morning(self):
        assert effective_weight("Opening_Range_Breakout", _MORNING) > effective_weight(
            "EMA_Crossover", _MORNING
        )

    def test_orb_decays_by_the_afternoon(self):
        morning = effective_weight("Opening_Range_Breakout", _MORNING)
        afternoon = effective_weight("Opening_Range_Breakout", _AFTERNOON)
        assert afternoon < morning

    def test_non_orb_strategies_do_not_decay(self):
        assert effective_weight("VWAP_Breakout", _MORNING) == effective_weight(
            "VWAP_Breakout", _AFTERNOON
        )

    def test_unknown_strategy_gets_a_neutral_weight(self):
        assert effective_weight("Some_New_Strategy", _MORNING) == 1.0


class TestConsensus:
    def test_unanimous_bullish_book_is_a_strong_buy_ce(self):
        signals = [
            _sig("Opening_Range_Breakout", SignalType.BUY_CE, 80),
            _sig("VWAP_Breakout", SignalType.BUY_CE, 75),
            _sig("Supertrend", SignalType.BUY_CE, 70),
        ]
        c = build_consensus(signals, _MORNING)
        assert c.signal == SignalType.BUY_CE
        assert c.conviction == "STRONG"
        assert c.agreement == 1.0
        assert c.is_actionable

    def test_unanimous_bearish_book_is_a_buy_pe(self):
        signals = [_sig(s, SignalType.BUY_PE, 75) for s in ("VWAP_Breakout", "Supertrend")]
        c = build_consensus(signals, _MORNING)
        assert c.signal == SignalType.BUY_PE
        assert c.is_actionable

    def test_a_split_book_refuses_to_pick_a_side(self):
        # Three bullish, three bearish — the tape is fighting itself.
        signals = [
            _sig("VWAP_Breakout", SignalType.BUY_CE, 70),
            _sig("Supertrend", SignalType.BUY_CE, 70),
            _sig("MACD_Momentum", SignalType.BUY_PE, 70),
            _sig("Bollinger_Squeeze", SignalType.BUY_PE, 70),
            _sig("EMA_Crossover", SignalType.BUY_PE, 70),
        ]
        c = build_consensus(signals, _MORNING)
        assert c.signal == SignalType.HOLD
        assert c.conviction == "NO_TRADE"
        assert not c.is_actionable
        assert "split" in c.rationale

    def test_mostly_hold_book_is_no_trade(self):
        signals = [
            _sig("VWAP_Breakout", SignalType.HOLD, 50),
            _sig("Supertrend", SignalType.HOLD, 50),
            _sig("MACD_Momentum", SignalType.HOLD, 50),
            _sig("EMA_Crossover", SignalType.BUY_CE, 60),
        ]
        c = build_consensus(signals, _MORNING)
        assert c.signal == SignalType.HOLD
        assert c.conviction == "NO_TRADE"
        assert "no setup" in c.rationale

    def test_no_new_entries_after_the_cutoff(self):
        """A weekly option bought at 15:10 has one session of value left and an
        overnight gap to survive — the indicators do not get a vote on that."""
        signals = [_sig(s, SignalType.BUY_CE, 90) for s in
                   ("Opening_Range_Breakout", "VWAP_Breakout", "Supertrend")]
        c = build_consensus(signals, _LATE)
        assert c.signal == SignalType.HOLD
        assert c.conviction == "NO_TRADE"
        assert "cutoff" in c.rationale

    def test_dissent_lowers_confidence_below_the_backers_own_conviction(self):
        strong_agreement = build_consensus(
            [_sig("VWAP_Breakout", SignalType.BUY_CE, 80),
             _sig("Supertrend", SignalType.BUY_CE, 80)],
            _MORNING,
        )
        with_dissent = build_consensus(
            [_sig("VWAP_Breakout", SignalType.BUY_CE, 80),
             _sig("Supertrend", SignalType.BUY_CE, 80),
             _sig("MACD_Momentum", SignalType.BUY_PE, 70)],
            _MORNING,
        )
        assert with_dissent.confidence < strong_agreement.confidence
        assert with_dissent.signal == SignalType.BUY_CE  # still bullish, just less sure

    def test_confidence_never_exceeds_the_backers_confidence(self):
        signals = [_sig(s, SignalType.BUY_CE, 60) for s in ("VWAP_Breakout", "Supertrend")]
        assert build_consensus(signals, _MORNING).confidence <= 60

    def test_dissenters_are_reported_for_sizing(self):
        c = build_consensus(
            [_sig("VWAP_Breakout", SignalType.BUY_CE, 85),
             _sig("Supertrend", SignalType.BUY_CE, 85),
             _sig("Opening_Range_Breakout", SignalType.BUY_CE, 85),
             _sig("MACD_Momentum", SignalType.BUY_PE, 60)],
            _MORNING,
        )
        assert [v.strategy for v in c.dissenters] == ["MACD_Momentum"]
        assert len(c.backers) == 3
        assert "Dissent: MACD_Momentum" in c.rationale

    def test_a_stale_orb_carries_less_of_the_afternoon_book(self):
        """The identical book is worth less in the afternoon: an opening-range break is
        the best read of the day at 10:00 and a description of ancient history at 14:00."""
        signals = [
            _sig("Opening_Range_Breakout", SignalType.BUY_CE, 85),
            _sig("VWAP_Breakout", SignalType.BUY_CE, 80),
            _sig("Supertrend", SignalType.BUY_CE, 75),
            _sig("MACD_Momentum", SignalType.BUY_PE, 70),
        ]
        morning = build_consensus(signals, _MORNING)
        afternoon = build_consensus(signals, _AFTERNOON)

        assert morning.signal == afternoon.signal == SignalType.BUY_CE
        assert afternoon.bullish_weight < morning.bullish_weight
        assert afternoon.confidence < morning.confidence

    def test_a_lone_voice_among_holds_is_too_thin_to_trade(self):
        signals = [
            _sig("VWAP_Breakout", SignalType.BUY_CE, 60),
            _sig("Supertrend", SignalType.HOLD, 50),
            _sig("MACD_Momentum", SignalType.HOLD, 50),
            _sig("Bollinger_Squeeze", SignalType.HOLD, 50),
            _sig("EMA_Crossover", SignalType.HOLD, 50),
            _sig("Opening_Range_Breakout", SignalType.HOLD, 50),
        ]
        c = build_consensus(signals, _MORNING)
        assert c.conviction == "NO_TRADE"

    def test_a_four_two_directional_split_still_trades(self):
        """The floor must reject knife-fights without rejecting ordinary dissent — a
        4-2 book is a normal, tradeable consensus."""
        signals = [
            _sig("Opening_Range_Breakout", SignalType.BUY_CE, 80),
            _sig("VWAP_Breakout", SignalType.BUY_CE, 80),
            _sig("Supertrend", SignalType.BUY_CE, 80),
            _sig("MACD_Momentum", SignalType.BUY_CE, 80),
            _sig("Bollinger_Squeeze", SignalType.BUY_PE, 70),
            _sig("EMA_Crossover", SignalType.BUY_PE, 70),
        ]
        c = build_consensus(signals, _MORNING)
        assert c.signal == SignalType.BUY_CE
        assert c.is_actionable

    def test_empty_book_is_no_trade(self):
        c = build_consensus([], _MORNING)
        assert c.signal == SignalType.HOLD
        assert c.conviction == "NO_TRADE"
