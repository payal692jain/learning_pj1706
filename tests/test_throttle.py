"""Tests for notification throttling — quiet on the clock, immediate on change."""

from datetime import datetime, timedelta

import pytest

from nifty_ai_agent.notifier.throttle import NotificationThrottle

T0 = datetime(2026, 8, 20, 10, 0)


def _throttle(**kw):
    return NotificationThrottle(**kw)


def _send(t, key="NIFTY", signal="BUY_CE", conviction="MODERATE",
          confidence=60, spot=24000.0, now=T0, force=False):
    return t.evaluate(key, signal=signal, conviction=conviction,
                      confidence=confidence, spot=spot, now=now, force=force)


class TestRoutineCadence:
    def test_first_signal_always_goes_out(self):
        assert _send(_throttle()).send

    def test_unchanged_signal_is_suppressed_inside_the_interval(self):
        t = _throttle(interval_minutes=15)
        _send(t)
        assert not _send(t, now=T0 + timedelta(minutes=5)).send
        assert not _send(t, now=T0 + timedelta(minutes=14)).send

    def test_interval_elapsing_releases_a_routine_update(self):
        t = _throttle(interval_minutes=15)
        _send(t)
        decision = _send(t, now=T0 + timedelta(minutes=15))
        assert decision.send and not decision.is_urgent
        assert decision.reason == "scheduled update"

    def test_the_clock_restarts_from_the_last_send_not_the_last_check(self):
        """A suppressed check must not push the next scheduled update further out."""
        t = _throttle(interval_minutes=15)
        _send(t)
        _send(t, now=T0 + timedelta(minutes=10))       # suppressed
        assert _send(t, now=T0 + timedelta(minutes=15)).send


class TestSuddenChange:
    def test_signal_flip_notifies_immediately(self):
        t = _throttle()
        _send(t, signal="BUY_CE")
        d = _send(t, signal="BUY_PE", now=T0 + timedelta(minutes=1))
        assert d.send and d.is_urgent
        assert "flipped" in d.reason

    def test_going_flat_is_also_a_flip(self):
        t = _throttle()
        _send(t, signal="BUY_CE")
        assert _send(t, signal="HOLD", now=T0 + timedelta(minutes=1)).send

    def test_conviction_upgrade_notifies(self):
        """The engine going from hedging to committing is worth interrupting for."""
        t = _throttle()
        _send(t, conviction="WEAK")
        d = _send(t, conviction="STRONG", now=T0 + timedelta(minutes=2))
        assert d.send and "WEAK → STRONG" in d.reason

    def test_conviction_downgrade_does_not_notify(self):
        """Losing conviction is not urgent — it resolves into silence, not a trade."""
        t = _throttle()
        _send(t, conviction="STRONG")
        assert not _send(t, conviction="WEAK", now=T0 + timedelta(minutes=2)).send

    def test_large_confidence_jump_notifies(self):
        t = _throttle(confidence_jump=15)
        _send(t, confidence=60)
        assert _send(t, confidence=78, now=T0 + timedelta(minutes=1)).send

    def test_small_confidence_drift_is_suppressed(self):
        t = _throttle(confidence_jump=15)
        _send(t, confidence=60)
        assert not _send(t, confidence=68, now=T0 + timedelta(minutes=1)).send

    def test_sharp_price_move_notifies(self):
        t = _throttle(move_pct=0.35)
        _send(t, spot=24000.0)
        d = _send(t, spot=24120.0, now=T0 + timedelta(minutes=2))   # +0.5%
        assert d.send and d.is_urgent and "moved up" in d.reason

    def test_price_move_works_in_both_directions(self):
        t = _throttle(move_pct=0.35)
        _send(t, spot=24000.0)
        assert "moved down" in _send(t, spot=23880.0, now=T0 + timedelta(minutes=2)).reason

    def test_small_drift_is_suppressed(self):
        t = _throttle(move_pct=0.35)
        _send(t, spot=24000.0)
        assert not _send(t, spot=24050.0, now=T0 + timedelta(minutes=2)).send   # +0.2%

    def test_move_is_measured_from_the_last_ALERT_not_the_last_check(self):
        """A slow grind must still trigger once it has travelled far enough."""
        t = _throttle(move_pct=0.35)
        _send(t, spot=24000.0)
        assert not _send(t, spot=24040.0, now=T0 + timedelta(minutes=2)).send
        assert not _send(t, spot=24070.0, now=T0 + timedelta(minutes=4)).send
        assert _send(t, spot=24095.0, now=T0 + timedelta(minutes=6)).send   # +0.40%

    def test_force_always_sends(self):
        t = _throttle()
        _send(t)
        d = _send(t, now=T0 + timedelta(minutes=1), force=True)
        assert d.send and d.is_urgent and "exit" in d.reason


class TestPerKeyIsolation:
    def test_indices_do_not_share_a_clock(self):
        """NIFTY going quiet says nothing about whether BANKNIFTY just broke."""
        t = _throttle()
        _send(t, key="NIFTY")
        assert _send(t, key="BANKNIFTY", now=T0 + timedelta(minutes=1)).send

    def test_one_key_suppression_does_not_suppress_another(self):
        t = _throttle()
        _send(t, key="NIFTY")
        _send(t, key="BANKNIFTY")
        assert not _send(t, key="NIFTY", now=T0 + timedelta(minutes=2)).send
        assert not _send(t, key="BANKNIFTY", now=T0 + timedelta(minutes=2)).send

    def test_reset_clears_history(self):
        t = _throttle()
        _send(t, key="NIFTY")
        t.reset("NIFTY")
        assert _send(t, key="NIFTY", now=T0 + timedelta(minutes=1)).send

    def test_reset_all(self):
        t = _throttle()
        _send(t, key="NIFTY")
        _send(t, key="SENSEX")
        t.reset()
        assert _send(t, key="NIFTY", now=T0 + timedelta(minutes=1)).send
        assert _send(t, key="SENSEX", now=T0 + timedelta(minutes=1)).send


class TestDecision:
    def test_decision_is_truthy_when_sending(self):
        assert bool(_send(_throttle())) is True

    def test_decision_is_falsy_when_suppressed(self):
        t = _throttle()
        _send(t)
        assert bool(_send(t, now=T0 + timedelta(minutes=1))) is False

    def test_zero_spot_does_not_crash_the_move_check(self):
        t = _throttle()
        _send(t, spot=0.0)
        assert not _send(t, spot=0.0, now=T0 + timedelta(minutes=1)).send


class TestCompactTradeCall:
    """Short by default: the decision and the trade, not the reasoning behind it."""

    @staticmethod
    def _call(compact):
        from datetime import time as dt_time

        from nifty_ai_agent.reports.trade_call import format_trade_call
        from nifty_ai_agent.risk.calculator import RiskCalculator
        from nifty_ai_agent.risk.margin import MarginCalculator
        from nifty_ai_agent.strategies.base import Signal, SignalType
        from nifty_ai_agent.strategies.consensus import build_consensus
        from nifty_ai_agent.strategies.global_analyser import GlobalSnapshot
        from nifty_ai_agent.strategies.option_analyser import ExpiryAnalysis, OptionLeg

        signals = [
            Signal(SignalType.BUY_CE, 72, "trend up", "EMA_Crossover"),
            Signal(SignalType.BUY_CE, 68, "above vwap", "VWAP_Breakout"),
            Signal(SignalType.BUY_CE, 75, "bullish", "Supertrend"),
            Signal(SignalType.HOLD, 50, "flat", "MACD_Momentum"),
            Signal(SignalType.HOLD, 50, "no range", "Opening_Range_Breakout"),
            Signal(SignalType.BUY_CE, 70, "squeeze", "Bollinger_Squeeze"),
        ]
        consensus = build_consensus(signals, now=dt_time(11, 0), intraday=True)
        analysis = ExpiryAnalysis(
            expiry="25-Aug-2026", spot=24200, atm_strike=24200, max_pain=24350,
            pcr=0.72, legs=[OptionLeg(24200, 123.1, 111.1, 5, 5, 0.12, 0.12, True)],
            call_oi_resistance=24500, put_oi_support=24000, bias="BULLISH",
            atm_ce_ltp=123.1, atm_pe_ltp=111.1,
        )
        return format_trade_call(
            "NIFTY", consensus,
            RiskCalculator().calculate(consensus.signal, 24200.0, 40.0),
            analysis, MarginCalculator(capital=200000), 75,
            GlobalSnapshot(global_bias="BULLISH", gift_nifty_pct=0.46, vix=11.3,
                           is_available=True),
            None, False, compact,
        )

    def test_compact_is_materially_shorter(self):
        full = self._call(compact=False)[1]
        compact = self._call(compact=True)[1]
        assert len(compact) < len(full) / 1.5

    def test_compact_keeps_the_actionable_trade(self):
        """Whatever is dropped, the contract and the levels must survive."""
        body = self._call(compact=True)[1]
        assert "BUY NIFTY 24200 CE" in body
        assert "Buy ₹" in body and "Sell ₹" in body and "Exit ₹" in body
        assert "SL" in body

    def test_compact_drops_the_research(self):
        body = self._call(compact=True)[1]
        assert "STRATEGIES" not in body
        assert "VIX" not in body

    def test_full_still_carries_the_research(self):
        body = self._call(compact=False)[1]
        assert "STRATEGIES" in body and "VIX" in body

    def test_both_keep_the_disclaimer(self):
        """Whatever else is trimmed, the risk warning is never one of the trims."""
        assert "not advice" in self._call(compact=True)[1].lower()
        assert "not advice" in self._call(compact=False)[1].lower()

    def test_title_is_unchanged_by_compaction(self):
        assert self._call(compact=True)[0] == self._call(compact=False)[0]


class TestShortRationale:
    def test_takes_the_first_clause(self):
        from nifty_ai_agent.reports.trade_call import _short_rationale
        assert _short_rationale("4/6 bullish. Dissent: MACD. Size down.") == "4/6 bullish"

    def test_truncates_an_over_long_clause(self):
        from nifty_ai_agent.reports.trade_call import _short_rationale
        out = _short_rationale("x" * 200)
        assert len(out) <= 70 and out.endswith("…")

    def test_short_input_is_untouched(self):
        from nifty_ai_agent.reports.trade_call import _short_rationale
        assert _short_rationale("no setup") == "no setup"


class TestDetailIsEarnedByActionability:
    """Full detail on a BUY or a SELL; compact on a heartbeat."""

    @staticmethod
    def _decide(is_actionable: bool, position_exit: bool, setting: bool = True) -> bool:
        """Mirror of main's rule — returns the `compact` flag passed to the report."""
        actionable_message = is_actionable or position_exit
        return setting and not actionable_message

    def test_a_buy_gets_full_detail(self):
        assert self._decide(is_actionable=True, position_exit=False) is False

    def test_a_sell_on_an_open_position_gets_full_detail(self):
        """A SELL is the most actionable message the agent sends."""
        assert self._decide(is_actionable=False, position_exit=True) is False

    def test_a_hold_heartbeat_is_compact(self):
        assert self._decide(is_actionable=False, position_exit=False) is True

    def test_setting_off_sends_everything_full(self):
        assert self._decide(is_actionable=False, position_exit=False, setting=False) is False

    def test_main_applies_the_rule(self):
        """Guards against the flag being reconnected straight to the setting again."""
        import inspect

        import main
        source = inspect.getsource(main.run_pipeline)
        assert "actionable_message = consensus.is_actionable or position_exit" in source
        assert "compact=settings.compact_notifications and not actionable_message" in source
