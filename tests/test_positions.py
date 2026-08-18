"""Tests for open-position tracking — HOLD/SELL verdicts and their persistence."""

from datetime import datetime, timezone

import pytest

from nifty_ai_agent.database.models import PositionRecord
from nifty_ai_agent.database.repository import DatabaseRepository
from nifty_ai_agent.positions import (
    evaluate_position,
    format_positions_for_notification,
    should_open_position,
)
from nifty_ai_agent.positions.tracker import HOLD, SELL
from nifty_ai_agent.strategies.base import SignalType


def _position(opt_type="CE", entry=25000.0, sl=24900.0, target=25200.0) -> PositionRecord:
    return PositionRecord(
        id=1, underlying="NIFTY", opt_type=opt_type, strike=25000.0,
        expiry="25-Aug-2026", lot_size=75, entry_premium=120.0,
        entry_spot=entry, stop_loss=sl, target=target,
        strategy="", conviction="STRONG", confidence=80,
        status="OPEN", opened_at=datetime.now(tz=timezone.utc),
    )


class TestShouldOpenPosition:
    def test_strong_directional_opens(self):
        assert should_open_position("STRONG", SignalType.BUY_CE)
        assert should_open_position("strong", SignalType.BUY_PE)

    def test_weaker_conviction_does_not_open(self):
        assert not should_open_position("MODERATE", SignalType.BUY_CE)
        assert not should_open_position("WEAK", SignalType.BUY_PE)

    def test_hold_is_never_a_position(self):
        assert not should_open_position("STRONG", SignalType.HOLD)


class TestCallVerdicts:
    def test_holds_between_levels(self):
        v = evaluate_position(_position(), 25050.0, SignalType.BUY_CE)
        assert v.action == HOLD
        assert v.move_pct == pytest.approx(0.2)

    def test_sells_on_target(self):
        v = evaluate_position(_position(), 25200.0, SignalType.BUY_CE)
        assert v.action == SELL and "target hit" in v.reason

    def test_sells_on_stop_loss(self):
        v = evaluate_position(_position(), 24900.0, SignalType.BUY_CE)
        assert v.action == SELL and "stop-loss hit" in v.reason

    def test_sells_on_reversal(self):
        v = evaluate_position(_position(), 25050.0, SignalType.BUY_PE)
        assert v.action == SELL and "reversed" in v.reason

    def test_hold_signal_is_not_a_reversal(self):
        """A HOLD means 'no fresh entry', not 'abandon what you hold'."""
        assert evaluate_position(_position(), 25050.0, SignalType.HOLD).action == HOLD

    def test_missing_signal_judges_on_levels_alone(self):
        assert evaluate_position(_position(), 25050.0, None).action == HOLD


class TestPutVerdicts:
    """A PE inverts every comparison — the most likely place for a sign bug."""

    def _pe(self):
        return _position(opt_type="PE", entry=25000.0, sl=25100.0, target=24800.0)

    def test_holds_between_levels(self):
        v = evaluate_position(self._pe(), 24950.0, SignalType.BUY_PE)
        assert v.action == HOLD
        # Falling underlying is a GAIN for a put — the sign must flip.
        assert v.move_pct == pytest.approx(0.2)

    def test_sells_on_target_below_entry(self):
        v = evaluate_position(self._pe(), 24800.0, SignalType.BUY_PE)
        assert v.action == SELL and "target hit" in v.reason

    def test_sells_on_stop_above_entry(self):
        v = evaluate_position(self._pe(), 25100.0, SignalType.BUY_PE)
        assert v.action == SELL and "stop-loss hit" in v.reason

    def test_sells_on_reversal_to_call(self):
        v = evaluate_position(self._pe(), 24950.0, SignalType.BUY_CE)
        assert v.action == SELL and "reversed" in v.reason


class TestExitPriority:
    def test_well_formed_levels_cannot_both_fire(self):
        """Sanity: with SL below and target above entry, one spot triggers at most one."""
        p = _position(entry=25000.0, sl=24900.0, target=25200.0)
        for spot in (24800.0, 24950.0, 25300.0):
            reason = evaluate_position(p, spot, None).reason
            assert not ("stop-loss hit" in reason and "target hit" in reason)

    def test_malformed_levels_degrade_to_stop_loss(self):
        """A stop on the target's side is corrupt; report the loss, the safer wrong answer."""
        p = _position(entry=25000.0, sl=25100.0, target=24900.0)  # both sides inverted
        assert "stop-loss hit" in evaluate_position(p, 25000.0, None).reason

    def test_levels_beat_reversal(self):
        v = evaluate_position(_position(), 25200.0, SignalType.BUY_PE)
        assert "target hit" in v.reason  # not "reversed"


class TestImplausibleSpotGuard:
    """A wrong spot must never become a confident SELL instruction."""

    def test_stock_priced_against_an_index_spot_refuses_a_verdict(self):
        p = _position(opt_type="PE", entry=270.0, sl=278.0, target=254.0)
        p.underlying = "POWERGRID"
        v = evaluate_position(p, 25060.0, SignalType.BUY_PE)  # NIFTY's spot, not the stock's
        assert v.action == HOLD
        assert "looks wrong" in v.reason

    def test_zero_spot_refuses_a_verdict(self):
        assert evaluate_position(_position(), 0.0, None).action == HOLD

    def test_a_plausible_move_still_gets_a_real_verdict(self):
        v = evaluate_position(_position(), 25200.0, SignalType.BUY_CE)
        assert v.action == SELL and "target hit" in v.reason


class TestFormatting:
    def test_empty_renders_nothing(self):
        assert format_positions_for_notification([]) == ""

    def test_renders_action_contract_and_move(self):
        out = format_positions_for_notification(
            [evaluate_position(_position(), 25050.0, SignalType.BUY_CE)]
        )
        assert "OPEN POSITIONS" in out
        assert "HOLD" in out and "NIFTY 25000CE" in out and "+0.20%" in out

    def test_exits_are_listed_first(self):
        hold = evaluate_position(_position(), 25050.0, SignalType.BUY_CE)
        sell = evaluate_position(_position(), 25200.0, SignalType.BUY_CE)
        out = format_positions_for_notification([hold, sell])
        assert out.index("SELL") < out.index("HOLD")


class TestManagePositionsWiring:
    """main._manage_positions — the bridge between the DB and the notification."""

    @pytest.fixture
    def db(self, tmp_path):
        return DatabaseRepository(f"sqlite:///{tmp_path / 'wiring.db'}")

    @staticmethod
    def _risk(valid=True):
        class _R:
            is_valid = valid
            stop_loss = 24900.0
            target = 25200.0
        return _R()

    def _call(self, db, spot, signal, conviction="STRONG", allow_open=True):
        import main
        return main._manage_positions(
            db, "NIFTY", spot, signal,
            opt_type="CE" if signal == SignalType.BUY_CE else "PE",
            conviction=conviction, confidence=80,
            strike=25000.0, expiry="25-Aug-2026",
            risk=self._risk(), lot_size=75, allow_open=allow_open,
        )

    def test_strong_call_opens_and_reports_nothing_yet(self, db):
        block, has_exit = self._call(db, 25000.0, SignalType.BUY_CE)
        assert block == "" and has_exit is False   # nothing was open to report on
        assert len(db.list_open_positions("NIFTY")) == 1

    def test_next_cycle_reports_hold(self, db):
        self._call(db, 25000.0, SignalType.BUY_CE)
        block, has_exit = self._call(db, 25050.0, SignalType.BUY_CE)
        assert "HOLD" in block and has_exit is False

    def test_target_reports_exit_and_closes(self, db):
        self._call(db, 25000.0, SignalType.BUY_CE)
        block, has_exit = self._call(db, 25250.0, SignalType.BUY_CE)
        assert "SELL" in block and has_exit is True
        assert db.list_open_positions("NIFTY") == []

    def test_exit_cycle_does_not_silently_re_enter(self, db):
        """SELL and a fresh entry in one message would be met as a HOLD next cycle."""
        self._call(db, 25000.0, SignalType.BUY_CE)
        self._call(db, 25250.0, SignalType.BUY_CE)          # target hit -> exit
        assert db.list_open_positions("NIFTY") == []

    def test_next_cycle_after_an_exit_may_re_enter(self, db):
        self._call(db, 25000.0, SignalType.BUY_CE)
        self._call(db, 25250.0, SignalType.BUY_CE)          # exit
        self._call(db, 25260.0, SignalType.BUY_CE)          # signal still strong
        assert len(db.list_open_positions("NIFTY")) == 1

    def test_allow_open_false_evaluates_without_opening(self, db):
        """A signal too weak to notify must not open a position behind the user's back."""
        block, _ = self._call(db, 25000.0, SignalType.BUY_CE, allow_open=False)
        assert block == "" and db.list_open_positions("NIFTY") == []

    def test_allow_open_false_still_reports_an_existing_position(self, db):
        self._call(db, 25000.0, SignalType.BUY_CE)
        block, has_exit = self._call(db, 25250.0, SignalType.BUY_CE, allow_open=False)
        assert "SELL" in block and has_exit is True

    def test_moderate_conviction_never_opens(self, db):
        self._call(db, 25000.0, SignalType.BUY_CE, conviction="MODERATE")
        assert db.list_open_positions("NIFTY") == []

    def test_db_failure_degrades_to_no_block(self, db, monkeypatch):
        """A DB fault must not take down the signal notification it rides on."""
        monkeypatch.setattr(
            db, "list_open_positions",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db down")),
        )
        assert self._call(db, 25000.0, SignalType.BUY_CE) == ("", False)


class TestPositionPersistence:
    @pytest.fixture
    def db(self, tmp_path):
        return DatabaseRepository(f"sqlite:///{tmp_path / 'test.db'}")

    def _open(self, db, underlying="NIFTY"):
        return db.open_position(
            underlying=underlying, opt_type="CE", strike=25000.0,
            expiry="25-Aug-2026", entry_spot=25000.0,
            stop_loss=24900.0, target=25200.0, conviction="STRONG", confidence=80,
        )

    def test_open_and_list(self, db):
        assert self._open(db) is not None
        open_positions = db.list_open_positions("NIFTY")
        assert len(open_positions) == 1
        assert open_positions[0].opt_type == "CE"

    def test_second_open_for_same_underlying_is_refused(self, db):
        """The 5-min loop must not stack a new position every bullish cycle."""
        self._open(db)
        assert self._open(db) is None
        assert len(db.list_open_positions("NIFTY")) == 1

    def test_other_underlyings_are_independent(self, db):
        self._open(db, "NIFTY")
        assert self._open(db, "BANKNIFTY") is not None
        assert len(db.list_open_positions()) == 2

    def test_close_removes_from_open_list_and_frees_the_slot(self, db):
        pid = self._open(db)
        db.close_position(pid, 25200.0, "target hit")
        assert db.list_open_positions("NIFTY") == []
        assert self._open(db) is not None  # slot is free again

    def test_closing_unknown_id_is_a_no_op(self, db):
        db.close_position(9999, 25000.0, "nope")  # must not raise

    def test_list_filters_by_underlying(self, db):
        self._open(db, "NIFTY")
        self._open(db, "RELIANCE")
        assert [p.underlying for p in db.list_open_positions("RELIANCE")] == ["RELIANCE"]
