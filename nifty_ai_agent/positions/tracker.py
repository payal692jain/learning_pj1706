"""Verdicts for open positions — HOLD it, or SELL it now.

The signal engine answers "should I enter?". Once an entry has been suggested the
next question is "what do I do with what I'm already holding?", and that is what
this module answers. It is deliberately stateless and pure: the caller loads the
open positions, passes in the current spot and the live consensus, and gets back a
verdict per position to render into the notification.

Exit rules (chosen deliberately, in priority order):
  1. Stop-loss hit on the underlying  → SELL, cut the loss
  2. Target hit on the underlying     → SELL, book the profit
  3. Consensus reversed against it    → SELL, the thesis is gone
  otherwise                           → HOLD

Levels are checked against a single spot, so on a well-formed position (stop and
target on opposite sides of entry) exactly one of rules 1 and 2 can fire. Stop
before target is therefore only a tie-break for a malformed record — a stop that
somehow sits on the target's side of the price — where reporting the loss is the
safer of two wrong answers. Levels are checked before a reversal so a position
that already reached its exit is reported as such rather than as a thesis change.
"""

import logging
from dataclasses import dataclass

from nifty_ai_agent.database.models import PositionRecord
from nifty_ai_agent.strategies.base import SignalType

logger = logging.getLogger(__name__)

HOLD = "HOLD"
SELL = "SELL"

# A position is only opened off a conviction this strong — a MODERATE call is
# still notified, it just does not become something the agent then manages.
_OPENING_CONVICTION = "STRONG"

_SIGNAL_FOR_OPT_TYPE = {"CE": SignalType.BUY_CE, "PE": SignalType.BUY_PE}

# A tracked position whose underlying has apparently moved further than this since
# entry is almost certainly being priced against the wrong symbol — a stock at 270
# judged against a 25,000 index spot, say. A position that genuinely moved this far
# would have hit its stop or target (a few percent away) many cycles earlier, so the
# realistic cause is a wiring mistake, and the safe answer is to refuse a verdict
# rather than emit a confident SELL off a nonsense price.
_IMPLAUSIBLE_MOVE_PCT = 50.0


@dataclass
class PositionVerdict:
    """What to do with one open position, and why."""

    position: PositionRecord
    action: str          # HOLD or SELL
    reason: str
    spot: float
    move_pct: float      # underlying move since entry, signed in the position's favour

    @property
    def is_exit(self) -> bool:
        return self.action == SELL


def should_open_position(conviction: str, signal: SignalType) -> bool:
    """True when a fresh signal is strong and directional enough to track.

    HOLD is not a position, and anything below STRONG conviction is a call the
    user is told about but the agent does not then babysit.
    """
    return (
        signal in (SignalType.BUY_CE, SignalType.BUY_PE)
        and conviction.upper() == _OPENING_CONVICTION
    )


def _favourable_move_pct(position: PositionRecord, spot: float) -> float:
    """Underlying move since entry as a %, signed so positive is always good.

    A PE gains when the underlying falls, so its sign is inverted — this keeps
    "+0.8%" meaning "in profit" for calls and puts alike.
    """
    if not position.entry_spot:
        return 0.0
    raw = (spot - position.entry_spot) / position.entry_spot * 100
    return raw if position.opt_type == "CE" else -raw


def _hit_stop(position: PositionRecord, spot: float) -> bool:
    if position.opt_type == "CE":
        return spot <= position.stop_loss
    return spot >= position.stop_loss


def _hit_target(position: PositionRecord, spot: float) -> bool:
    if position.opt_type == "CE":
        return spot >= position.target
    return spot <= position.target


def evaluate_position(
    position: PositionRecord,
    spot: float,
    current_signal: SignalType | None = None,
) -> PositionVerdict:
    """Return the HOLD/SELL verdict for one open *position* at *spot*.

    *current_signal* is the live consensus for the same underlying, used to detect
    a reversal; pass None when no fresh read is available (the position is then
    judged on its levels alone).
    """
    move = _favourable_move_pct(position, spot)

    def verdict(action: str, reason: str) -> PositionVerdict:
        return PositionVerdict(
            position=position, action=action, reason=reason, spot=spot, move_pct=move,
        )

    if spot <= 0 or abs(move) > _IMPLAUSIBLE_MOVE_PCT:
        logger.error(
            "Position %s %g%s: spot %.2f is implausible against entry %.2f (%.1f%%) "
            "— refusing a verdict; check the spot wired in for this underlying.",
            position.underlying, position.strike, position.opt_type,
            spot, position.entry_spot, move,
        )
        return verdict(
            HOLD,
            f"price for {position.underlying} looks wrong ({spot:,.2f} vs entry "
            f"{position.entry_spot:,.2f}) — no verdict until it is checked",
        )

    if _hit_stop(position, spot):
        return verdict(
            SELL,
            f"stop-loss hit — {position.underlying} at {spot:,.2f}, "
            f"SL was {position.stop_loss:,.2f}",
        )

    if _hit_target(position, spot):
        return verdict(
            SELL,
            f"target hit — {position.underlying} at {spot:,.2f}, "
            f"target was {position.target:,.2f}",
        )

    expected = _SIGNAL_FOR_OPT_TYPE.get(position.opt_type)
    reversed_now = (
        current_signal is not None
        and current_signal in (SignalType.BUY_CE, SignalType.BUY_PE)
        and current_signal is not expected
    )
    if reversed_now:
        return verdict(
            SELL,
            f"signal reversed to {current_signal.value} — the {position.opt_type} "
            f"thesis is gone",
        )

    return verdict(
        HOLD,
        f"{spot:,.2f} between SL {position.stop_loss:,.2f} "
        f"and target {position.target:,.2f}",
    )


def format_positions_for_notification(verdicts: list[PositionVerdict]) -> str:
    """Render verdicts as the OPEN POSITIONS block appended to a notification.

    Returns "" for an empty list so callers can append unconditionally.
    """
    if not verdicts:
        return ""

    lines = ["", "── OPEN POSITIONS ──"]
    for v in sorted(verdicts, key=lambda x: (not x.is_exit, x.position.underlying)):
        p = v.position
        icon = "🔴" if v.is_exit else "🟢"
        lines.append(
            f"{icon} {v.action}  {p.underlying} {p.strike:g}{p.opt_type} {p.expiry}"
        )
        lines.append(
            f"   entry {p.entry_spot:,.2f} → now {v.spot:,.2f} ({v.move_pct:+.2f}%)"
        )
        lines.append(f"   {v.reason}")
    return "\n".join(lines)
