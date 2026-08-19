"""Decide when a signal is worth interrupting someone for.

The pipeline runs every 5 minutes because that is how fast it needs to *see* the
market. That is not how often anyone wants to be told about it — twelve messages
an hour saying much the same thing is how a useful alert becomes one that gets
muted, and a muted alert is worth nothing on the day it matters.

So detection and notification are separated. The loop keeps its fast cadence; a
routine message goes out on a slower clock, and anything that actually *changed*
jumps the queue immediately:

    signal flipped        BUY_CE → BUY_PE, or either → HOLD
    conviction upgraded   the engine went from hedging to committing
    confidence jumped     a large move in either direction
    price moved sharply   the underlying travelled while the loop was quiet

Everything else waits for the interval. State is held per key (per index), because
NIFTY going quiet says nothing about whether BANKNIFTY just broke.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Ranked so "did conviction go UP" is a comparison rather than a lookup table.
_CONVICTION_RANK = {"NO_TRADE": 0, "WEAK": 1, "MODERATE": 2, "STRONG": 3}


@dataclass
class NotifyDecision:
    send: bool
    reason: str
    is_urgent: bool = False    # True when a change forced it, not the clock

    def __bool__(self) -> bool:
        return self.send


@dataclass
class _LastSent:
    at: datetime
    signal: str
    conviction: str
    confidence: int
    spot: float


@dataclass
class NotificationThrottle:
    """Per-key gate: routine messages on a clock, changes immediately.

    *move_pct* is deliberately small. It is measured against the price at the last
    notification, not against the day's range, so it answers "has the market moved
    since I last told you about it" rather than "is today volatile".
    """

    interval_minutes: int = 15
    move_pct: float = 0.35
    confidence_jump: int = 15
    _last: dict[str, _LastSent] = field(default_factory=dict)

    def evaluate(
        self,
        key: str,
        *,
        signal: str,
        conviction: str,
        confidence: int,
        spot: float,
        now: datetime | None = None,
        force: bool = False,
    ) -> NotifyDecision:
        """Should this cycle's notification for *key* go out?"""
        now = now or datetime.now()
        previous = self._last.get(key)

        if force:
            return self._allow(key, signal, conviction, confidence, spot, now,
                               "position exit", urgent=True)

        if previous is None:
            # Nothing has been sent for this key yet — the first read is news.
            return self._allow(key, signal, conviction, confidence, spot, now,
                               "first signal")

        if signal != previous.signal:
            return self._allow(
                key, signal, conviction, confidence, spot, now,
                f"signal flipped {previous.signal} → {signal}", urgent=True,
            )

        rank_now = _CONVICTION_RANK.get(conviction.upper(), 0)
        rank_before = _CONVICTION_RANK.get(previous.conviction.upper(), 0)
        if rank_now > rank_before:
            return self._allow(
                key, signal, conviction, confidence, spot, now,
                f"conviction {previous.conviction} → {conviction}", urgent=True,
            )

        if abs(confidence - previous.confidence) >= self.confidence_jump:
            return self._allow(
                key, signal, conviction, confidence, spot, now,
                f"confidence {previous.confidence}% → {confidence}%", urgent=True,
            )

        if previous.spot > 0 and spot > 0:
            moved = abs(spot - previous.spot) / previous.spot * 100
            if moved >= self.move_pct:
                direction = "up" if spot > previous.spot else "down"
                return self._allow(
                    key, signal, conviction, confidence, spot, now,
                    f"{key} moved {direction} {moved:.2f}% since last alert",
                    urgent=True,
                )

        if now - previous.at >= timedelta(minutes=self.interval_minutes):
            return self._allow(key, signal, conviction, confidence, spot, now,
                               "scheduled update")

        wait = self.interval_minutes - (now - previous.at).total_seconds() / 60
        logger.debug(
            "%s: suppressed, nothing changed (next in %.0f min)", key, max(wait, 0),
        )
        return NotifyDecision(send=False, reason="unchanged")

    def _allow(
        self, key: str, signal: str, conviction: str, confidence: int,
        spot: float, now: datetime, reason: str, urgent: bool = False,
    ) -> NotifyDecision:
        self._last[key] = _LastSent(
            at=now, signal=signal, conviction=conviction,
            confidence=confidence, spot=spot,
        )
        if urgent:
            logger.info("%s: notifying early — %s", key, reason)
        return NotifyDecision(send=True, reason=reason, is_urgent=urgent)

    def reset(self, key: str = "") -> None:
        """Forget one key's history, or all of it."""
        self._last.pop(key, None) if key else self._last.clear()
