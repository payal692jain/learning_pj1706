"""Did the agent's calls actually work? — hit rate from closed positions.

Every signal this system emits is a testable claim, and until now none of them
were ever scored. A confidence number that has never been checked against an
outcome is a number about the model's mood, not about the market.

Scoring is deliberately taken from *closed positions* rather than from the raw
signals table. A signal is only a claim; a position carries the entry, the levels
and the reason it ended, which is what makes it falsifiable. The exit reason
recorded by the position tracker is therefore the ground truth:

    "target hit"      → WIN
    "stop-loss hit"   → LOSS
    "signal reversed" → SCRATCH — the thesis was abandoned, not proved wrong

Scratches are reported separately and excluded from the hit rate. Counting them
as wins would flatter the number; counting them as losses would punish the
reversal rule for doing its job. They are their own thing and are shown as such.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

WIN, LOSS, SCRATCH, UNKNOWN = "WIN", "LOSS", "SCRATCH", "UNKNOWN"

# Minimum decided trades before a hit rate means anything. Below this the number
# is noise: 4 wins from 5 is "80%" and tells you nothing at all.
MIN_SAMPLE = 30


def classify(exit_reason: str) -> str:
    """Map a position's recorded exit reason onto an outcome."""
    reason = (exit_reason or "").lower()
    if "target hit" in reason:
        return WIN
    if "stop-loss hit" in reason:
        return LOSS
    if "reversed" in reason:
        return SCRATCH
    return UNKNOWN


@dataclass
class Scorecard:
    """Outcomes over a set of closed positions."""
    wins: int = 0
    losses: int = 0
    scratches: int = 0
    unknown: int = 0
    by_conviction: dict = field(default_factory=dict)   # conviction -> [wins, losses]
    by_underlying: dict = field(default_factory=dict)   # symbol -> [wins, losses]

    @property
    def decided(self) -> int:
        """Trades that actually resolved to a win or a loss."""
        return self.wins + self.losses

    @property
    def total(self) -> int:
        return self.wins + self.losses + self.scratches + self.unknown

    @property
    def hit_rate(self) -> float | None:
        """Wins as a share of decided trades, or None when nothing has resolved."""
        return None if not self.decided else self.wins / self.decided

    @property
    def is_significant(self) -> bool:
        return self.decided >= MIN_SAMPLE


def build_scorecard(positions) -> Scorecard:
    """Score an iterable of closed PositionRecords."""
    card = Scorecard()
    for p in positions:
        outcome = classify(getattr(p, "exit_reason", ""))
        if outcome == WIN:
            card.wins += 1
        elif outcome == LOSS:
            card.losses += 1
        elif outcome == SCRATCH:
            card.scratches += 1
            continue          # scratches skew per-bucket rates; keep them out
        else:
            card.unknown += 1
            continue

        for bucket, key in (
            (card.by_conviction, getattr(p, "conviction", "") or "?"),
            (card.by_underlying, getattr(p, "underlying", "") or "?"),
        ):
            tally = bucket.setdefault(key, [0, 0])
            tally[0 if outcome == WIN else 1] += 1
    return card


def format_scorecard(card: Scorecard) -> str:
    """Render the scorecard for a notification or a /performance reply."""
    if not card.total:
        return (
            "📈 Performance\n\nNo closed positions yet.\n"
            "Scoring starts once a tracked position hits its target or stop."
        )

    lines = ["📈 Performance", ""]
    rate = card.hit_rate
    if rate is None:
        lines.append("No trade has resolved to a win or loss yet.")
    else:
        lines.append(f"Hit rate: {rate * 100:.0f}%  ({card.wins}W / {card.losses}L)")
        if not card.is_significant:
            # Stated first and plainly: a small-sample number invites exactly the
            # over-confidence this report exists to prevent.
            lines.append(
                f"⚠️ Only {card.decided} decided trades — too few to mean anything. "
                f"Needs {MIN_SAMPLE}+."
            )
    if card.scratches:
        lines.append(f"Scratched on reversal: {card.scratches} (not scored)")
    if card.unknown:
        lines.append(f"Unclassified exits: {card.unknown}")

    if card.by_conviction:
        lines += ["", "By conviction:"]
        for name, (w, l) in sorted(
            card.by_conviction.items(), key=lambda kv: -(kv[1][0] + kv[1][1])
        ):
            total = w + l
            lines.append(f"  {name:<9} {w}W/{l}L  ({w / total * 100:.0f}%)")

    if card.by_underlying:
        ranked = sorted(
            card.by_underlying.items(), key=lambda kv: -(kv[1][0] + kv[1][1])
        )[:6]
        lines += ["", "Most traded:"]
        for name, (w, l) in ranked:
            lines.append(f"  {name:<11} {w}W/{l}L")

    return "\n".join(lines)
