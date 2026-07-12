"""Consensus engine — one actionable verdict from every strategy's opinion.

Six strategies each shouting a different signal is not advice, it is noise. This
folds them into a single weighted verdict: what to do, how strongly, and — when
they disagree — the honest answer that the tape is conflicted and the trade is to
stand aside.

Two intraday-specific rules that a naive vote count would miss:

1. **Strategies are not equal, and their edge moves with the clock.** An Opening
   Range Breakout signal at 09:45 is the highest-quality read of the day; the same
   signal at 14:30 is describing a range that stopped mattering hours ago. Weights
   decay accordingly.

2. **Late entries lose to theta.** A weekly option bought after 15:00 has one
   session of value left and an overnight gap to survive. Past the cutoff the
   engine returns NO_TRADE regardless of how bullish the indicators look.
"""

import logging
from dataclasses import dataclass
from datetime import time as dt_time

from nifty_ai_agent.strategies.base import Signal, SignalType

logger = logging.getLogger(__name__)

# Base weight per strategy, by how well it suits 5-minute index options.
# ORB and VWAP are genuine intraday edges; EMA crossover is a lagging trend filter
# that is right slowly — useful as confirmation, weak as a trigger.
_BASE_WEIGHTS: dict[str, float] = {
    "Opening_Range_Breakout": 1.3,
    "VWAP_Breakout":          1.2,
    "Supertrend":             1.1,
    "MACD_Momentum":          1.0,
    "Bollinger_Squeeze":      0.9,
    "EMA_Crossover":          0.8,
}
_DEFAULT_WEIGHT = 1.0

# The opening range is a morning instrument. Its weight is multiplied by these.
_ORB_PRIME_END = dt_time(11, 0)     # full strength while the break is fresh
_ORB_STALE_START = dt_time(13, 0)   # by the afternoon the range is history
_ORB_STALE_MULTIPLIER = 0.4

# No new intraday option entries after this — not enough session left to be paid.
_ENTRY_CUTOFF = dt_time(15, 0)

# Share of the DIRECTIONAL vote the winner must carry. Measured against bullish +
# bearish weight only, never against the whole book: with two possible directions the
# winner's share of the *total* is ≥50% by construction, so a floor applied to that
# could never reject anything — a 54/46 knife-fight would sail through as a signal.
_MIN_DIRECTIONAL_AGREEMENT = 0.65

# And enough of the book must actually be in the trade. One lone voice among five
# strategies seeing no setup is not a consensus, however loud it is.
_MIN_PARTICIPATION = 0.30


@dataclass
class StrategyVote:
    strategy: str
    signal: SignalType
    confidence: int
    weight: float          # effective weight after time-of-day adjustment
    reason: str

    @property
    def score(self) -> float:
        return self.confidence * self.weight


@dataclass
class Consensus:
    """The single recommendation, and the arithmetic behind it."""
    signal: SignalType
    confidence: int
    agreement: float           # 0.0–1.0 — weighted share backing the winning side
    conviction: str            # STRONG / MODERATE / WEAK / NO_TRADE
    votes: list[StrategyVote]
    bullish_weight: float
    bearish_weight: float
    hold_weight: float
    rationale: str

    @property
    def is_actionable(self) -> bool:
        return self.signal != SignalType.HOLD and self.conviction != "NO_TRADE"

    @property
    def backers(self) -> list[StrategyVote]:
        """The strategies that voted for the winning direction."""
        return [v for v in self.votes if v.signal == self.signal]

    @property
    def dissenters(self) -> list[StrategyVote]:
        """Strategies that voted for the OPPOSITE direction — the reason to size small."""
        opposite = (
            SignalType.BUY_PE if self.signal == SignalType.BUY_CE else SignalType.BUY_CE
        )
        return [v for v in self.votes if v.signal == opposite]


def effective_weight(strategy: str, now: dt_time) -> float:
    """Base weight for *strategy*, adjusted for what time of day it is."""
    weight = _BASE_WEIGHTS.get(strategy, _DEFAULT_WEIGHT)

    if strategy == "Opening_Range_Breakout":
        if now >= _ORB_STALE_START:
            weight *= _ORB_STALE_MULTIPLIER
        elif now >= _ORB_PRIME_END:
            weight *= 0.7

    return round(weight, 3)


def build_consensus(signals: list[Signal], now: dt_time) -> Consensus:
    """Fold every strategy's Signal into one verdict as of *now* (IST wall clock)."""
    votes = [
        StrategyVote(
            strategy=s.strategy,
            signal=s.signal,
            confidence=s.confidence,
            weight=effective_weight(s.strategy, now),
            reason=s.reason,
        )
        for s in signals
    ]

    if not votes:
        return Consensus(
            signal=SignalType.HOLD, confidence=0, agreement=0.0, conviction="NO_TRADE",
            votes=[], bullish_weight=0.0, bearish_weight=0.0, hold_weight=0.0,
            rationale="No strategies ran.",
        )

    bullish = sum(v.score for v in votes if v.signal == SignalType.BUY_CE)
    bearish = sum(v.score for v in votes if v.signal == SignalType.BUY_PE)
    hold = sum(v.score for v in votes if v.signal == SignalType.HOLD)
    total = bullish + bearish + hold

    # Too late in the session to open an intraday option position.
    if now >= _ENTRY_CUTOFF:
        return Consensus(
            signal=SignalType.HOLD, confidence=0, agreement=0.0, conviction="NO_TRADE",
            votes=votes, bullish_weight=bullish, bearish_weight=bearish, hold_weight=hold,
            rationale=(
                f"Past the {_ENTRY_CUTOFF.strftime('%H:%M')} entry cutoff — too little "
                "session left for an intraday option to pay, and an overnight hold is a "
                "different trade with different risk."
            ),
        )

    winner, winning_weight = max(
        (
            (SignalType.BUY_CE, bullish),
            (SignalType.BUY_PE, bearish),
            (SignalType.HOLD, hold),
        ),
        key=lambda pair: pair[1],
    )
    agreement = round(winning_weight / total, 2) if total > 0 else 0.0

    if winner == SignalType.HOLD:
        return Consensus(
            signal=SignalType.HOLD, confidence=0, agreement=agreement, conviction="NO_TRADE",
            votes=votes, bullish_weight=bullish, bearish_weight=bearish, hold_weight=hold,
            rationale=_hold_rationale(votes, bullish, bearish),
        )

    # Directional winner, but the losing direction is nearly as loud — a split tape.
    directional_total = bullish + bearish
    directional_agreement = (
        winning_weight / directional_total if directional_total > 0 else 0.0
    )
    if directional_agreement < _MIN_DIRECTIONAL_AGREEMENT:
        return Consensus(
            signal=SignalType.HOLD, confidence=0, agreement=agreement, conviction="NO_TRADE",
            votes=votes, bullish_weight=bullish, bearish_weight=bearish, hold_weight=hold,
            rationale=(
                f"Strategies are split — bullish {bullish:.0f} vs bearish {bearish:.0f} "
                f"({directional_agreement:.0%} for the winning side, below the "
                f"{_MIN_DIRECTIONAL_AGREEMENT:.0%} floor). A conflicted tape is a reason "
                "to stand aside, not to pick the marginally louder half."
            ),
        )

    # A direction nobody much participates in is not a consensus either.
    if agreement < _MIN_PARTICIPATION:
        return Consensus(
            signal=SignalType.HOLD, confidence=0, agreement=agreement, conviction="NO_TRADE",
            votes=votes, bullish_weight=bullish, bearish_weight=bearish, hold_weight=hold,
            rationale=(
                f"Only {agreement:.0%} of the book is in this trade — the rest see no "
                "setup. Too thin to pay premium for."
            ),
        )

    backers = [v for v in votes if v.signal == winner]
    weighted_confidence = (
        sum(v.score for v in backers) / sum(v.weight for v in backers)
    )
    # Scale the winners' own confidence by how much of the book agreed with them: a
    # 90%-confident lone voice outvoted by four abstentions is not a 90% trade.
    confidence = int(round(weighted_confidence * (0.5 + 0.5 * agreement)))
    confidence = max(10, min(95, confidence))
    conviction = _conviction(confidence, agreement)

    consensus = Consensus(
        signal=winner,
        confidence=confidence,
        agreement=agreement,
        conviction=conviction,
        votes=votes,
        bullish_weight=bullish,
        bearish_weight=bearish,
        hold_weight=hold,
        rationale=_directional_rationale(winner, backers, votes, agreement),
    )
    logger.info(
        "Consensus: %s %d%% (%s, agreement=%.0f%%) — bull=%.0f bear=%.0f hold=%.0f",
        winner.value, confidence, conviction, agreement * 100, bullish, bearish, hold,
    )
    return consensus


def _conviction(confidence: int, agreement: float) -> str:
    if confidence >= 70 and agreement >= 0.60:
        return "STRONG"
    if confidence >= 55 and agreement >= 0.50:
        return "MODERATE"
    return "WEAK"


def _hold_rationale(votes: list[StrategyVote], bullish: float, bearish: float) -> str:
    holders = [v.strategy for v in votes if v.signal == SignalType.HOLD]
    return (
        f"{len(holders)} of {len(votes)} strategies see no setup "
        f"({', '.join(holders)}). Directional weight is thin (bull {bullish:.0f} / "
        f"bear {bearish:.0f}) — no edge worth paying premium for."
    )


def _directional_rationale(
    winner: SignalType, backers: list[StrategyVote], votes: list[StrategyVote],
    agreement: float,
) -> str:
    side = "bullish" if winner == SignalType.BUY_CE else "bearish"
    opposite = SignalType.BUY_PE if winner == SignalType.BUY_CE else SignalType.BUY_CE
    against = [v.strategy for v in votes if v.signal == opposite]

    # Deliberately does not name the backers: the notification prints the full vote
    # table right underneath, and repeating it here just spends the character budget
    # that the risk and sizing lines need.
    text = (
        f"{len(backers)}/{len(votes)} strategies {side}, {agreement:.0%} weighted agreement."
    )
    if against:
        text += f" Dissent: {', '.join(against)} — size accordingly."
    return text
