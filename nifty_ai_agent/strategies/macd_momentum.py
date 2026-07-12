"""MACD momentum strategy — signal-line cross plus histogram expansion."""

import logging

import pandas as pd

from nifty_ai_agent.strategies.base import BaseStrategy, Signal, SignalType

logger = logging.getLogger(__name__)

# The histogram must be growing, not just positive. A positive-but-shrinking
# histogram is a trend losing steam — the worst moment to buy a decaying option.
_EXPANSION_BARS = 3


class MACDMomentumStrategy(BaseStrategy):
    """Generate signals from MACD position relative to its signal line.

    BUY_CE:  MACD > signal  AND  histogram positive and expanding
    BUY_PE:  MACD < signal  AND  histogram negative and expanding
    HOLD:    crossed but fading, or no cross
    """

    NAME = "MACD_Momentum"

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        required = {"macd", "macd_signal", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

        clean = df.dropna(subset=["macd", "macd_signal"])
        if len(clean) < _EXPANSION_BARS + 1:
            return Signal(
                signal=SignalType.HOLD,
                confidence=50,
                reason="Insufficient bars for MACD momentum analysis.",
                strategy=self.NAME,
            )

        histogram = clean["macd"] - clean["macd_signal"]
        current = float(histogram.iloc[-1])
        prior = float(histogram.iloc[-_EXPANSION_BARS - 1])
        macd = float(clean["macd"].iloc[-1])
        signal_line = float(clean["macd_signal"].iloc[-1])
        close = float(clean["close"].iloc[-1])

        # Expanding = the histogram has moved further from zero over the lookback.
        # A sign flip that overshoots the prior magnitude counts too: that is a fresh
        # cross carrying real momentum, which is exactly the entry this strategy wants.
        expanding = abs(current) > abs(prior)
        expansion = abs(current) - abs(prior)

        logger.debug(
            "MACDMomentum — macd=%.2f signal=%.2f hist=%.2f prior_hist=%.2f expanding=%s",
            macd, signal_line, current, prior, expanding,
        )

        if current > 0 and expanding:
            return Signal(
                signal=SignalType.BUY_CE,
                confidence=self._compute_confidence(current, expansion, close),
                reason=(
                    f"MACD ({macd:.1f}) is above its signal line ({signal_line:.1f}) and the "
                    f"histogram widened from {prior:.1f} to {current:.1f} over "
                    f"{_EXPANSION_BARS} bars — bullish momentum still building."
                ),
                strategy=self.NAME,
            )

        if current < 0 and expanding:
            return Signal(
                signal=SignalType.BUY_PE,
                confidence=self._compute_confidence(current, expansion, close),
                reason=(
                    f"MACD ({macd:.1f}) is below its signal line ({signal_line:.1f}) and the "
                    f"histogram widened from {prior:.1f} to {current:.1f} over "
                    f"{_EXPANSION_BARS} bars — bearish momentum still building."
                ),
                strategy=self.NAME,
            )

        fading = "fading" if abs(current) <= abs(prior) else "flat"
        return Signal(
            signal=SignalType.HOLD,
            confidence=50,
            reason=(
                f"MACD histogram is {fading} ({prior:.1f} → {current:.1f}) — momentum is "
                "not expanding, so no entry."
            ),
            strategy=self.NAME,
        )

    @staticmethod
    def _compute_confidence(histogram: float, expansion: float, close: float) -> int:
        """Score 50–95 from histogram size and expansion rate, both scaled by price.

        MACD values scale with the index level (a histogram of 12 means something
        very different on NIFTY at 24k than on SENSEX at 79k), so both inputs are
        normalised to basis points of the close before scoring.
        """
        base = 50
        if close <= 0:
            return base
        size_bps = abs(histogram) / close * 10_000
        expansion_bps = abs(expansion) / close * 10_000
        size_bonus = min(25, int(size_bps * 5))
        expansion_bonus = min(20, int(expansion_bps * 10))
        return min(95, base + size_bonus + expansion_bonus)
