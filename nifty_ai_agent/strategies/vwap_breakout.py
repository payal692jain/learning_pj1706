"""VWAP Breakout strategy — momentum entries on session VWAP breaks."""

import logging

import pandas as pd

from nifty_ai_agent.strategies.base import BaseStrategy, Signal, SignalType

logger = logging.getLogger(__name__)

# Configurable thresholds
_BREAKOUT_THRESHOLD_PCT = 0.15  # price must clear VWAP by at least 0.15%
_MOMENTUM_BARS = 3              # bars back to confirm the move is still building


class VWAPBreakoutStrategy(BaseStrategy):
    """Generate signals based on price breaking away from session VWAP with momentum.

    BUY_CE:  close > VWAP by >= 0.15%  AND  price has risen over the last 3 bars
    BUY_PE:  close < VWAP by >= 0.15%  AND  price has fallen over the last 3 bars
    HOLD:    all other cases
    """

    NAME = "VWAP_Breakout"

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        required = {"close", "vwap"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

        clean = df.dropna(subset=["close", "vwap"])
        if len(clean) < _MOMENTUM_BARS + 1:
            return Signal(
                signal=SignalType.HOLD,
                confidence=50,
                reason="Insufficient bars for VWAP breakout analysis.",
                strategy=self.NAME,
            )

        close = float(clean["close"].iloc[-1])
        vwap = float(clean["vwap"].iloc[-1])
        prior_close = float(clean["close"].iloc[-_MOMENTUM_BARS - 1])

        distance_pct = (close - vwap) / vwap * 100
        momentum_pct = (close - prior_close) / prior_close * 100

        logger.debug(
            "VWAPBreakout — close=%.2f vwap=%.2f distance_pct=%.3f momentum_pct=%.3f",
            close, vwap, distance_pct, momentum_pct,
        )

        if distance_pct > _BREAKOUT_THRESHOLD_PCT and momentum_pct > 0:
            confidence = self._compute_confidence(distance_pct, momentum_pct)
            return Signal(
                signal=SignalType.BUY_CE,
                confidence=confidence,
                reason=(
                    f"Price ({close:.0f}) is {distance_pct:.2f}% above VWAP ({vwap:.0f}) "
                    f"with {momentum_pct:.2f}% upward momentum — bullish breakout."
                ),
                strategy=self.NAME,
            )

        if distance_pct < -_BREAKOUT_THRESHOLD_PCT and momentum_pct < 0:
            confidence = self._compute_confidence(abs(distance_pct), abs(momentum_pct))
            return Signal(
                signal=SignalType.BUY_PE,
                confidence=confidence,
                reason=(
                    f"Price ({close:.0f}) is {abs(distance_pct):.2f}% below VWAP ({vwap:.0f}) "
                    f"with {abs(momentum_pct):.2f}% downward momentum — bearish breakdown."
                ),
                strategy=self.NAME,
            )

        return Signal(
            signal=SignalType.HOLD,
            confidence=50,
            reason=(
                f"No confirmed VWAP breakout. Close={close:.0f}, VWAP={vwap:.0f} "
                f"({distance_pct:+.2f}%)."
            ),
            strategy=self.NAME,
        )

    @staticmethod
    def _compute_confidence(distance_pct: float, momentum_pct: float) -> int:
        """Score confidence 50–95 based on breakout distance and momentum strength."""
        base = 50
        distance_bonus = min(25, int(distance_pct * 15))
        momentum_bonus = min(20, int(momentum_pct * 10))
        return min(95, base + distance_bonus + momentum_bonus)
