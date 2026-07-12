"""Supertrend strategy — ATR-band trend following."""

import logging

import pandas as pd

from nifty_ai_agent.strategies.base import BaseStrategy, Signal, SignalType

logger = logging.getLogger(__name__)

# How recent a trend flip must be to still count as a tradeable entry. Buying an
# option into the 40th bar of an established trend is buying the exhaustion, so
# stale trends decay in confidence rather than firing at full strength.
_FRESH_FLIP_BARS = 6
_MAX_TREND_AGE_BARS = 30


class SupertrendStrategy(BaseStrategy):
    """Generate signals from Supertrend direction, weighted by how fresh the flip is.

    BUY_CE:  supertrend_dir == +1  AND  trend is younger than 30 bars
    BUY_PE:  supertrend_dir == -1  AND  trend is younger than 30 bars
    HOLD:    no trend, or the trend is too old to chase
    """

    NAME = "Supertrend"

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        required = {"close", "supertrend", "supertrend_dir"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

        clean = df.dropna(subset=["supertrend", "supertrend_dir"])
        if clean.empty:
            return Signal(
                signal=SignalType.HOLD,
                confidence=50,
                reason="Insufficient bars for Supertrend analysis.",
                strategy=self.NAME,
            )

        direction = int(clean["supertrend_dir"].iloc[-1])
        band = float(clean["supertrend"].iloc[-1])
        close = float(clean["close"].iloc[-1])
        age = self._trend_age(clean["supertrend_dir"])

        logger.debug(
            "Supertrend — dir=%+d band=%.2f close=%.2f age=%d bars",
            direction, band, close, age,
        )

        if age > _MAX_TREND_AGE_BARS:
            return Signal(
                signal=SignalType.HOLD,
                confidence=50,
                reason=(
                    f"Supertrend has been {'up' if direction > 0 else 'down'} for "
                    f"{age} bars — too extended to chase a fresh entry."
                ),
                strategy=self.NAME,
            )

        # Distance from the band is the room left before the trend invalidates.
        band_distance_pct = abs(close - band) / band * 100
        confidence = self._compute_confidence(age, band_distance_pct)
        trend_word = "flipped" if age <= _FRESH_FLIP_BARS else "held"

        if direction > 0:
            return Signal(
                signal=SignalType.BUY_CE,
                confidence=confidence,
                reason=(
                    f"Supertrend {trend_word} bullish {age} bar(s) ago; close ({close:.0f}) "
                    f"is holding {band_distance_pct:.2f}% above the band ({band:.0f})."
                ),
                strategy=self.NAME,
            )

        return Signal(
            signal=SignalType.BUY_PE,
            confidence=confidence,
            reason=(
                f"Supertrend {trend_word} bearish {age} bar(s) ago; close ({close:.0f}) "
                f"is holding {band_distance_pct:.2f}% below the band ({band:.0f})."
            ),
            strategy=self.NAME,
        )

    @staticmethod
    def _trend_age(direction: pd.Series) -> int:
        """Bars since the last direction flip (1 = flipped on the latest bar)."""
        values = direction.to_numpy()
        latest = values[-1]
        age = 1
        for value in reversed(values[:-1]):
            if value != latest:
                break
            age += 1
        return age

    @staticmethod
    def _compute_confidence(age: int, band_distance_pct: float) -> int:
        """Score 50–95: fresh flips with the price clear of the band score highest."""
        base = 50
        # Freshness bonus decays linearly from +25 at a same-bar flip to 0 at the age cap.
        freshness_bonus = max(0, int(25 * (1 - age / _MAX_TREND_AGE_BARS)))
        distance_bonus = min(20, int(band_distance_pct * 20))
        return min(95, base + freshness_bonus + distance_bonus)
