"""Bollinger squeeze strategy — volatility contraction followed by expansion."""

import logging

import pandas as pd

from nifty_ai_agent.strategies.base import BaseStrategy, Signal, SignalType

logger = logging.getLogger(__name__)

_SQUEEZE_LOOKBACK = 40      # bars over which "narrow" is judged
_SQUEEZE_PERCENTILE = 0.25  # band width in the bottom quartile of the lookback = squeeze
_EXPANSION_RATIO = 1.15     # bands must be widening: current width vs. the squeeze width


class BollingerSqueezeStrategy(BaseStrategy):
    """Generate signals when a Bollinger squeeze resolves into a directional break.

    BUY_CE:  bands were squeezed, are now expanding, close breaks the upper band
    BUY_PE:  bands were squeezed, are now expanding, close breaks the lower band
    HOLD:    still squeezed (no direction yet), or no squeeze to resolve

    Buying the squeeze itself is a trap for option buyers — a flat market bleeds
    theta. This waits for the expansion *and* a directional break before firing.
    """

    NAME = "Bollinger_Squeeze"

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        required = {"close", "bb_upper", "bb_lower", "bb_width"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

        clean = df.dropna(subset=["bb_upper", "bb_lower", "bb_width"])
        if len(clean) < _SQUEEZE_LOOKBACK:
            return Signal(
                signal=SignalType.HOLD,
                confidence=50,
                reason="Insufficient bars for Bollinger squeeze analysis.",
                strategy=self.NAME,
            )

        window = clean["bb_width"].iloc[-_SQUEEZE_LOOKBACK:]
        squeeze_threshold = float(window.quantile(_SQUEEZE_PERCENTILE))
        recent_min = float(window.iloc[-_SQUEEZE_LOOKBACK // 2:].min())

        latest = clean.iloc[-1]
        close = float(latest["close"])
        upper = float(latest["bb_upper"])
        lower = float(latest["bb_lower"])
        width = float(latest["bb_width"])

        was_squeezed = recent_min <= squeeze_threshold
        expanding = recent_min > 0 and width >= recent_min * _EXPANSION_RATIO

        logger.debug(
            "BollingerSqueeze — width=%.3f squeeze_thr=%.3f recent_min=%.3f "
            "squeezed=%s expanding=%s close=%.2f",
            width, squeeze_threshold, recent_min, was_squeezed, expanding, close,
        )

        if not was_squeezed:
            return Signal(
                signal=SignalType.HOLD,
                confidence=50,
                reason=(
                    f"No recent volatility squeeze to resolve (band width {width:.2f}%) "
                    "— nothing coiled to break."
                ),
                strategy=self.NAME,
            )

        if not expanding:
            return Signal(
                signal=SignalType.HOLD,
                confidence=50,
                reason=(
                    f"Bands are still squeezed (width {width:.2f}%, floor {recent_min:.2f}%) "
                    "— waiting for expansion before taking a side."
                ),
                strategy=self.NAME,
            )

        expansion_ratio = width / recent_min

        if close > upper:
            return Signal(
                signal=SignalType.BUY_CE,
                confidence=self._compute_confidence(expansion_ratio, (close - upper) / upper * 100),
                reason=(
                    f"Squeeze resolved upward: bands expanded {expansion_ratio:.1f}× from their "
                    f"floor and close ({close:.0f}) broke the upper band ({upper:.0f})."
                ),
                strategy=self.NAME,
            )

        if close < lower:
            return Signal(
                signal=SignalType.BUY_PE,
                confidence=self._compute_confidence(expansion_ratio, (lower - close) / lower * 100),
                reason=(
                    f"Squeeze resolved downward: bands expanded {expansion_ratio:.1f}× from their "
                    f"floor and close ({close:.0f}) broke the lower band ({lower:.0f})."
                ),
                strategy=self.NAME,
            )

        return Signal(
            signal=SignalType.HOLD,
            confidence=50,
            reason=(
                f"Bands are expanding ({expansion_ratio:.1f}×) but close ({close:.0f}) is still "
                f"inside {lower:.0f}–{upper:.0f} — no direction confirmed yet."
            ),
            strategy=self.NAME,
        )

    @staticmethod
    def _compute_confidence(expansion_ratio: float, break_pct: float) -> int:
        """Score 50–95 from how violently the bands opened and how far price cleared them."""
        base = 50
        expansion_bonus = min(25, int((expansion_ratio - 1) * 50))
        break_bonus = min(20, int(break_pct * 40))
        return min(95, base + expansion_bonus + break_bonus)
