"""EMA Crossover strategy — the initial strategy per CLAUDE.md spec."""

import logging

import pandas as pd

from nifty_ai_agent.strategies.base import BaseStrategy, Signal, SignalType

logger = logging.getLogger(__name__)

# Configurable thresholds
_RSI_BULL_THRESHOLD = 60
_RSI_BEAR_THRESHOLD = 40
_MIN_EMA_SEPARATION_PCT = 0.05  # EMA20 must differ from EMA50 by at least 0.05%


class EMACrossoverStrategy(BaseStrategy):
    """Generate signals based on EMA20/EMA50 crossover + RSI confirmation.

    BUY_CE:  EMA20 > EMA50  AND  RSI > 60
    BUY_PE:  EMA20 < EMA50  AND  RSI < 40
    HOLD:    all other cases
    """

    NAME = "EMA_Crossover"

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        required = {"close", "ema_20", "ema_50", "rsi"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

        latest = df.dropna(subset=["ema_20", "ema_50", "rsi"]).iloc[-1]
        ema20 = float(latest["ema_20"])
        ema50 = float(latest["ema_50"])
        rsi = float(latest["rsi"])
        close = float(latest["close"])

        logger.debug(
            "EMACrossover — ema20=%.2f ema50=%.2f rsi=%.2f close=%.2f",
            ema20, ema50, rsi, close,
        )

        separation_pct = abs(ema20 - ema50) / ema50 * 100

        if ema20 > ema50 and rsi > _RSI_BULL_THRESHOLD:
            confidence = self._compute_confidence(
                ema_diff_pct=separation_pct,
                rsi=rsi,
                direction="bull",
            )
            return Signal(
                signal=SignalType.BUY_CE,
                confidence=confidence,
                reason=(
                    f"EMA20 ({ema20:.0f}) > EMA50 ({ema50:.0f}) "
                    f"with RSI at {rsi:.1f} — bullish momentum confirmed."
                ),
                strategy=self.NAME,
            )

        if ema20 < ema50 and rsi < _RSI_BEAR_THRESHOLD:
            confidence = self._compute_confidence(
                ema_diff_pct=separation_pct,
                rsi=rsi,
                direction="bear",
            )
            return Signal(
                signal=SignalType.BUY_PE,
                confidence=confidence,
                reason=(
                    f"EMA20 ({ema20:.0f}) < EMA50 ({ema50:.0f}) "
                    f"with RSI at {rsi:.1f} — bearish momentum confirmed."
                ),
                strategy=self.NAME,
            )

        return Signal(
            signal=SignalType.HOLD,
            confidence=50,
            reason=(
                f"No clear crossover. EMA20={ema20:.0f}, EMA50={ema50:.0f}, RSI={rsi:.1f}."
            ),
            strategy=self.NAME,
        )

    @staticmethod
    def _compute_confidence(ema_diff_pct: float, rsi: float, direction: str) -> int:
        """Score confidence 50–95 based on how far conditions are from thresholds."""
        base = 50

        # EMA separation bonus (up to +25)
        ema_bonus = min(25, int(ema_diff_pct * 10))

        # RSI strength bonus (up to +20)
        if direction == "bull":
            rsi_bonus = min(20, int((rsi - _RSI_BULL_THRESHOLD) * 0.8))
        else:
            rsi_bonus = min(20, int((_RSI_BEAR_THRESHOLD - rsi) * 0.8))

        return min(95, base + ema_bonus + rsi_bonus)
