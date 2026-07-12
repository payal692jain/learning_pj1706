"""Opening Range Breakout strategy — break of the first 15 minutes' high/low."""

import logging
from datetime import time as dt_time

import pandas as pd

from nifty_ai_agent.strategies.base import BaseStrategy, Signal, SignalType

logger = logging.getLogger(__name__)

_MARKET_OPEN = dt_time(9, 15)
_OPENING_RANGE_END = dt_time(9, 30)   # first 15 minutes = three 5m bars
_BREAKOUT_BUFFER_PCT = 0.05           # must clear the range edge by 0.05%, not just touch it
_VOLUME_CONFIRM_RATIO = 1.2           # breakout bar volume vs. the session average


class OpeningRangeBreakoutStrategy(BaseStrategy):
    """Generate signals when price breaks the 09:15–09:30 range of the current session.

    BUY_CE:  close > opening-range high + buffer
    BUY_PE:  close < opening-range low  - buffer
    HOLD:    inside the range, or the range has not formed yet

    The opening range is only meaningful for the session that produced it, so
    this reads the *latest* session in the DataFrame and ignores earlier days.
    """

    NAME = "Opening_Range_Breakout"

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        required = {"high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

        if not isinstance(df.index, pd.DatetimeIndex):
            return Signal(
                signal=SignalType.HOLD,
                confidence=50,
                reason="Opening range needs intraday timestamps — none available.",
                strategy=self.NAME,
            )

        session = df[df.index.date == df.index[-1].date()]
        opening_range = session[
            (session.index.time >= _MARKET_OPEN) & (session.index.time < _OPENING_RANGE_END)
        ]
        if opening_range.empty:
            return Signal(
                signal=SignalType.HOLD,
                confidence=50,
                reason="No 09:15–09:30 bars in this session yet — opening range not formed.",
                strategy=self.NAME,
            )

        or_high = float(opening_range["high"].max())
        or_low = float(opening_range["low"].min())
        latest = session.iloc[-1]
        close = float(latest["close"])

        if latest.name.time() < _OPENING_RANGE_END:
            return Signal(
                signal=SignalType.HOLD,
                confidence=50,
                reason=(
                    f"Still inside the opening range window — range so far "
                    f"{or_low:.0f}–{or_high:.0f}."
                ),
                strategy=self.NAME,
            )

        buffer_high = or_high * (1 + _BREAKOUT_BUFFER_PCT / 100)
        buffer_low = or_low * (1 - _BREAKOUT_BUFFER_PCT / 100)
        range_pct = (or_high - or_low) / or_low * 100 if or_low else 0.0
        volume_ratio = self._volume_ratio(session)

        logger.debug(
            "ORB — range=%.2f–%.2f close=%.2f range_pct=%.2f vol_ratio=%.2f",
            or_low, or_high, close, range_pct, volume_ratio,
        )

        if close > buffer_high:
            return Signal(
                signal=SignalType.BUY_CE,
                confidence=self._compute_confidence(
                    breakout_pct=(close - or_high) / or_high * 100,
                    range_pct=range_pct,
                    volume_ratio=volume_ratio,
                ),
                reason=(
                    f"Close ({close:.0f}) broke above the opening range high ({or_high:.0f}); "
                    f"range was {or_low:.0f}–{or_high:.0f} ({range_pct:.2f}% wide) on "
                    f"{volume_ratio:.1f}× average volume."
                ),
                strategy=self.NAME,
            )

        if close < buffer_low:
            return Signal(
                signal=SignalType.BUY_PE,
                confidence=self._compute_confidence(
                    breakout_pct=(or_low - close) / or_low * 100,
                    range_pct=range_pct,
                    volume_ratio=volume_ratio,
                ),
                reason=(
                    f"Close ({close:.0f}) broke below the opening range low ({or_low:.0f}); "
                    f"range was {or_low:.0f}–{or_high:.0f} ({range_pct:.2f}% wide) on "
                    f"{volume_ratio:.1f}× average volume."
                ),
                strategy=self.NAME,
            )

        return Signal(
            signal=SignalType.HOLD,
            confidence=50,
            reason=(
                f"Price ({close:.0f}) is still inside the opening range "
                f"{or_low:.0f}–{or_high:.0f} — no breakout."
            ),
            strategy=self.NAME,
        )

    @staticmethod
    def _volume_ratio(session: pd.DataFrame) -> float:
        """Latest bar's volume as a multiple of the session average (1.0 if no volume data).

        Index feeds routinely report zero volume — treat that as "no information"
        (neutral 1.0) rather than letting it veto an otherwise valid breakout.
        """
        if "volume" not in session.columns:
            return 1.0
        volumes = session["volume"].fillna(0)
        average = float(volumes.mean())
        if average <= 0:
            return 1.0
        return float(volumes.iloc[-1]) / average

    @staticmethod
    def _compute_confidence(
        breakout_pct: float, range_pct: float, volume_ratio: float,
    ) -> int:
        """Score 50–95. A tight opening range that breaks on volume is the strongest setup.

        A wide opening range means the move already happened inside it, so the
        breakout has less room left — wide ranges are penalised.
        """
        base = 50
        breakout_bonus = min(15, int(breakout_pct * 30))
        tightness_bonus = 15 if range_pct < 0.4 else (8 if range_pct < 0.8 else 0)
        volume_bonus = 15 if volume_ratio >= _VOLUME_CONFIRM_RATIO else 0
        return min(95, base + breakout_bonus + tightness_bonus + volume_bonus)
