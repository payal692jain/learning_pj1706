"""Rich RSI analysis beyond the binary threshold used by EMA crossover.

Provides zone classification, trend direction, and price-RSI divergence
detection.  The confidence adjustment function translates these into a
signed delta that the pipeline applies on top of the base strategy score.
"""

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

_DEEPLY_OVERSOLD  = 30
_OVERSOLD         = 40
_OVERBOUGHT       = 60
_DEEPLY_OVERBOUGHT = 70

_TREND_BARS      = 5   # compare RSI now vs N bars ago for trend
_DIV_LOOKBACK    = 14  # bars for divergence detection
_DIV_PRICE_PCT   = 0.2 # min % price move to consider a meaningful swing
_DIV_RSI_DELTA   = 3.0 # min RSI point difference for divergence


@dataclass
class RSIAnalysis:
    value: float
    zone: str       # DEEPLY_OVERSOLD / OVERSOLD / NEUTRAL / OVERBOUGHT / DEEPLY_OVERBOUGHT
    trend: str      # RISING / FALLING / FLAT
    divergence: str # BEARISH_DIV / BULLISH_DIV / NONE
    note: str       # human-readable one-liner


def analyse_rsi(df: pd.DataFrame) -> RSIAnalysis:
    """Compute zone, trend, and divergence from the 'rsi' and 'close' columns of *df*.

    Returns a neutral analysis if data is insufficient.
    """
    if "rsi" not in df.columns or df["rsi"].dropna().empty:
        return _neutral("RSI column unavailable")

    clean = df.dropna(subset=["rsi", "close"]).copy()
    if len(clean) < _TREND_BARS + 2:
        return _neutral("Insufficient bars for RSI analysis")

    current = float(clean["rsi"].iloc[-1])
    zone = _zone(current)
    trend = _trend(clean)
    divergence = _divergence(clean)
    note = _note(current, zone, trend, divergence)

    logger.info(
        "RSI analysis: %.1f  zone=%s  trend=%s  divergence=%s",
        current, zone, trend, divergence,
    )
    return RSIAnalysis(
        value=round(current, 1),
        zone=zone,
        trend=trend,
        divergence=divergence,
        note=note,
    )


def rsi_confidence_adjustment(analysis: RSIAnalysis, signal_type: str) -> tuple[int, str]:
    """Return *(confidence_delta, detail_text)* for adding to the signal reason.

    signal_type: "BUY_CE", "BUY_PE", or "HOLD".
    HOLD signals are returned unchanged.
    """
    if signal_type == "HOLD":
        return 0, ""

    delta = 0
    notes: list[str] = []
    bullish = signal_type == "BUY_CE"

    # ── Zone ───────────────────────────────────────────────────────────────────
    if bullish:
        if analysis.zone == "DEEPLY_OVERBOUGHT":
            delta += 8
            notes.append(f"RSI {analysis.value} — deeply overbought, strong bull momentum")
        elif analysis.zone == "OVERBOUGHT":
            delta += 4
            notes.append(f"RSI {analysis.value} — overbought zone, momentum confirmed")
        elif analysis.zone in ("OVERSOLD", "DEEPLY_OVERSOLD"):
            delta -= 6
            notes.append(f"RSI {analysis.value} — oversold, counter-trend CE entry risk")
    else:  # BUY_PE
        if analysis.zone == "DEEPLY_OVERSOLD":
            delta += 8
            notes.append(f"RSI {analysis.value} — deeply oversold, strong bear momentum")
        elif analysis.zone == "OVERSOLD":
            delta += 4
            notes.append(f"RSI {analysis.value} — oversold zone, momentum confirmed")
        elif analysis.zone in ("OVERBOUGHT", "DEEPLY_OVERBOUGHT"):
            delta -= 6
            notes.append(f"RSI {analysis.value} — overbought, counter-trend PE entry risk")

    # ── Trend ──────────────────────────────────────────────────────────────────
    if bullish:
        if analysis.trend == "RISING":
            delta += 3
            notes.append("RSI trending up")
        elif analysis.trend == "FALLING":
            delta -= 5
            notes.append("RSI turning down — momentum fading")
    else:
        if analysis.trend == "FALLING":
            delta += 3
            notes.append("RSI trending down")
        elif analysis.trend == "RISING":
            delta -= 5
            notes.append("RSI turning up — bear momentum fading")

    # ── Divergence (most important — overrides zone/trend partially) ────────────
    if bullish and analysis.divergence == "BEARISH_DIV":
        delta -= 12
        notes.append("bearish RSI divergence — price up, RSI not confirming")
    elif not bullish and analysis.divergence == "BULLISH_DIV":
        delta -= 12
        notes.append("bullish RSI divergence — price down, RSI not confirming")

    detail = (
        f" RSI ({analysis.zone.replace('_', ' ').title()}): {'; '.join(notes)}."
        if notes else ""
    )
    return delta, detail


# ── Internal helpers ────────────────────────────────────────────────────────────

def _zone(rsi: float) -> str:
    if rsi < _DEEPLY_OVERSOLD:
        return "DEEPLY_OVERSOLD"
    if rsi < _OVERSOLD:
        return "OVERSOLD"
    if rsi < _OVERBOUGHT:
        return "NEUTRAL"
    if rsi < _DEEPLY_OVERBOUGHT:
        return "OVERBOUGHT"
    return "DEEPLY_OVERBOUGHT"


def _trend(df: pd.DataFrame) -> str:
    now = float(df["rsi"].iloc[-1])
    prev = float(df["rsi"].iloc[-_TREND_BARS - 1])
    diff = now - prev
    if diff > 2.5:
        return "RISING"
    if diff < -2.5:
        return "FALLING"
    return "FLAT"


def _divergence(df: pd.DataFrame) -> str:
    """Compare price swing vs RSI swing over the last _DIV_LOOKBACK bars.

    Bearish divergence: price made a higher high but RSI made a lower high.
    Bullish divergence: price made a lower low but RSI made a higher low.
    """
    n = min(_DIV_LOOKBACK, len(df) - 1)
    window = df.iloc[-n - 1:]

    price_now = float(window["close"].iloc[-1])
    price_prev = float(window["close"].iloc[0])
    rsi_now = float(window["rsi"].iloc[-1])
    rsi_prev = float(window["rsi"].iloc[0])

    price_moved_up = price_now > price_prev * (1 + _DIV_PRICE_PCT / 100)
    price_moved_down = price_now < price_prev * (1 - _DIV_PRICE_PCT / 100)
    rsi_lower = rsi_now < rsi_prev - _DIV_RSI_DELTA
    rsi_higher = rsi_now > rsi_prev + _DIV_RSI_DELTA

    if price_moved_up and rsi_lower:
        return "BEARISH_DIV"
    if price_moved_down and rsi_higher:
        return "BULLISH_DIV"
    return "NONE"


def _note(rsi: float, zone: str, trend: str, divergence: str) -> str:
    parts = [f"RSI {rsi} — {zone.replace('_', ' ').title()}"]
    if trend != "FLAT":
        parts.append(f"trending {trend.lower()}")
    if divergence != "NONE":
        parts.append(divergence.replace("_", " ").lower())
    return ", ".join(parts)


def _neutral(reason: str) -> RSIAnalysis:
    return RSIAnalysis(
        value=50.0, zone="NEUTRAL", trend="FLAT",
        divergence="NONE", note=reason,
    )
