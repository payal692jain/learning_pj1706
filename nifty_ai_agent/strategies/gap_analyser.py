"""Gap statistics — what NIFTY has historically DONE after opening on a gap.

A gap forecast on its own is half an answer. "GIFT implies a 0.5% gap up" tells you
where the market opens; it says nothing about whether buying that open has ever
worked. This module supplies the missing half from NIFTY's own daily history: for
gaps in the same size bucket, how often did the day continue in the gap's direction,
and how often did it fade back and fill?

The base rates matter because the two cases demand opposite trades, and the naive
instinct (gap up → buy calls) is frequently the losing one: a large gap up often
opens at the day's high as overnight buyers take their profit from the people
chasing the open.

Everything here is computed from daily OHLC bars the pipeline already fetches. No
new data source, and no claim of predictive power beyond "this is what happened
last time", which is all a base rate ever is.
"""

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

# Must match the bucket boundaries in data/gift_nifty.py, or a gap would be
# forecast in one bucket and looked up in another.
_FLAT_PCT = 0.15
_LARGE_PCT = 0.75

# Below this many historical matches the percentages are noise dressed as insight.
_MIN_SAMPLE = 8


@dataclass
class GapStats:
    bucket: str
    sample: int              # how many historical days landed in this bucket
    continued: int           # closed further in the gap's direction than it opened
    faded: int               # closed back toward (or through) the previous close
    continuation_pct: float
    median_day_range_pct: float
    avg_close_vs_open_pct: float   # positive = the day paid gap-direction holders

    @property
    def is_reliable(self) -> bool:
        return self.sample >= _MIN_SAMPLE

    @property
    def verdict(self) -> str:
        """What the base rate actually recommends, in words."""
        if not self.is_reliable:
            return f"Only {self.sample} comparable day(s) on record — no usable base rate."
        if self.bucket == "FLAT":
            return (
                f"Flat opens gave no edge either way ({self.continuation_pct:.0f}% "
                f"continued, n={self.sample}) — wait for the range to establish."
            )
        direction = "higher" if self.bucket.endswith("UP") else "lower"
        against = "fade" if self.bucket.endswith("UP") else "bounce"
        if self.continuation_pct >= 60:
            return (
                f"{self.continuation_pct:.0f}% of {self.sample} comparable gaps kept "
                f"going {direction} — the gap has historically been worth following."
            )
        if self.continuation_pct <= 40:
            return (
                f"Only {self.continuation_pct:.0f}% of {self.sample} comparable gaps kept "
                f"going {direction} — these have mostly {against}d. Chasing the open "
                "has been the losing side."
            )
        return (
            f"A coin flip: {self.continuation_pct:.0f}% of {self.sample} comparable gaps "
            f"continued {direction}. No edge in the open itself."
        )


def classify_gap(gap_pct: float) -> str:
    """Bucket a gap by size and direction — the key both halves of this feature share."""
    if abs(gap_pct) < _FLAT_PCT:
        return "FLAT"
    size = "LARGE" if abs(gap_pct) >= _LARGE_PCT else "SMALL"
    return f"{size}_{'UP' if gap_pct > 0 else 'DOWN'}"


def analyse_gap_history(daily: pd.DataFrame, bucket: str) -> GapStats:
    """Base rate for *bucket*, computed from a daily OHLC DataFrame.

    Args:
        daily: Daily bars with open/high/low/close, oldest first.
        bucket: One of FLAT, SMALL_UP, LARGE_UP, SMALL_DOWN, LARGE_DOWN.

    Returns:
        GapStats. An empty/short history yields sample=0, which reports itself as
        unreliable rather than inventing a percentage from three data points.
    """
    required = {"open", "high", "low", "close"}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")

    df = daily.dropna(subset=["open", "high", "low", "close"]).copy()
    if len(df) < 2:
        return GapStats(bucket, 0, 0, 0, 0.0, 0.0, 0.0)

    prev_close = df["close"].shift(1)
    df["gap_pct"] = (df["open"] - prev_close) / prev_close * 100
    df["bucket"] = df["gap_pct"].apply(classify_gap)
    # How the day resolved, measured from the OPEN — that is where a gap trader
    # actually gets filled, not at yesterday's close.
    df["close_vs_open_pct"] = (df["close"] - df["open"]) / df["open"] * 100
    df["day_range_pct"] = (df["high"] - df["low"]) / df["open"] * 100

    matches = df[df["bucket"] == bucket].dropna(subset=["gap_pct"])
    sample = len(matches)
    if sample == 0:
        return GapStats(bucket, 0, 0, 0, 0.0, 0.0, 0.0)

    if bucket.endswith("UP"):
        continued_mask = matches["close_vs_open_pct"] > 0
    elif bucket.endswith("DOWN"):
        continued_mask = matches["close_vs_open_pct"] < 0
    else:
        # A flat open has no direction to continue; count green days so the number
        # still means something ("did the session go anywhere?").
        continued_mask = matches["close_vs_open_pct"] > 0

    continued = int(continued_mask.sum())
    stats = GapStats(
        bucket=bucket,
        sample=sample,
        continued=continued,
        faded=sample - continued,
        continuation_pct=round(continued / sample * 100, 1),
        median_day_range_pct=round(float(matches["day_range_pct"].median()), 2),
        avg_close_vs_open_pct=round(float(matches["close_vs_open_pct"].mean()), 2),
    )
    logger.info(
        "Gap history %s: n=%d continued=%d (%.0f%%) median range=%.2f%%",
        bucket, sample, continued, stats.continuation_pct, stats.median_day_range_pct,
    )
    return stats


@dataclass
class PivotLevels:
    """Classic floor-trader pivots — the levels tomorrow's open gets measured against."""
    pivot: float
    r1: float
    r2: float
    s1: float
    s2: float

    def context_for(self, price: float) -> str:
        """Where *price* sits in the pivot ladder."""
        if price >= self.r2:
            return f"above R2 ({self.r2:,.0f}) — extended"
        if price >= self.r1:
            return f"between R1 ({self.r1:,.0f}) and R2 ({self.r2:,.0f})"
        if price >= self.pivot:
            return f"between pivot ({self.pivot:,.0f}) and R1 ({self.r1:,.0f})"
        if price >= self.s1:
            return f"between S1 ({self.s1:,.0f}) and pivot ({self.pivot:,.0f})"
        if price >= self.s2:
            return f"between S2 ({self.s2:,.0f}) and S1 ({self.s1:,.0f})"
        return f"below S2 ({self.s2:,.0f}) — extended"


def compute_pivots(high: float, low: float, close: float) -> PivotLevels:
    """Standard pivots from the LAST completed session's high/low/close."""
    pivot = (high + low + close) / 3
    return PivotLevels(
        pivot=round(pivot, 2),
        r1=round(2 * pivot - low, 2),
        r2=round(pivot + (high - low), 2),
        s1=round(2 * pivot - high, 2),
        s2=round(pivot - (high - low), 2),
    )
