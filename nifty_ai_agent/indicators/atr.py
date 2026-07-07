"""ATR indicator — stateless, returns DataFrame with atr column."""

import pandas as pd


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add an 'atr' column to a copy of *df*.

    Uses Wilder's smoothing (EWM with alpha=1/period).
    """
    result = df.copy()
    high = result["high"]
    low = result["low"]
    prev_close = result["close"].shift(1)

    tr = pd.concat(
        [
            (high - low).rename("hl"),
            (high - prev_close).abs().rename("hc"),
            (low - prev_close).abs().rename("lc"),
        ],
        axis=1,
    ).max(axis=1)

    result["atr"] = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return result
