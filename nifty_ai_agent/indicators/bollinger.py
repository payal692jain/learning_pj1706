"""Bollinger Bands indicator — stateless, returns DataFrame with band columns."""

import pandas as pd


def compute_bollinger(
    df: pd.DataFrame, period: int = 20, std_devs: float = 2.0,
) -> pd.DataFrame:
    """Add 'bb_upper', 'bb_mid', 'bb_lower', and 'bb_width' columns to a copy of *df*.

    bb_width is the band spread as a percentage of the mid band — the value a
    squeeze strategy actually reads, since raw band distance is not comparable
    across indices trading at 24k and 79k.
    """
    result = df.copy()
    close = result["close"]

    mid = close.rolling(window=period, min_periods=period).mean()
    # Population std (ddof=0) — the convention Bollinger's own formulation uses;
    # pandas defaults to the sample std (ddof=1), which widens the bands slightly.
    sigma = close.rolling(window=period, min_periods=period).std(ddof=0)

    result["bb_mid"] = mid
    result["bb_upper"] = mid + std_devs * sigma
    result["bb_lower"] = mid - std_devs * sigma
    result["bb_width"] = (result["bb_upper"] - result["bb_lower"]) / mid * 100
    return result
