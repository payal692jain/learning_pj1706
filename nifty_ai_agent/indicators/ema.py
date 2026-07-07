"""EMA indicator — stateless, returns DataFrame with ema columns."""

import pandas as pd


def compute_ema(
    df: pd.DataFrame,
    periods: list[int] | None = None,
    price_col: str = "close",
) -> pd.DataFrame:
    """Add EMA columns to a copy of *df*.

    Args:
        df: OHLCV DataFrame.
        periods: List of EMA periods (default [20, 50]).
        price_col: Column to compute EMAs on.

    Returns:
        DataFrame with 'ema_{period}' columns for each period.
    """
    if periods is None:
        periods = [20, 50]
    result = df.copy()
    for period in periods:
        result[f"ema_{period}"] = (
            result[price_col].ewm(span=period, adjust=False).mean()
        )
    return result
