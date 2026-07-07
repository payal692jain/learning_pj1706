"""RSI indicator — stateless, returns DataFrame column."""

import pandas as pd


def compute_rsi(df: pd.DataFrame, period: int = 14, price_col: str = "close") -> pd.DataFrame:
    """Add an 'rsi' column to a copy of *df*.

    Args:
        df: OHLCV DataFrame with a *price_col* column.
        period: Lookback window (default 14).
        price_col: Column to compute RSI on.

    Returns:
        DataFrame with additional 'rsi' column.
    """
    result = df.copy()
    delta = result[price_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    # When avg_loss is 0 (no down-moves), RSI = 100
    rsi = pd.Series(index=result.index, dtype=float)
    nonzero = avg_loss != 0
    rs = avg_gain[nonzero] / avg_loss[nonzero]
    rsi[nonzero] = 100 - (100 / (1 + rs))
    rsi[~nonzero & avg_gain.notna()] = 100.0
    result["rsi"] = rsi
    return result
