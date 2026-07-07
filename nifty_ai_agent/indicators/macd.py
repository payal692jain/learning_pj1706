"""MACD indicator — stateless, returns DataFrame with macd columns."""

import pandas as pd


def compute_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    price_col: str = "close",
) -> pd.DataFrame:
    """Add MACD columns to a copy of *df*.

    Adds columns: 'macd', 'macd_signal', 'macd_histogram'.
    """
    result = df.copy()
    ema_fast = result[price_col].ewm(span=fast, adjust=False).mean()
    ema_slow = result[price_col].ewm(span=slow, adjust=False).mean()
    result["macd"] = ema_fast - ema_slow
    result["macd_signal"] = result["macd"].ewm(span=signal, adjust=False).mean()
    result["macd_histogram"] = result["macd"] - result["macd_signal"]
    return result
