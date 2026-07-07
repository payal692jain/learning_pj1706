"""VWAP indicator — stateless, returns DataFrame with vwap column.

For daily OHLCV data VWAP is approximated as the cumulative
(typical_price × volume) / cumulative_volume within each trading day.
On a 1-day lookback this reduces to the simple typical-price × volume ratio.
"""

import pandas as pd


def compute_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'vwap' column to a copy of *df*.

    Works on any interval; for daily bars this is the session VWAP.
    """
    result = df.copy()
    typical = (result["high"] + result["low"] + result["close"]) / 3
    cum_tp_vol = (typical * result["volume"]).cumsum()
    cum_vol = result["volume"].cumsum()
    result["vwap"] = cum_tp_vol / cum_vol.replace(0, float("nan"))
    return result
