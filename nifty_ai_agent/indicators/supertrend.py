"""Supertrend indicator — stateless, returns DataFrame with supertrend columns."""

import pandas as pd

from nifty_ai_agent.indicators.atr import compute_atr


def compute_supertrend(
    df: pd.DataFrame, period: int = 10, multiplier: float = 3.0,
) -> pd.DataFrame:
    """Add 'supertrend' and 'supertrend_dir' columns to a copy of *df*.

    supertrend_dir is +1 while the trend is up (price above the band) and -1
    while it is down; the flip bar is where a trade is signalled.

    The bands are "locked": the upper band can only ratchet down and the lower
    band only up while the trend is unchanged. Without that lock the bands
    breathe with every ATR tick and the direction whipsaws on noise, which is
    the whole reason Supertrend is used over a raw ATR channel.
    """
    result = df.copy()

    # Supertrend needs its own ATR at *period*, which differs from the 14-period
    # ATR the risk engine uses — compute it here rather than borrowing that column.
    atr = compute_atr(result, period=period)["atr"]
    hl2 = (result["high"] + result["low"]) / 2
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    close = result["close"]
    n = len(result)
    upper = [float("nan")] * n
    lower = [float("nan")] * n
    direction = [1] * n

    for i in range(n):
        if pd.isna(atr.iloc[i]):
            continue

        prev_upper = upper[i - 1] if i > 0 and not pd.isna(upper[i - 1]) else upper_basic.iloc[i]
        prev_lower = lower[i - 1] if i > 0 and not pd.isna(lower[i - 1]) else lower_basic.iloc[i]
        prev_close = close.iloc[i - 1] if i > 0 else close.iloc[i]

        upper[i] = (
            upper_basic.iloc[i]
            if upper_basic.iloc[i] < prev_upper or prev_close > prev_upper
            else prev_upper
        )
        lower[i] = (
            lower_basic.iloc[i]
            if lower_basic.iloc[i] > prev_lower or prev_close < prev_lower
            else prev_lower
        )

        prev_dir = direction[i - 1] if i > 0 else 1
        if close.iloc[i] > upper[i]:
            direction[i] = 1
        elif close.iloc[i] < lower[i]:
            direction[i] = -1
        else:
            direction[i] = prev_dir

    result["supertrend_dir"] = direction
    result["supertrend"] = [
        lower[i] if direction[i] == 1 else upper[i] for i in range(n)
    ]
    # Bars before ATR warms up carry no trend — blank both so strategies that
    # dropna() on 'supertrend' don't read the placeholder direction as real.
    result.loc[atr.isna(), ["supertrend", "supertrend_dir"]] = float("nan")
    return result
