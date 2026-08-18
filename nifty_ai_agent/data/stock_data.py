"""Bulk OHLCV fetch for the NIFTY 50 stock universe.

The stock scanner needs intraday bars for ~50 names at once. Consistent with the
index path, Upstox is the primary source: when a token is configured, each name's
cash-market OHLCV is pulled from Upstox's authenticated API (resolving the equity
key via the instrument master). Any name Upstox cannot serve — and the whole
universe when no token is set — falls back to yfinance.

The yfinance fallback pulls the survivors in a single bulk download and splits them
per symbol — the same bulk pattern nifty50_stocks.fetch_nifty50_movers relies on,
extended from close-only to full OHLCV.
"""

import logging

import pandas as pd
import yfinance as yf

from nifty_ai_agent.data.instrument_master import InstrumentMaster

logger = logging.getLogger(__name__)

_OHLCV = ["open", "high", "low", "close", "volume"]

# yfinance tickers carry a ".NS" suffix; Upstox / the instrument master key stocks
# by the bare NSE trading symbol ("RELIANCE", "INFY").
_NS_SUFFIX = ".NS"


def fetch_stock_histories(
    symbols: list[str],
    period: str = "5d",
    interval: str = "5m",
    *,
    upstox_token: str = "",
    master: InstrumentMaster | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    """Return *(histories, spots)* for *symbols* (yfinance tickers, e.g. 'RELIANCE.NS').

    ``histories[sym]`` is an OHLCV DataFrame (lowercase columns, DatetimeIndex named
    'datetime'); ``spots[sym]`` is that symbol's most recent close. Symbols with no
    usable data are simply omitted from both dicts — a scan over the survivors is
    more useful than an all-or-nothing failure when a source drops a name or two.

    When *upstox_token* and *master* are supplied, Upstox is tried first per symbol;
    any name it cannot serve falls back to a single bulk yfinance download.
    """
    if not symbols:
        return {}, {}

    histories: dict[str, pd.DataFrame] = {}
    spots: dict[str, float] = {}

    remaining = list(symbols)
    if upstox_token and master is not None:
        histories, spots = _fetch_via_upstox(remaining, upstox_token, master, period, interval)
        remaining = [s for s in symbols if s not in histories]
        if remaining:
            logger.info(
                "Upstox stock histories: %d/%d served — %d falling back to yfinance",
                len(histories), len(symbols), len(remaining),
            )

    if remaining:
        yf_hist, yf_spots = _fetch_via_yfinance(remaining, period, interval)
        histories.update(yf_hist)
        spots.update(yf_spots)

    logger.info("Stock histories: %d/%d symbols with usable data", len(histories), len(symbols))
    return histories, spots


def _fetch_via_upstox(
    symbols: list[str],
    token: str,
    master: InstrumentMaster,
    period: str,
    interval: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    """Per-symbol Upstox OHLCV. Upstox has no bulk endpoint, so this loops; any name
    that raises (unresolvable key, auth error, no candles) is dropped for the caller
    to retry via yfinance rather than failing the whole scan."""
    from nifty_ai_agent.data.upstox_provider import UpstoxClient

    client = UpstoxClient(token)
    days = _period_to_days(period)
    histories: dict[str, pd.DataFrame] = {}
    spots: dict[str, float] = {}

    for sym in symbols:
        bare = sym[: -len(_NS_SUFFIX)] if sym.endswith(_NS_SUFFIX) else sym
        try:
            # equity_key() follows the rename table, so a ticker retired by a
            # demerger resolves to its successor here rather than falling through
            # to yfinance and 404-ing on a symbol that no longer exists.
            key = master.equity_key(bare)
            if not key:
                continue
            df = client.get_historical_ohlcv_by_key(key, days=days, interval=interval)
            df = df.dropna(how="all")
            if df.empty or not set(_OHLCV).issubset(df.columns):
                continue
            df = df[_OHLCV].copy()
            df.index.name = "datetime"
            close = float(df["close"].iloc[-1])
            if close <= 0:
                continue
            histories[sym] = df
            spots[sym] = close
        except Exception as exc:
            logger.debug("Upstox stock history failed for %s (%s) — trying yfinance", sym, exc)

    return histories, spots


def _fetch_via_yfinance(
    symbols: list[str],
    period: str,
    interval: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    """Single bulk yfinance download for the whole (residual) universe, split per symbol."""
    logger.info(
        "Fetching %d stock histories via yfinance (period=%s interval=%s)",
        len(symbols), period, interval,
    )
    raw = yf.download(
        symbols,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=True,
        group_by="ticker",
    )
    if raw is None or raw.empty:
        logger.warning("yfinance returned no stock history for the universe")
        return {}, {}

    multi = isinstance(raw.columns, pd.MultiIndex)
    histories: dict[str, pd.DataFrame] = {}
    spots: dict[str, float] = {}

    for sym in symbols:
        try:
            sdf = raw[sym] if multi else raw
            sdf = sdf.dropna(how="all")
            if sdf.empty:
                continue
            sdf = sdf.rename(columns=str.lower)
            if not set(_OHLCV).issubset(sdf.columns):
                continue
            sdf = sdf[_OHLCV].copy()
            sdf.index.name = "datetime"
            close = float(sdf["close"].iloc[-1])
            if close <= 0:
                continue
            histories[sym] = sdf
            spots[sym] = close
        except Exception as exc:
            logger.debug("Skipping %s — no usable history (%s)", sym, exc)

    return histories, spots


def _period_to_days(period: str) -> int:
    """Convert a yfinance-style period string ('5d', '10d') to a day count for Upstox.

    Upstox's historical API takes an explicit date range rather than a period token,
    so the '<n>d' form is parsed to <n>; anything else defaults to 5 days.
    """
    period = period.strip().lower()
    if period.endswith("d"):
        try:
            return max(1, int(period[:-1]))
        except ValueError:
            pass
    return 5
