"""OHLCV for the NSE currency pairs that carry options.

Currency options (NCD_FO) are struck against the *futures* price, not an
interbank spot rate, so the near-month FUT contract is the underlying this module
returns. Quoting a USDINR option chain off a spot rate from elsewhere would put
the ATM strike in the wrong place whenever the forward premium is meaningful,
which for USDINR it usually is.

Upstox is the only source here — yfinance's INR crosses are interbank spot and
would reintroduce exactly that basis error.
"""

import logging

import pandas as pd

from nifty_ai_agent.data.instrument_master import SEGMENT_CURRENCY_FO, InstrumentMaster

logger = logging.getLogger(__name__)

# The four pairs NSE lists options on.
CURRENCY_PAIRS: list[str] = ["USDINR", "EURINR", "GBPINR", "JPYINR"]

_OHLCV = ["open", "high", "low", "close", "volume"]

# How many expiries to try per pair before giving up. Enough to step past a few
# dead weeklies onto the liquid monthly, without turning one missing pair into a
# dozen wasted API calls every scan.
_MAX_FUT_ATTEMPTS = 5


def live_futures(master: InstrumentMaster, pair: str) -> list[dict]:
    """Every unexpired FUT row for *pair*, soonest expiry first.

    NSE lists weekly *and* monthly currency futures. The weeklies on the non-USD
    pairs frequently have no trades at all, so the nearest contract is not always
    the one with a usable price history — callers should walk this list rather than
    trusting the front month.
    """
    from datetime import date

    from nifty_ai_agent.data.instrument_master import _epoch_ms_to_date

    today = date.today()
    futures: list[tuple] = []
    expiring_today: list[tuple] = []
    for row in master._load():  # noqa: SLF001 — same package, no public row accessor
        if (
            row.get("segment") != SEGMENT_CURRENCY_FO
            or row.get("asset_symbol") != pair
            or row.get("instrument_type") != "FUT"
        ):
            continue
        expiry = _epoch_ms_to_date(row.get("expiry"))
        if expiry is None or expiry < today:
            continue
        # A contract settling today stops printing new candles partway through the
        # session, so it quotes a stale price while a later expiry is still live.
        (expiring_today if expiry == today else futures).append((expiry, row))

    if not futures and not expiring_today:
        logger.warning("Currency: no live %s future in the instrument master", pair)
    # Today's expiry is a last resort, never the first choice.
    ordered = sorted(futures, key=lambda er: er[0]) + sorted(
        expiring_today, key=lambda er: er[0]
    )
    return [row for _, row in ordered]


def near_month_future(master: InstrumentMaster, pair: str) -> dict | None:
    """The soonest-expiring FUT row for *pair*, or None when the master has none."""
    futures = live_futures(master, pair)
    return futures[0] if futures else None


def fetch_currency_histories(
    upstox_token: str,
    master: InstrumentMaster,
    pairs: list[str] | None = None,
    period: str = "10d",
    interval: str = "5m",
) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    """Return *(histories, spots)* keyed by pair name ("USDINR", …).

    A pair Upstox cannot serve is omitted from both dicts rather than failing the
    batch — a three-pair read is more useful than none. Requires a token: the
    currency candle endpoints are authenticated.
    """
    pairs = pairs or CURRENCY_PAIRS
    if not upstox_token:
        logger.info("Currency: no Upstox token — skipping (no unauthenticated source)")
        return {}, {}

    from nifty_ai_agent.data.stock_data import _period_to_days
    from nifty_ai_agent.data.upstox_provider import UpstoxClient

    client = UpstoxClient(upstox_token)
    days = _period_to_days(period)
    histories: dict[str, pd.DataFrame] = {}
    spots: dict[str, float] = {}

    for pair in pairs:
        # Walk expiries outward until one has a usable history: the weekly futures
        # on EUR/GBP/JPY INR are routinely untraded, and stopping at the front month
        # would drop three of the four pairs without ever saying why.
        for attempt, future in enumerate(live_futures(master, pair)[:_MAX_FUT_ATTEMPTS]):
            try:
                df = client.get_historical_ohlcv_by_key(
                    future["instrument_key"], days=days, interval=interval,
                ).dropna(how="all")
                if df.empty or not set(_OHLCV).issubset(df.columns):
                    continue
                df = df[_OHLCV].copy()
                df.index.name = "datetime"
                close = float(df["close"].iloc[-1])
                if close <= 0:
                    continue
                histories[pair] = df
                spots[pair] = close
                if attempt:
                    logger.info(
                        "Currency: %s priced off %s — %d nearer contract(s) had no candles",
                        pair, future.get("trading_symbol", "?"), attempt,
                    )
                break
            except Exception as exc:
                logger.debug(
                    "Currency: %s via %s failed — %s",
                    pair, future.get("trading_symbol", "?"), exc,
                )
        else:
            logger.warning("Currency: no tradeable %s future had a price history", pair)

    logger.info("Currency histories: %d/%d pairs with usable data", len(histories), len(pairs))
    return histories, spots
