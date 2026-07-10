"""BSE SENSEX market data provider — yfinance for OHLCV, Upstox for live option chain.

BSE's own website exposes no reliable public option chain API, so live data
comes from Upstox (same as NIFTY) when UPSTOX_ACCESS_TOKEN is configured.
Falls back to a VIX-based synthetic estimate otherwise.

Key differences from NIFTY:
  - Strike step: ₹100 (NIFTY uses ₹50)
  - Weekly expiry: Thursday (confirmed live via Upstox — SENSEX moved off Friday,
    same as NIFTY moved off Thursday onto Tuesday)
  - yfinance symbol: ^BSESN
"""

import logging
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

from nifty_ai_agent.data.base import MarketDataProvider, OptionChainData, SpotData
from nifty_ai_agent.data.nse_provider import (
    _compute_max_pain,
    _compute_pcr,
    _identify_expiries,
    _iso_to_nse_date,
    _nse_date_to_iso,
)

logger = logging.getLogger(__name__)

_STRIKE_STEP = 100   # SENSEX options use ₹100 strike increments
_VIX_SYMBOL  = "^INDIAVIX"
_RETRY_COUNT = 3
_RETRY_DELAY = 3


def _retry(fn, retries: int = _RETRY_COUNT, delay: float = _RETRY_DELAY):
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            logger.warning("BSE attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def _next_thursday() -> str:
    """Return nearest upcoming Thursday as 'DD-Mon-YYYY' (BSE SENSEX weekly expiry)."""
    today = date.today()
    days_ahead = (3 - today.weekday()) % 7   # 3 = Thursday
    if days_ahead == 0:
        days_ahead = 7
    return (today + timedelta(days=days_ahead)).strftime("%d-%b-%Y")


class BSEDataProvider(MarketDataProvider):
    """Fetches SENSEX spot + OHLCV via yfinance; option chain via Upstox (or VIX proxy)."""

    def __init__(self, symbol: str = "^BSESN", upstox_access_token: str = "") -> None:
        self._symbol = symbol
        self._upstox_token = upstox_access_token

    # ── Public interface ────────────────────────────────────────────

    def get_spot_data(self) -> SpotData:
        """Fetch live spot price — Upstox first (if configured), then yfinance."""
        if self._upstox_token:
            try:
                return self._get_spot_data_via_upstox()
            except Exception as exc:
                logger.warning(
                    "Upstox SENSEX spot fetch failed (%s: %s) — falling back to yfinance",
                    type(exc).__name__, exc,
                )

        logger.info("Fetching SENSEX spot data via yfinance")

        def _fetch() -> SpotData:
            ticker = yf.Ticker(self._symbol)
            info = ticker.fast_info
            hist = ticker.history(period="1d", interval="1m")
            if hist.empty:
                raise ValueError("yfinance returned empty history for SENSEX spot")
            latest = hist.iloc[-1]
            return SpotData(
                symbol=self._symbol,
                price=float(info.last_price),
                timestamp=datetime.now(tz=timezone.utc),
                open=float(latest.get("Open", 0)),
                high=float(latest.get("High", 0)),
                low=float(latest.get("Low", 0)),
                volume=int(latest.get("Volume", 0)),
            )

        return _retry(_fetch)

    def _get_spot_data_via_upstox(self) -> SpotData:
        from nifty_ai_agent.data.upstox_provider import UpstoxClient

        quote = UpstoxClient(self._upstox_token).get_quote("SENSEX")
        logger.info("SENSEX spot fetched via Upstox (live): %.2f", quote["price"])
        return SpotData(
            symbol=self._symbol,
            price=quote["price"],
            timestamp=datetime.now(tz=timezone.utc),
            open=quote["open"],
            high=quote["high"],
            low=quote["low"],
            volume=int(quote["volume"]),
        )

    def get_option_chain(self) -> OptionChainData:
        """Fetch the live SENSEX option chain via Upstox; fall back to a VIX estimate.

        BSE's own website exposes no reliable public option chain API, so there is
        no scrape-based middle tier here (unlike NSEDataProvider) — it's Upstox or
        the synthetic estimate.
        """
        if self._upstox_token:
            try:
                data = self._fetch_option_chain_via_upstox()
                logger.info("SENSEX option chain fetched via Upstox (live)")
                return data
            except Exception as exc:
                logger.warning(
                    "Upstox SENSEX option chain fetch failed (%s: %s) — using VIX-based synthetic chain",
                    type(exc).__name__, exc,
                )
        else:
            logger.info("UPSTOX_ACCESS_TOKEN not set — using VIX-based synthetic SENSEX chain")

        return self._synthetic_option_chain()

    def _fetch_option_chain_via_upstox(self) -> OptionChainData:
        """Fetch weekly + monthly SENSEX option chains from Upstox's authenticated API."""
        from nifty_ai_agent.data.upstox_provider import UpstoxClient

        client = UpstoxClient(self._upstox_token)
        expiries_iso = client.get_expiries("SENSEX")
        if not expiries_iso:
            raise RuntimeError("Upstox returned no expiry dates for SENSEX")

        expiry_dates_nse_fmt = [_iso_to_nse_date(d) for d in expiries_iso]
        weekly_expiry, monthly_expiry = _identify_expiries(expiry_dates_nse_fmt)

        weekly_df = client.get_option_chain("SENSEX", _nse_date_to_iso(weekly_expiry))
        monthly_df = pd.DataFrame()
        if monthly_expiry and monthly_expiry != weekly_expiry:
            monthly_df = client.get_option_chain("SENSEX", _nse_date_to_iso(monthly_expiry))

        weekly_pcr = _compute_pcr(weekly_df)
        weekly_max_pain = _compute_max_pain(weekly_df) if not weekly_df.empty else 0.0

        return OptionChainData(
            symbol="SENSEX",
            expiry=weekly_expiry,
            monthly_expiry=monthly_expiry if not monthly_df.empty else "",
            timestamp=datetime.now(tz=timezone.utc),
            strikes=weekly_df,
            monthly_strikes=monthly_df,
            pcr=round(weekly_pcr, 3),
            max_pain=weekly_max_pain,
        )

    def get_historical_data(self, days: int = 60, interval: str = "5m") -> pd.DataFrame:
        """Fetch OHLCV data — Upstox first (if configured), then yfinance."""
        if self._upstox_token:
            try:
                from nifty_ai_agent.data.upstox_provider import UpstoxClient
                df = UpstoxClient(self._upstox_token).get_historical_ohlcv(
                    "SENSEX", days=days, interval=interval,
                )
                logger.info("SENSEX OHLCV fetched via Upstox (live): %d bars", len(df))
                return df
            except Exception as exc:
                logger.warning(
                    "Upstox SENSEX historical fetch failed (%s: %s) — falling back to yfinance",
                    type(exc).__name__, exc,
                )

        logger.info(
            "Fetching %d days of SENSEX OHLCV via yfinance (interval=%s)", days, interval
        )

        def _fetch() -> pd.DataFrame:
            if interval in ("1m", "2m", "5m", "15m", "30m", "60m", "90m"):
                df = yf.download(
                    self._symbol,
                    period=f"{min(days, 59)}d",
                    interval=interval,
                    progress=False,
                    auto_adjust=True,
                )
            else:
                end = datetime.now(tz=timezone.utc)
                start = end - timedelta(days=days + 10)
                df = yf.download(
                    self._symbol,
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    interval=interval,
                    progress=False,
                    auto_adjust=True,
                )
            if df.empty:
                raise ValueError("yfinance returned empty SENSEX historical data")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            df.columns = [c.lower() for c in df.columns]
            df.index.name = "datetime"
            return df

        return _retry(_fetch)

    # ── Private ─────────────────────────────────────────────────────

    def _synthetic_option_chain(self) -> OptionChainData:
        spot = 0.0
        try:
            spot = float(yf.Ticker(self._symbol).fast_info.last_price)
        except Exception:
            pass

        vix = 15.0
        try:
            vix = float(yf.Ticker(_VIX_SYMBOL).fast_info.last_price)
            logger.info("India VIX %.2f → SENSEX synthetic PCR", vix)
        except Exception:
            logger.warning("India VIX unavailable — using default VIX=15.0 for SENSEX")

        if vix < 12:
            pcr = 0.65
        elif vix < 15:
            pcr = 0.85
        elif vix < 18:
            pcr = 1.00
        elif vix < 22:
            pcr = 1.25
        else:
            pcr = 1.55

        atm = int(round(spot / _STRIKE_STEP) * _STRIKE_STEP) if spot else 80000
        expiry = _next_thursday()

        logger.info(
            "SENSEX synthetic chain: VIX=%.2f PCR=%.2f ATM=%d expiry=%s",
            vix, pcr, atm, expiry,
        )
        return OptionChainData(
            symbol="SENSEX",
            expiry=expiry,
            monthly_expiry="",
            timestamp=datetime.now(tz=timezone.utc),
            strikes=pd.DataFrame(),
            monthly_strikes=pd.DataFrame(),
            pcr=round(pcr, 2),
            max_pain=float(atm),
            iv_proxy=round(vix / 100, 4),
        )
