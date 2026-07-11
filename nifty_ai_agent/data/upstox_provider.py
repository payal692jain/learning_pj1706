"""Upstox v2 API client — live option chain, quotes, and historical candles.

Unlike NSE's website, Upstox's API is an authenticated REST API — no bot
detection to route around. It requires an access token that expires nightly
(~3:30 AM IST); see scripts/upstox_login.py for the daily refresh flow.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Any
from urllib.parse import quote as urlquote

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_UPSTOX_BASE = "https://api.upstox.com/v2"
_TIMEOUT = 15

# Upstox instrument keys for index option chains — confirmed live.
_INSTRUMENT_KEYS = {
    "NIFTY": "NSE_INDEX|Nifty 50",
    "SENSEX": "BSE_INDEX|SENSEX",
    "BANKNIFTY": "NSE_INDEX|Nifty Bank",
}

# Upstox's historical-candle API only accepts these intraday granularities —
# anything finer (e.g. the 5-minute bars the indicator engine uses) is built
# by fetching 1-minute candles and resampling locally.
_RESAMPLE_FREQ = {
    "1m": "1min", "5m": "5min", "15m": "15min",
    "30m": "30min", "60m": "60min", "90m": "90min",
}


def drop_expiring_today(expiries_iso: list[str]) -> list[str]:
    """Drop any expiry dated today from a sorted 'YYYY-MM-DD' list.

    An option expiring within hours has no meaningful time left to trade, so
    the "weekly" pick should roll forward to the next available expiry
    instead of the one about to go worthless. Falls back to the original
    list if that would leave nothing (e.g. Upstox only returned today).
    """
    today_iso = date.today().isoformat()
    filtered = [d for d in expiries_iso if d != today_iso]
    return filtered if filtered else expiries_iso


class UpstoxAuthError(Exception):
    """Raised when the access token is missing, expired, or rejected."""


class UpstoxClient:
    """Fetches live option chain, quote, and historical candle data from Upstox."""

    def __init__(self, access_token: str) -> None:
        self._access_token = access_token

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Accept": "application/json",
        }

    def _require_instrument_key(self, index_name: str) -> str:
        if not self._access_token:
            raise UpstoxAuthError("UPSTOX_ACCESS_TOKEN is not set")
        instrument_key = _INSTRUMENT_KEYS.get(index_name)
        if not instrument_key:
            raise ValueError(f"No Upstox instrument key configured for {index_name!r}")
        return instrument_key

    # ── Option chain ─────────────────────────────────────────────────────

    def get_expiries(self, index_name: str) -> list[str]:
        """Return available expiry dates ('YYYY-MM-DD', sorted ascending) for *index_name*.

        Drops an expiry dated today — see drop_expiring_today().
        """
        instrument_key = self._require_instrument_key(index_name)

        resp = requests.get(
            f"{_UPSTOX_BASE}/option/contract",
            params={"instrument_key": instrument_key},
            headers=self._headers(),
            timeout=_TIMEOUT,
        )
        self._raise_for_auth_error(resp)
        resp.raise_for_status()
        payload = resp.json()

        expiries = sorted({
            item["expiry"] for item in payload.get("data", []) if item.get("expiry")
        })
        return drop_expiring_today(expiries)

    def get_lot_size(self, index_name: str) -> int:
        """Return the option contract lot size for *index_name* from live contract data.

        Lot sizes are revised by SEBI/exchanges periodically, so reading them from
        the live contract feed beats hardcoding.
        """
        instrument_key = self._require_instrument_key(index_name)

        resp = requests.get(
            f"{_UPSTOX_BASE}/option/contract",
            params={"instrument_key": instrument_key},
            headers=self._headers(),
            timeout=_TIMEOUT,
        )
        self._raise_for_auth_error(resp)
        resp.raise_for_status()
        for item in resp.json().get("data", []):
            if item.get("lot_size"):
                return int(item["lot_size"])
        raise RuntimeError(f"No lot_size in Upstox contract data for {index_name}")

    def get_option_chain(self, index_name: str, expiry_date: str) -> pd.DataFrame:
        """Return a strikes DataFrame for *index_name* at *expiry_date* ('YYYY-MM-DD').

        Columns: strike, ce_oi, pe_oi, ce_ltp, pe_ltp, ce_iv, pe_iv — matching the
        shape NSEDataProvider already produces, so downstream analysis is unchanged.
        """
        instrument_key = self._require_instrument_key(index_name)

        resp = requests.get(
            f"{_UPSTOX_BASE}/option/chain",
            params={"instrument_key": instrument_key, "expiry_date": expiry_date},
            headers=self._headers(),
            timeout=_TIMEOUT,
        )
        self._raise_for_auth_error(resp)
        resp.raise_for_status()
        payload = resp.json()

        if payload.get("status") != "success":
            raise RuntimeError(f"Upstox option chain error: {payload}")

        rows: list[dict[str, Any]] = []
        for entry in payload.get("data", []):
            call = entry.get("call_options") or {}
            put = entry.get("put_options") or {}
            call_md = call.get("market_data") or {}
            put_md = put.get("market_data") or {}
            call_greeks = call.get("option_greeks") or {}
            put_greeks = put.get("option_greeks") or {}
            rows.append({
                "strike": entry.get("strike_price", 0),
                "ce_oi": call_md.get("oi", 0),
                "pe_oi": put_md.get("oi", 0),
                "ce_ltp": call_md.get("ltp", 0),
                "pe_ltp": put_md.get("ltp", 0),
                "ce_iv": call_greeks.get("iv", 0),
                "pe_iv": put_greeks.get("iv", 0),
            })

        logger.info(
            "Upstox option chain: %s expiry=%s (%d strikes)",
            index_name, expiry_date, len(rows),
        )
        return pd.DataFrame(rows)

    # ── Live quote (spot price) ─────────────────────────────────────────

    def get_quote(self, index_name: str) -> dict[str, float]:
        """Return {'price', 'open', 'high', 'low', 'volume'} for *index_name*."""
        instrument_key = self._require_instrument_key(index_name)

        resp = requests.get(
            f"{_UPSTOX_BASE}/market-quote/quotes",
            params={"instrument_key": instrument_key},
            headers=self._headers(),
            timeout=_TIMEOUT,
        )
        self._raise_for_auth_error(resp)
        resp.raise_for_status()
        payload = resp.json()

        data = payload.get("data") or {}
        if not data:
            raise RuntimeError(f"Upstox quote error: {payload}")
        # Upstox keys the response by "EXCHANGE:Symbol" (colon), not the
        # "EXCHANGE|Symbol" (pipe) instrument_key we sent — take the one value.
        quote_data = next(iter(data.values()))
        ohlc = quote_data.get("ohlc") or {}
        return {
            "price": float(quote_data.get("last_price", 0.0)),
            "open": float(ohlc.get("open", 0.0)),
            "high": float(ohlc.get("high", 0.0)),
            "low": float(ohlc.get("low", 0.0)),
            "volume": float(quote_data.get("volume") or 0),
        }

    # ── Historical OHLCV (for indicators) ───────────────────────────────

    def get_historical_ohlcv(self, index_name: str, days: int, interval: str = "5m") -> pd.DataFrame:
        """Return an OHLCV DataFrame for *index_name* covering the last *days* days.

        Upstox has no native 5-minute granularity — 1-minute candles (historical
        + today's intraday) are fetched and resampled locally to match *interval*.
        Daily bars ("1d") use Upstox's native "day" interval directly instead.
        """
        instrument_key = self._require_instrument_key(index_name)
        today = date.today()
        from_date = today - timedelta(days=days)

        if interval in ("1d", "day"):
            candles = self._fetch_candles(instrument_key, "day", from_date, today)
            df = self._candles_to_df(candles)
        else:
            freq = _RESAMPLE_FREQ.get(interval, "5min")
            historical = self._fetch_candles(
                instrument_key, "1minute", from_date, today - timedelta(days=1)
            )
            intraday = self._fetch_intraday_candles(instrument_key, "1minute")
            df = self._candles_to_df(historical + intraday)
            if not df.empty:
                df = df.resample(freq).agg({
                    "open": "first", "high": "max", "low": "min",
                    "close": "last", "volume": "sum",
                }).dropna(subset=["open"])

        if df.empty:
            raise ValueError(f"Upstox returned no historical candles for {index_name}")

        logger.info(
            "Upstox historical OHLCV: %s interval=%s (%d bars)",
            index_name, interval, len(df),
        )
        return df

    def _fetch_candles(
        self, instrument_key: str, interval: str, from_date: date, to_date: date,
    ) -> list[list]:
        if from_date > to_date:
            return []
        url = (
            f"{_UPSTOX_BASE}/historical-candle/{urlquote(instrument_key, safe='')}/"
            f"{interval}/{to_date.isoformat()}/{from_date.isoformat()}"
        )
        resp = requests.get(url, headers=self._headers(), timeout=_TIMEOUT)
        self._raise_for_auth_error(resp)
        resp.raise_for_status()
        return resp.json().get("data", {}).get("candles", [])

    def _fetch_intraday_candles(self, instrument_key: str, interval: str) -> list[list]:
        url = f"{_UPSTOX_BASE}/historical-candle/intraday/{urlquote(instrument_key, safe='')}/{interval}"
        resp = requests.get(url, headers=self._headers(), timeout=_TIMEOUT)
        self._raise_for_auth_error(resp)
        resp.raise_for_status()
        return resp.json().get("data", {}).get("candles", [])

    @staticmethod
    def _candles_to_df(candles: list[list]) -> pd.DataFrame:
        """Convert Upstox's [timestamp, open, high, low, close, volume, oi] rows
        into a DataFrame sorted ascending by time, with tz stripped (IST wall-clock
        time is kept as naive, matching the rest of the pipeline)."""
        if not candles:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        rows = []
        for ts, o, h, low_, c, vol, *_rest in candles:
            rows.append({
                "datetime": datetime.fromisoformat(ts).replace(tzinfo=None),
                "open": o, "high": h, "low": low_, "close": c, "volume": vol,
            })
        return pd.DataFrame(rows).set_index("datetime").sort_index()

    @staticmethod
    def _raise_for_auth_error(resp: requests.Response) -> None:
        if resp.status_code in (401, 403):
            raise UpstoxAuthError(
                f"Upstox rejected the access token (HTTP {resp.status_code}) — "
                "it has likely expired; re-run scripts/upstox_login.py"
            )
