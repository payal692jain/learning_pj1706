"""Upstox v2 API client for live option chain data.

Unlike NSE's website, Upstox's API is an authenticated REST API — no bot
detection to route around. It requires an access token that expires nightly
(~3:30 AM IST); see scripts/upstox_login.py for the daily refresh flow.
"""

import logging
from datetime import date
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_UPSTOX_BASE = "https://api.upstox.com/v2"
_TIMEOUT = 15

# Upstox instrument keys for index option chains — confirmed live.
_INSTRUMENT_KEYS = {
    "NIFTY": "NSE_INDEX|Nifty 50",
    "SENSEX": "BSE_INDEX|SENSEX",
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


class UpstoxOptionChainClient:
    """Fetches live option chain data (strikes, OI, LTP, IV) from Upstox."""

    def __init__(self, access_token: str) -> None:
        self._access_token = access_token

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Accept": "application/json",
        }

    def get_expiries(self, index_name: str) -> list[str]:
        """Return available expiry dates ('YYYY-MM-DD', sorted ascending) for *index_name*.

        Drops an expiry dated today — see drop_expiring_today().
        """
        if not self._access_token:
            raise UpstoxAuthError("UPSTOX_ACCESS_TOKEN is not set")

        instrument_key = _INSTRUMENT_KEYS.get(index_name)
        if not instrument_key:
            raise ValueError(f"No Upstox instrument key configured for {index_name!r}")

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

    def get_option_chain(self, index_name: str, expiry_date: str) -> pd.DataFrame:
        """Return a strikes DataFrame for *index_name* at *expiry_date* ('YYYY-MM-DD').

        Columns: strike, ce_oi, pe_oi, ce_ltp, pe_ltp, ce_iv, pe_iv — matching the
        shape NSEDataProvider already produces, so downstream analysis is unchanged.
        """
        if not self._access_token:
            raise UpstoxAuthError("UPSTOX_ACCESS_TOKEN is not set")

        instrument_key = _INSTRUMENT_KEYS.get(index_name)
        if not instrument_key:
            raise ValueError(f"No Upstox instrument key configured for {index_name!r}")

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

    @staticmethod
    def _raise_for_auth_error(resp: requests.Response) -> None:
        if resp.status_code in (401, 403):
            raise UpstoxAuthError(
                f"Upstox rejected the access token (HTTP {resp.status_code}) — "
                "it has likely expired; re-run scripts/upstox_login.py"
            )
