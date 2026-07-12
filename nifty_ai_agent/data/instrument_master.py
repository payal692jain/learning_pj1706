"""Upstox NSE instrument master — resolves tradable contracts by symbol.

Upstox identifies every instrument by an opaque key ("NSE_EQ|INE040A01034",
"NSE_FO|99849"), not by ticker. Index option chains can be fetched without one
because the three index keys are known constants, but STOCK options cannot: their
key is derived from the company's ISIN, and ISINs are not guessable. Hardcoding
them is how you end up quoting the wrong company — Kotak Bank's ISIN, for one, is
INE237A01036 and not the INE237A01028 that several stale listings still show.

So the master is downloaded from Upstox's public (unauthenticated) feed and cached
on disk for the day. ~2 MB gzipped, ~86k rows, fetched at most once per session.
"""

import gzip
import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_MASTER_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
_TIMEOUT = 60
_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
_CACHE_FILE = _CACHE_DIR / "upstox_nse_instruments.json"


@dataclass(frozen=True)
class OptionContract:
    instrument_key: str
    trading_symbol: str
    asset_symbol: str          # underlying, e.g. "HDFCBANK"
    strike: float
    opt_type: str              # CE / PE
    expiry: date
    lot_size: int

    @property
    def days_to_expiry(self) -> int:
        return max(0, (self.expiry - date.today()).days)


class InstrumentMaster:
    """Symbol → instrument-key lookups, backed by a day-cached copy of the NSE master."""

    def __init__(self, cache_file: Path = _CACHE_FILE) -> None:
        self._cache_file = cache_file
        self._rows: list[dict] | None = None

    # ── Loading ──────────────────────────────────────────────────────────────

    def _load(self) -> list[dict]:
        if self._rows is not None:
            return self._rows

        if self._is_cache_fresh():
            try:
                self._rows = json.loads(self._cache_file.read_text(encoding="utf-8"))
                logger.info("Instrument master: %d rows from cache", len(self._rows))
                return self._rows
            except Exception as exc:
                logger.warning("Instrument master cache unreadable (%s) — refetching", exc)

        self._rows = self._download()
        self._write_cache(self._rows)
        return self._rows

    def _is_cache_fresh(self) -> bool:
        """Fresh means written today. Contracts are added/expired daily, so a stale
        master silently quotes strikes that no longer trade."""
        if not self._cache_file.exists():
            return False
        modified = datetime.fromtimestamp(self._cache_file.stat().st_mtime, tz=timezone.utc)
        return modified.date() == datetime.now(tz=timezone.utc).date()

    def _download(self) -> list[dict]:
        logger.info("Downloading Upstox NSE instrument master…")
        resp = requests.get(_MASTER_URL, timeout=_TIMEOUT)
        resp.raise_for_status()
        rows = json.loads(gzip.decompress(resp.content))
        logger.info("Instrument master: %d rows downloaded", len(rows))
        return rows

    def _write_cache(self, rows: list[dict]) -> None:
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            self._cache_file.write_text(json.dumps(rows), encoding="utf-8")
        except Exception as exc:
            # A cache we cannot write is a performance problem, not a correctness one.
            logger.warning("Could not cache instrument master: %s", exc)

    # ── Lookups ──────────────────────────────────────────────────────────────

    def equity_key(self, symbol: str) -> str | None:
        """Instrument key for a cash-market stock, e.g. 'HDFCBANK' → 'NSE_EQ|INE040A01034'."""
        for row in self._load():
            if row.get("segment") == "NSE_EQ" and row.get("trading_symbol") == symbol:
                return row.get("instrument_key")
        logger.warning("Instrument master: no NSE_EQ row for %s", symbol)
        return None

    def option_contracts(self, asset_symbol: str) -> list[OptionContract]:
        """Every live CE/PE contract on *asset_symbol*, expiries in the past excluded."""
        today = date.today()
        contracts: list[OptionContract] = []

        for row in self._load():
            if (
                row.get("segment") != "NSE_FO"
                or row.get("asset_symbol") != asset_symbol
                or row.get("instrument_type") not in ("CE", "PE")
            ):
                continue
            expiry = _epoch_ms_to_date(row.get("expiry"))
            if expiry is None or expiry < today:
                continue
            contracts.append(
                OptionContract(
                    instrument_key=row["instrument_key"],
                    trading_symbol=row.get("trading_symbol", ""),
                    asset_symbol=asset_symbol,
                    strike=float(row.get("strike_price", 0)),
                    opt_type=row["instrument_type"],
                    expiry=expiry,
                    lot_size=int(row.get("lot_size", 0)),
                )
            )
        return contracts

    def nearest_expiry(self, asset_symbol: str) -> date | None:
        """The soonest expiry that is not today.

        An option expiring in hours has no time value left to trade, so a
        suggestion to buy one is a suggestion to buy a lottery ticket.
        """
        today = date.today()
        expiries = {c.expiry for c in self.option_contracts(asset_symbol) if c.expiry > today}
        return min(expiries) if expiries else None

    def atm_contract(
        self, asset_symbol: str, spot: float, opt_type: str, expiry: date | None = None,
    ) -> OptionContract | None:
        """The contract whose strike sits closest to *spot* for the given expiry."""
        expiry = expiry or self.nearest_expiry(asset_symbol)
        if expiry is None:
            return None

        candidates = [
            c for c in self.option_contracts(asset_symbol)
            if c.opt_type == opt_type and c.expiry == expiry and c.strike > 0
        ]
        if not candidates:
            logger.warning("No %s contracts for %s at %s", opt_type, asset_symbol, expiry)
            return None
        return min(candidates, key=lambda c: abs(c.strike - spot))


def _epoch_ms_to_date(value) -> date | None:
    """Upstox stores expiry as epoch milliseconds."""
    if not value:
        return None
    try:
        return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc).date()
    except (TypeError, ValueError, OSError):
        return None


_shared: InstrumentMaster | None = None


def get_instrument_master() -> InstrumentMaster:
    """Process-wide master — the 2 MB download is not worth repeating per index."""
    global _shared
    if _shared is None:
        _shared = InstrumentMaster()
    return _shared
