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

import difflib
import gzip
import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import requests

from nifty_ai_agent.config import get_settings

logger = logging.getLogger(__name__)

_MASTER_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
_TIMEOUT = 60
_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
_CACHE_FILE = _CACHE_DIR / "upstox_nse_instruments.json"


# Upstox segments this master can serve options from.
SEGMENT_EQUITY_FO = "NSE_FO"    # index + single-stock options
SEGMENT_CURRENCY_FO = "NCD_FO"  # USDINR / EURINR / GBPINR / JPYINR options

# Tickers retired by a corporate action, mapped to the successor that still
# trades. Kept explicit rather than guessed: a demerger splits one listing into
# several, and only one of them is usually the F&O name, so the choice has to be
# made deliberately. Tata Motors demerged into TMPV (passenger vehicles, which
# carries the F&O book) and TMCV (commercial vehicles, cash only).
RENAMED_SYMBOLS: dict[str, str] = {
    "TATAMOTORS": "TMPV",
}


@dataclass(frozen=True)
class EquityMatch:
    """A resolved cash-market listing, and what was originally asked for."""
    instrument_key: str
    trading_symbol: str
    name: str
    requested: str

    @property
    def is_exact(self) -> bool:
        return self.trading_symbol == self.requested


@dataclass(frozen=True)
class OptionContract:
    instrument_key: str
    trading_symbol: str
    asset_symbol: str          # underlying, e.g. "HDFCBANK"
    strike: float
    opt_type: str              # CE / PE
    expiry: date
    lot_size: int
    # Units per lot_size unit. 1.0 for equity/index F&O, but currency options carry
    # lot_size=1 with qty_multiplier=1000 — quoting cost off lot_size alone would
    # understate a USDINR contract by 1000x.
    qty_multiplier: float = 1.0

    @property
    def days_to_expiry(self) -> int:
        return max(0, (self.expiry - date.today()).days)

    @property
    def contract_size(self) -> int:
        """Units actually bought per contract — what a premium must be multiplied by."""
        return int(round(self.lot_size * self.qty_multiplier))


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
        """Instrument key for a cash-market stock, e.g. 'HDFCBANK' → 'NSE_EQ|INE040A01034'.

        Resolves known renames (see RENAMED_SYMBOLS) so a ticker retired by a
        demerger still finds its successor instead of silently returning nothing.
        """
        match = self.resolve_equity(symbol)
        return match.instrument_key if match else None

    def resolve_equity(self, symbol: str) -> EquityMatch | None:
        """Resolve *symbol* to a live NSE_EQ instrument.

        Exact trading symbol first, then the rename table. Deliberately does NOT
        guess by similarity: substituting a merely similar company into a scan is
        how the agent would end up quoting a contract on the wrong business. Use
        suggest_symbols() when a human is there to choose.
        """
        rows = [r for r in self._load() if r.get("segment") == "NSE_EQ"]
        wanted = symbol.strip().upper()

        for row in rows:
            if row.get("trading_symbol") == wanted:
                return EquityMatch(
                    instrument_key=row["instrument_key"],
                    trading_symbol=wanted,
                    name=row.get("name", ""),
                    requested=wanted,
                )

        successor = RENAMED_SYMBOLS.get(wanted)
        if successor:
            for row in rows:
                if row.get("trading_symbol") == successor:
                    logger.info(
                        "Instrument master: %s is retired — using %s (%s)",
                        wanted, successor, row.get("name", ""),
                    )
                    return EquityMatch(
                        instrument_key=row["instrument_key"],
                        trading_symbol=successor,
                        name=row.get("name", ""),
                        requested=wanted,
                    )

        logger.warning("Instrument master: no NSE_EQ row for %s", wanted)
        return None

    def suggest_symbols(self, query: str, limit: int = 5) -> list[tuple[str, str]]:
        """Return [(trading_symbol, company name)] closest to *query*.

        Matches on ticker similarity and on company-name substring, so both a typo
        ("RELAINCE") and a plain-English guess ("tata motors") land somewhere useful.
        """
        wanted = query.strip().upper()
        rows = [r for r in self._load() if r.get("segment") == "NSE_EQ"]
        by_symbol = {r["trading_symbol"]: r.get("name", "") for r in rows if r.get("trading_symbol")}

        ranked: list[tuple[str, str]] = []
        for symbol, name in by_symbol.items():
            if wanted in symbol or (name and wanted in name.upper()):
                ranked.append((symbol, name))
        ranked.sort(key=lambda pair: len(pair[0]))

        if len(ranked) < limit:
            close = difflib.get_close_matches(wanted, list(by_symbol), n=limit, cutoff=0.6)
            for symbol in close:
                if all(symbol != s for s, _ in ranked):
                    ranked.append((symbol, by_symbol[symbol]))
        return ranked[:limit]

    def option_contracts(
        self, asset_symbol: str, segment: str = SEGMENT_EQUITY_FO
    ) -> list[OptionContract]:
        """Every live CE/PE contract on *asset_symbol*, expiries in the past excluded.

        *segment* selects the derivatives book: NSE_FO for index/stock options,
        NCD_FO for currency. The row shape is identical across both.
        """
        today = date.today()
        contracts: list[OptionContract] = []

        for row in self._load():
            if (
                row.get("segment") != segment
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
                    qty_multiplier=float(row.get("qty_multiplier") or 1.0),
                )
            )
        return contracts

    def nearest_expiry(
        self,
        asset_symbol: str,
        allow_expiry_day: bool | None = None,
        segment: str = SEGMENT_EQUITY_FO,
    ) -> date | None:
        """The soonest expiry that is not today.

        An option expiring in hours has no time value left to trade, so a
        suggestion to buy one is a suggestion to buy a lottery ticket.

        Set ALLOW_EXPIRY_DAY_OPTIONS=true to include today's expiry anyway;
        *allow_expiry_day* overrides that setting when passed explicitly.
        """
        if allow_expiry_day is None:
            allow_expiry_day = get_settings().allow_expiry_day_options

        today = date.today()
        expiries = {
            c.expiry for c in self.option_contracts(asset_symbol, segment)
            if (c.expiry >= today if allow_expiry_day else c.expiry > today)
        }
        return min(expiries) if expiries else None

    def atm_contract(
        self,
        asset_symbol: str,
        spot: float,
        opt_type: str,
        expiry: date | None = None,
        allow_expiry_day: bool | None = None,
        segment: str = SEGMENT_EQUITY_FO,
    ) -> OptionContract | None:
        """The contract whose strike sits closest to *spot* for the given expiry."""
        expiry = expiry or self.nearest_expiry(
            asset_symbol, allow_expiry_day=allow_expiry_day, segment=segment,
        )
        if expiry is None:
            return None

        candidates = [
            c for c in self.option_contracts(asset_symbol, segment)
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
