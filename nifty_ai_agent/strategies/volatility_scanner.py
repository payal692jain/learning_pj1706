"""Volatile-stock straddle scanner — long-volatility CE+PE ideas.

Where the single-stock scanner takes a directional view (one CE *or* PE on a
consensus signal), this scanner is direction-agnostic: it ranks the stock
universe by realised volatility (ATR as a % of spot) and, for the most volatile
names, suggests buying the ATM straddle — the call *and* the put on the same
strike/expiry. A straddle profits from a large move in either direction, so the
most volatile stocks are the ones with the best chance of covering both legs'
premium. The trade-off, spelled out in the report, is that a straddle risks the
full combined premium if the stock sits still (theta bleed).

The universe and its OHLCV come from the same fetch the directional scanner uses;
only the selection (volatility, not signal) and the idea (two legs, not one)
differ.
"""

import logging
from dataclasses import dataclass

import pandas as pd

from nifty_ai_agent.data.instrument_master import InstrumentMaster, OptionContract
from nifty_ai_agent.strategies.option_analyser import compute_atm_theoretical_prices
from nifty_ai_agent.strategies.pipeline import compute_all_indicators

logger = logging.getLogger(__name__)

# yfinance tickers carry a ".NS" suffix; the instrument master keys stocks by the
# bare NSE trading symbol ("RELIANCE", "M&M").
_NS_SUFFIX = ".NS"


@dataclass
class StraddleIdea:
    """One ATM long-straddle suggestion on a volatile stock."""
    symbol: str            # bare NSE symbol, e.g. "RELIANCE"
    spot: float
    atr_pct: float         # ATR as % of spot — the volatility rank
    strike: float
    expiry: str            # DD-Mon-YYYY (stock options are monthly)
    lot_size: int
    ce_premium: float
    pe_premium: float
    ce_is_live: bool       # False when the CE premium is a Black-Scholes estimate
    pe_is_live: bool
    total_premium: float   # per share: ce_premium + pe_premium (the debit paid)
    breakeven_low: float   # strike - total_premium
    breakeven_high: float  # strike + total_premium
    breakeven_move_pct: float  # move from spot to a breakeven, as % — the hurdle
    lot_cost: float        # total_premium * lot_size — cash to open one straddle


@dataclass
class VolatilityScanResult:
    ideas: list[StraddleIdea]  # top-N, most volatile first
    scanned: int               # symbols with usable spot data
    ranked: int                # symbols that produced a volatility score
    errors: int                # symbols that raised while being processed


def _premium_lookup(instrument_keys: list[str]) -> dict[str, float]:
    """Default no-op premium source — every scan without a live client estimates."""
    return {}


def scan_volatile_straddles(
    histories: dict[str, pd.DataFrame],
    spots: dict[str, float],
    master: InstrumentMaster,
    *,
    premium_lookup=_premium_lookup,
    default_iv: float = 0.30,
    top_n: int = 5,
    allow_expiry_day: bool | None = None,
) -> VolatilityScanResult:
    """Rank *histories* by ATR% and return ATM straddle ideas on the most volatile.

    Args:
        histories: symbol → OHLCV DataFrame (yfinance ticker keys, e.g. 'RELIANCE.NS').
        spots: symbol → current spot price.
        master: instrument master for resolving the ATM CE/PE contracts + lot size.
        premium_lookup: instrument_key list → {key: ltp}. Defaults to estimate-only.
        default_iv: annualised IV for the Black-Scholes fallback when no live premium.
        top_n: how many straddle ideas to return.
        allow_expiry_day: passed through to the contract picker.
    """
    scored: list[tuple[str, float, float]] = []  # (symbol, spot, atr_pct)
    scanned = errors = 0

    for symbol, hist in histories.items():
        spot = spots.get(symbol, 0.0)
        if spot <= 0:
            continue
        scanned += 1
        try:
            atr_pct = _atr_pct(hist, spot)
        except Exception as exc:
            errors += 1
            logger.warning("Volatility scan: %s scoring failed — %s", symbol, exc)
            continue
        if atr_pct is not None:
            scored.append((symbol, spot, atr_pct))

    # Most volatile first — highest expected move gives a straddle the best shot.
    scored.sort(key=lambda t: t[2], reverse=True)

    ideas: list[StraddleIdea] = []
    for symbol, spot, atr_pct in scored:
        if len(ideas) >= top_n:
            break
        try:
            idea = _build_straddle(
                symbol, spot, atr_pct, master,
                premium_lookup=premium_lookup, default_iv=default_iv,
                allow_expiry_day=allow_expiry_day,
            )
        except Exception as exc:
            errors += 1
            logger.warning("Volatility scan: %s straddle build failed — %s", symbol, exc)
            continue
        if idea is not None:
            ideas.append(idea)

    logger.info(
        "Volatility scan: %d scanned, %d ranked, %d errors → top %d straddles",
        scanned, len(scored), errors, len(ideas),
    )
    return VolatilityScanResult(
        ideas=ideas, scanned=scanned, ranked=len(scored), errors=errors,
    )


def _atr_pct(hist: pd.DataFrame, spot: float) -> float | None:
    """ATR of the latest bar as a % of spot — the volatility rank. None if no ATR."""
    df = compute_all_indicators(hist)
    usable = df.dropna(subset=["atr"])
    if usable.empty:
        return None
    atr = float(usable["atr"].iloc[-1])
    if atr <= 0:
        return None
    return round(atr / spot * 100, 2)


def _build_straddle(
    symbol: str,
    spot: float,
    atr_pct: float,
    master: InstrumentMaster,
    *,
    premium_lookup,
    default_iv: float,
    allow_expiry_day: bool | None,
) -> StraddleIdea | None:
    """Resolve the ATM CE+PE, price both legs, and frame the breakevens. None if no
    tradable contract or no usable premium."""
    asset = symbol[: -len(_NS_SUFFIX)] if symbol.endswith(_NS_SUFFIX) else symbol

    ce = master.atm_contract(asset, spot, "CE", allow_expiry_day=allow_expiry_day)
    if ce is None:
        logger.info("Volatility scan: %s — no CE contract, skipping", asset)
        return None
    # Pin the put to the call's exact strike/expiry so both legs form one straddle
    # (passing the strike as the spot forces the nearest match to be that strike).
    pe = master.atm_contract(
        asset, ce.strike, "PE", expiry=ce.expiry, allow_expiry_day=allow_expiry_day,
    )
    if pe is None:
        logger.info("Volatility scan: %s — no matching PE contract, skipping", asset)
        return None

    strike = ce.strike
    expiry_str = ce.expiry.strftime("%d-%b-%Y")
    ce_prem, ce_live, pe_prem, pe_live = _straddle_premiums(
        ce, pe, spot, int(round(strike)), expiry_str, premium_lookup, default_iv,
    )
    if ce_prem <= 0 or pe_prem <= 0:
        logger.info("Volatility scan: %s — no usable straddle premium, skipping", asset)
        return None

    total = round(ce_prem + pe_prem, 2)
    return StraddleIdea(
        symbol=asset,
        spot=round(spot, 2),
        atr_pct=atr_pct,
        strike=strike,
        expiry=expiry_str,
        lot_size=ce.lot_size,
        ce_premium=ce_prem,
        pe_premium=pe_prem,
        ce_is_live=ce_live,
        pe_is_live=pe_live,
        total_premium=total,
        breakeven_low=round(strike - total, 2),
        breakeven_high=round(strike + total, 2),
        breakeven_move_pct=round(total / spot * 100, 2) if spot else 0.0,
        lot_cost=round(total * ce.lot_size, 2),
    )


def _straddle_premiums(
    ce: OptionContract,
    pe: OptionContract,
    spot: float,
    strike: int,
    expiry_str: str,
    premium_lookup,
    default_iv: float,
) -> tuple[float, bool, float, bool]:
    """Return *(ce_premium, ce_is_live, pe_premium, pe_is_live)*.

    Each leg is a live LTP when available, otherwise its Black-Scholes estimate —
    resolved independently, so one live leg and one estimated leg is possible (and
    flagged as such rather than silently mixed)."""
    live: dict[str, float] = {}
    try:
        live = premium_lookup([ce.instrument_key, pe.instrument_key]) or {}
    except Exception as exc:
        logger.debug("Volatility scan: LTP lookup failed for %s (%s)", ce.asset_symbol, exc)

    bs_ce, bs_pe = compute_atm_theoretical_prices(spot, strike, expiry_str, default_iv)

    ce_ltp = live.get(ce.instrument_key, 0.0)
    pe_ltp = live.get(pe.instrument_key, 0.0)
    ce_prem, ce_live = (round(float(ce_ltp), 2), True) if ce_ltp and ce_ltp > 0 else (bs_ce, False)
    pe_prem, pe_live = (round(float(pe_ltp), 2), True) if pe_ltp and pe_ltp > 0 else (bs_pe, False)
    return ce_prem, ce_live, pe_prem, pe_live
