"""Single-stock option scanner — monthly CE/PE ideas across a stock universe.

Runs the same indicator + strategy + consensus stack the index loop uses, but
over individual F&O stocks instead of the indices. For each stock with an
actionable consensus it resolves the monthly ATM contract (stock options in
India are monthly-only), prices its premium, and applies the risk engine on the
underlying. The best ideas by confidence are returned for a single Pushover
digest.

Breadth and global-context adjustments are deliberately NOT applied here: they
describe the index tape as a whole, not one constituent, so folding them into a
single-stock read would be borrowing conviction the stock hasn't earned.
"""

import logging
from dataclasses import dataclass
from datetime import time as dt_time

import pandas as pd

from nifty_ai_agent.data.instrument_master import InstrumentMaster, OptionContract
from nifty_ai_agent.risk.calculator import RiskCalculator
from nifty_ai_agent.strategies.base import BaseStrategy, SignalType
from nifty_ai_agent.strategies.consensus import build_consensus
from nifty_ai_agent.strategies.option_analyser import compute_atm_theoretical_prices
from nifty_ai_agent.strategies.pipeline import DEFAULT_STRATEGIES, compute_all_indicators

logger = logging.getLogger(__name__)

# yfinance tickers carry a ".NS" suffix; Upstox keys stocks by the bare NSE
# trading symbol ("RELIANCE", "M&M", "BAJAJ-AUTO").
_NS_SUFFIX = ".NS"

_REQUIRED_COLS = ["ema_20", "ema_50", "rsi", "atr"]


@dataclass
class StockIdea:
    """One actionable single-stock option suggestion."""
    symbol: str            # bare NSE symbol, e.g. "RELIANCE"
    signal: str            # BUY_CE / BUY_PE
    confidence: int
    conviction: str        # STRONG / MODERATE / WEAK
    opt_type: str          # CE / PE
    strike: float
    expiry: str            # DD-Mon-YYYY
    lot_size: int
    entry_premium: float
    is_live: bool          # False when the premium is a Black-Scholes estimate
    spot: float
    target: float          # target on the UNDERLYING
    stop_loss: float       # stop-loss on the UNDERLYING
    rr: float


@dataclass
class ScanResult:
    ideas: list[StockIdea]  # top-N, highest confidence first
    scanned: int            # symbols with usable data
    actionable: int         # symbols that produced a (pre-ranking) idea
    errors: int             # symbols that raised while being processed


def _premium_lookup(instrument_keys: list[str]) -> dict[str, float]:
    """Default no-op premium source — every scan without a live client estimates."""
    return {}


def scan_stocks(
    histories: dict[str, pd.DataFrame],
    spots: dict[str, float],
    master: InstrumentMaster,
    risk_calculator: RiskCalculator,
    *,
    now: dt_time,
    premium_lookup=_premium_lookup,
    default_iv: float = 0.30,
    top_n: int = 5,
    strategies: list[BaseStrategy] | None = None,
    allow_expiry_day: bool | None = None,
    intraday: bool = False,
) -> ScanResult:
    """Scan every symbol in *histories* and return the best monthly option ideas.

    Args:
        histories: symbol → OHLCV DataFrame (yfinance ticker keys, e.g. 'RELIANCE.NS').
        spots: symbol → current spot price.
        master: instrument master for resolving the ATM contract + lot size.
        risk_calculator: sizes SL/target on the underlying.
        now: current IST wall-clock time — the consensus engine weights strategies
            by time of day and refuses new entries past its cutoff.
        premium_lookup: instrument_key list → {key: ltp}. Defaults to estimate-only.
        default_iv: annualised IV for the Black-Scholes fallback when no live premium.
        top_n: how many ideas to return.
        strategies: strategy book to run (defaults to the shared DEFAULT_STRATEGIES).
        allow_expiry_day: passed through to the contract picker.
        intraday: apply the consensus engine's intraday-only rules (time-of-day
            weighting and the late-session entry cutoff). Defaults False, because a
            monthly stock option is not disqualified by an afternoon entry.
    """
    strategies = strategies or DEFAULT_STRATEGIES
    ideas: list[StockIdea] = []
    scanned = errors = 0

    for symbol, hist in histories.items():
        spot = spots.get(symbol, 0.0)
        if spot <= 0:
            continue
        scanned += 1
        try:
            idea = _build_idea(
                symbol, hist, spot, master, risk_calculator, strategies,
                now=now, premium_lookup=premium_lookup, default_iv=default_iv,
                allow_expiry_day=allow_expiry_day, intraday=intraday,
            )
        except Exception as exc:
            errors += 1
            logger.warning("Stock scan: %s failed — %s", symbol, exc)
            continue
        if idea is not None:
            ideas.append(idea)

    ideas.sort(key=lambda i: i.confidence, reverse=True)
    logger.info(
        "Stock scan: %d scanned, %d actionable, %d errors → top %d",
        scanned, len(ideas), errors, min(top_n, len(ideas)),
    )
    return ScanResult(
        ideas=ideas[:top_n], scanned=scanned, actionable=len(ideas), errors=errors,
    )


def _build_idea(
    symbol: str,
    hist: pd.DataFrame,
    spot: float,
    master: InstrumentMaster,
    risk_calculator: RiskCalculator,
    strategies: list[BaseStrategy],
    *,
    now: dt_time,
    premium_lookup,
    default_iv: float,
    allow_expiry_day: bool | None,
    intraday: bool,
) -> StockIdea | None:
    """Run the full pipeline for one stock; return an idea or None (no trade)."""
    df = compute_all_indicators(hist)
    usable = df.dropna(subset=_REQUIRED_COLS)
    if usable.empty:
        logger.debug("Stock scan: %s has no complete indicator rows — skipping", symbol)
        return None
    latest = usable.iloc[-1]
    atr = float(latest["atr"])

    signals = [s.generate_signal(df) for s in strategies]
    consensus = build_consensus(signals, now=now, intraday=intraday)
    if not consensus.is_actionable:
        return None

    asset = symbol[: -len(_NS_SUFFIX)] if symbol.endswith(_NS_SUFFIX) else symbol
    opt_type = "CE" if consensus.signal == SignalType.BUY_CE else "PE"

    contract = master.atm_contract(
        asset, spot, opt_type, allow_expiry_day=allow_expiry_day,
    )
    if contract is None:
        logger.info("Stock scan: %s — no %s contract available, skipping", asset, opt_type)
        return None

    expiry_str = contract.expiry.strftime("%d-%b-%Y")
    entry_premium, is_live = _resolve_premium(
        contract, spot, expiry_str, opt_type, premium_lookup, default_iv,
    )
    if entry_premium <= 0:
        logger.info("Stock scan: %s — no usable premium, skipping", asset)
        return None

    risk = risk_calculator.calculate(consensus.signal, spot, atr)

    return StockIdea(
        symbol=asset,
        signal=consensus.signal.value,
        confidence=consensus.confidence,
        conviction=consensus.conviction,
        opt_type=opt_type,
        strike=contract.strike,
        expiry=expiry_str,
        lot_size=contract.lot_size,
        entry_premium=entry_premium,
        is_live=is_live,
        spot=round(spot, 2),
        target=risk.target,
        stop_loss=risk.stop_loss,
        rr=risk.risk_reward_ratio,
    )


def _resolve_premium(
    contract: OptionContract,
    spot: float,
    expiry_str: str,
    opt_type: str,
    premium_lookup,
    default_iv: float,
) -> tuple[float, bool]:
    """Return *(premium, is_live)* — live LTP if available, else a BS estimate.

    The instrument master resolves the strike/lot/expiry without a token, but the
    traded premium needs an authenticated quote. When that is unavailable the
    Black-Scholes estimate keeps a "Buy ₹" figure on the alert rather than a blank.
    """
    try:
        ltp = premium_lookup([contract.instrument_key]).get(contract.instrument_key, 0.0)
        if ltp and ltp > 0:
            return round(float(ltp), 2), True
    except Exception as exc:
        logger.debug("Stock scan: LTP lookup failed for %s (%s)", contract.trading_symbol, exc)

    ce, pe = compute_atm_theoretical_prices(spot, int(round(contract.strike)), expiry_str, default_iv)
    return (ce if opt_type == "CE" else pe), False
