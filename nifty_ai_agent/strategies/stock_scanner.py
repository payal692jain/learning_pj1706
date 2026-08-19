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
from dataclasses import dataclass, field
from datetime import time as dt_time

import pandas as pd

from nifty_ai_agent.data.fundamentals import volume_pace_ratio
from nifty_ai_agent.data.instrument_master import (
    SEGMENT_EQUITY_FO,
    InstrumentMaster,
    OptionContract,
)
from nifty_ai_agent.risk.calculator import RiskCalculator
from nifty_ai_agent.strategies.base import BaseStrategy, SignalType
from nifty_ai_agent.strategies.consensus import build_consensus
from nifty_ai_agent.strategies.option_analyser import (
    compute_atm_theoretical_prices,
    estimate_premium_at_spot,
)
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
    # What the CONTRACT is worth if the underlying reaches those levels today.
    # The underlying target answers "where is the stock going"; these answer the
    # question an intraday option trade actually turns on — what to sell at.
    # Same-session estimate: no overnight theta is taken out.
    target_premium: float = 0.0
    stop_premium: float = 0.0
    # Event/fundamental context — None when nothing was fetched for this name.
    fundamentals: object | None = None
    volume_ratio: float | None = None   # pace-adjusted vs 20d average
    # Levels for trading the SHARES rather than the option. Derived from the DAILY
    # ATR, not the intraday one: a 5-minute ATR puts the target ~0.4% away, which
    # is a scalp, not a reason to own a stock. 0.0 when daily bars were unavailable.
    cash_target: float = 0.0
    cash_stop: float = 0.0


@dataclass
class HoldRead:
    """A symbol that scanned cleanly but produced no trade.

    Kept rather than discarded so a small watchlist can show its current read
    instead of the symbol silently vanishing from the digest — on a two-name book,
    "BSE is undecided" and "BSE was never scanned" look identical otherwise.
    """
    symbol: str
    confidence: int
    conviction: str
    spot: float
    reason: str = ""   # e.g. "results in 2d" when an event blocked the entry


@dataclass
class ScanResult:
    ideas: list[StockIdea]  # top-N, highest confidence first
    scanned: int            # symbols with usable data
    actionable: int         # symbols that produced a (pre-ranking) idea
    errors: int             # symbols that raised while being processed
    holds: list[HoldRead] = field(default_factory=list)  # scanned, but no trade


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
    segment: str = SEGMENT_EQUITY_FO,
    fundamentals: dict | None = None,
    earnings_blackout_days: int = 0,
    trend_tf: str = "",
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
        segment: derivatives book the contracts live in — NSE_FO for stocks,
            NCD_FO for currency pairs.
        fundamentals: {bare symbol: StockFundamentals} for event context.
        earnings_blackout_days: refuse a new entry when results are this many
            days away or fewer. 0 disables the guard.
        trend_tf: slower timeframe ('60m') an entry must agree with. Empty
            disables the filter.
    """
    strategies = strategies or DEFAULT_STRATEGIES
    ideas: list[StockIdea] = []
    holds: list[HoldRead] = []
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
                allow_expiry_day=allow_expiry_day, intraday=intraday, segment=segment,
                fundamentals=fundamentals or {},
                earnings_blackout_days=earnings_blackout_days,
                trend_tf=trend_tf,
            )
        except Exception as exc:
            errors += 1
            logger.warning("Stock scan: %s failed — %s", symbol, exc)
            continue
        if isinstance(idea, HoldRead):
            holds.append(idea)
        elif idea is not None:
            ideas.append(idea)

    ideas.sort(key=lambda i: i.confidence, reverse=True)
    logger.info(
        "Stock scan: %d scanned, %d actionable, %d errors → top %d",
        scanned, len(ideas), errors, min(top_n, len(ideas)),
    )
    return ScanResult(
        ideas=ideas[:top_n], scanned=scanned, actionable=len(ideas), errors=errors,
        holds=holds,
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
    segment: str = SEGMENT_EQUITY_FO,
    fundamentals: dict | None = None,
    earnings_blackout_days: int = 0,
    trend_tf: str = "",
) -> StockIdea | HoldRead | None:
    """Run the full pipeline for one stock.

    Returns a StockIdea when the consensus is actionable and a contract prices,
    a HoldRead when the symbol scanned fine but the engine wants no trade, and
    None only when the data itself was unusable.
    """
    df = compute_all_indicators(hist)
    usable = df.dropna(subset=_REQUIRED_COLS)
    if usable.empty:
        logger.debug("Stock scan: %s has no complete indicator rows — skipping", symbol)
        return None
    latest = usable.iloc[-1]
    atr = float(latest["atr"])

    signals = [s.generate_signal(df) for s in strategies]
    consensus = build_consensus(signals, now=now, intraday=intraday)
    asset = symbol[: -len(_NS_SUFFIX)] if symbol.endswith(_NS_SUFFIX) else symbol

    # Confirm against a slower timeframe before calling anything actionable.
    # Backtested over 12 symbols: 30m entries alone returned +0.021% expectancy,
    # 30m confirmed by 60m returned +0.031% on roughly half the trades.
    if trend_tf and consensus.is_actionable:
        trend = _trend_signal(hist, trend_tf, strategies, now)
        if trend is not None and trend is not consensus.signal:
            return HoldRead(
                symbol=asset, confidence=consensus.confidence,
                conviction=consensus.conviction, spot=round(spot, 2),
                reason=f"{trend_tf} trend does not confirm",
            )
    if not consensus.is_actionable:
        return HoldRead(
            symbol=asset, confidence=consensus.confidence,
            conviction=consensus.conviction, spot=round(spot, 2),
        )

    # ── Event guard ──────────────────────────────────────────────────────────
    # A chart cannot see a results date. Long premium into a print is a different
    # trade: the gap can run through any stop, and the post-result IV collapse
    # takes value out even when the direction was right. Refuse the entry and say
    # why, rather than emitting a technical signal that does not know about it.
    fund = (fundamentals or {}).get(asset)
    if earnings_blackout_days and fund is not None:
        dte = fund.days_to_earnings
        if dte is not None and dte <= earnings_blackout_days:
            logger.info("Stock scan: %s blocked — results in %dd", asset, dte)
            return HoldRead(
                symbol=asset, confidence=consensus.confidence,
                conviction=consensus.conviction, spot=round(spot, 2),
                reason=f"results in {dte}d" if dte else "results today",
            )

    opt_type = "CE" if consensus.signal == SignalType.BUY_CE else "PE"
    contract = master.atm_contract(
        asset, spot, opt_type, allow_expiry_day=allow_expiry_day, segment=segment,
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

    volume_ratio = None
    if fund is not None and "volume" in hist.columns:
        last_day = hist[hist.index.date == hist.index[-1].date()]
        volume_ratio = volume_pace_ratio(
            float(last_day["volume"].sum()), fund.avg_volume, now,
        )

    risk = risk_calculator.calculate(consensus.signal, spot, atr)
    cash_target, cash_stop = _cash_levels(
        hist, spot, bullish=consensus.signal == SignalType.BUY_CE,
    )
    target_premium, stop_premium = _project_premiums(
        entry_premium, spot, risk, contract, expiry_str, opt_type, default_iv,
    )

    return StockIdea(
        symbol=asset,
        signal=consensus.signal.value,
        confidence=consensus.confidence,
        conviction=consensus.conviction,
        opt_type=opt_type,
        strike=contract.strike,
        expiry=expiry_str,
        lot_size=contract.contract_size,
        entry_premium=entry_premium,
        is_live=is_live,
        spot=round(spot, 2),
        target=risk.target,
        stop_loss=risk.stop_loss,
        rr=risk.risk_reward_ratio,
        target_premium=target_premium,
        stop_premium=stop_premium,
        fundamentals=fund,
        volume_ratio=volume_ratio,
        cash_target=cash_target,
        cash_stop=cash_stop,
    )


def _trend_signal(hist, trend_tf: str, strategies, now):
    """Consensus on *hist* resampled up to *trend_tf*, or None when unavailable.

    None means "no opinion", and the caller treats that as permission to proceed
    rather than as a veto — a missing slow read is missing information, and
    blocking every entry on it would be indistinguishable from a dead scan.
    """
    from nifty_ai_agent.strategies.multi_timeframe import resample_ohlcv

    try:
        bars = resample_ohlcv(hist, trend_tf)
        if len(bars) < 60:
            return None
        slow = compute_all_indicators(bars)
        if slow[_REQUIRED_COLS].iloc[-1].isna().any():
            return None
        return build_consensus(
            [s.generate_signal(slow) for s in strategies], now=now, intraday=False,
        ).signal
    except Exception as exc:
        logger.debug("Trend read on %s failed: %s", trend_tf, exc)
        return None


def _cash_levels(hist, spot: float, bullish: bool, atr_mult: float = 1.5,
                 rr: float = 2.0) -> tuple[float, float]:
    """Swing-scale target and stop for holding the SHARES, from the daily ATR.

    The intraday risk engine sizes for an option scalp — roughly 0.4% on a
    5-minute ATR. Presented as a stock buy/sell price that is actively
    misleading: nobody takes delivery for 0.4%, and a stop that tight is inside
    a single day's noise. Recomputing on daily bars gives levels someone could
    hold a position against.

    Returns (0.0, 0.0) when there are too few daily bars to trust an ATR.
    """
    try:
        daily = hist.resample("1D").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna(how="any")
        # ATR(14) needs its window plus a bar to be worth anything.
        if len(daily) < 15:
            return 0.0, 0.0
        from nifty_ai_agent.indicators.atr import compute_atr

        atr = float(compute_atr(daily)["atr"].iloc[-1])
        if not atr or atr <= 0:
            return 0.0, 0.0
        risk = atr * atr_mult
        if bullish:
            return round(spot + risk * rr, 2), round(spot - risk, 2)
        return round(spot - risk * rr, 2), round(spot + risk, 2)
    except Exception as exc:
        logger.debug("Cash levels unavailable: %s", exc)
        return 0.0, 0.0


def _project_premiums(
    entry_premium: float,
    spot: float,
    risk,
    contract: OptionContract,
    expiry_str: str,
    opt_type: str,
    iv: float,
) -> tuple[float, float]:
    """Premium if the underlying reaches the target, and if it reaches the stop.

    Anchored on the live entry premium and moved by the Black-Scholes *delta* over
    that spot move, so any model-vs-market basis cancels out. Held flat in time:
    an intraday trade closes the same session, so taking days of theta out of the
    exit price would understate what the position is actually worth when the level
    prints. Returns (0.0, 0.0) when risk levels are unusable, and the callers
    render those as blank rather than as a Rs 0 target.
    """
    if not risk.is_valid or entry_premium <= 0 or spot <= 0:
        return 0.0, 0.0
    strike = int(round(contract.strike))
    try:
        return (
            estimate_premium_at_spot(
                entry_premium, spot, risk.target, strike, expiry_str, iv, opt_type,
            ),
            estimate_premium_at_spot(
                entry_premium, spot, risk.stop_loss, strike, expiry_str, iv, opt_type,
            ),
        )
    except Exception as exc:
        logger.debug("Premium projection failed for %s: %s", contract.trading_symbol, exc)
        return 0.0, 0.0


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
