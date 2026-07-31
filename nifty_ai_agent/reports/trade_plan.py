"""Trade plan — one Pushover message covering NIFTY, SENSEX, BANKNIFTY.

For each index with an actionable signal it shows: which CE/PE to buy, the live
entry premium, and the estimated premium to SELL at when the index hits the risk
target (and to exit at, if the stop-loss hits instead).

Prices only. Position sizing lives with the risk engine, not here — how many lots
fit an account is a property of the account, and mixing it into the plan buried
the three numbers that actually matter under lot economics.

The sell prices are Black-Scholes re-pricings of the live premium at the risk
engine's index target/SL — estimates, not promises. Options lose the SL amount
just as readily as they gain the target amount.
"""

import logging
from dataclasses import dataclass

from nifty_ai_agent.risk.calculator import RiskParameters
from nifty_ai_agent.strategies.base import SignalType
from nifty_ai_agent.strategies.option_analyser import (
    ExpiryAnalysis,
    atm_iv,
    estimate_premium_at_spot,
)

logger = logging.getLogger(__name__)

# Used when the Upstox contract feed is unavailable (e.g. token expired).
# Confirmed live from Upstox contract data (Jul 2026); the live feed value wins.
FALLBACK_LOT_SIZES = {"NIFTY": 65, "SENSEX": 20, "BANKNIFTY": 30}


@dataclass
class TradeIdea:
    index_name: str
    signal: str            # BUY_CE / BUY_PE
    confidence: int
    strike: int
    opt_type: str          # CE / PE
    expiry: str
    entry_premium: float
    target_sell: float     # estimated premium at the index risk target
    sl_sell: float         # estimated premium at the index stop-loss
    is_live: bool          # False when premiums come from the VIX-based estimate


def build_trade_idea(
    index_name: str,
    signal_type: SignalType,
    confidence: int,
    analysis: ExpiryAnalysis,
    risk: RiskParameters,
) -> TradeIdea | None:
    """Turn a signal + option chain analysis + risk levels into a TradeIdea.

    Returns None for HOLD signals or when no usable entry premium exists.
    """
    if signal_type == SignalType.HOLD:
        return None

    bullish = signal_type == SignalType.BUY_CE
    opt_type = "CE" if bullish else "PE"
    entry = analysis.atm_ce_ltp if bullish else analysis.atm_pe_ltp
    if not entry or entry <= 0:
        return None

    iv = atm_iv(analysis, opt_type)
    target_sell = estimate_premium_at_spot(
        entry, analysis.spot, risk.target, analysis.atm_strike,
        analysis.expiry, iv, opt_type,
    )
    sl_sell = estimate_premium_at_spot(
        entry, analysis.spot, risk.stop_loss, analysis.atm_strike,
        analysis.expiry, iv, opt_type,
    )

    return TradeIdea(
        index_name=index_name,
        signal=signal_type.value,
        confidence=confidence,
        strike=analysis.atm_strike,
        opt_type=opt_type,
        expiry=analysis.expiry,
        entry_premium=entry,
        target_sell=target_sell,
        sl_sell=sl_sell,
        is_live=analysis.is_live,
    )


def _short_name(name: str) -> str:
    """Fit an index name into a narrow table column header."""
    return {"BANKNIFTY": "BANKNIF"}.get(name, name)[:7]


def _prem(value: float) -> str:
    """Premium for a table cell — whole rupees, but two decimals below ₹10 so
    near-expiry paise premiums don't collapse to a misleading '0'."""
    if value and value < 10:
        return f"{value:.2f}"
    return f"{value:,.0f}"


def format_trade_plan(ideas: list[TradeIdea], holds: list[str]) -> tuple[str, str]:
    """Return (title, body) for the combined three-index trade-plan notification.

    Prices only — what to buy at, what to sell at, where to exit. Position sizing
    is deliberately not here: it depends on the account, not on the setup.

    Rendered in Pushover monospace mode as a column-per-index table so all
    three indices line up and read at a glance.
    """
    summary = " | ".join(
        [f"{i.index_name} {i.opt_type}" for i in ideas] + [f"{h} —" for h in holds]
    )
    title = f"🎯 Trade Plan — {summary}" if summary else "🎯 Trade Plan"

    lines: list[str] = []

    if ideas:
        def row(label: str, cells: list[str]) -> str:
            return f"{label:<8}" + "".join(f"{c:>9}" for c in cells)

        lines += [
            row("", [_short_name(i.index_name) + ("*" if not i.is_live else "") for i in ideas]),
            row("Option", [f"{i.strike}{i.opt_type}" for i in ideas]),
            row("Expiry", [i.expiry[:6] for i in ideas]),
            row("Buy ₹", [_prem(i.entry_premium) for i in ideas]),
            row("Sell ₹", [_prem(i.target_sell) for i in ideas]),
            row("Exit ₹", [_prem(i.sl_sell) for i in ideas]),
            "",
        ]

    for name in holds:
        lines.append(f"⏸ {name}: HOLD — no edge; staying out IS the plan.")
    if holds:
        lines.append("")

    if ideas:
        lines.append("(Buy=entry, Sell=at target, Exit=at stop-loss)")
    lines.append(
        "⚠️ Estimates, not guarantees — expect losing days; never risk "
        "money you can't afford to lose."
    )
    return title, "\n".join(lines)
