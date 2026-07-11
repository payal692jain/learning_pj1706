"""Capital-aware trade plan — one Pushover message covering NIFTY, SENSEX, BANKNIFTY.

For each index with an actionable signal it shows: which CE/PE to buy, the live
entry premium, the estimated premium to SELL at when the index hits the risk
target (and at the stop-loss), lot economics, and how many lots the configured
capital can actually afford versus how many the daily profit target would need.

The sell prices are Black-Scholes re-pricings of the live premium at the risk
engine's index target/SL — estimates, not promises. The formatter deliberately
flags when the profit target is NOT reachable with the available capital rather
than pretending otherwise: a daily target of 20% of capital is aggressive, and
options lose the SL amount just as readily as they gain the target amount.
"""

import logging
import math
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
    lot_size: int
    is_live: bool          # False when premiums come from the VIX-based estimate

    @property
    def cost_per_lot(self) -> float:
        return self.entry_premium * self.lot_size

    @property
    def pnl_target_per_lot(self) -> float:
        return (self.target_sell - self.entry_premium) * self.lot_size

    @property
    def pnl_sl_per_lot(self) -> float:
        return (self.sl_sell - self.entry_premium) * self.lot_size


def build_trade_idea(
    index_name: str,
    signal_type: SignalType,
    confidence: int,
    analysis: ExpiryAnalysis,
    risk: RiskParameters,
    lot_size: int,
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
        lot_size=lot_size,
        is_live=analysis.is_live,
    )


def format_trade_plan(
    ideas: list[TradeIdea],
    holds: list[str],
    capital: float,
    profit_target: float,
) -> tuple[str, str]:
    """Return (title, body) for the combined three-index trade-plan notification."""
    summary = " | ".join(
        [f"{i.index_name} {i.opt_type}" for i in ideas] + [f"{h} —" for h in holds]
    )
    title = f"🎯 Trade Plan — {summary}" if summary else "🎯 Trade Plan"

    lines: list[str] = [
        f"Capital ₹{capital:,.0f} · Daily target ₹{profit_target:,.0f}",
        "",
    ]

    for idea in ideas:
        lines.extend(_format_idea(idea, capital, profit_target))
        lines.append("")

    for name in holds:
        lines.append(f"⏸ {name}: HOLD — no edge right now, staying out IS the plan.")
    if holds:
        lines.append("")

    lines.append(
        "⚠️ Sell prices are model estimates at the risk target/SL — signals, "
        "not guarantees. A ₹{:,.0f}/day goal on ₹{:,.0f} is {:.0f}%/day; expect "
        "losing days and never risk money you can't afford to lose.".format(
            profit_target, capital, profit_target / capital * 100 if capital else 0,
        )
    )
    return title, "\n".join(lines)


def _format_idea(idea: TradeIdea, capital: float, profit_target: float) -> list[str]:
    icon = "📈" if idea.opt_type == "CE" else "📉"
    est = "" if idea.is_live else " (Est.)"
    lines = [
        f"{icon} {idea.index_name}: BUY {idea.strike} {idea.opt_type}  {idea.expiry}"
        f"  @ ₹{idea.entry_premium:g}{est}  ({idea.confidence}%)",
        f"   SELL @ ₹{idea.target_sell:g} target  |  EXIT @ ₹{idea.sl_sell:g} stop-loss",
        f"   1 lot = {idea.lot_size} qty = ₹{idea.cost_per_lot:,.0f}"
        f"  →  +₹{idea.pnl_target_per_lot:,.0f} at target / −₹{abs(idea.pnl_sl_per_lot):,.0f} at SL",
    ]

    affordable = int(capital // idea.cost_per_lot) if idea.cost_per_lot > 0 else 0
    if affordable == 0:
        lines.append(
            f"   ✗ 1 lot costs ₹{idea.cost_per_lot:,.0f} — more than your ₹{capital:,.0f} capital."
        )
        return lines

    lines.append(
        f"   Max {affordable} lot(s) with capital → "
        f"+₹{idea.pnl_target_per_lot * affordable:,.0f} at target / "
        f"−₹{abs(idea.pnl_sl_per_lot) * affordable:,.0f} at SL"
    )

    if idea.pnl_target_per_lot > 0:
        needed = math.ceil(profit_target / idea.pnl_target_per_lot)
        if needed <= affordable:
            lines.append(
                f"   ✓ ₹{profit_target:,.0f} target needs {needed} lot(s)"
                f" (₹{idea.cost_per_lot * needed:,.0f})"
            )
        else:
            lines.append(
                f"   ✗ ₹{profit_target:,.0f} target needs {needed} lot(s)"
                f" (₹{idea.cost_per_lot * needed:,.0f}) — NOT reachable with"
                f" ₹{capital:,.0f}; max realistic: ₹{idea.pnl_target_per_lot * affordable:,.0f}"
            )
    return lines
