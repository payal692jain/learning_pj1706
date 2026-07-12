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


def _short_name(name: str) -> str:
    """Fit an index name into a narrow table column header."""
    return {"BANKNIFTY": "BANKNIF"}.get(name, name)[:7]


def _inr(value: float) -> str:
    """Compact INR for a table cell: 6760 → '6.8k', 102193 → '102k'."""
    if abs(value) >= 100_000:
        return f"{value / 1000:,.0f}k"
    if abs(value) >= 1_000:
        return f"{value / 1000:.1f}k"
    return f"{value:,.0f}"


def _prem(value: float) -> str:
    """Premium for a table cell — whole rupees, but two decimals below ₹10 so
    near-expiry paise premiums don't collapse to a misleading '0'."""
    if value and value < 10:
        return f"{value:.2f}"
    return f"{value:,.0f}"


def format_trade_plan(
    ideas: list[TradeIdea],
    holds: list[str],
    capital: float,
    profit_target: float,
) -> tuple[str, str]:
    """Return (title, body) for the combined three-index trade-plan notification.

    Rendered in Pushover monospace mode as a column-per-index table so all
    three indices line up and read at a glance.
    """
    summary = " | ".join(
        [f"{i.index_name} {i.opt_type}" for i in ideas] + [f"{h} —" for h in holds]
    )
    title = f"🎯 Trade Plan — {summary}" if summary else "🎯 Trade Plan"

    lines: list[str] = [
        f"Capital ₹{capital:,.0f} · Target ₹{profit_target:,.0f}/day",
        "",
    ]

    if ideas:
        def row(label: str, cells: list[str]) -> str:
            return f"{label:<8}" + "".join(f"{c:>9}" for c in cells)

        aff = [
            int(capital // i.cost_per_lot) if i.cost_per_lot > 0 else 0 for i in ideas
        ]
        lines += [
            row("", [_short_name(i.index_name) + ("*" if not i.is_live else "") for i in ideas]),
            row("Option", [f"{i.strike}{i.opt_type}" for i in ideas]),
            row("Expiry", [i.expiry[:6] for i in ideas]),
            row("Buy ₹", [_prem(i.entry_premium) for i in ideas]),
            row("Sell ₹", [_prem(i.target_sell) for i in ideas]),
            row("Exit ₹", [_prem(i.sl_sell) for i in ideas]),
            row("Lot qty", [str(i.lot_size) for i in ideas]),
            row("1lot ₹", [_inr(i.cost_per_lot) for i in ideas]),
            row("Lots/cap", [str(a) for a in aff]),
            row("P/L tgt", [
                f"+{_inr(i.pnl_target_per_lot * a)}" if a else "0"
                for i, a in zip(ideas, aff)
            ]),
            row("P/L SL", [
                f"-{_inr(abs(i.pnl_sl_per_lot) * a)}" if a else "0"
                for i, a in zip(ideas, aff)
            ]),
            "",
        ]

        # Per-index reachability of the daily target — the honest part.
        for idea, a in zip(ideas, aff):
            lines.append(_reachability_note(idea, a, capital, profit_target))
        lines.append("")

    for name in holds:
        lines.append(f"⏸ {name}: HOLD — no edge; staying out IS the plan.")
    if holds:
        lines.append("")

    if ideas:
        lines.append("(Sell=at target, Exit=at stop-loss; P/L for max lots)")
    pct = profit_target / capital * 100 if capital else 0
    lines.append(
        f"⚠️ Estimates, not guarantees. ₹{profit_target:,.0f}/day on "
        f"₹{capital:,.0f} is {pct:.0f}%/day — expect losing days; never risk "
        "money you can't afford to lose."
    )
    return title, "\n".join(lines)


def _reachability_note(idea: TradeIdea, affordable: int, capital: float, profit_target: float) -> str:
    est = "*" if not idea.is_live else ""
    if affordable == 0:
        return f"✗ {idea.index_name}{est}: 1 lot ₹{idea.cost_per_lot:,.0f} > capital ₹{capital:,.0f}"
    if idea.pnl_target_per_lot <= 0:
        return f"• {idea.index_name}{est}: no upside at target — skip"
    needed = math.ceil(profit_target / idea.pnl_target_per_lot)
    if needed <= affordable:
        return (
            f"✓ {idea.index_name}{est}: ₹{profit_target:,.0f} needs {needed} lot(s)"
            f" (₹{idea.cost_per_lot * needed:,.0f})"
        )
    return (
        f"✗ {idea.index_name}{est}: ₹{profit_target:,.0f} needs {needed} lot(s) —"
        f" NOT reachable; max ₹{idea.pnl_target_per_lot * affordable:,.0f}"
    )
