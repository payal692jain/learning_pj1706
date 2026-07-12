"""The intraday trade call — one verdict per index, with the strategy book behind it.

This replaces the old "here are six strategies, you decide" notification. Six
opinions arriving at once is not a call, it is a research dump; the trader still has
to do the aggregation on a phone screen while the move happens. This message leads
with the decision, then shows the vote that produced it, so the reasoning is
auditable without being the headline.

Where a signal is actionable it carries the whole trade: which contract, the entry
premium, the premium to sell at when the index reaches target, the premium to exit at
if it stops out, and how many lots the risk rules actually permit — which is
frequently zero, and says so.
"""

import logging

from nifty_ai_agent.data.bank_options import BankOptionIdea, format_bank_options
from nifty_ai_agent.reports.layout import fit_sections, inr, premium
from nifty_ai_agent.risk.calculator import RiskParameters
from nifty_ai_agent.risk.margin import MarginCalculator, option_buy_margin
from nifty_ai_agent.strategies.base import SignalType
from nifty_ai_agent.strategies.consensus import Consensus
from nifty_ai_agent.strategies.global_analyser import GlobalSnapshot
from nifty_ai_agent.strategies.option_analyser import (
    ExpiryAnalysis,
    atm_iv,
    estimate_premium_at_spot,
)

logger = logging.getLogger(__name__)

_CONVICTION_ICON = {
    "STRONG": "🟢",
    "MODERATE": "🟡",
    "WEAK": "🟠",
    "NO_TRADE": "⏸",
}
_VOTE_ICON = {SignalType.BUY_CE: "▲", SignalType.BUY_PE: "▼", SignalType.HOLD: "·"}

# Strategy names are verbose in code and useless in a 9-character notification column.
_SHORT_STRATEGY = {
    "Opening_Range_Breakout": "ORB",
    "VWAP_Breakout": "VWAP",
    "Supertrend": "Supertrnd",
    "MACD_Momentum": "MACD",
    "Bollinger_Squeeze": "BollSqz",
    "EMA_Crossover": "EMA",
}


def format_trade_call(
    index_name: str,
    consensus: Consensus,
    risk: RiskParameters,
    analysis: ExpiryAnalysis | None,
    margin: MarginCalculator,
    lot_size: int,
    global_snapshot: GlobalSnapshot | None = None,
    bank_ideas: list[BankOptionIdea] | None = None,
    prediction: bool = False,
) -> tuple[str, str]:
    """Return (title, body) for one index's trade call."""
    icon = _CONVICTION_ICON.get(consensus.conviction, "")
    action = (
        f"{consensus.signal.value} — {consensus.conviction} {consensus.confidence}%"
        if consensus.is_actionable
        else "NO TRADE"
    )
    prefix = "📊 PREDICTION" if prediction else icon
    title = f"{prefix} {index_name} {action}"

    essential: list[str] = []
    if prediction:
        essential.append("⚠️ Market closed — outlook for the next session.")

    if consensus.is_actionable and analysis:
        essential += _contract_lines(
            index_name, consensus, risk, analysis, margin, lot_size,
        )
    else:
        essential.append(f"⏸ {consensus.rationale}")
    essential.append("")

    optional = _strategy_table(consensus)
    if global_snapshot and global_snapshot.is_available:
        optional += _global_lines(global_snapshot)
    if bank_ideas:
        optional += format_bank_options(bank_ideas, margin.capital) + [""]

    footer = ["⚠️ Estimates, not advice. Options can lose 100%."]
    return title, fit_sections(essential, optional, footer)


def _contract_lines(
    index_name: str,
    consensus: Consensus,
    risk: RiskParameters,
    analysis: ExpiryAnalysis,
    margin: MarginCalculator,
    lot_size: int,
) -> list[str]:
    """The trade itself: contract, premiums at target and stop, and permitted size."""
    opt_type = "CE" if consensus.signal == SignalType.BUY_CE else "PE"
    entry = (
        (analysis.atm_ce_ltp or analysis.theoretical_ce_atm) if opt_type == "CE"
        else (analysis.atm_pe_ltp or analysis.theoretical_pe_atm)
    )
    est = "*" if not analysis.is_live else ""

    lines = [
        f"📌 BUY {index_name} {analysis.atm_strike} {opt_type}  {analysis.expiry[:6]}{est}",
    ]

    if not entry or entry <= 0 or not risk.is_valid:
        lines.append(f"[No sizing — {risk.rejection_reason or 'no live premium'}]")
        lines.append(consensus.rationale)
        return lines

    iv = atm_iv(analysis, opt_type)
    target_premium = estimate_premium_at_spot(
        entry, analysis.spot, risk.target, analysis.atm_strike, analysis.expiry, iv, opt_type,
    )
    sl_premium = estimate_premium_at_spot(
        entry, analysis.spot, risk.stop_loss, analysis.atm_strike, analysis.expiry, iv, opt_type,
    )
    loss_per_lot = max(0.0, entry - sl_premium) * lot_size

    sizing = margin.size(
        option_buy_margin(index_name, analysis.atm_strike, opt_type, entry, lot_size, analysis.spot),
        loss_per_lot_at_sl=loss_per_lot,
    )

    lines += [
        f"Buy ₹{premium(entry)} → Sell ₹{premium(target_premium)} "
        f"| Exit ₹{premium(sl_premium)}",
        f"Index {risk.entry_price:,.0f} → {risk.target:,.0f} | SL {risk.stop_loss:,.0f} "
        f"(RR 1:{risk.risk_reward_ratio})",
    ]

    if sizing.is_tradeable:
        lines.append(
            f"Size {sizing.lots} lot(s) · margin ₹{inr(sizing.margin_used)} · "
            f"risk ₹{inr(sizing.risk_at_sl)} ({sizing.risk_pct_of_capital:.1f}%)"
        )
    else:
        lines.append(f"⛔ 0 lots — {sizing.blocked_reason}")

    lines.append(consensus.rationale)
    return lines


def _strategy_table(consensus: Consensus) -> list[str]:
    """Every strategy's vote — the working behind the verdict."""
    lines = [
        f"── STRATEGIES ({consensus.agreement:.0%} agree) ──",
    ]
    # Loudest voices first: a reader scanning this wants the votes that moved the needle.
    for vote in sorted(consensus.votes, key=lambda v: v.score, reverse=True):
        icon = _VOTE_ICON.get(vote.signal, "·")
        name = _SHORT_STRATEGY.get(vote.strategy, vote.strategy)[:9]
        verdict = "HOLD" if vote.signal == SignalType.HOLD else vote.signal.value[-2:]
        lines.append(f"{icon} {name:<9} {verdict:<4} {vote.confidence:>3}% ×{vote.weight:.1f}")
    lines.append("")
    return lines


def _global_lines(snapshot: GlobalSnapshot) -> list[str]:
    parts = [f"🌍 {snapshot.global_bias}"]
    if snapshot.gift_nifty_pct:
        parts.append(f"GIFT {snapshot.gift_nifty_pct:+.1f}%")
    if snapshot.vix:
        parts.append(f"VIX {snapshot.vix:.1f} ({snapshot.vix_regime})")
    if snapshot.news.headlines:
        parts.append(f"news {snapshot.news.label.lower()}")
    return [" · ".join(parts), ""]
