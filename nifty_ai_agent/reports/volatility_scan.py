"""Volatile-stock straddle scan — one Pushover digest of long-volatility ideas.

A ranked table of the most volatile stocks (by ATR %) and, for each, the ATM
straddle to buy: the combined CE+PE premium, the strike, and the move needed to
break even. Underlying breakeven levels follow per idea. Premiums are estimates
unless a live quote was available (marked with *).
"""

from nifty_ai_agent.reports.layout import fit_sections, inr, premium, row
from nifty_ai_agent.strategies.volatility_scanner import StraddleIdea, VolatilityScanResult


def _idea_block(idea: StraddleIdea) -> list[str]:
    """A two-line entry: the volatility/cost row, then the straddle's breakeven levels."""
    est = "*" if not (idea.ce_is_live and idea.pe_is_live) else ""
    header = row(
        f"🎢{idea.symbol}",
        [f"{idea.atr_pct:g}%", premium(idea.total_premium) + est, f"±{idea.breakeven_move_pct:g}%"],
        label_width=11, cell_width=9,
    )
    levels = (
        f"   {idea.strike:g} CE+PE (₹{premium(idea.ce_premium)}+₹{premium(idea.pe_premium)}) · "
        f"BE {idea.breakeven_low:g}/{idea.breakeven_high:g} · lot ₹{inr(idea.lot_cost)}"
    )
    return [header, levels]


def format_volatility_scan(result: VolatilityScanResult) -> tuple[str, str]:
    """Return (title, body) for the volatility straddle Pushover notification."""
    ideas = result.ideas

    if not ideas:
        title = "⏸ Volatile Straddles — none"
        body = fit_sections(
            [f"No tradable straddles ({result.scanned} scanned)."],
            [],
            ["⚠️ Estimates, not advice. A straddle can lose its full premium."],
        )
        return title, body

    summary = " · ".join(i.symbol for i in ideas[:3])
    title = f"🎢 Volatile Straddles — {summary}"

    expiry = ideas[0].expiry
    essential = [
        f"Monthly expiry {expiry} · {result.ranked} ranked by ATR%, top {len(ideas)}:",
        row("", ["ATR%", "Cost₹", "BE±%"], label_width=11, cell_width=9),
    ]

    optional: list[str] = []
    for idea in ideas:
        optional += _idea_block(idea) + [""]

    any_est = any(not (i.ce_is_live and i.pe_is_live) for i in ideas)
    footer = [
        "* estimated premium — no live quote" if any_est else "",
        "Buy CE+PE: profits on a big move either way; loses if it sits still.",
        "⚠️ Estimates, not advice. A straddle can lose its full premium.",
    ]
    footer = [line for line in footer if line]
    return title, fit_sections(essential, optional, footer)
