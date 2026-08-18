"""Stock option scan — one Pushover digest of the best single-stock CE/PE ideas.

A ranked table: which stock, which monthly contract to buy, the entry premium,
and the consensus confidence behind it. Underlying target/stop-loss levels follow
per idea. Prices are estimates unless a live quote was available (marked with *).
"""

from nifty_ai_agent.reports.layout import fit_sections, premium, row
from nifty_ai_agent.strategies.stock_scanner import ScanResult, StockIdea

_SIDE_ICON = {"BUY_CE": "📈", "BUY_PE": "📉"}


def _idea_block(idea: StockIdea) -> list[str]:
    """Contract/price row, the underlying levels, then the premium to exit at.

    The underlying target says where the stock has to go; the premium line says
    what the option is worth when it gets there — which is the number an intraday
    trade is actually closed on.
    """
    est = "*" if not idea.is_live else ""
    contract = f"{idea.strike:g}{idea.opt_type}"
    header = row(
        f"{_SIDE_ICON.get(idea.signal, '')}{idea.symbol}",
        [contract, premium(idea.entry_premium) + est, f"{idea.confidence}%"],
        label_width=11, cell_width=9,
    )
    lines = [
        header,
        f"   {idea.spot:g} → tgt {idea.target:g} · SL {idea.stop_loss:g} "
        f"(1:{idea.rr:g}, {idea.conviction.title()})",
    ]
    if idea.target_premium:
        lines.append(
            f"   sell ₹{premium(idea.target_premium)} · exit ₹{premium(idea.stop_premium)}"
        )
    return lines


def format_stock_scan(result: ScanResult) -> tuple[str, str]:
    """Return (title, body) for the stock scan Pushover notification."""
    ideas = result.ideas

    if not ideas:
        title = "⏸ Stock Options — no setups"
        body = fit_sections(
            [f"No actionable stock setups ({result.scanned} scanned)."],
            [],
            ["⚠️ Estimates, not advice. Options can lose 100%."],
        )
        return title, body

    summary = " · ".join(f"{i.symbol} {i.opt_type}" for i in ideas[:3])
    title = f"📈 Stock Options — {summary}"

    expiry = ideas[0].expiry
    essential = [
        f"Monthly expiry {expiry} · {result.actionable} setups, top {len(ideas)}:",
        row("", ["Opt", "Buy ₹", "Conf"], label_width=11, cell_width=9),
    ]

    optional: list[str] = []
    for idea in ideas:
        optional += _idea_block(idea) + [""]

    footer = [
        "* estimated premium — no live quote" if any(not i.is_live for i in ideas) else "",
        "⚠️ Estimates, not advice. Options can lose 100%.",
    ]
    footer = [line for line in footer if line]
    return title, fit_sections(essential, optional, footer)
