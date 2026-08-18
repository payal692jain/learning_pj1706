"""One digest covering BSE Ltd stock options and the NSE currency pairs.

Two books that share nothing except being outside the index/NIFTY-50 universe the
rest of the agent watches, delivered as a single notification so they cost one
buzz rather than two.

Premiums differ in kind between the halves and the formatter must not blur that:
BSE Ltd trades in rupees per share (lot 200), while a currency premium is quoted
in paise per unit against a 1000-unit contract. Cost per lot is therefore always
rendered from the contract size, never from the premium alone.
"""

from nifty_ai_agent.reports.layout import fit_sections, premium, row
from nifty_ai_agent.strategies.stock_scanner import ScanResult, StockIdea

_SIDE_ICON = {"BUY_CE": "📈", "BUY_PE": "📉"}

# Currency premiums are small (a USDINR option trades in paise), so the shared
# premium() helper's rupee rounding would collapse them all to "0".
_CURRENCY_PAIRS = {"USDINR", "EURINR", "GBPINR", "JPYINR"}


def _fmt_premium(idea: StockIdea, value: float) -> str:
    """Format *value* at the precision this idea's book actually quotes in."""
    if idea.symbol in _CURRENCY_PAIRS:
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return premium(value)


def _premium_cell(idea: StockIdea) -> str:
    return _fmt_premium(idea, idea.entry_premium)


def _idea_block(idea: StockIdea) -> list[str]:
    """Contract line, then the underlying levels and what one lot actually costs."""
    est = "" if idea.is_live else "*"
    header = row(
        f"{_SIDE_ICON.get(idea.signal, '')}{idea.symbol}",
        [f"{idea.strike:g}{idea.opt_type}", _premium_cell(idea) + est, f"{idea.confidence}%"],
        label_width=11, cell_width=9,
    )
    cost = idea.entry_premium * idea.lot_size
    lines = [
        header,
        f"   {idea.spot:g} → tgt {idea.target:g} · SL {idea.stop_loss:g} "
        f"(1:{idea.rr:g}, {idea.conviction.title()})",
    ]
    if idea.target_premium:
        gain = (idea.target_premium - idea.entry_premium) * idea.lot_size
        lines.append(
            f"   sell {_fmt_premium(idea, idea.target_premium)} · "
            f"exit {_fmt_premium(idea, idea.stop_premium)} (₹{gain:+,.0f}/lot)"
        )
    lines.append(
        f"   {idea.lot_size:,} × {_premium_cell(idea)} = ₹{cost:,.0f}/lot · exp {idea.expiry}"
    )
    return lines


def _watch_line(hold) -> str:
    """One line for a scanned-but-flat underlying.

    These are the whole reason the book stays legible: on a five-name watchlist a
    symbol that simply disappears when the engine is undecided reads as a broken
    scan, not as a considered "no trade".
    """
    return f"   {hold.symbol:<9} no trade ({hold.spot:g})"


def format_bse_currency_scan(
    bse: ScanResult, currency: ScanResult
) -> tuple[str, str]:
    """Return (title, body) for the combined BSE Ltd + currency notification."""
    ideas = list(bse.ideas) + list(currency.ideas)
    holds = list(bse.holds) + list(currency.holds)

    if ideas:
        summary = " · ".join(f"{i.symbol} {i.opt_type}" for i in ideas[:3])
        title = f"💱 BSE + Currency — {summary}"
        essential = [row("", ["Opt", "Buy", "Conf"], label_width=11, cell_width=9)]
    else:
        title = "⏸ BSE + Currency — no setups"
        essential = [
            f"No actionable setups ({bse.scanned + currency.scanned} scanned)."
        ]

    optional: list[str] = []
    for label, result in (("BSE LTD", bse), ("CURRENCY (NSE)", currency)):
        if not (result.ideas or result.holds):
            continue
        optional += [f"── {label} ──"]
        for idea in result.ideas:
            optional += _idea_block(idea) + [""]
        # Watch lines last within a section, so a real idea is never trimmed in
        # favour of a "nothing here" line when the body has to shrink.
        for hold in result.holds:
            optional.append(_watch_line(hold))
        if result.holds:
            optional.append("")

    footer = [
        "* estimated premium — no live quote" if any(not i.is_live for i in ideas) else "",
        "⚠️ Currency ETDs need underlying exposure (RBI).",
        "⚠️ Estimates, not advice. Options can lose 100%.",
    ]
    return title, fit_sections(essential, optional, [f for f in footer if f])
