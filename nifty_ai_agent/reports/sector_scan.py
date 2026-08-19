"""Scan results grouped by sector, with cash-market levels alongside the option.

Two things a flat list of stock ideas does not tell you:

  * **Where the money is moving.** Four bullish calls that are all banks is one
    bet on financials, not four independent ideas. Grouping by sector makes that
    concentration visible before it is discovered by losing on all four at once.
  * **What to do if you do not trade options.** Every idea already carries levels
    on the underlying; they were just never surfaced as a cash trade.

The cash line is deliberately asymmetric. A CE signal maps cleanly onto "buy the
shares" — entry, target, stop. A PE signal does NOT: you cannot buy a stock the
model expects to fall, and retail cash accounts cannot short it either. So a
bearish read renders as exit/avoid with the level that would invalidate it,
never as a buy price. Printing "buy 2,280" under a bearish call would be the
worst kind of wrong: precise, actionable, and backwards.
"""

from dataclasses import dataclass, field

from nifty_ai_agent.reports.layout import fit_sections, premium
from nifty_ai_agent.strategies.stock_scanner import ScanResult, StockIdea

_UNKNOWN_SECTOR = "Other"

# Long yfinance sector names eat the width a phone notification has.
_SHORT_SECTOR = {
    "Financial Services": "Financials",
    "Consumer Cyclical": "Consumer Cyc",
    "Consumer Defensive": "Consumer Def",
    "Communication Services": "Telecom",
    "Basic Materials": "Materials",
}


@dataclass
class SectorView:
    name: str
    ideas: list[StockIdea] = field(default_factory=list)

    @property
    def calls(self) -> int:
        return sum(1 for i in self.ideas if i.opt_type == "CE")

    @property
    def puts(self) -> int:
        return sum(1 for i in self.ideas if i.opt_type == "PE")

    @property
    def lean(self) -> str:
        if self.calls > self.puts:
            return "BULLISH"
        if self.puts > self.calls:
            return "BEARISH"
        return "MIXED"

    @property
    def icon(self) -> str:
        return {"BULLISH": "🟢", "BEARISH": "🔴", "MIXED": "🟡"}[self.lean]

    @property
    def strength(self) -> int:
        """How lopsided the sector is — used to rank, so 4-0 outranks 2-1."""
        return abs(self.calls - self.puts)


def sector_of(idea: StockIdea) -> str:
    """Sector name for *idea*, or "Other" when fundamentals were unavailable."""
    fundamentals = getattr(idea, "fundamentals", None)
    sector = getattr(fundamentals, "sector", "") if fundamentals else ""
    return sector or _UNKNOWN_SECTOR


def group_by_sector(ideas: list[StockIdea]) -> list[SectorView]:
    """Group ideas into sectors, most lopsided first."""
    buckets: dict[str, SectorView] = {}
    for idea in ideas:
        name = sector_of(idea)
        buckets.setdefault(name, SectorView(name)).ideas.append(idea)
    return sorted(
        buckets.values(), key=lambda s: (s.strength, len(s.ideas)), reverse=True,
    )


def cash_line(idea: StockIdea) -> str:
    """The equity trade, as distinct from the option trade.

    A bullish read becomes a buy with a target and a stop. A bearish one becomes
    exit/avoid — there is no honest "buy price" for a stock the model expects to
    fall.
    """
    target = idea.cash_target or idea.target
    stop = idea.cash_stop or idea.stop_loss
    swing = "" if idea.cash_target else " (intraday)"

    if idea.opt_type == "CE":
        return f"   stock: BUY {idea.spot:g} → {target:g} · SL {stop:g}{swing}"
    return f"   stock: AVOID/EXIT · downside {target:g} · invalid {stop:g}{swing}"


def _idea_lines(idea: StockIdea) -> list[str]:
    side = "📈" if idea.opt_type == "CE" else "📉"
    est = "" if idea.is_live else "*"
    return [
        f" {side}{idea.symbol:<11}{idea.strike:g}{idea.opt_type} "
        f"₹{premium(idea.entry_premium)}{est} {idea.confidence}%",
        cash_line(idea),
    ]


def format_sector_scan(result: ScanResult, top_sectors: int = 4) -> tuple[str, str]:
    """Return (title, body) for the sector-grouped view of a scan."""
    if not result.ideas:
        return (
            "🏭 Sectors — no setups",
            fit_sections(
                [f"No actionable stock setups ({result.scanned} scanned)."],
                [], ["⚠️ Not advice. Options can lose 100%."],
            ),
        )

    sectors = group_by_sector(result.ideas)
    bullish = sum(1 for s in sectors if s.lean == "BULLISH")
    bearish = sum(1 for s in sectors if s.lean == "BEARISH")

    lead = sectors[0]
    title = f"🏭 Sectors — {_short(lead.name)} {lead.lean.lower()}"

    essential = [
        f"{len(sectors)} sectors active · {bullish} bullish / {bearish} bearish",
    ]

    optional: list[str] = []
    for sector in sectors[:top_sectors]:
        optional.append("")
        optional.append(
            f"{sector.icon} {_short(sector.name)} · {sector.calls}↑ {sector.puts}↓"
        )
        for idea in sector.ideas:
            optional += _idea_lines(idea)

    footer = [
        "",
        "* estimated premium" if any(not i.is_live for i in result.ideas) else "",
        "⚠️ Not advice. Options can lose 100%.",
    ]
    return title, fit_sections(essential, optional, [f for f in footer if f != ""] or [""])


def _short(sector: str) -> str:
    return _SHORT_SECTOR.get(sector, sector)
