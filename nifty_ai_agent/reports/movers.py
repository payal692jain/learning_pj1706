"""Top gainers and losers digest.

Pushover rejects a body over 1024 characters outright, and twenty movers with
prices does not fit, so the list degrades by dropping rows from the bottom of
each half rather than truncating mid-line. The extremes are what a movers list
is for: the tenth gainer matters less than the first.
"""

from nifty_ai_agent.data.market_movers import Mover, MoversSnapshot
from nifty_ai_agent.reports.layout import fit_sections


def _row(mover: Mover) -> str:
    # The sign has to live inside the field width, not before it — formatting it
    # separately leaves "+  7.94%" with the sign floating away from its number.
    return (
        f"{mover.symbol:<12}{mover.change_pct:>+7.2f}%  {mover.last_price:>9,.1f}"
    )


def format_movers(snapshot: MoversSnapshot, top_n: int = 10) -> tuple[str, str]:
    """Return (title, body) for the gainers/losers notification."""
    if not snapshot.gainers and not snapshot.losers:
        return (
            "📊 Market Movers — unavailable",
            fit_sections(
                ["No quote data available for the F&O universe right now."],
                [], ["⚠️ Data only, not advice."],
            ),
        )

    lead = snapshot.gainers[0] if snapshot.gainers else None
    drag = snapshot.losers[0] if snapshot.losers else None
    headline = " · ".join(
        f"{m.symbol} {m.change_pct:+.1f}%" for m in (lead, drag) if m
    )
    title = f"📊 Movers — {headline}"

    essential = [
        f"F&O universe · {snapshot.scanned} stocks · "
        f"{snapshot.advances}↑ / {snapshot.declines}↓ (A/D {snapshot.advance_decline_ratio})",
    ]

    optional: list[str] = []
    if snapshot.gainers:
        optional.append("")
        optional.append("🟢 TOP GAINERS")
        optional += [_row(m) for m in snapshot.gainers[:top_n]]
    if snapshot.losers:
        optional.append("")
        optional.append("🔴 TOP LOSERS")
        optional += [_row(m) for m in snapshot.losers[:top_n]]

    return title, fit_sections(essential, optional, ["", "⚠️ Data only, not advice."])
