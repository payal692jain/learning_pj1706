"""The next-session outlook — what GIFT Nifty says about tomorrow's open.

Sent twice, timed to the two moments GIFT actually tells you something new:

    17:00 IST  Session 2 has opened (16:35). First read on tomorrow, priced
               before the US session even starts.
    06:45 IST  Session 1 has opened (06:30). The final pre-open read, with
               Wall Street's whole day now in the price and 2h30m left to plan.

The message deliberately pairs the forecast with its base rate. "GIFT implies a
0.6% gap up" is only half an answer; the half that decides the trade is what NIFTY
has historically DONE after opening 0.6% up — and the answer is often "faded",
because a big gap up frequently opens at the day's high.
"""

import logging

from nifty_ai_agent.data.gift_nifty import OpenOutlook
from nifty_ai_agent.reports.layout import fit_sections
from nifty_ai_agent.strategies.gap_analyser import GapStats, PivotLevels

logger = logging.getLogger(__name__)

_GAP_ICON = {"GAP_UP": "🟢", "GAP_DOWN": "🔴", "FLAT": "⚪"}

_SESSION_LABEL = {
    "SESSION_1": "Session 1 (06:30–15:40) — final pre-open read",
    "SESSION_2": "Session 2 (16:35–02:45) — overnight read",
    "CLOSED": "GIFT closed — last traded price",
}


def format_next_session(
    outlook: OpenOutlook,
    stats: GapStats,
    pivots: PivotLevels | None = None,
) -> tuple[str, str]:
    """Return (title, body) for the next-session outlook notification."""
    gift = outlook.gift
    icon = _GAP_ICON.get(outlook.direction, "")
    arrow = "+" if outlook.gap_points > 0 else ""

    title = (
        f"{icon} NIFTY tomorrow: {outlook.direction.replace('_', ' ').title()} "
        f"{arrow}{outlook.gap_points:,.0f} pts"
    )

    essential = [
        f"GIFT {gift.price:,.1f} ({gift.change_pct:+.2f}%) · {gift.expiry[:6]}",
        _SESSION_LABEL.get(gift.session, gift.session),
        "",
        f"Prev close   {outlook.nifty_prev_close:>9,.0f}",
        f"Implied open {outlook.implied_open:>9,.0f}",
        f"Gap          {arrow + format(outlook.gap_points, ',.0f'):>9} "
        f"({outlook.gap_pct:+.2f}%)",
        "",
        "── WHAT THIS GAP USUALLY DOES ──",
        stats.verdict,
        "",
    ]

    optional: list[str] = []
    if stats.is_reliable:
        optional += [
            f"Continued {stats.continued}/{stats.sample} · faded {stats.faded}/{stats.sample}",
            f"Median day range {stats.median_day_range_pct:.2f}% · "
            f"avg close-vs-open {stats.avg_close_vs_open_pct:+.2f}%",
            "",
        ]

    if pivots:
        optional += [
            "── LEVELS FOR TOMORROW ──",
            f"R2 {pivots.r2:>9,.0f}   S1 {pivots.s1:>9,.0f}",
            f"R1 {pivots.r1:>9,.0f}   S2 {pivots.s2:>9,.0f}",
            f"PP {pivots.pivot:>9,.0f}",
            f"Open lands {pivots.context_for(outlook.implied_open)}",
            "",
        ]

    footer = [
        "⚠️ A gap forecast is not a signal — the 09:15 open is where the",
        "strategies get their first real bar. Base rates describe the past.",
    ]
    if gift.timestamp:
        footer.insert(0, f"GIFT as of {gift.timestamp}")

    return title, fit_sections(essential, optional, footer)
