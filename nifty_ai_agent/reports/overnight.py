"""Overnight market analysis — one pre-open digest of the global backdrop.

Sent at 07:45, ahead of the fuller 08:00 morning report. Consolidates how the
overnight/global markets moved (US, Asia, Europe, India VIX) with GIFT Nifty's
implied open and the historical base rate for that gap size — the single "here's
the overnight setup" read to wake up to, rather than the multi-message 08:00 brief.
"""

from nifty_ai_agent.data.gift_nifty import OpenOutlook
from nifty_ai_agent.data.market_context import IndexSnapshot
from nifty_ai_agent.data.news_fetcher import NewsItem, format_news_for_notification
from nifty_ai_agent.reports.layout import fit_sections
from nifty_ai_agent.strategies.gap_analyser import GapStats

_GAP_ICON = {"GAP_UP": "🟢", "GAP_DOWN": "🔴", "FLAT": "⚪"}


def _pct(by: dict[str, IndexSnapshot], name: str) -> str:
    snap = by.get(name)
    return f"{snap.change_pct:+.2f}%" if snap else "n/a"


def format_overnight_analysis(
    indices: list[IndexSnapshot],
    global_bias: str,
    outlook: OpenOutlook | None,
    stats: GapStats | None,
    news: list[NewsItem],
) -> tuple[str, str]:
    """Return (title, body) for the overnight analysis notification.

    Indices are rendered grouped (US / Asia / Europe / VIX) to stay compact; the
    implied-open block and news are added when available. Everything composes
    through fit_sections so the body respects the Pushover length limit.
    """
    by = {s.name: s for s in indices}

    if outlook is not None:
        icon = _GAP_ICON.get(outlook.direction, "🌍")
        arrow = "+" if outlook.gap_points > 0 else ""
        title = f"{icon} Overnight: {global_bias} · NIFTY {arrow}{outlook.gap_points:,.0f} pts"
    else:
        title = f"🌍 Overnight: {global_bias}"

    essential = [
        f"Global bias: {global_bias}",
        f"US    S&P {_pct(by, 'S&P 500')} · Dow {_pct(by, 'Dow Jones')} · Nasdaq {_pct(by, 'NASDAQ')}",
        f"Asia  Nikkei {_pct(by, 'Nikkei 225')} · HangSeng {_pct(by, 'Hang Seng')}",
        f"Euro  FTSE {_pct(by, 'FTSE 100')} · DAX {_pct(by, 'DAX')}",
    ]
    vix = by.get("India VIX")
    if vix is not None:
        essential.append(f"India VIX {vix.price:,.2f} ({vix.change_pct:+.2f}%)")

    if outlook is not None:
        gift = outlook.gift
        essential += [
            "",
            f"GIFT {gift.price:,.0f} ({gift.change_pct:+.2f}%)",
            f"Implied open {outlook.implied_open:,.0f} ({outlook.gap_pct:+.2f}%)",
        ]
        if stats is not None:
            essential.append(f"Usually: {stats.verdict}")

    optional: list[str] = []
    if news:
        optional += ["", "── OVERNIGHT HEADLINES ──", format_news_for_notification(news, limit=3)]

    footer = [
        "",
        "⚠️ Overnight backdrop, not a signal — the 09:15 open is the first real bar.",
    ]
    return title, fit_sections(essential, optional, footer)
