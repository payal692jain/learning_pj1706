"""On-demand analysis of one stock — the answer to "should I buy this?".

Two different questions get asked about the same stock and they have different
answers, so this report gives both rather than blurring them:

  1. DELIVERY  — is this worth owning the shares? Trend, valuation, where it sits
     in its 52-week range, and what event is coming.
  2. INTRADAY  — is there an option trade in it today? The consensus signal, the
     ATM contract, and what to sell at.

A stock can be a poor delivery buy and a fine intraday long, or the reverse, and
collapsing that into one verdict is how a swing view gets traded on a 5-minute
signal.
"""

from nifty_ai_agent.data.fundamentals import StockFundamentals
from nifty_ai_agent.strategies.base import SignalType
from nifty_ai_agent.strategies.stock_scanner import StockIdea

# 52-week position bands. A stock in the top decile of its range is not "cheap"
# however good the P/E looks, and one at the bottom is not "safe" however bad.
_NEAR_HIGH = 0.85
_NEAR_LOW = 0.15


def _fmt(value: float, suffix: str = "", places: int = 1) -> str:
    return f"{value:,.{places}f}{suffix}" if value else "n/a"


def _delivery_verdict(
    fund: StockFundamentals | None, spot: float, signal: SignalType, confidence: int
) -> tuple[str, list[str]]:
    """Return (verdict, reasons) for owning the shares outright.

    Deliberately conservative: the strategy stack is intraday and technical, so
    it is evidence about today's drift, not about whether a business is worth
    owning. A positive verdict therefore needs the trend AND the event calendar
    to agree, and any unknown counts against acting rather than for it.
    """
    reasons: list[str] = []
    positives = negatives = 0

    if signal == SignalType.BUY_CE:
        positives += 1
        reasons.append(f"trend up ({confidence}% consensus)")
    elif signal == SignalType.BUY_PE:
        negatives += 1
        reasons.append(f"trend down ({confidence}% consensus)")
    else:
        reasons.append("trend flat — no directional edge")

    if fund is not None:
        position = fund.position_in_range(spot)
        if position is not None:
            pct = position * 100
            if position >= _NEAR_HIGH:
                negatives += 1
                reasons.append(f"near 52w high ({pct:.0f}% of range)")
            elif position <= _NEAR_LOW:
                reasons.append(f"near 52w low ({pct:.0f}% of range)")
            else:
                reasons.append(f"mid 52w range ({pct:.0f}%)")

        dte = fund.days_to_earnings
        if dte is not None and dte <= 7:
            negatives += 1
            reasons.append(f"results in {dte}d — gap risk")

        if fund.event_tags:
            reasons.append("news: " + ", ".join(fund.event_tags[:3]))

    if negatives:
        verdict = "AVOID FOR NOW" if negatives > positives else "WAIT"
    elif positives:
        verdict = "ACCUMULATE"
    else:
        verdict = "NO EDGE"
    return verdict, reasons


def format_stock_analysis(
    symbol: str,
    spot: float,
    signal: SignalType,
    confidence: int,
    conviction: str,
    reason: str,
    fund: StockFundamentals | None = None,
    idea: StockIdea | None = None,
    volume_ratio: float | None = None,
    blocked_reason: str = "",
) -> str:
    """Render the full on-demand analysis for one stock as plain text."""
    lines = [f"📊 {symbol} — ₹{spot:,.2f}", ""]

    # ── 1. Delivery / cash view ──────────────────────────────────────────────
    verdict, reasons = _delivery_verdict(fund, spot, signal, confidence)
    lines.append(f"BUY THE STOCK?  {verdict}")
    lines += [f"  · {r}" for r in reasons]

    # ── 2. Valuation ─────────────────────────────────────────────────────────
    if fund is not None:
        lines.append("")
        lines.append("VALUATION")
        lines.append(
            f"  P/E {_fmt(fund.pe)} · fwd {_fmt(fund.forward_pe)} · "
            f"P/B {_fmt(fund.price_to_book)}"
        )
        if fund.peg or fund.eps:
            lines.append(f"  PEG {_fmt(fund.peg, places=2)} · EPS {_fmt(fund.eps)}")
        if fund.sector:
            lines.append(f"  {fund.sector}")
        if fund.week52_high and fund.week52_low:
            lines.append(
                f"  52w {fund.week52_low:,.0f}–{fund.week52_high:,.0f}"
            )
        if volume_ratio:
            note = "  ⚠ unusual" if volume_ratio >= 2 else ""
            lines.append(f"  volume {volume_ratio:g}× avg (pace){note}")

    # ── 3. Events ────────────────────────────────────────────────────────────
    if fund is not None:
        dte = fund.days_to_earnings
        if dte is not None or fund.headlines:
            lines.append("")
            lines.append("EVENTS")
        if dte is not None:
            flag = "  ⚠ blackout" if dte <= 3 else ""
            lines.append(f"  Results {fund.next_earnings} (in {dte}d){flag}")
        for h in fund.headlines[:3]:
            tag = f"[{'/'.join(h.tags)}] " if h.tags else ""
            lines.append(f"  · {tag}{h.title[:70]}")

    # ── 4. Intraday option view ──────────────────────────────────────────────
    lines.append("")
    lines.append("INTRADAY OPTION")
    if blocked_reason:
        lines.append(f"  NO TRADE — {blocked_reason}")
    elif idea is not None:
        lines.append(
            f"  {idea.signal}  {idea.strike:g}{idea.opt_type}  exp {idea.expiry}"
        )
        lines.append(
            f"  buy ₹{idea.entry_premium:g}"
            + (f" · sell ₹{idea.target_premium:g} · exit ₹{idea.stop_premium:g}"
               if idea.target_premium else "")
        )
        lines.append(
            f"  underlying tgt {idea.target:g} · SL {idea.stop_loss:g} (1:{idea.rr:g})"
        )
        cost = idea.entry_premium * idea.lot_size
        lines.append(f"  {idea.lot_size:,} qty = ₹{cost:,.0f}/lot")
        if not idea.is_live:
            lines.append("  * premium estimated — no live quote")
    else:
        lines.append(f"  NO TRADE — {conviction} ({confidence}%)")
        if reason:
            lines.append(f"  {reason[:90]}")

    lines.append("")
    lines.append("⚠️ Analysis, not advice. Options can lose 100%.")
    return "\n".join(lines)
