"""Risk & margin notification — sent every cycle, whatever the signals say.

This is the "can I even take a trade, and what would it cost me" message. It is
deliberately signal-independent: it prices the ATM call AND the ATM put on every
index, plus the future, so the numbers are there before a signal arrives rather
than only alongside one. A HOLD cycle still tells you your margin position.

Every margin figure is a local SPAN estimate (see risk/margin.py) — the broker's
blocked amount will differ, especially when the exchange hikes margins intraday.
"""

import logging
from dataclasses import dataclass

from nifty_ai_agent.reports.layout import PUSHOVER_LIMIT as _PUSHOVER_LIMIT
from nifty_ai_agent.reports.layout import fit_sections as _fit
from nifty_ai_agent.reports.layout import inr as _inr
from nifty_ai_agent.reports.layout import premium as _prem
from nifty_ai_agent.reports.layout import row as _row
from nifty_ai_agent.risk.margin import (
    MarginCalculator,
    MarginRequirement,
    PositionSizing,
    futures_margin,
    option_buy_margin,
    option_sell_margin,
)
from nifty_ai_agent.strategies.option_analyser import (
    ExpiryAnalysis,
    atm_iv,
    estimate_premium_at_spot,
)

logger = logging.getLogger(__name__)


@dataclass
class LegView:
    """One tradeable option leg: what it costs, what it loses, how many lots fit."""
    opt_type: str            # CE / PE
    premium: float
    buy: MarginRequirement
    sell: MarginRequirement
    sizing: PositionSizing

    @property
    def loss_per_lot(self) -> float:
        return self.sizing.loss_per_lot_at_sl


@dataclass
class IndexMarginView:
    index_name: str
    spot: float
    lot_size: int
    strike: int
    sl_points: float
    futures: MarginRequirement
    futures_sizing: PositionSizing
    ce: LegView | None
    pe: LegView | None
    is_live: bool


def build_index_margin_view(
    index_name: str,
    analysis: ExpiryAnalysis,
    atr: float,
    lot_size: int,
    calculator: MarginCalculator,
    atr_sl_multiplier: float = 1.5,
    realised_loss_today: float = 0.0,
) -> IndexMarginView:
    """Price the future and both ATM option legs for *index_name* and size each one.

    The stop-loss distance is the same ATR-derived figure the risk engine uses
    (1.5 × ATR), so the loss-per-lot here matches the SL a signal would carry.
    """
    spot = analysis.spot or 0.0
    sl_points = atr * atr_sl_multiplier

    fut = futures_margin(index_name, spot, lot_size)
    # A future loses one rupee per point per unit — the SL distance IS the loss.
    fut_sizing = calculator.size(fut, sl_points * lot_size, realised_loss_today)

    legs: dict[str, LegView | None] = {"CE": None, "PE": None}
    for opt_type in ("CE", "PE"):
        premium = analysis.atm_ce_ltp if opt_type == "CE" else analysis.atm_pe_ltp
        premium = premium or (
            analysis.theoretical_ce_atm if opt_type == "CE" else analysis.theoretical_pe_atm
        )
        if not premium or premium <= 0 or spot <= 0:
            continue

        # Where the index sits when this leg's stop-loss triggers: down for a
        # call, up for a put.
        sl_spot = spot - sl_points if opt_type == "CE" else spot + sl_points
        premium_at_sl = estimate_premium_at_spot(
            premium, spot, sl_spot, analysis.atm_strike, analysis.expiry,
            atm_iv(analysis, opt_type), opt_type,
        )
        loss_per_lot = max(0.0, premium - premium_at_sl) * lot_size

        buy = option_buy_margin(
            index_name, analysis.atm_strike, opt_type, premium, lot_size, spot,
        )
        sell = option_sell_margin(
            index_name, analysis.atm_strike, opt_type, premium, lot_size, spot,
        )
        legs[opt_type] = LegView(
            opt_type=opt_type,
            premium=premium,
            buy=buy,
            sell=sell,
            sizing=calculator.size(buy, loss_per_lot, realised_loss_today),
        )

    return IndexMarginView(
        index_name=index_name,
        spot=spot,
        lot_size=lot_size,
        strike=analysis.atm_strike,
        sl_points=sl_points,
        futures=fut,
        futures_sizing=fut_sizing,
        ce=legs["CE"],
        pe=legs["PE"],
        is_live=analysis.is_live,
    )


def _short_name(name: str) -> str:
    return {"BANKNIFTY": "BANKNIF"}.get(name, name)[:7]


def format_margin_report(
    views: list[IndexMarginView],
    calculator: MarginCalculator,
    realised_loss_today: float = 0.0,
) -> tuple[str, str]:
    """Return (title, body) for the standalone risk & margin notification.

    The verdict leads: what you can actually take is the answer, and the tables
    below it are the working. If the body would breach Pushover's size limit the
    detail tables are dropped from the bottom up — the verdict and the risk
    disclaimer always survive.
    """
    title = "🛡 Risk & Margin — " + " · ".join(_short_name(v.index_name) for v in views)

    remaining_day = max(0.0, calculator.daily_loss_budget - max(0.0, realised_loss_today))
    header = [
        f"Capital ₹{calculator.capital:,.0f} · "
        f"Risk/trade {calculator.max_risk_per_trade_pct:.0f}% = ₹{calculator.risk_budget:,.0f}",
        f"Day stop {calculator.daily_loss_limit_pct:.0f}% = "
        f"₹{calculator.daily_loss_budget:,.0f} (₹{remaining_day:,.0f} left)",
        "",
    ]

    if not views:
        return title, "\n".join(header + ["No index data this cycle — margins unavailable."])

    verdict = ["── VERDICT ──"]
    for view in views:
        verdict.append(_verdict_line(view))
    verdict.append("")

    footer = [
        f"c=affordable r={calculator.max_risk_per_trade_pct:.0f}%-rule →=take the smaller",
        "⚠️ SPAN estimates, not your broker's blocked amount.",
    ]
    if any(not v.is_live for v in views):
        footer.append("* premiums estimated — no live chain.")

    # The option tables are the primary content — the strategy engine trades options,
    # so those numbers must always survive. Futures margin is comparative context and
    # is the section that gets dropped first if the message runs out of room.
    essential = header + verdict + _spot_table(views) + _options_table(views)
    droppable = _futures_table(views)

    return title, _fit(essential, droppable, footer)


def _spot_table(views: list[IndexMarginView]) -> list[str]:
    return [
        _row("", [_short_name(v.index_name) + ("*" if not v.is_live else "") for v in views]),
        _row("Spot", [f"{v.spot:,.0f}" for v in views]),
        _row("Lot/SLpt", [f"{v.lot_size}/{v.sl_points:,.0f}" for v in views]),
        "",
    ]


def _lots_cell(sizing: PositionSizing) -> str:
    """'295/5/5' — affordable / risk-permitted / recommended, the whole sizing story."""
    return f"{sizing.lots_by_margin}/{sizing.lots_by_risk}/{sizing.lots}"


def _futures_table(views: list[IndexMarginView]) -> list[str]:
    return [
        "── FUTURES (1 lot) ──",
        _row("Margin", [_inr(v.futures.total_per_lot) for v in views]),
        _row("SL loss", [_inr(v.futures_sizing.loss_per_lot_at_sl) for v in views]),
        _row("Lots c/r/→", [_lots_cell(v.futures_sizing) for v in views]),
        "",
    ]


def _options_table(views: list[IndexMarginView]) -> list[str]:
    """CE and PE side by side in one section.

    Deliberately not two separate tables: a risk report that shows the call leg
    and silently omits the put leg (because the message ran out of room) is worse
    than one that shows neither. They live or die together.
    """
    if not any(v.ce or v.pe for v in views):
        return []

    def cell(opt_type: str, getter) -> list[str]:
        legs = [(v.ce if opt_type == "CE" else v.pe) for v in views]
        return [getter(leg) if leg else "-" for leg in legs]

    lines = [
        "── ATM OPTIONS (BUY) ──",
        _row("Strike", [f"{v.strike:,}" if (v.ce or v.pe) else "-" for v in views]),
    ]
    for opt_type in ("CE", "PE"):
        lines += [
            _row(
                f"{opt_type} prm/lot",
                cell(opt_type, lambda l: f"{_prem(l.premium)}/{_inr(l.buy.total_per_lot)}"),
            ),
            _row(f"{opt_type} SL loss", cell(opt_type, lambda l: _inr(l.loss_per_lot))),
            _row(f"{opt_type} lots", cell(opt_type, lambda l: _lots_cell(l.sizing))),
        ]
    lines.append(
        _row("Short mrg", cell("CE", lambda l: _inr(l.sell.total_per_lot)))
    )
    lines.append("")
    return lines


def _verdict_line(view: IndexMarginView) -> str:
    """One line per index: the size it can actually take, or why it can take none."""
    legs = [
        (name, sizing)
        for name, sizing in (
            ("FUT", view.futures_sizing),
            ("CE", view.ce.sizing if view.ce else None),
            ("PE", view.pe.sizing if view.pe else None),
        )
        if sizing is not None
    ]
    allowed = [(name, s) for name, s in legs if s.is_tradeable]

    if not allowed:
        # Every leg is blocked. Prefer an option leg's reason: futures always block
        # on raw margin, which hides the more useful risk-rule explanation.
        reason = next(
            (s.blocked_reason for name, s in legs if name != "FUT" and s.blocked_reason),
            legs[0][1].blocked_reason if legs else "",
        )
        return f"✗ {view.index_name}: 0 lots — {reason}"

    sizes = " · ".join(f"{name} {s.lots}" for name, s in allowed)
    capped_by = allowed[0][1].binding_constraint
    return f"✓ {view.index_name}: {sizes} lot(s) — capped by {capped_by}"
