"""Margin engine — what a futures or option position actually costs to hold,
and how many lots the capital and the risk rules jointly allow.

Margin is computed locally (no broker call): futures and short options use a
SPAN + exposure estimate as a percentage of contract notional, long options use
the premium debit, which IS the whole margin for a buyer.

The percentages below are estimates calibrated to typical exchange SPAN for
index F&O. They move with volatility — the exchange raises them in stressed
markets — so every number this module produces is a planning figure, not the
broker's final blocked amount. Everything that surfaces it must say so.

The load-bearing output is not the margin itself but *lots*: the smaller of what
the capital can fund and what the risk rule permits. Those two disagree often,
and when the risk rule says zero the honest answer is "this trade does not fit
this account" — see PositionSizing.blocked_reason.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarginRates:
    """SPAN and exposure margin as a percentage of contract notional."""
    span_pct: float
    exposure_pct: float

    @property
    def total_pct(self) -> float:
        return self.span_pct + self.exposure_pct


# Typical exchange SPAN + exposure for index F&O. BANKNIFTY runs richer than
# NIFTY because it is the more volatile underlying; SENSEX sits between them.
INDEX_MARGIN_RATES: dict[str, MarginRates] = {
    "NIFTY":     MarginRates(span_pct=10.5, exposure_pct=2.0),
    "BANKNIFTY": MarginRates(span_pct=12.0, exposure_pct=3.0),
    "SENSEX":    MarginRates(span_pct=11.0, exposure_pct=2.5),
}
_DEFAULT_RATES = MarginRates(span_pct=12.0, exposure_pct=3.0)

# A short option can never be margined below this share of notional, however far
# out of the money it is — mirrors the exchange's minimum-margin floor and stops
# the OTM discount below from returning an absurdly small (or negative) number.
_SHORT_OPTION_MARGIN_FLOOR_PCT = 5.0


def margin_rates(index_name: str) -> MarginRates:
    """SPAN/exposure rates for *index_name*, defaulting to the conservative pair."""
    return INDEX_MARGIN_RATES.get(index_name.upper(), _DEFAULT_RATES)


def compact_inr(value: float) -> str:
    """Indian-notation money for a narrow cell: 6760 → '6.8k', 340080 → '3.40L'."""
    if abs(value) >= 100_000:
        return f"{value / 100_000:.2f}L"
    if abs(value) >= 1_000:
        return f"{value / 1000:.1f}k"
    return f"{value:,.0f}"


@dataclass
class MarginRequirement:
    """What one lot of a specific instrument costs to put on."""
    instrument: str          # "NIFTY FUT" / "NIFTY 24200 CE"
    kind: str                # "FUTURES" / "OPTION_BUY" / "OPTION_SELL"
    lot_size: int
    notional_per_lot: float  # contract value — the exposure the position controls
    span: float              # per lot
    exposure: float          # per lot
    premium: float           # per lot — debit for a buy, credit for a sell
    basis: str               # human-readable explanation of how it was derived

    @property
    def total_per_lot(self) -> float:
        """Cash blocked to hold one lot."""
        return self.span + self.exposure + max(0.0, self.premium)

    @property
    def leverage(self) -> float:
        """Notional controlled per rupee of margin — the number that flatters, then kills."""
        if self.total_per_lot <= 0:
            return 0.0
        return self.notional_per_lot / self.total_per_lot


def futures_margin(index_name: str, spot: float, lot_size: int) -> MarginRequirement:
    """Margin to hold one lot of the near-month future on *index_name*."""
    rates = margin_rates(index_name)
    notional = spot * lot_size
    return MarginRequirement(
        instrument=f"{index_name} FUT",
        kind="FUTURES",
        lot_size=lot_size,
        notional_per_lot=notional,
        span=notional * rates.span_pct / 100,
        exposure=notional * rates.exposure_pct / 100,
        premium=0.0,
        basis=f"SPAN {rates.span_pct:.1f}% + exposure {rates.exposure_pct:.1f}% of notional (est.)",
    )


def option_buy_margin(
    index_name: str, strike: int, opt_type: str, premium: float,
    lot_size: int, spot: float,
) -> MarginRequirement:
    """Cash to buy one lot of an option — the premium, in full, and nothing else.

    A long option carries no SPAN: the most a buyer can lose is the premium, so
    the exchange has nothing further to collateralise.
    """
    return MarginRequirement(
        instrument=f"{index_name} {strike} {opt_type}",
        kind="OPTION_BUY",
        lot_size=lot_size,
        notional_per_lot=spot * lot_size,
        span=0.0,
        exposure=0.0,
        premium=premium * lot_size,
        basis=f"premium debit ₹{premium:,.2f} × {lot_size}",
    )


def option_sell_margin(
    index_name: str, strike: int, opt_type: str, premium: float,
    lot_size: int, spot: float,
) -> MarginRequirement:
    """Margin to short one lot of an option — SPAN on notional, discounted for OTM distance.

    Approximates the exchange formula: start from the futures-equivalent margin,
    subtract the amount by which the strike is out of the money (that distance is
    cushion the exchange gives back), and floor the result. The premium received
    is a credit, not a debit, so it does not add to the blocked amount.
    """
    rates = margin_rates(index_name)
    notional = spot * lot_size

    otm_points = (strike - spot) if opt_type.upper() == "CE" else (spot - strike)
    otm_credit = max(0.0, otm_points) * lot_size

    gross = notional * rates.total_pct / 100
    floor = notional * _SHORT_OPTION_MARGIN_FLOOR_PCT / 100
    blocked = max(floor, gross - otm_credit)

    # Report it split in the same SPAN/exposure shape as a future so the two are
    # directly comparable in the notification table.
    exposure = notional * rates.exposure_pct / 100
    span = max(0.0, blocked - exposure)

    return MarginRequirement(
        instrument=f"{index_name} {strike} {opt_type}",
        kind="OPTION_SELL",
        lot_size=lot_size,
        notional_per_lot=notional,
        span=span,
        exposure=exposure,
        premium=-premium * lot_size,  # credit received
        basis=(
            f"SPAN+exposure {rates.total_pct:.1f}% of notional less "
            f"₹{otm_credit:,.0f} OTM cushion, floored at "
            f"{_SHORT_OPTION_MARGIN_FLOOR_PCT:.0f}% (est.)"
        ),
    )


@dataclass
class PositionSizing:
    """How many lots to actually take, and why that number and not a bigger one."""
    requirement: MarginRequirement
    capital: float
    risk_budget: float             # rupees allowed to lose on this trade
    loss_per_lot_at_sl: float      # rupees lost per lot if the stop-loss hits
    lots_by_margin: int            # what the capital can fund
    lots_by_risk: int              # what the risk-per-trade rule permits
    lots: int                      # the recommendation — the smaller of the two
    margin_used: float
    margin_free: float
    risk_at_sl: float              # rupees at risk across the recommended lots
    blocked_reason: str = ""       # non-empty when lots == 0

    @property
    def is_tradeable(self) -> bool:
        return self.lots > 0

    @property
    def margin_utilisation_pct(self) -> float:
        return (self.margin_used / self.capital * 100) if self.capital > 0 else 0.0

    @property
    def risk_pct_of_capital(self) -> float:
        return (self.risk_at_sl / self.capital * 100) if self.capital > 0 else 0.0

    @property
    def binding_constraint(self) -> str:
        """Which rule capped the size — the thing to fix if you want more lots."""
        if self.lots == 0:
            return "BLOCKED"
        if self.lots_by_risk <= self.lots_by_margin:
            return "RISK"
        return "MARGIN"


class MarginCalculator:
    """Sizes positions against capital, the per-trade risk cap, and the daily loss limit."""

    def __init__(
        self,
        capital: float,
        max_risk_per_trade_pct: float = 1.0,
        daily_loss_limit_pct: float = 3.0,
        max_margin_utilisation_pct: float = 100.0,
    ) -> None:
        self.capital = capital
        self.max_risk_per_trade_pct = max_risk_per_trade_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_margin_utilisation_pct = max_margin_utilisation_pct

    @property
    def risk_budget(self) -> float:
        """Rupees allowed to lose on a single trade."""
        return self.capital * self.max_risk_per_trade_pct / 100

    @property
    def daily_loss_budget(self) -> float:
        """Rupees allowed to lose across the whole day before trading stops."""
        return self.capital * self.daily_loss_limit_pct / 100

    @property
    def deployable_capital(self) -> float:
        """Capital available for margin after the utilisation cap."""
        return self.capital * self.max_margin_utilisation_pct / 100

    def min_capital_for_risk_rule(self, loss_per_lot_at_sl: float) -> float:
        """Capital needed before even ONE lot fits inside the per-trade risk cap.

        The number that explains most "0 lots" verdicts: at a 1% risk cap, a lot
        that loses ₹3,770 at its stop-loss needs a ₹377,000 account to be legal.
        """
        if self.max_risk_per_trade_pct <= 0:
            return 0.0
        return loss_per_lot_at_sl * 100 / self.max_risk_per_trade_pct

    def size(
        self,
        requirement: MarginRequirement,
        loss_per_lot_at_sl: float,
        realised_loss_today: float = 0.0,
    ) -> PositionSizing:
        """Recommend a lot count for *requirement* given its stop-loss loss per lot.

        Args:
            requirement: The instrument's per-lot margin.
            loss_per_lot_at_sl: Rupees lost on one lot if the stop-loss is hit.
                For a long option that is (entry premium − premium at SL) × lot size.
            realised_loss_today: Losses already taken today, as a positive number.
                Once these reach the daily limit, sizing returns zero lots.

        Returns:
            PositionSizing — lots, margin, risk, and the reason if the answer is zero.
        """
        per_lot = requirement.total_per_lot
        lots_by_margin = int(self.deployable_capital // per_lot) if per_lot > 0 else 0
        lots_by_risk = (
            int(self.risk_budget // loss_per_lot_at_sl) if loss_per_lot_at_sl > 0 else lots_by_margin
        )

        remaining_day_budget = max(0.0, self.daily_loss_budget - max(0.0, realised_loss_today))
        lots_by_day_limit = (
            int(remaining_day_budget // loss_per_lot_at_sl)
            if loss_per_lot_at_sl > 0 else lots_by_margin
        )

        lots = max(0, min(lots_by_margin, lots_by_risk, lots_by_day_limit))

        # Reason precedence matters: a lot that breaches the per-trade risk cap will
        # usually breach the (larger) day budget too, but the risk cap is the rule
        # that actually bit. Only blame the day limit when risk alone would have allowed a lot.
        # Kept terse: these strings are rendered inside a Pushover notification with a
        # hard 1024-character budget, and three indices each carry one.
        blocked_reason = ""
        if lots == 0:
            if lots_by_margin == 0:
                blocked_reason = (
                    f"1 lot needs ₹{compact_inr(per_lot)} margin vs "
                    f"₹{compact_inr(self.capital)} capital"
                )
            elif lots_by_risk == 0:
                needed = self.min_capital_for_risk_rule(loss_per_lot_at_sl)
                blocked_reason = (
                    f"risks ₹{compact_inr(loss_per_lot_at_sl)}/lot at SL vs a "
                    f"₹{compact_inr(self.risk_budget)} risk cap "
                    f"({self.max_risk_per_trade_pct:.0f}%); needs ₹{compact_inr(needed)}"
                )
            else:
                blocked_reason = (
                    f"daily loss limit — only ₹{compact_inr(remaining_day_budget)} of the "
                    f"₹{compact_inr(self.daily_loss_budget)} day budget left"
                )

        margin_used = lots * per_lot
        sizing = PositionSizing(
            requirement=requirement,
            capital=self.capital,
            risk_budget=self.risk_budget,
            loss_per_lot_at_sl=loss_per_lot_at_sl,
            lots_by_margin=lots_by_margin,
            lots_by_risk=lots_by_risk,
            lots=lots,
            margin_used=margin_used,
            margin_free=self.capital - margin_used,
            risk_at_sl=lots * loss_per_lot_at_sl,
            blocked_reason=blocked_reason,
        )

        logger.info(
            "Margin | %s per_lot=₹%.0f lots(margin=%d risk=%d day=%d) → %d  %s",
            requirement.instrument, per_lot, lots_by_margin, lots_by_risk,
            lots_by_day_limit, lots, blocked_reason or "OK",
        )
        return sizing
