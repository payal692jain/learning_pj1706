"""Risk management — stop-loss, target, and RR ratio calculations."""

import logging
from dataclasses import dataclass

from nifty_ai_agent.strategies.base import SignalType

logger = logging.getLogger(__name__)


@dataclass
class RiskParameters:
    signal: SignalType
    entry_price: float
    stop_loss: float
    target: float
    risk_reward_ratio: float
    risk_amount: float       # absolute points at risk
    risk_pct: float          # risk as % of entry price
    is_valid: bool           # meets minimum RR requirement
    rejection_reason: str = ""


class RiskCalculator:
    """Calculate trade risk parameters for a given signal and ATR.

    Uses ATR to set a natural stop-loss distance (1.5× ATR by default).
    Target is sized so that RR >= min_rr.
    """

    def __init__(
        self,
        max_risk_pct: float = 1.0,
        daily_loss_limit_pct: float = 3.0,
        min_rr: float = 2.0,
        atr_sl_multiplier: float = 1.5,
    ) -> None:
        self.max_risk_pct = max_risk_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.min_rr = min_rr
        self.atr_sl_multiplier = atr_sl_multiplier

    def calculate(
        self,
        signal: SignalType,
        entry_price: float,
        atr: float,
    ) -> RiskParameters:
        """Compute SL, target, and RR for *signal* at *entry_price*.

        Args:
            signal: BUY_CE, BUY_PE, or HOLD.
            entry_price: Current NIFTY spot price.
            atr: Current ATR value.

        Returns:
            RiskParameters dataclass.
        """
        if signal == SignalType.HOLD:
            return RiskParameters(
                signal=signal,
                entry_price=entry_price,
                stop_loss=0.0,
                target=0.0,
                risk_reward_ratio=0.0,
                risk_amount=0.0,
                risk_pct=0.0,
                is_valid=False,
                rejection_reason="HOLD signal — no trade.",
            )

        sl_distance = atr * self.atr_sl_multiplier
        target_distance = sl_distance * self.min_rr

        if signal == SignalType.BUY_CE:
            stop_loss = entry_price - sl_distance
            target = entry_price + target_distance
        else:  # BUY_PE
            stop_loss = entry_price + sl_distance
            target = entry_price - target_distance

        risk_amount = abs(entry_price - stop_loss)
        risk_pct = (risk_amount / entry_price) * 100
        rr = round(target_distance / sl_distance, 2) if sl_distance > 0 else 0.0

        is_valid = rr >= self.min_rr and risk_pct <= self.max_risk_pct
        rejection_reason = ""
        if rr < self.min_rr:
            rejection_reason = f"RR {rr:.1f} below minimum {self.min_rr}"
        elif risk_pct > self.max_risk_pct:
            rejection_reason = (
                f"Risk {risk_pct:.2f}% exceeds max {self.max_risk_pct}%"
            )

        params = RiskParameters(
            signal=signal,
            entry_price=round(entry_price, 2),
            stop_loss=round(stop_loss, 2),
            target=round(target, 2),
            risk_reward_ratio=rr,
            risk_amount=round(risk_amount, 2),
            risk_pct=round(risk_pct, 4),
            is_valid=is_valid,
            rejection_reason=rejection_reason,
        )

        logger.info(
            "Risk | signal=%s entry=%.2f SL=%.2f target=%.2f RR=%.2f valid=%s",
            signal.value, entry_price, stop_loss, target, rr, is_valid,
        )
        return params
