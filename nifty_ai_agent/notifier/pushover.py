"""Pushover notifier — sends iPhone/Android push notifications via Pushover API.

Sign up at https://pushover.net to get:
  - PUSHOVER_USER_KEY  (your user key)
  - PUSHOVER_API_TOKEN (your app token — create one app per project)
"""

import logging
import time
from datetime import date, timedelta

import requests

from nifty_ai_agent.risk.calculator import RiskParameters
from nifty_ai_agent.strategies.base import Signal, SignalType
from nifty_ai_agent.strategies.option_analyser import ExpiryAnalysis, OptionLeg


def _format_price(ltp: float) -> str:
    """Format a premium as '@ ₹X' ('@ ₹X.XX' below ₹10 to avoid a misleading
    '@ ₹0' on near-expiry paise-level prices), or '@ market price' if unknown."""
    if not ltp:
        return "@ market price"
    if ltp < 10:
        return f"@ ₹{ltp:.2f}"
    return f"@ ₹{ltp:.0f}"


def _find_itm_legs(analysis: ExpiryAnalysis) -> tuple[OptionLeg | None, OptionLeg | None]:
    """Return (itm_call_leg, itm_put_leg) — one strike deeper in-the-money than
    the ATM strike already shown in the main contract line.

    Anchored on the ATM *strike* (not spot) so these are always distinct from
    the ATM line above them — if anchored on spot instead, the ATM strike
    itself is sometimes already slightly ITM, making the "ITM" line a
    redundant repeat of the ATM one.

    ITM call = nearest strike below ATM. ITM put = nearest strike above ATM.
    Returns (None, None) when there's no real strike data to draw from (e.g.
    the VIX-based synthetic estimate has no strikes at all).
    """
    if not analysis.legs:
        return None, None
    atm = analysis.atm_strike
    below = [leg for leg in analysis.legs if leg.strike < atm]
    above = [leg for leg in analysis.legs if leg.strike > atm]
    itm_call = max(below, key=lambda l: l.strike) if below else None
    itm_put = min(above, key=lambda l: l.strike) if above else None
    return itm_call, itm_put


def _next_expiry(weekday: int) -> str:
    """Return nearest upcoming *weekday* (0=Mon … 6=Sun) as 'DD-Mon-YYYY'.

    NIFTY weekly expiry: Tuesday (weekday=1)
    SENSEX weekly expiry: Thursday (weekday=3)
    """
    today = date.today()
    days_ahead = (weekday - today.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return (today + timedelta(days=days_ahead)).strftime("%d-%b-%Y")

logger = logging.getLogger(__name__)

_PUSHOVER_URL = "https://api.pushover.net/1/messages.json"
_RETRY_COUNT = 3
_RETRY_DELAY = 2

# Pushover priority levels
_PRIORITY_HIGH = 1    # bypasses quiet hours, requires acknowledgement
_PRIORITY_NORMAL = 0
_PRIORITY_LOW = -1    # no sound/vibration


class PushoverNotifier:
    """Send Pushover push notifications to your iPhone lock screen."""

    def __init__(self, user_key: str, api_token: str) -> None:
        self._user_key = user_key
        self._api_token = api_token

    def send_signal(
        self,
        signal: Signal,
        risk: RiskParameters,
        ai_explanation: str = "",
        option_analysis: ExpiryAnalysis | None = None,
        prediction: bool = False,
        index_name: str = "NIFTY",
        strike_step: int = 50,
        expiry_weekday: int = 1,  # 1=Tuesday (NIFTY), 3=Thursday (SENSEX)
        monthly_option_analysis: ExpiryAnalysis | None = None,
    ) -> bool:
        """Send a formatted signal alert. Returns True on success."""
        title, body, priority = self._format_signal(
            signal, risk, ai_explanation, option_analysis, prediction,
            index_name, strike_step, expiry_weekday, monthly_option_analysis,
        )
        # EOD predictions and BUY/PE signals play sound.
        # Intraday HOLD is shown silently (lock-screen banner, no noise).
        sound = "none" if (not prediction and signal.signal == SignalType.HOLD) else ""
        return self._send(title=title, message=body, priority=priority, sound=sound)

    def send_multi_signal(
        self,
        results: list[tuple[Signal, RiskParameters, str]],
        option_analysis: ExpiryAnalysis | None = None,
        prediction: bool = False,
        index_name: str = "NIFTY",
        strike_step: int = 50,
        expiry_weekday: int = 1,
        monthly_option_analysis: ExpiryAnalysis | None = None,
    ) -> bool:
        """Send one notification listing every strategy's prediction for this cycle.

        *results* is a list of (signal, risk, ai_explanation) tuples, one per
        strategy that ran this cycle. Returns True on success.
        """
        title, body, priority = self._format_multi_signal(
            results, option_analysis, prediction, index_name, strike_step, expiry_weekday,
            monthly_option_analysis,
        )
        any_actionable = any(signal.signal != SignalType.HOLD for signal, _, _ in results)
        sound = "none" if (not prediction and not any_actionable) else ""
        return self._send(title=title, message=body, priority=priority, sound=sound)

    def send_text(self, title: str, message: str, priority: int = _PRIORITY_NORMAL) -> bool:
        """Send a raw push notification."""
        return self._send(title=title, message=message, priority=priority)

    def _send(
        self,
        title: str,
        message: str,
        priority: int = _PRIORITY_NORMAL,
        sound: str = "",
    ) -> bool:
        payload: dict = {
            "token": self._api_token,
            "user": self._user_key,
            "title": title,
            "message": message,
            "priority": priority,
        }
        if sound:
            payload["sound"] = sound
        # High-priority messages require retry/expire params
        if priority == _PRIORITY_HIGH:
            payload["retry"] = 60
            payload["expire"] = 3600

        for attempt in range(1, _RETRY_COUNT + 1):
            try:
                resp = requests.post(_PUSHOVER_URL, data=payload, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                if data.get("status") != 1:
                    raise ValueError(f"Pushover API error: {data.get('errors')}")
                logger.info("Pushover notification sent (attempt %d)", attempt)
                return True
            except (requests.RequestException, ValueError) as exc:
                logger.warning(
                    "Pushover attempt %d/%d failed: %s", attempt, _RETRY_COUNT, exc
                )
                if attempt < _RETRY_COUNT:
                    time.sleep(_RETRY_DELAY)

        logger.error("All Pushover attempts failed.")
        return False

    @staticmethod
    def _format_contract_lines(
        signal: Signal,
        risk: RiskParameters,
        option_analysis: ExpiryAnalysis | None,
        monthly_option_analysis: ExpiryAnalysis | None,
        index_name: str,
        strike_step: int,
        expiry_weekday: int,
    ) -> list[str]:
        """Return the '📌 Buy' line(s) for a BUY_CE/BUY_PE signal.

        Shows the weekly contract, and the monthly contract alongside it
        whenever a monthly option chain is available, so the two premiums
        are never confused for one another. Each also gets a follow-up line
        with the nearest in-the-money call and put strikes, when real strike
        data is available (not the VIX-based synthetic estimate).
        """
        if signal.signal == SignalType.HOLD:
            return []

        opt_type = "CE" if signal.signal == SignalType.BUY_CE else "PE"

        def _lines_for(analysis: ExpiryAnalysis, label: str) -> list[str]:
            ltp = (
                (analysis.atm_ce_ltp or analysis.theoretical_ce_atm)
                if signal.signal == SignalType.BUY_CE
                else (analysis.atm_pe_ltp or analysis.theoretical_pe_atm)
            )
            tag = label if analysis.is_live else f"{label}, Est."
            result = [
                f"📌 Buy ({tag}): {index_name} {analysis.atm_strike} {opt_type}  "
                f"{analysis.expiry}  {_format_price(ltp)}"
            ]

            itm_call, itm_put = _find_itm_legs(analysis)
            itm_parts = []
            if itm_call and itm_call.ce_ltp:
                itm_parts.append(f"ITM CE {itm_call.strike} {_format_price(itm_call.ce_ltp)}")
            if itm_put and itm_put.pe_ltp:
                itm_parts.append(f"ITM PE {itm_put.strike} {_format_price(itm_put.pe_ltp)}")
            if itm_parts:
                result.append("   " + "  |  ".join(itm_parts))

            return result

        lines: list[str] = []
        if option_analysis:
            lines.extend(_lines_for(option_analysis, "Weekly"))
        else:
            # Option chain unavailable — derive ATM from spot (entry price)
            strike = int(round(risk.entry_price / strike_step) * strike_step)
            expiry = _next_expiry(expiry_weekday)
            lines.append(f"📌 Buy (Weekly): {index_name} {strike} {opt_type}  {expiry}  @ market price")

        if monthly_option_analysis:
            lines.extend(_lines_for(monthly_option_analysis, "Monthly"))

        return lines

    @staticmethod
    def _format_signal(
        signal: Signal,
        risk: RiskParameters,
        ai_explanation: str,
        option_analysis: ExpiryAnalysis | None = None,
        prediction: bool = False,
        index_name: str = "NIFTY",
        strike_step: int = 50,
        expiry_weekday: int = 1,
        monthly_option_analysis: ExpiryAnalysis | None = None,
    ) -> tuple[str, str, int]:
        """Return (title, body, priority) for Pushover."""
        icons = {
            SignalType.BUY_CE: "📈",
            SignalType.BUY_PE: "📉",
            SignalType.HOLD: "⏸",
        }
        icon = icons.get(signal.signal, "")
        if prediction:
            title = f"📊 PREDICTION | {index_name} {signal.signal.value} — {signal.confidence}%"
        else:
            title = f"{icon} {index_name} {signal.signal.value} — {signal.confidence}%"

        lines: list[str] = []

        if prediction:
            lines.append("⚠️ Market closed — based on last session's closing data.")
            lines.append("Outlook for next trading session:\n")

        # ── Suggested contract(s) — weekly + monthly (always shown for BUY signals) ──
        if signal.signal != SignalType.HOLD:
            lines.extend(PushoverNotifier._format_contract_lines(
                signal, risk, option_analysis, monthly_option_analysis,
                index_name, strike_step, expiry_weekday,
            ))
            lines.append("")

        # ── Risk parameters ────────────────────────────────────────────────────
        lines.append(f"Strategy: {signal.strategy}")
        if risk.is_valid:
            lines += [
                f"Entry:  {risk.entry_price:,.0f}",
                f"SL:     {risk.stop_loss:,.0f}",
                f"Target: {risk.target:,.0f}",
                f"RR:     1:{risk.risk_reward_ratio}",
            ]
        else:
            lines.append(f"[No trade — {risk.rejection_reason or 'HOLD'}]")

        lines.append(f"\n{signal.reason}")

        if ai_explanation:
            lines.append(f"\n{ai_explanation}")

        body = "\n".join(lines)

        # All signals use normal priority so they appear on the lock screen.
        # The caller controls sound via the `sound` payload (HOLD intraday = silent).
        priority = _PRIORITY_NORMAL
        return title, body, priority

    @staticmethod
    def _format_multi_signal(
        results: list[tuple[Signal, RiskParameters, str]],
        option_analysis: ExpiryAnalysis | None = None,
        prediction: bool = False,
        index_name: str = "NIFTY",
        strike_step: int = 50,
        expiry_weekday: int = 1,
        monthly_option_analysis: ExpiryAnalysis | None = None,
    ) -> tuple[str, str, int]:
        """Return (title, body, priority) listing every strategy's prediction."""
        icons = {
            SignalType.BUY_CE: "📈",
            SignalType.BUY_PE: "📉",
            SignalType.HOLD: "⏸",
        }

        summary = " | ".join(
            f"{signal.strategy}: {signal.signal.value} {signal.confidence}%"
            for signal, _, _ in results
        )
        prefix = "📊 PREDICTION" if prediction else "🔔"
        title = f"{prefix} {index_name} — {summary}"

        lines: list[str] = []
        if prediction:
            lines.append("⚠️ Market closed — based on last session's closing data.")
            lines.append("Outlook for next trading session:\n")

        for i, (signal, risk, ai_explanation) in enumerate(results):
            icon = icons.get(signal.signal, "")
            lines.append(f"{icon} {signal.strategy} — {signal.signal.value} ({signal.confidence}%)")

            lines.extend(PushoverNotifier._format_contract_lines(
                signal, risk, option_analysis, monthly_option_analysis,
                index_name, strike_step, expiry_weekday,
            ))

            if risk.is_valid:
                lines.append(
                    f"Entry {risk.entry_price:,.0f}  SL {risk.stop_loss:,.0f}  "
                    f"Target {risk.target:,.0f}  RR 1:{risk.risk_reward_ratio}"
                )
            else:
                lines.append(f"[No trade — {risk.rejection_reason or 'HOLD'}]")

            lines.append(signal.reason)
            if ai_explanation:
                lines.append(ai_explanation)

            if i < len(results) - 1:
                lines.append("─" * 24)

        body = "\n".join(lines)
        priority = _PRIORITY_NORMAL
        return title, body, priority
