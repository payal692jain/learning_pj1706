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
from nifty_ai_agent.strategies.option_analyser import (
    ExpiryAnalysis,
    OptionLeg,
    atm_iv,
    estimate_premium_at_spot,
)


def _fmt_cell(value: float) -> str:
    """Format a premium for a table cell.

    Two decimals below ₹10 so near-expiry paise premiums (e.g. 0.10) don't
    round to a misleading "0"; 'mkt' when the premium is unknown.
    """
    if not value:
        return "mkt"
    if value < 10:
        return f"{value:.2f}"
    return f"{value:,.0f}"


def _short_expiry(expiry: str) -> str:
    """'14-Jul-2026' → '14-Jul' — the year wastes table width."""
    return expiry[:6] if len(expiry) >= 6 else expiry


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
        return self._send(
            title=title, message=body, priority=priority, sound=sound, monospace=True,
        )

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
        return self._send(
            title=title, message=body, priority=priority, sound=sound, monospace=True,
        )

    def send_text(
        self, title: str, message: str, priority: int = _PRIORITY_NORMAL,
        monospace: bool = False,
    ) -> bool:
        """Send a raw push notification."""
        return self._send(title=title, message=message, priority=priority, monospace=monospace)

    def _send(
        self,
        title: str,
        message: str,
        priority: int = _PRIORITY_NORMAL,
        sound: str = "",
        monospace: bool = False,
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
        if monospace:
            # Fixed-width rendering so the contract tables stay aligned.
            payload["monospace"] = 1
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
    def _format_contract_table(
        signal: Signal,
        risk: RiskParameters,
        option_analysis: ExpiryAnalysis | None,
        monthly_option_analysis: ExpiryAnalysis | None,
        index_name: str,
        strike_step: int,
        expiry_weekday: int,
    ) -> list[str]:
        """Return an aligned Weekly/Monthly table for a BUY_CE/BUY_PE signal.

        Rendered in Pushover's monospace mode, so column alignment holds:

            📌 BUY NIFTY 24200 CE
                       Weekly    Monthly
            Expiry     14-Jul     28-Jul
            Buy ₹         104        269
            Sell ₹        329        459   ← at index target
            Exit ₹         46        195   ← at stop-loss
            ITM CE  24150@135  24150@292
            ITM PE  24250@111  24250@235
        """
        if signal.signal == SignalType.HOLD:
            return []

        opt_type = "CE" if signal.signal == SignalType.BUY_CE else "PE"

        cols: list[tuple[str, ExpiryAnalysis]] = []
        if option_analysis:
            cols.append(("Weekly", option_analysis))
        if monthly_option_analysis:
            cols.append(("Monthly", monthly_option_analysis))
        if not cols:
            # Option chain unavailable — derive ATM from spot (entry price)
            strike = int(round(risk.entry_price / strike_step) * strike_step)
            expiry = _next_expiry(expiry_weekday)
            return [f"📌 Buy: {index_name} {strike} {opt_type}  {expiry}  @ market price"]

        def row(label: str, cells: list[str]) -> str:
            return f"{label:<7}" + "".join(f"{c:>11}" for c in cells)

        entries = [
            (a.atm_ce_ltp or a.theoretical_ce_atm) if opt_type == "CE"
            else (a.atm_pe_ltp or a.theoretical_pe_atm)
            for _, a in cols
        ]

        lines = [
            f"📌 BUY {index_name} {cols[0][1].atm_strike} {opt_type}",
            row("", [label + ("*" if not a.is_live else "") for label, a in cols]),
            row("Expiry", [_short_expiry(a.expiry) for _, a in cols]),
            row("Buy ₹", [_fmt_cell(e) for e in entries]),
        ]

        # Premium to SELL at when the index reaches the risk target, and to
        # EXIT at if it hits the stop-loss — per expiry, since weekly and
        # monthly premiums move differently for the same index move.
        if risk.is_valid:
            sells, exits = [], []
            for (_, a), entry in zip(cols, entries):
                if entry and a.spot > 0:
                    iv = atm_iv(a, opt_type)
                    sells.append(_fmt_cell(estimate_premium_at_spot(
                        entry, a.spot, risk.target, a.atm_strike, a.expiry, iv, opt_type,
                    )))
                    exits.append(_fmt_cell(estimate_premium_at_spot(
                        entry, a.spot, risk.stop_loss, a.atm_strike, a.expiry, iv, opt_type,
                    )))
                else:
                    sells.append("-")
                    exits.append("-")
            lines.append(row("Sell ₹", sells))
            lines.append(row("Exit ₹", exits))

        itm_ce_cells, itm_pe_cells = [], []
        for _, a in cols:
            itm_call, itm_put = _find_itm_legs(a)
            itm_ce_cells.append(
                f"{itm_call.strike}@{itm_call.ce_ltp:.0f}" if itm_call and itm_call.ce_ltp else "-"
            )
            itm_pe_cells.append(
                f"{itm_put.strike}@{itm_put.pe_ltp:.0f}" if itm_put and itm_put.pe_ltp else "-"
            )
        if any(c != "-" for c in itm_ce_cells + itm_pe_cells):
            lines.append(row("ITM CE", itm_ce_cells))
            lines.append(row("ITM PE", itm_pe_cells))

        lines.append("(Sell=at target, Exit=at stop-loss)")
        if any(not a.is_live for _, a in cols):
            lines.append("* estimated — no live chain")

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

        # ── Suggested contract table — weekly + monthly (BUY signals only) ──────
        if signal.signal != SignalType.HOLD:
            lines.extend(PushoverNotifier._format_contract_table(
                signal, risk, option_analysis, monthly_option_analysis,
                index_name, strike_step, expiry_weekday,
            ))
            lines.append("")

        # ── Risk parameters (index levels) ──────────────────────────────────────
        lines.append(f"Strategy: {signal.strategy}")
        if risk.is_valid:
            lines += [
                f"Entry:  {risk.entry_price:>9,.0f}   SL: {risk.stop_loss:>9,.0f}",
                f"Target: {risk.target:>9,.0f}   RR: {'1:' + str(risk.risk_reward_ratio):>9}",
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

            lines.extend(PushoverNotifier._format_contract_table(
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
