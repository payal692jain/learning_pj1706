"""Upstox token health — detect expiry, alert once, and resume automatically.

What "auto-resume" can and cannot mean here, honestly:

Upstox's OAuth tokens expire nightly (~03:30 IST) and the authorization_code grant
needs a human at a browser. There is no refresh_token to trade in, so a daily token
CANNOT be renewed unattended — any code claiming otherwise would be quietly logging
in as you, which is neither possible nor desirable.

What this module does instead:

  * Detects the dead token the moment a call is rejected, instead of letting the
    agent silently fall back to synthetic VIX-based prices for the rest of the day
    while the notifications keep arriving as if nothing changed.
  * Pushes ONE alert (not one per five-minute cycle) telling you to refresh.
  * Resumes on its own the instant a fresh token lands in .env — get_settings()
    re-reads the file on every call, so no restart is needed. The health check
    notices the token working again and says so.

The permanent fix is the Analytics Access Token (read-only, market-data only, valid
~1 year), which is why the alert names it.
"""

import logging
from dataclasses import dataclass
from datetime import date

from nifty_ai_agent.data.upstox_provider import UpstoxAuthError, UpstoxClient

logger = logging.getLogger(__name__)

_LOGIN_HINT = (
    "Refresh it: run `python scripts/upstox_login.py`, or paste a ~1-year "
    "Analytics Access Token into UPSTOX_ACCESS_TOKEN to stop this recurring."
)


@dataclass
class TokenStatus:
    is_valid: bool
    detail: str

    @property
    def is_missing(self) -> bool:
        return not self.is_valid and "not set" in self.detail


class TokenMonitor:
    """Tracks Upstox token health across cycles and alerts on transitions only.

    Stateful by design: the agent runs a pipeline every five minutes, and a dead
    token would otherwise fire ~75 identical push notifications in a session. The
    alert fires on the EDGE — working→broken, and broken→working — not on the level.
    """

    def __init__(self, notifier=None) -> None:
        self._notifier = notifier
        self._last_valid: bool | None = None
        self._alerted_on: date | None = None

    def check(self, access_token: str) -> TokenStatus:
        """Probe the token with a cheap authenticated call."""
        if not access_token:
            return TokenStatus(False, "UPSTOX_ACCESS_TOKEN is not set")

        try:
            UpstoxClient(access_token).get_expiries("NIFTY")
            return TokenStatus(True, "Upstox token is live")
        except UpstoxAuthError as exc:
            return TokenStatus(False, str(exc))
        except Exception as exc:
            # A network blip is not an expired token — do not cry wolf.
            logger.warning("Token check inconclusive (treating as valid): %s", exc)
            return TokenStatus(True, f"check inconclusive: {exc}")

    def check_and_alert(self, access_token: str) -> TokenStatus:
        """Check the token and push a notification only when its state CHANGES."""
        status = self.check(access_token)
        today = date.today()

        newly_broken = not status.is_valid and self._last_valid is not False
        # Re-alert once a day even if it was already broken yesterday, so a token
        # that expired overnight is flagged each morning rather than never again.
        stale_alert = not status.is_valid and self._alerted_on != today

        if newly_broken or stale_alert:
            self._alerted_on = today
            self._notify(
                "🔑 Upstox token expired",
                f"{status.detail}\n\n"
                f"{_LOGIN_HINT}\n\n"
                "Until then the agent keeps running on estimated option prices — "
                "signals still fire, but premiums are theoretical, not traded.",
            )
        elif status.is_valid and self._last_valid is False:
            self._notify(
                "✅ Upstox token restored",
                "Live option chain data is back — premiums are real again. "
                "No restart was needed.",
            )

        self._last_valid = status.is_valid
        return status

    def _notify(self, title: str, message: str) -> None:
        logger.info("Token monitor: %s", title)
        if not self._notifier:
            return
        try:
            self._notifier.send_text(title=title, message=message)
        except Exception as exc:
            logger.error("Token alert failed to send: %s", exc)
