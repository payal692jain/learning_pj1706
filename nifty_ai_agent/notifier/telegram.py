"""Telegram Bot API notifier — broadcasts signals to many subscribers.

Where Pushover targets one personal device, a Telegram bot fans a single message
out to every subscriber's chat, which is how the agent serves multiple users off
one backend and one market-data token. Create a bot via @BotFather to obtain the
TELEGRAM_BOT_TOKEN.

This is a thin HTTP client (like the Pushover notifier) — no bot framework, so it
stays dependency-light and unit-testable by mocking `requests`.
"""

import logging
import time
from collections.abc import Iterable

import requests

logger = logging.getLogger(__name__)

_API_BASE = "https://api.telegram.org/bot"
_RETRY_COUNT = 3
_RETRY_DELAY = 2  # seconds
_TIMEOUT = 15
# Telegram rejects a message body over 4096 characters.
TELEGRAM_LIMIT = 4096


class TelegramBlockedError(Exception):
    """Raised when a chat has blocked/deleted the bot (HTTP 403) — the caller
    should deactivate that subscriber rather than retry a dead chat."""


class TelegramNotifier:
    """Sends messages and reads updates via the Telegram Bot HTTP API."""

    def __init__(self, bot_token: str) -> None:
        self._token = bot_token

    def _url(self, method: str) -> str:
        return f"{_API_BASE}{self._token}/{method}"

    def send_message(
        self, chat_id: int | str, text: str, *, monospace: bool = False, silent: bool = False,
    ) -> bool:
        """Send *text* to one chat. Returns True on success, False on a soft failure.

        *monospace* wraps the body in an HTML <pre> block so the option tables stay
        aligned, mirroring the Pushover monospace behaviour. *silent* delivers without
        a notification sound (a HOLD heartbeat lands in the chat without buzzing).
        Raises TelegramBlockedError when the chat blocked the bot, so the broadcaster
        can prune that subscriber.
        """
        if monospace:
            text = f"<pre>{_html_escape(text)}</pre>"
        payload = {
            "chat_id": chat_id,
            "text": text[:TELEGRAM_LIMIT],
            "disable_web_page_preview": True,
        }
        if monospace:
            payload["parse_mode"] = "HTML"
        if silent:
            payload["disable_notification"] = True

        for attempt in range(1, _RETRY_COUNT + 1):
            try:
                resp = requests.post(self._url("sendMessage"), data=payload, timeout=_TIMEOUT)
                if resp.status_code == 403:
                    raise TelegramBlockedError(f"chat {chat_id} blocked the bot")
                resp.raise_for_status()
                if not resp.json().get("ok"):
                    raise ValueError(resp.json().get("description"))
                return True
            except TelegramBlockedError:
                raise
            except (requests.RequestException, ValueError) as exc:
                logger.warning(
                    "Telegram send attempt %d/%d to %s failed: %s",
                    attempt, _RETRY_COUNT, chat_id, exc,
                )
                if attempt < _RETRY_COUNT:
                    time.sleep(_RETRY_DELAY)
        return False

    def get_updates(self, offset: int | None = None, timeout: int = 25) -> list[dict]:
        """Long-poll for new updates. Returns the raw update dicts (may be empty)."""
        params: dict = {"timeout": timeout}
        if offset is not None:
            params["offset"] = offset
        try:
            resp = requests.get(self._url("getUpdates"), params=params, timeout=timeout + 10)
            resp.raise_for_status()
            return resp.json().get("result", [])
        except (requests.RequestException, ValueError) as exc:
            logger.warning("Telegram getUpdates failed: %s", exc)
            return []


def broadcast(
    notifier: TelegramNotifier,
    chat_ids: Iterable[int],
    text: str,
    *,
    monospace: bool = True,
    silent: bool = False,
) -> tuple[int, list[int]]:
    """Send *text* to every chat in *chat_ids*.

    Returns *(delivered, blocked)* — the count sent successfully and the chat ids
    that blocked the bot, so the caller can deactivate those subscribers. Fanning
    out one precomputed message is the whole point: an index signal is identical
    for every user, so it is built once and delivered many times. *silent* delivers
    without a buzz (used for the HOLD heartbeat).
    """
    delivered = 0
    blocked: list[int] = []
    for cid in chat_ids:
        try:
            if notifier.send_message(cid, text, monospace=monospace, silent=silent):
                delivered += 1
        except TelegramBlockedError:
            blocked.append(cid)
    return delivered, blocked


def _html_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
