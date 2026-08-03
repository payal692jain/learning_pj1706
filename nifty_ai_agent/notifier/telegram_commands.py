"""Telegram command router — turns an incoming update into a reply.

Kept separate from the HTTP client and the polling loop so the whole command
surface is a pure, unit-testable transform: (update, subscriber store) → reply
text, with the store the only side effect. The store is duck-typed (anything with
the DatabaseRepository subscriber methods) so tests can pass a fake.
"""

import logging

logger = logging.getLogger(__name__)

_MAX_RISK_PCT = 10.0  # a per-trade risk above this is almost certainly a typo

WELCOME = (
    "✅ Subscribed to NIFTY/SENSEX/BANKNIFTY option signals.\n\n"
    "You'll get the consensus call each cycle during market hours, plus the "
    "stock and straddle scans. Set your sizing:\n"
    "• /setcapital 100000 — deployable capital\n"
    "• /risk 1 — % risked per trade\n"
    "• /status — your settings   • /stop — unsubscribe\n\n"
    "⚠️ Educational signals only, not investment advice. Options can lose 100%."
)

HELP = (
    "Commands:\n"
    "/start — subscribe\n"
    "/stop — unsubscribe\n"
    "/setcapital 100000 — set deployable capital (₹)\n"
    "/risk 1 — set % risked per trade (0–10)\n"
    "/status — show your settings\n"
    "/help — this message"
)


def parse_command(text: str) -> tuple[str, str]:
    """'/setcapital 100000' → ('setcapital', '100000'); non-commands → ('', '').

    Strips a trailing bot @mention ('/start@MyBot' → 'start') so the bot behaves
    the same in groups, where Telegram appends the mention.
    """
    text = (text or "").strip()
    if not text.startswith("/"):
        return "", ""
    parts = text[1:].split(maxsplit=1)
    command = parts[0].split("@", 1)[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""
    return command, arg


def handle_update(update: dict, store) -> tuple[int, str] | None:
    """Turn one Telegram update into *(chat_id, reply_text)*, or None to ignore.

    Ignores anything without a message + chat id (edited messages, channel posts,
    callback queries — none of which the subscribe flow needs).
    """
    message = update.get("message") or {}
    chat = message.get("chat") or {}
    chat_id = chat.get("id")
    if chat_id is None:
        return None

    username = chat.get("username") or chat.get("first_name") or ""
    command, arg = parse_command(message.get("text", ""))
    if not command:
        return chat_id, "Send /help to see what I can do."
    return chat_id, _dispatch(command, arg, chat_id, username, store)


def _dispatch(command: str, arg: str, chat_id: int, username: str, store) -> str:
    if command in ("start", "subscribe"):
        store.add_or_reactivate_subscriber(chat_id, username)
        return WELCOME

    if command in ("stop", "unsubscribe"):
        store.deactivate_subscriber(chat_id)
        return "Unsubscribed. Send /start any time to resume."

    if command == "setcapital":
        value = _parse_positive(arg)
        if value is None:
            return "Usage: /setcapital 100000"
        store.update_subscriber_capital(chat_id, value)
        return f"Capital set to ₹{value:,.0f}."

    if command == "risk":
        value = _parse_positive(arg)
        if value is None or value > _MAX_RISK_PCT:
            return f"Usage: /risk 1   (percent per trade, 0–{_MAX_RISK_PCT:g})"
        store.update_subscriber_risk(chat_id, value)
        return f"Risk per trade set to {value:g}%."

    if command == "status":
        sub = store.get_subscriber(chat_id)
        if sub is None or not sub.active:
            return "You're not subscribed. Send /start to begin."
        return f"Subscribed. Capital ₹{sub.capital:,.0f}, risk {sub.risk_pct:g}% per trade."

    if command in ("help", "commands"):
        return HELP

    return "Unknown command. Send /help."


def _parse_positive(arg: str) -> float | None:
    """Parse a strictly-positive number, tolerating commas ('1,00,000'). None if invalid."""
    try:
        value = float(arg.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None
    return value if value > 0 else None
