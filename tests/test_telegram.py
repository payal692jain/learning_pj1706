"""Tests for the Telegram broadcast bot — notifier, broadcast, commands, store."""

import pytest

from nifty_ai_agent.database.repository import DatabaseRepository
from nifty_ai_agent.notifier import telegram as tg
from nifty_ai_agent.notifier.telegram import (
    TelegramBlockedError,
    TelegramNotifier,
    broadcast,
)
from nifty_ai_agent.notifier.telegram_commands import (
    HELP,
    WELCOME,
    handle_update,
    parse_command,
)


# ── Fakes ───────────────────────────────────────────────────────────────────────

class _Resp:
    def __init__(self, status_code=200, ok=True, description=None, result=None):
        self.status_code = status_code
        self._payload = {"ok": ok, "description": description, "result": result or []}

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def _update(text, chat_id=42, username="alice"):
    return {"message": {"chat": {"id": chat_id, "username": username}, "text": text}}


# ── TelegramNotifier ─────────────────────────────────────────────────────────────

class TestTelegramNotifier:
    def test_send_message_success(self, monkeypatch):
        captured = {}

        def _post(url, data, timeout):
            captured["url"] = url
            captured["data"] = data
            return _Resp(ok=True)

        monkeypatch.setattr(tg.requests, "post", _post)
        assert TelegramNotifier("TOK").send_message(42, "hi") is True
        assert captured["data"]["chat_id"] == 42
        assert captured["data"]["text"] == "hi"
        assert "sendMessage" in captured["url"] and "TOK" in captured["url"]

    def test_monospace_wraps_in_pre_and_escapes(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            tg.requests, "post",
            lambda url, data, timeout: captured.update(data) or _Resp(ok=True),
        )
        TelegramNotifier("TOK").send_message(42, "a < b & c", monospace=True)
        assert captured["parse_mode"] == "HTML"
        assert captured["text"] == "<pre>a &lt; b &amp; c</pre>"

    def test_silent_sets_disable_notification(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            tg.requests, "post",
            lambda url, data, timeout: captured.update(data) or _Resp(ok=True),
        )
        TelegramNotifier("TOK").send_message(42, "heartbeat", silent=True)
        assert captured.get("disable_notification") is True

    def test_default_is_not_silent(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            tg.requests, "post",
            lambda url, data, timeout: captured.update(data) or _Resp(ok=True),
        )
        TelegramNotifier("TOK").send_message(42, "buzz")
        assert "disable_notification" not in captured

    def test_blocked_raises(self, monkeypatch):
        monkeypatch.setattr(tg.requests, "post", lambda *a, **k: _Resp(status_code=403))
        with pytest.raises(TelegramBlockedError):
            TelegramNotifier("TOK").send_message(42, "hi")

    def test_retries_then_returns_false(self, monkeypatch):
        calls = {"n": 0}

        def _post(*a, **k):
            calls["n"] += 1
            return _Resp(ok=False, description="boom")

        monkeypatch.setattr(tg.requests, "post", _post)
        monkeypatch.setattr(tg.time, "sleep", lambda *_: None)  # no real backoff
        assert TelegramNotifier("TOK").send_message(42, "hi") is False
        assert calls["n"] == 3

    def test_get_updates_returns_results(self, monkeypatch):
        monkeypatch.setattr(
            tg.requests, "get",
            lambda url, params, timeout: _Resp(result=[{"update_id": 1}]),
        )
        assert TelegramNotifier("TOK").get_updates(offset=5) == [{"update_id": 1}]


# ── broadcast ────────────────────────────────────────────────────────────────────

class TestBroadcast:
    def test_fans_out_and_counts_delivered(self, monkeypatch):
        sent = []
        monkeypatch.setattr(tg.requests, "post", lambda url, data, timeout: sent.append(data["chat_id"]) or _Resp(ok=True))
        delivered, blocked = broadcast(TelegramNotifier("TOK"), [1, 2, 3], "signal")
        assert delivered == 3 and blocked == []
        assert sent == [1, 2, 3]

    def test_collects_blocked_chats(self, monkeypatch):
        def _post(url, data, timeout):
            return _Resp(status_code=403) if data["chat_id"] == 2 else _Resp(ok=True)

        monkeypatch.setattr(tg.requests, "post", _post)
        delivered, blocked = broadcast(TelegramNotifier("TOK"), [1, 2, 3], "signal")
        assert delivered == 2 and blocked == [2]


# ── parse_command ────────────────────────────────────────────────────────────────

class TestParseCommand:
    @pytest.mark.parametrize("text,expected", [
        ("/start", ("start", "")),
        ("/setcapital 100000", ("setcapital", "100000")),
        ("/start@MyIndexBot", ("start", "")),
        ("/RISK 1.5", ("risk", "1.5")),
        ("hello there", ("", "")),
        ("", ("", "")),
    ])
    def test_parse(self, text, expected):
        assert parse_command(text) == expected


# ── handle_update against a real (temp) repository ───────────────────────────────

@pytest.fixture
def store(tmp_path):
    return DatabaseRepository(f"sqlite:///{tmp_path}/subs.db")


class TestHandleUpdate:
    def test_start_subscribes(self, store):
        chat_id, reply = handle_update(_update("/start"), store)
        assert chat_id == 42
        assert reply == WELCOME
        assert [s.chat_id for s in store.list_active_subscribers()] == [42]

    def test_stop_deactivates(self, store):
        handle_update(_update("/start"), store)
        _, reply = handle_update(_update("/stop"), store)
        assert "Unsubscribed" in reply
        assert store.list_active_subscribers() == []

    def test_setcapital_and_risk_persist(self, store):
        handle_update(_update("/start"), store)
        handle_update(_update("/setcapital 1,00,000"), store)
        handle_update(_update("/risk 2"), store)
        sub = store.get_subscriber(42)
        assert sub.capital == pytest.approx(100000.0)
        assert sub.risk_pct == pytest.approx(2.0)

    def test_risk_rejects_out_of_range(self, store):
        handle_update(_update("/start"), store)
        _, reply = handle_update(_update("/risk 50"), store)
        assert "Usage" in reply
        assert store.get_subscriber(42).risk_pct == pytest.approx(1.0)  # unchanged default

    def test_setcapital_rejects_garbage(self, store):
        handle_update(_update("/start"), store)
        _, reply = handle_update(_update("/setcapital abc"), store)
        assert "Usage" in reply

    def test_status_reports_settings(self, store):
        handle_update(_update("/start"), store)
        _, reply = handle_update(_update("/status"), store)
        assert "Subscribed" in reply and "50,000" in reply

    def test_help_and_unknown(self, store):
        _, reply = handle_update(_update("/help"), store)
        assert reply == HELP
        _, reply = handle_update(_update("/frobnicate"), store)
        assert "Unknown command" in reply

    def test_reactivate_keeps_single_row(self, store):
        handle_update(_update("/start"), store)
        handle_update(_update("/stop"), store)
        handle_update(_update("/start"), store)
        assert len(store.list_active_subscribers()) == 1

    def test_non_message_update_ignored(self, store):
        assert handle_update({"edited_message": {}}, store) is None
