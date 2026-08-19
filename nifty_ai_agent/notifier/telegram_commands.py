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
    "📊 ANALYSE\n"
    "/analyse NIFTY — index: direction, votes, levels, ATM contract\n"
    "/analyse RELIANCE — stock: buy the shares? intraday option?\n"
    "   works for NIFTY · BANKNIFTY · SENSEX and any NSE F&O stock\n"
    "\n⚙️ YOUR SETTINGS\n"
    "/setcapital 100000 — deployable capital (₹)\n"
    "/risk 1 — % risked per trade (0–10)\n"
    "/status — show your settings\n"
    "\n📈 MARKET\n"
    "/movers — top 10 gainers and losers (F&O universe)\n"
    "/performance — hit rate of closed calls\n"
    "\n🔔 SUBSCRIPTION\n"
    "/start — subscribe   /stop — unsubscribe\n"
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
    text = (message.get("text") or "").strip()
    command, arg = parse_command(text)
    if not command:
        # Plain text, no slash: if it names a real instrument, just analyse it.
        # Making people remember a command name is the main thing that stops a
        # bot like this from being used.
        if _is_tradeable(text):
            return chat_id, _analyse(text)
        return chat_id, "Send /help to see what I can do, or a symbol like NIFTY."
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

    if command in ("sectors", "sector"):
        return _sectors()

    if command in ("movers", "gainers", "losers", "topmovers"):
        return _movers()

    if command in ("performance", "stats", "score"):
        from nifty_ai_agent.backtesting.scorecard import (
            build_scorecard, format_scorecard,
        )
        return format_scorecard(build_scorecard(store.list_closed_positions()))

    if command in ("analyse", "analyze", "stock"):
        if not arg:
            return (
                "What should I analyse?\n\n"
                "  /analyse NIFTY\n"
                "  /analyse BANKNIFTY\n"
                "  /analyse RELIANCE\n\n"
                "Any of NIFTY · BANKNIFTY · SENSEX, or an NSE F&O stock symbol."
            )
        return _analyse(arg)

    if command in ("help", "commands"):
        return HELP

    # A bare symbol is what people type first ("/nifty", "/reliance"), so treat it
    # as an analysis request — but only if it actually resolves to a tradeable
    # instrument. Guessing from shape alone would send every mistyped command
    # ("/frobnicate") through a slow network analysis to reach a confusing error.
    if _is_tradeable(command):
        return _analyse(command)

    return "Unknown command. Send /help, or a symbol like NIFTY or RELIANCE."


def _sectors() -> str:
    """Stock scan grouped by sector. Lazily imported like the other heavy commands."""
    from datetime import datetime

    import pytz

    from nifty_ai_agent.config import get_settings
    from nifty_ai_agent.data.fundamentals import fetch_fundamentals
    from nifty_ai_agent.data.instrument_master import get_instrument_master
    from nifty_ai_agent.data.nifty50_stocks import NIFTY50_SYMBOLS
    from nifty_ai_agent.data.stock_data import fetch_stock_histories
    from nifty_ai_agent.reports.sector_scan import format_sector_scan
    from nifty_ai_agent.risk.calculator import RiskCalculator
    from nifty_ai_agent.strategies.stock_scanner import scan_stocks

    settings = get_settings()
    lookup = None
    if settings.upstox_access_token:
        from nifty_ai_agent.data.upstox_provider import UpstoxClient
        lookup = UpstoxClient(settings.upstox_access_token).get_ltp
    try:
        master = get_instrument_master()
        histories, spots = fetch_stock_histories(
            NIFTY50_SYMBOLS, period=f"{settings.historical_days}d", interval=settings.data_interval,
            upstox_token=settings.upstox_access_token, master=master,
        )
        if not histories:
            return "No stock data available right now."
        result = scan_stocks(
            histories, spots, master=master,
            risk_calculator=RiskCalculator(
                max_risk_pct=settings.max_risk_per_trade_pct,
                min_rr=settings.min_risk_reward_ratio,
                atr_sl_multiplier=settings.atr_sl_multiplier,
            ),
            now=datetime.now(pytz.timezone("Asia/Kolkata")).time(),
            premium_lookup=lookup or (lambda keys: {}),
            default_iv=settings.stock_default_iv, top_n=12,
            fundamentals=fetch_fundamentals(list(histories)),
            earnings_blackout_days=settings.earnings_blackout_days,
            trend_tf=settings.stock_trend_timeframe,
        )
    except Exception as exc:
        logger.error("Sectors command failed: %s", exc)
        return f"Could not build the sector view ({type(exc).__name__})."
    return format_sector_scan(result)[1]


def _movers() -> str:
    """Top gainers and losers. Imported lazily, like the other heavy commands."""
    from nifty_ai_agent.config import get_settings
    from nifty_ai_agent.data.instrument_master import get_instrument_master
    from nifty_ai_agent.data.market_movers import fetch_market_movers
    from nifty_ai_agent.reports.movers import format_movers

    settings = get_settings()
    if not settings.upstox_access_token:
        return "Movers need an Upstox token — quotes are an authenticated endpoint."
    try:
        snapshot = fetch_market_movers(
            settings.upstox_access_token, get_instrument_master(),
            top_n=settings.movers_top_n,
        )
    except Exception as exc:
        logger.error("Movers command failed: %s", exc)
        return f"Could not fetch movers ({type(exc).__name__}). Try again shortly."
    return format_movers(snapshot, top_n=settings.movers_top_n)[1]


def _is_tradeable(word: str) -> bool:
    """True when *word* names an index or a live NSE equity.

    Resolved rather than pattern-matched, so a typo'd command is reported as an
    unknown command instead of being sent through a network analysis that fails
    slowly and confusingly. Index aliases are checked first because they are an
    in-memory dict; the instrument master is only consulted for the rest, and it
    is process-cached after its first load.
    """
    candidate = " ".join(word.strip().split())
    if not candidate or len(candidate) > 20:
        return False

    from nifty_ai_agent.reports.analyse_index import resolve_index

    if resolve_index(candidate):
        return True
    if not candidate.replace(".", "").isalnum():
        return False

    try:
        from nifty_ai_agent.data.instrument_master import get_instrument_master

        bare = candidate.upper().removesuffix(".NS")
        return get_instrument_master().resolve_equity(bare) is not None
    except Exception as exc:
        # A master we cannot read should not turn every word into a symbol.
        logger.debug("Symbol check failed for %r: %s", word, exc)
        return False


def _analyse(symbol: str) -> str:
    """Run the on-demand analysis for *symbol* — index or stock.

    Indices are checked first: they have their own path (live option chain, no
    fundamentals) and "NIFTY" would otherwise be looked up as an equity ticker
    and come back as "no price data".

    Imported lazily: this pulls in pandas, yfinance and the whole strategy stack,
    which the command parser has no other reason to load — and which would make
    every unit test of this module pay for them.
    """
    from nifty_ai_agent.config import get_settings
    from nifty_ai_agent.reports.analyse_index import analyse_index, resolve_index

    settings = get_settings()

    index = resolve_index(symbol)
    if index:
        return analyse_index(index, settings)

    from nifty_ai_agent.reports.analyse_one import analyse_stock

    lookup = None
    if settings.upstox_access_token:
        from nifty_ai_agent.data.upstox_provider import UpstoxClient
        client = UpstoxClient(settings.upstox_access_token)
        lookup = client.get_ltp
    return analyse_stock(symbol, settings, premium_lookup=lookup)


def _parse_positive(arg: str) -> float | None:
    """Parse a strictly-positive number, tolerating commas ('1,00,000'). None if invalid."""
    try:
        value = float(arg.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None
    return value if value > 0 else None
