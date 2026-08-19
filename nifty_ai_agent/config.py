"""Application configuration — loaded from environment / .env file."""

import logging
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Anthropic ──────────────────────────────────────────────
    anthropic_api_key: str = Field(default="", description="Anthropic Claude API key (optional)")
    claude_model: str = Field(default="claude-opus-4-8", description="Claude model ID")

    # ── Pushover ───────────────────────────────────────────────
    pushover_enabled: bool = Field(
        default=True,
        description=(
            "Send notifications to Pushover. Set false to deliver via Telegram only "
            "(e.g. once the Telegram bot is your delivery channel)."
        ),
    )
    pushover_user_key: str = Field(default="", description="Pushover user key (optional if disabled)")
    pushover_api_token: str = Field(default="", description="Pushover application API token (optional if disabled)")

    # ── Telegram broadcast bot (multi-user delivery) ───────────
    telegram_bot_token: str = Field(
        default="",
        description=(
            "Telegram Bot API token from @BotFather. When set, the agent runs a "
            "broadcast bot: users /start it to subscribe and every signal is fanned "
            "out to all active subscribers. Leave blank to stay single-user (Pushover)."
        ),
    )

    # ── Database ───────────────────────────────────────────────
    database_url: str = Field(
        default="sqlite:///nifty_ai_agent.db",
        description="SQLAlchemy database URL",
    )

    # ── Upstox (live weekly + monthly option chain data) ────────
    upstox_api_key: str = Field(default="", description="Upstox developer app API key")
    upstox_api_secret: str = Field(default="", description="Upstox developer app API secret")
    upstox_redirect_uri: str = Field(default="", description="Upstox OAuth redirect URI")
    upstox_access_token: str = Field(
        default="",
        description=(
            "Upstox access token — a long-lived (~1yr) Analytics Access Token, "
            "or a daily token from scripts/upstox_login.py"
        ),
    )

    # ── Market data ────────────────────────────────────────────
    nifty_symbol: str = Field(default="^NSEI", description="yfinance NIFTY symbol")
    sensex_symbol: str = Field(default="^BSESN", description="yfinance SENSEX symbol")
    banknifty_symbol: str = Field(default="^NSEBANK", description="yfinance BANKNIFTY symbol")
    data_fetch_interval_minutes: int = Field(
        default=5, description="How often to run the signal loop"
    )
    historical_days: int = Field(
        default=30,
        description=(
            "Days of OHLC history to fetch. Must be enough for the SLOWEST "
            "timeframe read, not just the base bars: at 10 days a 5-minute fetch "
            "resamples to only ~46 hourly bars, below the 60 needed for EMA50, so "
            "the 60m trend filter silently reported 'no read' and confirmed "
            "nothing. 30 days yields ~150 hourly bars"
        ),
    )
    data_interval: str = Field(
        default="5m",
        description="yfinance bar interval: 5m for live intraday, 1d for EOD/backtesting",
    )

    allow_expiry_day_options: bool = Field(
        default=False,
        description=(
            "Allow suggesting the contract that expires today. Off by default: on expiry "
            "day the extrinsic value has collapsed and gamma/theta dominate, so a "
            "directional signal with an SL and a 1:2 RR target rarely survives. "
            "Turn on only if you deliberately trade expiry-day scalps."
        ),
    )

    # ── Capital ────────────────────────────────────────────────
    trading_capital: float = Field(
        default=50000.0,
        description=(
            "Deployable capital in INR — sizes the margin/lots section of the signal "
            "notification. The trade plan itself is prices-only and ignores this."
        ),
    )

    # ── Notification filtering ─────────────────────────────────
    min_signal_confidence: int = Field(
        default=0,
        description=(
            "Only notify when a signal's confidence is at least this %. Gates the "
            "intraday trade call (including the EOD prediction), the trade plan, and "
            "the stock scan — lower-conviction signals are still recorded, just not "
            "pushed. 0 (default) notifies on every actionable signal, no filtering."
        ),
    )

    compact_notifications: bool = Field(
        default=True,
        description=(
            "Trim HOLD heartbeats to the decision alone, dropping the strategy "
            "vote table, global cues and bank-constituent ideas. Actionable "
            "messages — a BUY, or a SELL on an open position — always keep the "
            "full detail regardless, because those are the ones you might trade "
            "from. Set false to send everything in full"
        ),
    )
    notify_interval_minutes: int = Field(
        default=15,
        description=(
            "Minimum minutes between ROUTINE index signal notifications. The "
            "pipeline still runs every data_fetch_interval_minutes so it sees "
            "the market at full speed — this only governs how often an "
            "unchanged read is repeated. Sudden changes bypass it entirely"
        ),
    )
    notify_on_move_pct: float = Field(
        default=0.35,
        description=(
            "Percent the index must travel since the LAST alert to force an "
            "immediate notification, regardless of the interval"
        ),
    )
    notify_on_confidence_jump: int = Field(
        default=15,
        description=(
            "Confidence points, up or down, that force an immediate "
            "notification regardless of the interval"
        ),
    )

    # ── Risk ───────────────────────────────────────────────────
    max_risk_per_trade_pct: float = Field(
        default=1.0, description="Maximum risk per trade as % of capital"
    )
    daily_loss_limit_pct: float = Field(
        default=3.0, description="Daily loss limit as % of capital"
    )
    min_risk_reward_ratio: float = Field(
        default=2.0, description="Minimum acceptable RR ratio"
    )
    atr_sl_multiplier: float = Field(
        default=1.5,
        description="Stop-loss distance as a multiple of ATR — shared by the risk and margin engines",
    )

    # ── Next-session outlook (GIFT Nifty) ──────────────────────
    gap_history_days: int = Field(
        default=400,
        description=(
            "Calendar days of NIFTY daily bars used to compute gap base rates "
            "(~275 trading sessions — enough to fill the rarer large-gap buckets)"
        ),
    )

    # ── Margin ─────────────────────────────────────────────────
    max_margin_utilisation_pct: float = Field(
        default=100.0,
        description="Share of capital that may be committed as margin (below 100 keeps a buffer)",
    )

    # ── Stock option scan (NIFTY 50 constituents) ──────────────
    stock_scan_enabled: bool = Field(
        default=True,
        description="Run the scheduled single-stock monthly-option scan across NIFTY 50 constituents",
    )
    stock_scan_interval_minutes: int = Field(
        default=30,
        description=(
            "How often (minutes) to run the single-stock CE/PE scan during market "
            "hours. Runs are market-hours guarded, so this only fires 09:15–15:30."
        ),
    )
    stock_scan_top_n: int = Field(
        default=5, description="How many top stock ideas to include in the scan notification"
    )
    stock_default_iv: float = Field(
        default=0.30,
        description=(
            "Annualised IV (decimal) for the Black-Scholes premium fallback when no "
            "live stock-option quote is available — stocks run hotter than the index, "
            "so this defaults above the index VIX"
        ),
    )

    earnings_blackout_days: int = Field(
        default=3,
        description=(
            "Refuse a new single-stock option entry when quarterly results are "
            "this many calendar days away or fewer. Long premium into a print is "
            "a different trade: the gap can run through any stop, and the "
            "post-result IV collapse drains value even when direction is right. "
            "Set 0 to disable the guard"
        ),
    )
    stock_trend_timeframe: str = Field(
        default="60m",
        description=(
            "Slower timeframe a single-stock entry must agree with before it "
            "becomes actionable. Backtested over 12 symbols, 30m entries alone "
            "returned +0.021% expectancy and 30m confirmed by 60m returned "
            "+0.031% on about half the trades. Empty string disables the filter"
        ),
    )
    stock_news_limit: int = Field(
        default=4, description="Headlines fetched per stock for event detection"
    )

    # ── Top gainers / losers digest ─────────────────────────────
    movers_enabled: bool = Field(
        default=True,
        description="Send a top gainers/losers digest across the F&O universe",
    )
    movers_interval_minutes: int = Field(
        default=60, description="Minutes between gainers/losers digests"
    )
    movers_top_n: int = Field(
        default=10, description="How many gainers and losers to list"
    )

    # ── BSE Ltd + NSE currency scan ─────────────────────────────
    bse_currency_scan_enabled: bool = Field(
        default=True,
        description="Scan BSE Ltd stock options and the NSE currency pairs",
    )
    bse_currency_scan_interval_minutes: int = Field(
        default=30, description="Minutes between BSE Ltd + currency scans"
    )
    currency_default_iv: float = Field(
        default=0.06,
        description=(
            "Annualised IV (decimal) for the Black-Scholes premium fallback on "
            "currency options — INR pairs realise far lower vol than equities, so "
            "the equity default would overstate these premiums several-fold"
        ),
    )

    # ── Volatile-stock straddle scan (long-volatility plays) ────
    volatility_scan_enabled: bool = Field(
        default=True,
        description=(
            "Run the scheduled scan that ranks NIFTY 50 constituents by realised "
            "volatility (ATR %) and suggests an ATM long straddle (buy CE + PE) on "
            "the most volatile names — a direction-agnostic play on a big move"
        ),
    )
    volatility_scan_interval_minutes: int = Field(
        default=30,
        description=(
            "How often (minutes) to run the volatile-stock straddle scan during "
            "market hours. Runs are market-hours guarded, so this only fires 09:15–15:30."
        ),
    )
    volatility_scan_top_n: int = Field(
        default=5, description="How many top volatile-stock straddles to include in the scan notification"
    )

    # ── Logging ────────────────────────────────────────────────
    log_level: str = Field(default="INFO", description="Python logging level")
    log_file: str = Field(default="nifty_ai_agent/logs/agent.log", description="Log file path")


def get_settings() -> Settings:
    """Re-read `.env` on every call.

    Intentionally not cached: UPSTOX_ACCESS_TOKEN may be updated (e.g. a daily
    scripts/upstox_login.py refresh, or swapping in a new Analytics token)
    while the agent keeps running for days at a time — a cached Settings object
    would freeze that token (and any other .env edit) at process-start value.
    """
    return Settings()


def configure_logging(settings: Settings | None = None) -> None:
    if settings is None:
        settings = get_settings()

    numeric_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(settings.log_file, encoding="utf-8"),
        ],
    )
