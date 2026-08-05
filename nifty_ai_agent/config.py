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
        default=10, description="Days of OHLC history to fetch (5m bars: 10d = ~750 candles)"
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
