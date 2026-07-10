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
    pushover_user_key: str = Field(..., description="Pushover user key")
    pushover_api_token: str = Field(..., description="Pushover application API token")

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
        default="", description="Upstox access token — refreshed daily via scripts/upstox_login.py"
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

    # ── Logging ────────────────────────────────────────────────
    log_level: str = Field(default="INFO", description="Python logging level")
    log_file: str = Field(default="nifty_ai_agent/logs/agent.log", description="Log file path")


def get_settings() -> Settings:
    """Re-read `.env` on every call.

    Intentionally not cached: UPSTOX_ACCESS_TOKEN is refreshed daily by
    scripts/upstox_login.py while the agent keeps running for days at a
    time — a cached Settings object would freeze that token (and any other
    .env edit) at whatever it was when the process started.
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
