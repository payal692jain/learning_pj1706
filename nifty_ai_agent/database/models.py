"""SQLAlchemy ORM models matching the CLAUDE.md database schema."""

from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class MarketDataRecord(Base):
    __tablename__ = "market_data"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    datetime: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, default="NIFTY")
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class SignalRecord(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    datetime: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    signal: Mapped[str] = mapped_column(String(20), nullable=False)
    confidence: Mapped[int] = mapped_column(Integer, nullable=False)
    strategy: Mapped[str] = mapped_column(String(50), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    ai_explanation: Mapped[str] = mapped_column(Text, nullable=True)
    entry_price: Mapped[float] = mapped_column(Float, nullable=True)
    stop_loss: Mapped[float] = mapped_column(Float, nullable=True)
    target: Mapped[float] = mapped_column(Float, nullable=True)
    risk_reward: Mapped[float] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="OPEN")


class Subscriber(Base):
    """A Telegram user who receives broadcast signals.

    chat_id is a BigInteger because Telegram group/channel ids exceed 32 bits.
    Only per-user knobs live here (capital, risk %) — the signals themselves are
    global, computed once and fanned out to every active subscriber.
    """

    __tablename__ = "subscribers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, nullable=False, unique=True, index=True)
    username: Mapped[str] = mapped_column(String(64), nullable=True)
    capital: Mapped[float] = mapped_column(Float, nullable=False, default=50000.0)
    risk_pct: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    subscribed_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class PositionRecord(Base):
    """One suggested option position the agent is tracking to a HOLD/SELL verdict.

    Opened when a STRONG-conviction BUY_CE/BUY_PE fires for an underlying that has
    no open position, so a 5-minute signal loop cannot pile up dozens of phantom
    positions. Closed when the underlying reaches the target or stop-loss, or when
    the consensus reverses.
    """

    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # "NIFTY" / "BANKNIFTY" / "RELIANCE" — the underlying, not the contract.
    underlying: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    opt_type: Mapped[str] = mapped_column(String(2), nullable=False)      # CE / PE
    strike: Mapped[float] = mapped_column(Float, nullable=False)
    expiry: Mapped[str] = mapped_column(String(20), nullable=False)       # DD-Mon-YYYY
    lot_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    entry_premium: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    entry_spot: Mapped[float] = mapped_column(Float, nullable=False)
    stop_loss: Mapped[float] = mapped_column(Float, nullable=False)       # on the underlying
    target: Mapped[float] = mapped_column(Float, nullable=False)          # on the underlying

    strategy: Mapped[str] = mapped_column(String(50), nullable=False, default="")
    conviction: Mapped[str] = mapped_column(String(20), nullable=False, default="")
    confidence: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    status: Mapped[str] = mapped_column(String(20), nullable=False, default="OPEN", index=True)
    opened_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    closed_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    exit_spot: Mapped[float] = mapped_column(Float, nullable=True)
    exit_reason: Mapped[str] = mapped_column(String(120), nullable=True)


class TradeRecord(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    signal_id: Mapped[int] = mapped_column(Integer, nullable=True)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    exit_price: Mapped[float] = mapped_column(Float, nullable=True)
    pnl: Mapped[float] = mapped_column(Float, nullable=True)
    strategy: Mapped[str] = mapped_column(String(50), nullable=False)
    result: Mapped[str] = mapped_column(String(20), nullable=True)  # WIN / LOSS / OPEN
    entry_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    exit_time: Mapped[datetime] = mapped_column(DateTime, nullable=True)
