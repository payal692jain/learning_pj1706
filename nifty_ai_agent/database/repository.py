"""Database repository — all DB operations go through this class."""

import logging
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from nifty_ai_agent.database.models import Base, MarketDataRecord, SignalRecord, TradeRecord
from nifty_ai_agent.risk.calculator import RiskParameters
from nifty_ai_agent.strategies.base import Signal

logger = logging.getLogger(__name__)


class DatabaseRepository:
    """Thin wrapper around SQLAlchemy providing domain-level CRUD."""

    def __init__(self, database_url: str) -> None:
        self._engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False} if "sqlite" in database_url else {},
        )
        Base.metadata.create_all(self._engine)
        logger.info("Database initialised at %s", database_url)

    # ── Market data ──────────────────────────────────────────────────

    def save_market_data(self, df: pd.DataFrame, symbol: str = "NIFTY") -> None:
        """Persist OHLCV rows from *df* — skips rows already in DB."""
        with Session(self._engine) as session:
            for ts, row in df.iterrows():
                dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
                record = MarketDataRecord(
                    datetime=dt,
                    symbol=symbol,
                    open=float(row.get("open", 0)),
                    high=float(row.get("high", 0)),
                    low=float(row.get("low", 0)),
                    close=float(row.get("close", 0)),
                    volume=int(row.get("volume", 0)),
                )
                session.merge(record)
            session.commit()
        logger.debug("Saved %d market data rows for %s", len(df), symbol)

    def get_latest_market_data(self, limit: int = 60) -> list[MarketDataRecord]:
        with Session(self._engine) as session:
            stmt = (
                select(MarketDataRecord)
                .order_by(MarketDataRecord.datetime.desc())
                .limit(limit)
            )
            return list(session.scalars(stmt).all())

    # ── Signals ──────────────────────────────────────────────────────

    def save_signal(
        self,
        signal: Signal,
        risk: RiskParameters,
        ai_explanation: str = "",
    ) -> int:
        """Persist a generated signal and return its DB id."""
        with Session(self._engine) as session:
            record = SignalRecord(
                datetime=datetime.now(tz=timezone.utc),
                signal=signal.signal.value,
                confidence=signal.confidence,
                strategy=signal.strategy,
                reason=signal.reason,
                ai_explanation=ai_explanation,
                entry_price=risk.entry_price if risk else None,
                stop_loss=risk.stop_loss if risk else None,
                target=risk.target if risk else None,
                risk_reward=risk.risk_reward_ratio if risk else None,
                status="OPEN",
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            logger.info("Saved signal id=%d %s", record.id, signal.signal.value)
            return record.id

    def get_recent_signals(self, limit: int = 10) -> list[SignalRecord]:
        with Session(self._engine) as session:
            stmt = (
                select(SignalRecord)
                .order_by(SignalRecord.datetime.desc())
                .limit(limit)
            )
            return list(session.scalars(stmt).all())

    # ── Trades ───────────────────────────────────────────────────────

    def save_trade(
        self,
        entry_price: float,
        strategy: str,
        signal_id: int | None = None,
    ) -> int:
        with Session(self._engine) as session:
            record = TradeRecord(
                signal_id=signal_id,
                entry_price=entry_price,
                strategy=strategy,
                result="OPEN",
                entry_time=datetime.now(tz=timezone.utc),
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return record.id

    def close_trade(
        self,
        trade_id: int,
        exit_price: float,
        result: str,
    ) -> None:
        with Session(self._engine) as session:
            record = session.get(TradeRecord, trade_id)
            if record is None:
                logger.warning("Trade id=%d not found", trade_id)
                return
            record.exit_price = exit_price
            record.result = result
            record.exit_time = datetime.now(tz=timezone.utc)
            record.pnl = exit_price - record.entry_price
            session.commit()
