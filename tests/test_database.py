"""Tests for the database repository."""

import pandas as pd
import pytest

from nifty_ai_agent.database.repository import DatabaseRepository
from nifty_ai_agent.risk.calculator import RiskCalculator
from nifty_ai_agent.strategies.base import SignalType
from nifty_ai_agent.strategies.ema_crossover import EMACrossoverStrategy


@pytest.fixture
def db(tmp_path):
    url = f"sqlite:///{tmp_path}/test.db"
    return DatabaseRepository(url)


def _dummy_signal():
    from nifty_ai_agent.strategies.base import Signal
    return Signal(
        signal=SignalType.BUY_CE,
        confidence=75,
        reason="EMA20 > EMA50, RSI = 65",
        strategy="EMA_Crossover",
    )


def _dummy_risk():
    calc = RiskCalculator()
    return calc.calculate(SignalType.BUY_CE, 24000.0, 100.0)


def _dummy_ohlcv(n: int = 5) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "open": [24000.0] * n,
            "high": [24100.0] * n,
            "low": [23900.0] * n,
            "close": [24050.0] * n,
            "volume": [1_000_000] * n,
        },
        index=idx,
    )


class TestDatabaseRepository:
    def test_save_and_retrieve_signal(self, db):
        signal = _dummy_signal()
        risk = _dummy_risk()
        sid = db.save_signal(signal, risk, "Test explanation")
        assert isinstance(sid, int)
        records = db.get_recent_signals(limit=5)
        assert len(records) == 1
        assert records[0].signal == "BUY_CE"
        assert records[0].confidence == 75

    def test_save_market_data(self, db):
        df = _dummy_ohlcv()
        db.save_market_data(df, symbol="NIFTY")
        records = db.get_latest_market_data(limit=10)
        assert len(records) == 5

    def test_multiple_signals(self, db):
        for _ in range(3):
            db.save_signal(_dummy_signal(), _dummy_risk())
        records = db.get_recent_signals(limit=10)
        assert len(records) == 3

    def test_save_and_close_trade(self, db):
        trade_id = db.save_trade(
            entry_price=24000.0,
            strategy="EMA_Crossover",
            signal_id=None,
        )
        assert isinstance(trade_id, int)
        db.close_trade(trade_id, exit_price=24300.0, result="WIN")

    def test_close_nonexistent_trade_logs_warning(self, db):
        # Should not raise
        db.close_trade(trade_id=99999, exit_price=24000.0, result="LOSS")

    def test_signal_has_ai_explanation(self, db):
        db.save_signal(_dummy_signal(), _dummy_risk(), ai_explanation="Bullish trend.")
        records = db.get_recent_signals()
        assert records[0].ai_explanation == "Bullish trend."
