"""Base strategy interface and shared data models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import pandas as pd


class SignalType(str, Enum):
    BUY_CE = "BUY_CE"
    BUY_PE = "BUY_PE"
    HOLD = "HOLD"


@dataclass
class Signal:
    signal: SignalType
    confidence: int  # 0–100
    reason: str
    strategy: str


class BaseStrategy(ABC):
    """All trading strategies must implement this interface."""

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """Analyse *df* (OHLCV + indicators) and return a Signal.

        Strategies must NEVER place trades — signal generation only.

        Args:
            df: DataFrame with OHLCV columns plus pre-computed indicator columns.

        Returns:
            Signal with type, confidence percentage, and human-readable reason.
        """
