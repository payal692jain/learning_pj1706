"""Abstract base classes and data models for the market data layer."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd


@dataclass
class SpotData:
    symbol: str
    price: float
    timestamp: datetime
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    volume: int = 0


@dataclass
class OptionChainData:
    symbol: str
    expiry: str           # nearest weekly expiry
    timestamp: datetime
    strikes: pd.DataFrame = field(default_factory=pd.DataFrame)
    pcr: float = 0.0
    max_pain: float = 0.0
    monthly_expiry: str = ""
    monthly_strikes: pd.DataFrame = field(default_factory=pd.DataFrame)
    iv_proxy: float = 0.0   # India VIX / 100 — for Black-Scholes when strikes are empty


class MarketDataProvider(ABC):
    """Interface every market data source must implement."""

    @abstractmethod
    def get_spot_data(self) -> SpotData:
        """Return current NIFTY spot price and OHLC."""

    @abstractmethod
    def get_option_chain(self) -> OptionChainData:
        """Return current NIFTY option chain with PCR and max-pain."""

    @abstractmethod
    def get_historical_data(self, days: int = 60) -> pd.DataFrame:
        """Return OHLCV DataFrame for the last *days* trading days.

        Expected columns: open, high, low, close, volume.
        Index: DatetimeIndex.
        """
