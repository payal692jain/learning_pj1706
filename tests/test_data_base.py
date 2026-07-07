"""Tests for data base models and provider interface."""

from datetime import datetime, timezone

import pandas as pd
import pytest

from nifty_ai_agent.data.base import MarketDataProvider, OptionChainData, SpotData


class TestSpotData:
    def test_creation(self):
        sd = SpotData(
            symbol="NIFTY",
            price=24000.0,
            timestamp=datetime.now(tz=timezone.utc),
        )
        assert sd.price == 24000.0
        assert sd.symbol == "NIFTY"

    def test_defaults(self):
        sd = SpotData("NIFTY", 24000.0, datetime.now(tz=timezone.utc))
        assert sd.open == 0.0
        assert sd.high == 0.0
        assert sd.low == 0.0
        assert sd.volume == 0


class TestOptionChainData:
    def test_creation(self):
        ocd = OptionChainData(
            symbol="NIFTY",
            expiry="27-Jun-2024",
            timestamp=datetime.now(tz=timezone.utc),
            pcr=1.2,
            max_pain=24000.0,
        )
        assert ocd.pcr == 1.2
        assert ocd.max_pain == 24000.0
        assert isinstance(ocd.strikes, pd.DataFrame)


class TestMarketDataProviderInterface:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            MarketDataProvider()  # type: ignore[abstract]

    def test_concrete_must_implement_all(self):
        class Incomplete(MarketDataProvider):
            def get_spot_data(self):
                return None

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]
