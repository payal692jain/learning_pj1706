"""Tests for the Supertrend, MACD momentum, ORB, and Bollinger squeeze strategies."""

import numpy as np
import pandas as pd

from nifty_ai_agent.indicators.bollinger import compute_bollinger
from nifty_ai_agent.indicators.supertrend import compute_supertrend
from nifty_ai_agent.strategies.base import SignalType
from nifty_ai_agent.strategies.bollinger_squeeze import BollingerSqueezeStrategy
from nifty_ai_agent.strategies.macd_momentum import MACDMomentumStrategy
from nifty_ai_agent.strategies.orb import OpeningRangeBreakoutStrategy
from nifty_ai_agent.strategies.supertrend import SupertrendStrategy


def _ohlc(closes: list[float], start: str = "2024-01-01 09:15") -> pd.DataFrame:
    close = np.array(closes, dtype=float)
    idx = pd.date_range(start, periods=len(close), freq="5min")
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 15,
            "low": close - 15,
            "close": close,
            "volume": np.full(len(close), 100_000),
        },
        index=idx,
    )


class TestSupertrendIndicator:
    def test_direction_is_up_in_a_rising_market(self):
        df = compute_supertrend(_ohlc(list(np.linspace(24_000, 24_600, 60))))
        assert df["supertrend_dir"].iloc[-1] == 1
        assert df["supertrend"].iloc[-1] < df["close"].iloc[-1]

    def test_direction_is_down_in_a_falling_market(self):
        df = compute_supertrend(_ohlc(list(np.linspace(24_600, 24_000, 60))))
        assert df["supertrend_dir"].iloc[-1] == -1
        assert df["supertrend"].iloc[-1] > df["close"].iloc[-1]

    def test_warmup_bars_have_no_trend(self):
        df = compute_supertrend(_ohlc(list(np.linspace(24_000, 24_600, 60))), period=10)
        assert df["supertrend_dir"].iloc[:9].isna().all()


class TestSupertrendStrategy:
    def setup_method(self):
        self.strategy = SupertrendStrategy()

    def test_buy_ce_on_a_fresh_bullish_flip(self):
        # Down, then a sharp reversal up — flips the trend near the end of the series.
        closes = list(np.linspace(24_500, 24_100, 40)) + list(np.linspace(24_110, 24_500, 12))
        signal = self.strategy.generate_signal(compute_supertrend(_ohlc(closes)))
        assert signal.signal == SignalType.BUY_CE
        assert signal.confidence > 50

    def test_buy_pe_on_a_fresh_bearish_flip(self):
        closes = list(np.linspace(24_100, 24_500, 40)) + list(np.linspace(24_490, 24_100, 12))
        signal = self.strategy.generate_signal(compute_supertrend(_ohlc(closes)))
        assert signal.signal == SignalType.BUY_PE

    def test_hold_when_the_trend_is_too_old_to_chase(self):
        closes = list(np.linspace(24_000, 25_500, 90))  # one long uninterrupted trend
        signal = self.strategy.generate_signal(compute_supertrend(_ohlc(closes)))
        assert signal.signal == SignalType.HOLD
        assert "extended" in signal.reason

    def test_hold_when_bars_are_insufficient(self):
        signal = self.strategy.generate_signal(compute_supertrend(_ohlc([24_000] * 5)))
        assert signal.signal == SignalType.HOLD


class TestMACDMomentumStrategy:
    def setup_method(self):
        self.strategy = MACDMomentumStrategy()

    def _with_macd(self, macd: list[float], signal_line: list[float]) -> pd.DataFrame:
        df = _ohlc([24_000.0] * len(macd))
        df["macd"] = macd
        df["macd_signal"] = signal_line
        return df

    def test_buy_ce_when_histogram_is_positive_and_expanding(self):
        df = self._with_macd(macd=[10, 14, 18, 24, 32], signal_line=[10, 10, 10, 10, 10])
        signal = self.strategy.generate_signal(df)
        assert signal.signal == SignalType.BUY_CE

    def test_buy_pe_when_histogram_is_negative_and_expanding(self):
        df = self._with_macd(macd=[-10, -14, -18, -24, -32], signal_line=[-10] * 5)
        signal = self.strategy.generate_signal(df)
        assert signal.signal == SignalType.BUY_PE

    def test_hold_when_a_positive_histogram_is_fading(self):
        # Above the signal line, but momentum is draining — the exhaustion trap.
        df = self._with_macd(macd=[40, 34, 26, 18, 12], signal_line=[10] * 5)
        signal = self.strategy.generate_signal(df)
        assert signal.signal == SignalType.HOLD
        assert "fading" in signal.reason

    def test_confidence_is_scale_invariant_across_indices(self):
        # The same relative move on NIFTY and SENSEX should score the same.
        nifty = self._with_macd([10, 14, 18, 24, 32], [10] * 5)
        nifty["close"] = 24_000.0
        sensex = self._with_macd([33, 46, 59, 79, 105], [33] * 5)
        sensex["close"] = 79_000.0
        assert abs(
            self.strategy.generate_signal(nifty).confidence
            - self.strategy.generate_signal(sensex).confidence
        ) <= 2


class TestOpeningRangeBreakout:
    def setup_method(self):
        self.strategy = OpeningRangeBreakoutStrategy()

    def test_buy_ce_when_price_clears_the_opening_range_high(self):
        # 09:15–09:30 chops in a tight band, then breaks out upward.
        closes = [24_000, 24_010, 23_995] + [24_060, 24_090, 24_120]
        signal = self.strategy.generate_signal(_ohlc(closes))
        assert signal.signal == SignalType.BUY_CE
        assert "above the opening range" in signal.reason

    def test_buy_pe_when_price_breaks_the_opening_range_low(self):
        closes = [24_000, 24_010, 23_995] + [23_930, 23_900, 23_860]
        signal = self.strategy.generate_signal(_ohlc(closes))
        assert signal.signal == SignalType.BUY_PE

    def test_hold_while_price_stays_inside_the_range(self):
        closes = [24_000, 24_010, 23_995, 24_005, 24_000, 23_998]
        signal = self.strategy.generate_signal(_ohlc(closes))
        assert signal.signal == SignalType.HOLD
        assert "inside the opening range" in signal.reason

    def test_hold_while_the_opening_range_is_still_forming(self):
        signal = self.strategy.generate_signal(_ohlc([24_000, 24_010]))
        assert signal.signal == SignalType.HOLD

    def test_only_the_latest_session_range_is_used(self):
        # Yesterday ranged 23,000–23,030; today opens far higher and breaks out.
        yesterday = _ohlc([23_000, 23_010, 23_030], start="2024-01-01 09:15")
        today = _ohlc([24_000, 24_010, 23_995, 24_120], start="2024-01-02 09:15")
        signal = self.strategy.generate_signal(pd.concat([yesterday, today]))
        assert signal.signal == SignalType.BUY_CE
        assert "24" in signal.reason  # today's range, not yesterday's 23k band

    def test_hold_without_intraday_timestamps(self):
        df = _ohlc([24_000] * 5).reset_index(drop=True)
        signal = self.strategy.generate_signal(df)
        assert signal.signal == SignalType.HOLD


class TestBollingerSqueezeStrategy:
    def setup_method(self):
        self.strategy = BollingerSqueezeStrategy()

    def _squeeze_then(self, breakout: list[float]) -> pd.DataFrame:
        # 60 near-flat bars collapse the bands, then the breakout leg expands them.
        rng = np.random.default_rng(7)
        quiet = 24_000 + rng.normal(0, 1.5, 60)
        return compute_bollinger(_ohlc(list(quiet) + breakout))

    def test_buy_ce_when_a_squeeze_resolves_upward(self):
        signal = self.strategy.generate_signal(
            self._squeeze_then([24_020, 24_060, 24_110, 24_170])
        )
        assert signal.signal == SignalType.BUY_CE
        assert "upward" in signal.reason

    def test_buy_pe_when_a_squeeze_resolves_downward(self):
        signal = self.strategy.generate_signal(
            self._squeeze_then([23_980, 23_940, 23_890, 23_830])
        )
        assert signal.signal == SignalType.BUY_PE

    def test_hold_while_still_squeezed(self):
        rng = np.random.default_rng(11)
        quiet = 24_000 + rng.normal(0, 1.5, 70)
        signal = self.strategy.generate_signal(compute_bollinger(_ohlc(list(quiet))))
        assert signal.signal == SignalType.HOLD
        assert "squeez" in signal.reason

    def test_hold_when_bars_are_insufficient(self):
        signal = self.strategy.generate_signal(compute_bollinger(_ohlc([24_000] * 25)))
        assert signal.signal == SignalType.HOLD
