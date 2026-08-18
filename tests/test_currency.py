"""Tests for currency-pair OHLCV sourcing (NSE currency futures)."""

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from nifty_ai_agent.data import currency as ccy


def _epoch_ms(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)


def _fut_row(pair: str, expiry: date, key: str) -> dict:
    return {
        "segment": "NCD_FO", "asset_symbol": pair, "instrument_type": "FUT",
        "expiry": _epoch_ms(expiry), "instrument_key": key,
        "trading_symbol": f"{pair} FUT {expiry:%d %b %y}".upper(),
    }


class _Master:
    def __init__(self, rows):
        self._rows = rows

    def _load(self):
        return self._rows


def _frame(last: float, bars: int = 5) -> pd.DataFrame:
    idx = pd.date_range("2026-08-14 10:00", periods=bars, freq="5min")
    close = np.linspace(last - 0.1, last, bars)
    return pd.DataFrame(
        {"open": close, "high": close + 0.05, "low": close - 0.05,
         "close": close, "volume": np.full(bars, 100.0)},
        index=idx,
    )


class TestLiveFutures:
    def test_orders_by_expiry(self):
        today = date.today()
        rows = [
            _fut_row("USDINR", today + timedelta(days=21), "K3"),
            _fut_row("USDINR", today + timedelta(days=7), "K1"),
            _fut_row("USDINR", today + timedelta(days=14), "K2"),
        ]
        keys = [r["instrument_key"] for r in ccy.live_futures(_Master(rows), "USDINR")]
        assert keys == ["K1", "K2", "K3"]

    def test_expired_contracts_are_dropped(self):
        today = date.today()
        rows = [
            _fut_row("USDINR", today - timedelta(days=1), "OLD"),
            _fut_row("USDINR", today + timedelta(days=7), "LIVE"),
        ]
        keys = [r["instrument_key"] for r in ccy.live_futures(_Master(rows), "USDINR")]
        assert keys == ["LIVE"]

    def test_todays_expiry_is_ranked_last(self):
        """A contract settling today quotes a stale price while a later one is live."""
        today = date.today()
        rows = [
            _fut_row("USDINR", today, "TODAY"),
            _fut_row("USDINR", today + timedelta(days=7), "NEXT"),
        ]
        keys = [r["instrument_key"] for r in ccy.live_futures(_Master(rows), "USDINR")]
        assert keys == ["NEXT", "TODAY"]

    def test_todays_expiry_still_offered_when_it_is_all_there_is(self):
        today = date.today()
        rows = [_fut_row("USDINR", today, "TODAY")]
        keys = [r["instrument_key"] for r in ccy.live_futures(_Master(rows), "USDINR")]
        assert keys == ["TODAY"]

    def test_other_pairs_are_not_returned(self):
        today = date.today()
        rows = [
            _fut_row("EURINR", today + timedelta(days=7), "EUR"),
            _fut_row("USDINR", today + timedelta(days=7), "USD"),
        ]
        keys = [r["instrument_key"] for r in ccy.live_futures(_Master(rows), "USDINR")]
        assert keys == ["USD"]


class TestFetchCurrencyHistories:
    @pytest.fixture
    def rows(self):
        today = date.today()
        return [
            _fut_row("USDINR", today + timedelta(days=7), "DEAD"),
            _fut_row("USDINR", today + timedelta(days=14), "LIVE"),
        ]

    def _client(self, served: dict):
        class _C:
            def __init__(self, token):
                pass

            def get_historical_ohlcv_by_key(self, key, days, interval="5m"):
                if key not in served:
                    raise ValueError(f"no candles for {key}")
                return served[key]

        return _C

    def test_falls_through_an_untraded_weekly_to_the_next_expiry(self, rows, monkeypatch):
        """EUR/GBP/JPY weeklies are routinely untraded — stopping there loses the pair."""
        import nifty_ai_agent.data.upstox_provider as up
        monkeypatch.setattr(up, "UpstoxClient", self._client({"LIVE": _frame(95.4)}))

        hist, spots = ccy.fetch_currency_histories(
            "token", _Master(rows), pairs=["USDINR"],
        )
        assert list(hist) == ["USDINR"]
        assert spots["USDINR"] == pytest.approx(95.4)

    def test_pair_with_no_tradeable_history_is_omitted_not_fatal(self, rows, monkeypatch):
        import nifty_ai_agent.data.upstox_provider as up
        monkeypatch.setattr(up, "UpstoxClient", self._client({}))

        assert ccy.fetch_currency_histories(
            "token", _Master(rows), pairs=["USDINR"],
        ) == ({}, {})

    def test_no_token_returns_empty(self, rows):
        """Currency candles are an authenticated endpoint — there is no free fallback."""
        assert ccy.fetch_currency_histories("", _Master(rows)) == ({}, {})

    def test_zero_close_is_rejected(self, rows, monkeypatch):
        import nifty_ai_agent.data.upstox_provider as up
        monkeypatch.setattr(up, "UpstoxClient", self._client({"LIVE": _frame(0.0)}))
        assert ccy.fetch_currency_histories(
            "token", _Master(rows), pairs=["USDINR"],
        ) == ({}, {})
