"""Tests for the single-stock option scanner, its report, and the bulk fetcher."""

from datetime import date, time

import numpy as np
import pandas as pd
import pytest

from nifty_ai_agent.data.instrument_master import OptionContract
from nifty_ai_agent.risk.calculator import RiskCalculator
from nifty_ai_agent.reports.stock_scan import format_stock_scan
from nifty_ai_agent.strategies.stock_scanner import ScanResult, StockIdea, scan_stocks


# ── Fixtures / helpers ──────────────────────────────────────────────────────────

def _trend_frame(start: float, end: float, periods: int = 120) -> pd.DataFrame:
    """A clean 5-minute OHLCV frame trending from *start* to *end*."""
    idx = pd.date_range("2026-07-27 09:15", periods=periods, freq="5min")
    close = np.linspace(start, end, periods) + np.sin(np.linspace(0, 12, periods)) * 0.3
    df = pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.linspace(2000, 5000, periods),
        },
        index=idx,
    )
    df.index.name = "datetime"
    return df


class FakeMaster:
    """Instrument master stub — returns a deterministic ATM contract per call."""

    def __init__(self, lot_size: int = 250, raises: bool = False) -> None:
        self.lot_size = lot_size
        self.raises = raises

    def atm_contract(self, asset, spot, opt_type, allow_expiry_day=None):
        if self.raises:
            raise RuntimeError("master unavailable")
        strike = round(spot / 10) * 10
        return OptionContract(
            instrument_key=f"NSE_FO|{asset}",
            trading_symbol=f"{asset}{strike}{opt_type}",
            asset_symbol=asset,
            strike=float(strike),
            opt_type=opt_type,
            expiry=date(2026, 8, 27),
            lot_size=self.lot_size,
        )


def _live_lookup(price: float):
    return lambda keys: {k: price for k in keys}


def _ohlcv_frame(last_close: float, periods: int = 3) -> pd.DataFrame:
    """A tiny OHLCV frame whose final close is *last_close* (mimics an Upstox fetch)."""
    idx = pd.date_range("2026-07-27 09:15", periods=periods, freq="5min")
    close = np.linspace(last_close - 2, last_close, periods)
    df = pd.DataFrame(
        {"open": close - 0.2, "high": close + 0.5, "low": close - 0.5,
         "close": close, "volume": np.linspace(1000, 2000, periods)},
        index=idx,
    )
    df.index.name = "datetime"
    return df


class _EquityMaster:
    """Instrument-master stub for equity_key lookups (bare symbols it knows)."""

    def __init__(self, known: set[str]) -> None:
        self._known = known

    def equity_key(self, symbol: str) -> str | None:
        return f"NSE_EQ|{symbol}" if symbol in self._known else None


def _FakeUpstoxClient(served: dict[str, pd.DataFrame]):
    """Return a UpstoxClient replacement serving *served* keyed by bare symbol."""

    class _Client:
        def __init__(self, token):
            self._token = token

        def get_historical_ohlcv_by_key(self, key, days, interval="5m"):
            symbol = key.split("|", 1)[1]  # "NSE_EQ|RELIANCE" → "RELIANCE"
            if symbol not in served:
                raise ValueError(f"no candles for {symbol}")
            return served[symbol]

    return _Client


# ── scan_stocks ───────────────────────────────────────────────────────────────

class TestScanStocks:
    def test_uptrend_yields_bullish_idea_with_live_premium(self):
        df = _trend_frame(100, 130)
        res = scan_stocks(
            {"RELIANCE.NS": df},
            {"RELIANCE.NS": 130.0},
            master=FakeMaster(),
            risk_calculator=RiskCalculator(),
            now=time(10, 0),
            premium_lookup=_live_lookup(42.5),
        )
        assert res.scanned == 1
        assert res.actionable == 1
        assert len(res.ideas) == 1
        idea = res.ideas[0]
        assert idea.symbol == "RELIANCE"          # ".NS" stripped
        assert idea.signal == "BUY_CE"
        assert idea.opt_type == "CE"
        assert idea.expiry == "27-Aug-2026"        # monthly
        assert idea.lot_size == 250
        assert idea.entry_premium == 42.5
        assert idea.is_live is True
        assert idea.target > idea.spot > idea.stop_loss  # bullish levels

    def test_no_live_client_falls_back_to_estimate(self):
        df = _trend_frame(100, 130)
        res = scan_stocks(
            {"RELIANCE.NS": df},
            {"RELIANCE.NS": 130.0},
            master=FakeMaster(),
            risk_calculator=RiskCalculator(),
            now=time(10, 0),
            default_iv=0.30,
        )
        assert len(res.ideas) == 1
        idea = res.ideas[0]
        assert idea.is_live is False
        assert idea.entry_premium > 0              # Black-Scholes estimate

    def test_afternoon_scan_still_produces_monthly_idea(self):
        # Monthly stock options are NOT subject to the intraday 15:20 entry cutoff,
        # so a late-session scan of a trending stock still yields an idea.
        df = _trend_frame(100, 130)
        res = scan_stocks(
            {"RELIANCE.NS": df},
            {"RELIANCE.NS": 130.0},
            master=FakeMaster(),
            risk_calculator=RiskCalculator(),
            now=time(15, 30),
        )
        assert res.actionable == 1
        assert len(res.ideas) == 1

    def test_intraday_true_reinstates_the_entry_cutoff(self):
        # Opting back into intraday rules restores the after-15:20 NO_TRADE guard.
        df = _trend_frame(100, 130)
        res = scan_stocks(
            {"RELIANCE.NS": df},
            {"RELIANCE.NS": 130.0},
            master=FakeMaster(),
            risk_calculator=RiskCalculator(),
            now=time(15, 30),
            intraday=True,
        )
        assert res.actionable == 0
        assert res.ideas == []

    def test_top_n_caps_and_ranks_by_confidence(self):
        frames = {f"S{i}.NS": _trend_frame(100, 130) for i in range(4)}
        spots = {sym: 130.0 for sym in frames}
        res = scan_stocks(
            frames, spots, master=FakeMaster(), risk_calculator=RiskCalculator(),
            now=time(10, 0), premium_lookup=_live_lookup(10.0), top_n=2,
        )
        assert res.actionable == 4
        assert len(res.ideas) == 2                 # capped
        confidences = [i.confidence for i in res.ideas]
        assert confidences == sorted(confidences, reverse=True)

    def test_master_failure_is_counted_not_fatal(self):
        df = _trend_frame(100, 130)
        res = scan_stocks(
            {"RELIANCE.NS": df, "INFY.NS": df},
            {"RELIANCE.NS": 130.0, "INFY.NS": 130.0},
            master=FakeMaster(raises=True),
            risk_calculator=RiskCalculator(),
            now=time(10, 0),
        )
        assert res.scanned == 2
        assert res.errors == 2
        assert res.ideas == []

    def test_nonpositive_spot_is_skipped(self):
        df = _trend_frame(100, 130)
        res = scan_stocks(
            {"RELIANCE.NS": df},
            {"RELIANCE.NS": 0.0},
            master=FakeMaster(),
            risk_calculator=RiskCalculator(),
            now=time(10, 0),
        )
        assert res.scanned == 0
        assert res.ideas == []


# ── format_stock_scan ───────────────────────────────────────────────────────────

class TestFormatStockScan:
    def _idea(self, symbol="RELIANCE", conf=73, live=True) -> StockIdea:
        return StockIdea(
            symbol=symbol, signal="BUY_CE", confidence=conf, conviction="STRONG",
            opt_type="CE", strike=2500.0, expiry="27-Aug-2026", lot_size=250,
            entry_premium=42.5, is_live=live, spot=2498.0, target=2530.0,
            stop_loss=2482.0, rr=2.0,
        )

    def test_with_ideas_lists_contracts_and_stays_under_limit(self):
        res = ScanResult(ideas=[self._idea()], scanned=48, actionable=1, errors=0)
        title, body = format_stock_scan(res)
        assert "RELIANCE" in title
        assert "2500CE" in body
        assert "27-Aug-2026" in body
        assert len(body) <= 1024

    def test_estimate_flag_adds_disclaimer(self):
        res = ScanResult(ideas=[self._idea(live=False)], scanned=48, actionable=1, errors=0)
        _, body = format_stock_scan(res)
        assert "*" in body
        assert "estimated premium" in body

    def test_all_live_omits_estimate_note(self):
        res = ScanResult(ideas=[self._idea(live=True)], scanned=48, actionable=1, errors=0)
        _, body = format_stock_scan(res)
        assert "estimated premium" not in body

    def test_empty_result_is_a_clean_no_setup_message(self):
        res = ScanResult(ideas=[], scanned=50, actionable=0, errors=1)
        title, body = format_stock_scan(res)
        assert "no setups" in title
        assert "50 scanned" in body


# ── fetch_stock_histories ─────────────────────────────────────────────────────

class TestFetchStockHistories:
    def test_empty_symbol_list_short_circuits(self):
        from nifty_ai_agent.data import stock_data
        assert stock_data.fetch_stock_histories([]) == ({}, {})

    def test_splits_multiindex_and_extracts_spot(self, monkeypatch):
        from nifty_ai_agent.data import stock_data

        idx = pd.date_range("2026-07-27 09:15", periods=3, freq="5min")
        cols = pd.MultiIndex.from_product(
            [["RELIANCE.NS", "INFY.NS"], ["Open", "High", "Low", "Close", "Volume"]]
        )
        data = np.array([
            [100, 101, 99, 100.5, 10, 200, 201, 199, 200.5, 20],
            [101, 102, 100, 101.5, 11, 201, 202, 200, 201.5, 21],
            [102, 103, 101, 102.5, 12, 202, 203, 201, 202.5, 22],
        ], dtype=float)
        raw = pd.DataFrame(data, index=idx, columns=cols)

        monkeypatch.setattr(stock_data.yf, "download", lambda *a, **k: raw)
        histories, spots = stock_data.fetch_stock_histories(["RELIANCE.NS", "INFY.NS"])

        assert set(histories) == {"RELIANCE.NS", "INFY.NS"}
        assert list(histories["RELIANCE.NS"].columns) == ["open", "high", "low", "close", "volume"]
        assert spots["RELIANCE.NS"] == pytest.approx(102.5)
        assert spots["INFY.NS"] == pytest.approx(202.5)

    def test_empty_download_returns_empty(self, monkeypatch):
        from nifty_ai_agent.data import stock_data
        monkeypatch.setattr(stock_data.yf, "download", lambda *a, **k: pd.DataFrame())
        assert stock_data.fetch_stock_histories(["RELIANCE.NS"]) == ({}, {})

    def test_upstox_primary_skips_yfinance_when_all_served(self, monkeypatch):
        import nifty_ai_agent.data.upstox_provider as up
        from nifty_ai_agent.data import stock_data

        served = {"RELIANCE": _ohlcv_frame(102.5), "INFY": _ohlcv_frame(202.5)}
        monkeypatch.setattr(up, "UpstoxClient", _FakeUpstoxClient(served))
        # yfinance must not be consulted when Upstox serves the whole universe.
        monkeypatch.setattr(
            stock_data.yf, "download",
            lambda *a, **k: pytest.fail("yfinance should not be called"),
        )

        histories, spots = stock_data.fetch_stock_histories(
            ["RELIANCE.NS", "INFY.NS"],
            upstox_token="tok",
            master=_EquityMaster({"RELIANCE", "INFY"}),
        )

        assert set(histories) == {"RELIANCE.NS", "INFY.NS"}
        assert list(histories["RELIANCE.NS"].columns) == ["open", "high", "low", "close", "volume"]
        assert spots["RELIANCE.NS"] == pytest.approx(102.5)
        assert spots["INFY.NS"] == pytest.approx(202.5)

    def test_falls_back_to_yfinance_for_symbols_upstox_misses(self, monkeypatch):
        import nifty_ai_agent.data.upstox_provider as up
        from nifty_ai_agent.data import stock_data

        # Upstox serves RELIANCE; INFY has no equity key → must come from yfinance.
        monkeypatch.setattr(up, "UpstoxClient", _FakeUpstoxClient({"RELIANCE": _ohlcv_frame(102.5)}))

        idx = pd.date_range("2026-07-27 09:15", periods=2, freq="5min")
        cols = pd.MultiIndex.from_product([["INFY.NS"], ["Open", "High", "Low", "Close", "Volume"]])
        raw = pd.DataFrame(
            [[200, 201, 199, 200.5, 20], [201, 202, 200, 202.5, 21]], index=idx, columns=cols,
        )
        calls: list[list[str]] = []

        def _download(symbols, *a, **k):
            calls.append(symbols)
            return raw

        monkeypatch.setattr(stock_data.yf, "download", _download)

        histories, spots = stock_data.fetch_stock_histories(
            ["RELIANCE.NS", "INFY.NS"],
            upstox_token="tok",
            master=_EquityMaster({"RELIANCE"}),  # INFY absent
        )

        assert set(histories) == {"RELIANCE.NS", "INFY.NS"}
        assert calls == [["INFY.NS"]]  # only the residual went to yfinance
        assert spots["RELIANCE.NS"] == pytest.approx(102.5)
        assert spots["INFY.NS"] == pytest.approx(202.5)

    def test_no_token_uses_yfinance_only(self, monkeypatch):
        import nifty_ai_agent.data.upstox_provider as up
        from nifty_ai_agent.data import stock_data

        monkeypatch.setattr(
            up, "UpstoxClient",
            lambda *a, **k: pytest.fail("Upstox should not be used without a token"),
        )
        monkeypatch.setattr(stock_data.yf, "download", lambda *a, **k: pd.DataFrame())
        assert stock_data.fetch_stock_histories(
            ["RELIANCE.NS"], master=_EquityMaster({"RELIANCE"}),
        ) == ({}, {})

    def test_period_to_days_parsing(self):
        from nifty_ai_agent.data.stock_data import _period_to_days
        assert _period_to_days("5d") == 5
        assert _period_to_days("10d") == 10
        assert _period_to_days("weird") == 5
