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

    def __init__(
        self, lot_size: int = 250, raises: bool = False, qty_multiplier: float = 1.0
    ) -> None:
        self.lot_size = lot_size
        self.raises = raises
        self.qty_multiplier = qty_multiplier
        self.segments: list[str] = []   # records what the scanner asked for

    def atm_contract(
        self, asset, spot, opt_type, allow_expiry_day=None, segment="NSE_FO",
    ):
        if self.raises:
            raise RuntimeError("master unavailable")
        self.segments.append(segment)
        strike = round(spot / 10) * 10
        return OptionContract(
            instrument_key=f"{segment}|{asset}",
            trading_symbol=f"{asset}{strike}{opt_type}",
            asset_symbol=asset,
            strike=float(strike),
            opt_type=opt_type,
            expiry=date(2026, 8, 27),
            lot_size=self.lot_size,
            qty_multiplier=self.qty_multiplier,
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


class TestPreOpenStockScan:
    """The 09:10 pre-open job swaps the market-hours guard for a weekday guard."""

    def _patched(self, monkeypatch, *, market_hours: bool, weekday: bool):
        """Patch main's guards + fetch; return the fetch spy."""
        import main

        calls: list[str] = []
        monkeypatch.setattr(main, "_is_market_hours", lambda: market_hours)
        monkeypatch.setattr(main, "_is_trading_weekday", lambda: weekday)
        monkeypatch.setattr(
            main, "fetch_stock_histories",
            lambda *a, **k: (calls.append("fetched"), ({}, {}))[1],
        )
        return main, calls

    def test_pre_open_runs_outside_market_hours(self, monkeypatch):
        main, calls = self._patched(monkeypatch, market_hours=False, weekday=True)
        main._run_stock_scan(pre_open=True)
        assert calls == ["fetched"]

    def test_regular_scan_still_blocked_outside_market_hours(self, monkeypatch):
        main, calls = self._patched(monkeypatch, market_hours=False, weekday=True)
        main._run_stock_scan()
        assert calls == []

    def test_pre_open_skips_on_weekend(self, monkeypatch):
        main, calls = self._patched(monkeypatch, market_hours=False, weekday=False)
        main._run_stock_scan(pre_open=True)
        assert calls == []


class TestCurrencySegment:
    """Currency options live in NCD_FO and size differently from equity F&O."""

    def test_segment_is_passed_through_to_the_master(self):
        master = FakeMaster(lot_size=1, qty_multiplier=1000.0)
        res = scan_stocks(
            {"USDINR": _trend_frame(85.0, 87.5)}, {"USDINR": 87.5},
            master=master, risk_calculator=RiskCalculator(),
            now=time(11, 0), segment="NCD_FO",
        )
        assert res.actionable == 1
        assert master.segments == ["NCD_FO"]

    def test_currency_contract_size_uses_qty_multiplier(self):
        """lot_size=1 with qty_multiplier=1000 is a 1000-unit contract, not a 1-unit one."""
        master = FakeMaster(lot_size=1, qty_multiplier=1000.0)
        res = scan_stocks(
            {"USDINR": _trend_frame(85.0, 87.5)}, {"USDINR": 87.5},
            master=master, risk_calculator=RiskCalculator(),
            now=time(11, 0), segment="NCD_FO",
        )
        assert res.ideas[0].lot_size == 1000

    def test_equity_contract_size_is_unchanged(self):
        master = FakeMaster(lot_size=200)   # qty_multiplier defaults to 1.0
        res = scan_stocks(
            {"BSE.NS": _trend_frame(3200.0, 3400.0)}, {"BSE.NS": 3400.0},
            master=master, risk_calculator=RiskCalculator(), now=time(11, 0),
        )
        assert res.ideas[0].lot_size == 200
        assert res.ideas[0].symbol == "BSE"   # .NS stripped
        assert master.segments == ["NSE_FO"]


class TestOptionContractSizing:
    def test_contract_size_multiplies_lot_by_multiplier(self):
        c = OptionContract(
            instrument_key="NCD_FO|1", trading_symbol="USDINR 87.75 CE",
            asset_symbol="USDINR", strike=87.75, opt_type="CE",
            expiry=date(2026, 8, 27), lot_size=1, qty_multiplier=1000.0,
        )
        assert c.contract_size == 1000

    def test_contract_size_defaults_to_lot_size(self):
        c = OptionContract(
            instrument_key="NSE_FO|1", trading_symbol="BSE 3400 CE",
            asset_symbol="BSE", strike=3400.0, opt_type="CE",
            expiry=date(2026, 8, 27), lot_size=200,
        )
        assert c.contract_size == 200


class TestPremiumProjection:
    """The exit price an intraday option trade is actually closed on."""

    def _idea(self, frames_from, frames_to, spot, premium=20.0):
        res = scan_stocks(
            {"RELIANCE.NS": _trend_frame(frames_from, frames_to)},
            {"RELIANCE.NS": spot},
            master=FakeMaster(), risk_calculator=RiskCalculator(max_risk_pct=5.0),
            now=time(11, 0), premium_lookup=_live_lookup(premium),
        )
        assert res.ideas, "expected an actionable idea"
        return res.ideas[0]

    def test_call_gains_when_the_underlying_reaches_target(self):
        idea = self._idea(100, 130, 130.0)
        assert idea.signal == "BUY_CE"
        assert idea.target_premium > idea.entry_premium
        assert idea.stop_premium < idea.entry_premium

    def test_put_gains_when_the_underlying_falls_to_target(self):
        """A put must gain on a DOWN move — the sign trap for projections."""
        idea = self._idea(130, 100, 100.0)
        assert idea.signal == "BUY_PE"
        assert idea.target_premium > idea.entry_premium
        assert idea.stop_premium < idea.entry_premium

    def test_projection_is_bounded_by_the_underlying_move(self):
        """Delta <= 1, so an option cannot gain more than the underlying did."""
        idea = self._idea(100, 130, 130.0)
        underlying_move = abs(idea.target - idea.spot)
        assert (idea.target_premium - idea.entry_premium) < underlying_move

    def test_premium_never_goes_negative(self):
        idea = self._idea(100, 130, 130.0, premium=0.10)
        assert idea.stop_premium > 0

    def test_invalid_risk_yields_no_projection(self):
        """A rejected risk calc has no levels to project to — render blank, not Rs 0."""
        from nifty_ai_agent.strategies.stock_scanner import _project_premiums

        class _Risk:
            is_valid = False
            target = 0.0
            stop_loss = 0.0

        contract = OptionContract(
            instrument_key="NSE_FO|X", trading_symbol="X", asset_symbol="X",
            strike=100.0, opt_type="CE", expiry=date(2026, 8, 27), lot_size=100,
        )
        assert _project_premiums(
            20.0, 100.0, _Risk(), contract, "27-Aug-2026", "CE", 0.30,
        ) == (0.0, 0.0)


class TestStockScanReportPremiums:
    def test_sell_and_exit_lines_are_rendered(self):
        res = scan_stocks(
            {"RELIANCE.NS": _trend_frame(100, 130)}, {"RELIANCE.NS": 130.0},
            master=FakeMaster(), risk_calculator=RiskCalculator(max_risk_pct=5.0),
            now=time(11, 0), premium_lookup=_live_lookup(20.0),
        )
        _, body = format_stock_scan(res)
        assert "sell ₹" in body and "exit ₹" in body

    def test_line_omitted_when_there_is_no_projection(self):
        idea = StockIdea(
            symbol="X", signal="BUY_CE", confidence=70, conviction="STRONG",
            opt_type="CE", strike=100.0, expiry="27-Aug-2026", lot_size=100,
            entry_premium=20.0, is_live=True, spot=100.0,
            target=0.0, stop_loss=0.0, rr=0.0,
        )
        _, body = format_stock_scan(
            ScanResult(ideas=[idea], scanned=1, actionable=1, errors=0)
        )
        assert "sell ₹" not in body


class TestEarningsBlackout:
    """Long premium into a results print is a different trade; the chart can't see it."""

    @staticmethod
    def _fund(days_away):
        from datetime import date, timedelta
        from nifty_ai_agent.data.fundamentals import StockFundamentals
        return StockFundamentals(
            symbol="RELIANCE",
            next_earnings=(date.today() + timedelta(days=days_away)).isoformat(),
        )

    def _scan(self, days_away, blackout=3):
        return scan_stocks(
            {"RELIANCE.NS": _trend_frame(100, 130)}, {"RELIANCE.NS": 130.0},
            master=FakeMaster(), risk_calculator=RiskCalculator(max_risk_pct=5.0),
            now=time(11, 0), premium_lookup=_live_lookup(20.0),
            fundamentals={"RELIANCE": self._fund(days_away)},
            earnings_blackout_days=blackout,
        )

    def test_entry_blocked_inside_the_window(self):
        res = self._scan(days_away=2)
        assert res.ideas == []
        assert res.holds and "results in 2d" in res.holds[0].reason

    def test_results_today_is_blocked(self):
        res = self._scan(days_away=0)
        assert res.ideas == []
        assert "results today" in res.holds[0].reason

    def test_entry_allowed_outside_the_window(self):
        res = self._scan(days_away=30)
        assert len(res.ideas) == 1
        assert res.holds == []

    def test_boundary_day_is_blocked(self):
        assert self._scan(days_away=3, blackout=3).ideas == []
        assert len(self._scan(days_away=4, blackout=3).ideas) == 1

    def test_guard_disabled_by_zero(self):
        assert len(self._scan(days_away=1, blackout=0).ideas) == 1

    def test_unknown_earnings_date_does_not_block(self):
        """No date must mean 'no known event', never 'assume the worst and freeze'."""
        from nifty_ai_agent.data.fundamentals import StockFundamentals
        res = scan_stocks(
            {"RELIANCE.NS": _trend_frame(100, 130)}, {"RELIANCE.NS": 130.0},
            master=FakeMaster(), risk_calculator=RiskCalculator(max_risk_pct=5.0),
            now=time(11, 0), premium_lookup=_live_lookup(20.0),
            fundamentals={"RELIANCE": StockFundamentals(symbol="RELIANCE")},
            earnings_blackout_days=3,
        )
        assert len(res.ideas) == 1

    def test_no_fundamentals_at_all_does_not_block(self):
        res = scan_stocks(
            {"RELIANCE.NS": _trend_frame(100, 130)}, {"RELIANCE.NS": 130.0},
            master=FakeMaster(), risk_calculator=RiskCalculator(max_risk_pct=5.0),
            now=time(11, 0), premium_lookup=_live_lookup(20.0),
            earnings_blackout_days=3,
        )
        assert len(res.ideas) == 1


class TestSymbolResolution:
    """A ticker retired by a corporate action must not silently become a 404."""

    @staticmethod
    def _master(rows):
        from nifty_ai_agent.data.instrument_master import InstrumentMaster
        m = InstrumentMaster()
        m._rows = rows
        return m

    @staticmethod
    def _eq(symbol, name, key=None):
        return {"segment": "NSE_EQ", "trading_symbol": symbol, "name": name,
                "instrument_key": key or f"NSE_EQ|{symbol}"}

    def _rows(self):
        return [
            self._eq("TMPV", "TATA MOTORS PASS VEH LTD"),
            self._eq("TMCV", "TATA MOTORS LIMITED"),
            self._eq("RELIANCE", "RELIANCE INDUSTRIES LTD"),
            self._eq("TATAPOWER", "TATA POWER CO LTD"),
        ]

    def test_exact_symbol_resolves(self):
        m = self._master(self._rows())
        match = m.resolve_equity("RELIANCE")
        assert match.trading_symbol == "RELIANCE" and match.is_exact

    def test_retired_symbol_follows_the_rename(self):
        m = self._master(self._rows())
        match = m.resolve_equity("TATAMOTORS")
        assert match.trading_symbol == "TMPV"
        assert not match.is_exact
        assert match.requested == "TATAMOTORS"

    def test_equity_key_uses_the_successor(self):
        m = self._master(self._rows())
        assert m.equity_key("TATAMOTORS") == "NSE_EQ|TMPV"

    def test_lowercase_input_resolves(self):
        assert self._master(self._rows()).resolve_equity("reliance").trading_symbol == "RELIANCE"

    def test_unknown_symbol_returns_none(self):
        assert self._master(self._rows()).resolve_equity("NOSUCHCO") is None

    def test_resolution_never_guesses_by_similarity(self):
        """Substituting a merely similar company into a scan would trade the wrong business."""
        assert self._master(self._rows()).resolve_equity("TATAPOWR") is None

    def test_suggestions_rank_the_intended_name_first(self):
        m = self._master(self._rows())
        assert m.suggest_symbols("RELAINCE")[0][0] == "RELIANCE"

    def test_suggestions_match_on_company_name(self):
        m = self._master(self._rows())
        found = [s for s, _ in m.suggest_symbols("TATA MOTORS")]
        assert "TMPV" in found and "TMCV" in found

    def test_no_suggestions_for_nonsense(self):
        assert self._master(self._rows()).suggest_symbols("ZZZZQQ") == []


class TestUniverseIsCurrent:
    def test_no_retired_tickers_in_the_scan_universe(self):
        from nifty_ai_agent.data.instrument_master import RENAMED_SYMBOLS
        from nifty_ai_agent.data.nifty50_stocks import NIFTY50_SYMBOLS

        bare = {s[:-3] if s.endswith(".NS") else s for s in NIFTY50_SYMBOLS}
        stale = bare & set(RENAMED_SYMBOLS)
        assert not stale, f"universe still lists retired tickers: {sorted(stale)}"
