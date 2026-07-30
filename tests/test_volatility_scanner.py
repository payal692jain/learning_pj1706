"""Tests for the volatile-stock straddle scanner and its report."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from nifty_ai_agent.data.instrument_master import OptionContract
from nifty_ai_agent.reports.volatility_scan import format_volatility_scan
from nifty_ai_agent.strategies.volatility_scanner import (
    StraddleIdea,
    VolatilityScanResult,
    scan_volatile_straddles,
)


# ── Fixtures / helpers ──────────────────────────────────────────────────────────

def _vol_frame(spot: float, half_range: float, periods: int = 60) -> pd.DataFrame:
    """A flat-close OHLCV frame whose ATR resolves to ~2*half_range (deterministic).

    With close held flat, prev_close == close so true range == high-low == 2*half_range
    on every bar, and Wilder's ATR converges to that. atr_pct is then a known value.
    """
    idx = pd.date_range("2026-07-27 09:15", periods=periods, freq="5min")
    close = np.full(periods, spot, dtype=float)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + half_range,
            "low": close - half_range,
            "close": close,
            "volume": np.full(periods, 1000.0),
        },
        index=idx,
    )
    df.index.name = "datetime"
    return df


class _StraddleMaster:
    """Instrument-master stub returning an ATM contract at round(spot/10)*10."""

    def __init__(self, lot_size: int = 250, known: set[str] | None = None) -> None:
        self.lot_size = lot_size
        self.known = known  # None → every asset resolves

    def atm_contract(self, asset, spot, opt_type, expiry=None, allow_expiry_day=None):
        if self.known is not None and asset not in self.known:
            return None
        strike = round(spot / 10) * 10
        return OptionContract(
            instrument_key=f"NSE_FO|{asset}{opt_type}",
            trading_symbol=f"{asset}{strike}{opt_type}",
            asset_symbol=asset,
            strike=float(strike),
            opt_type=opt_type,
            expiry=expiry or date(2026, 8, 27),
            lot_size=self.lot_size,
        )


def _live_lookup(price: float):
    return lambda keys: {k: price for k in keys}


# ── scan_volatile_straddles ─────────────────────────────────────────────────────

class TestScanVolatileStraddles:
    def test_ranks_by_atr_and_builds_straddle(self):
        histories = {
            "A.NS": _vol_frame(100, 5),   # atr ≈ 10 → 10%
            "B.NS": _vol_frame(200, 2),   # atr ≈ 4  → 2%
        }
        spots = {"A.NS": 100.0, "B.NS": 200.0}
        res = scan_volatile_straddles(
            histories, spots, master=_StraddleMaster(),
            premium_lookup=_live_lookup(40.0), top_n=5,
        )

        assert res.scanned == 2
        assert res.ranked == 2
        assert [i.symbol for i in res.ideas] == ["A", "B"]      # most volatile first
        assert res.ideas[0].atr_pct == pytest.approx(10.0, abs=0.5)
        assert res.ideas[1].atr_pct == pytest.approx(2.0, abs=0.5)

        top = res.ideas[0]
        assert top.strike == 100.0
        assert top.total_premium == pytest.approx(80.0)         # 40 + 40
        assert top.breakeven_low == pytest.approx(20.0)
        assert top.breakeven_high == pytest.approx(180.0)
        assert top.breakeven_move_pct == pytest.approx(80.0)
        assert top.lot_cost == pytest.approx(80.0 * 250)
        assert top.ce_is_live and top.pe_is_live

    def test_estimate_when_no_live_quote(self):
        histories = {"A.NS": _vol_frame(100, 5)}
        res = scan_volatile_straddles(
            histories, {"A.NS": 100.0}, master=_StraddleMaster(),
        )
        assert len(res.ideas) == 1
        idea = res.ideas[0]
        assert idea.ce_is_live is False and idea.pe_is_live is False
        assert idea.ce_premium > 0 and idea.pe_premium > 0      # Black-Scholes
        assert idea.total_premium == pytest.approx(idea.ce_premium + idea.pe_premium)

    def test_top_n_caps_the_result(self):
        histories = {f"S{i}.NS": _vol_frame(100 + i, 5) for i in range(4)}
        spots = {sym: 100.0 + i for i, sym in enumerate(histories)}
        res = scan_volatile_straddles(
            histories, spots, master=_StraddleMaster(),
            premium_lookup=_live_lookup(10.0), top_n=2,
        )
        assert res.ranked == 4
        assert len(res.ideas) == 2

    def test_nonpositive_spot_is_skipped(self):
        res = scan_volatile_straddles(
            {"A.NS": _vol_frame(100, 5)}, {"A.NS": 0.0}, master=_StraddleMaster(),
        )
        assert res.scanned == 0
        assert res.ideas == []

    def test_missing_contract_is_skipped(self):
        # Master knows only A → B is volatility-ranked but yields no straddle.
        histories = {"A.NS": _vol_frame(100, 2), "B.NS": _vol_frame(100, 5)}
        spots = {"A.NS": 100.0, "B.NS": 100.0}
        res = scan_volatile_straddles(
            histories, spots, master=_StraddleMaster(known={"A"}),
            premium_lookup=_live_lookup(10.0), top_n=5,
        )
        assert res.ranked == 2                       # both scored
        assert [i.symbol for i in res.ideas] == ["A"]  # only A tradable

    def test_one_live_one_estimated_leg_flags_not_live(self):
        # Live quote for the CE leg only; the PE leg falls back to an estimate.
        def _ce_only(keys):
            return {k: 33.0 for k in keys if k.endswith("CE")}

        res = scan_volatile_straddles(
            {"A.NS": _vol_frame(100, 5)}, {"A.NS": 100.0},
            master=_StraddleMaster(), premium_lookup=_ce_only,
        )
        idea = res.ideas[0]
        assert idea.ce_is_live is True
        assert idea.pe_is_live is False
        assert idea.ce_premium == pytest.approx(33.0)


# ── format_volatility_scan ──────────────────────────────────────────────────────

class TestFormatVolatilityScan:
    def _idea(self, symbol="RELIANCE", live=True) -> StraddleIdea:
        return StraddleIdea(
            symbol=symbol, spot=1400.0, atr_pct=3.8, strike=1400.0,
            expiry="27-Aug-2026", lot_size=250, ce_premium=42.0, pe_premium=38.0,
            ce_is_live=live, pe_is_live=live, total_premium=80.0,
            breakeven_low=1320.0, breakeven_high=1480.0, breakeven_move_pct=5.7,
            lot_cost=20000.0,
        )

    def test_no_ideas_message(self):
        res = VolatilityScanResult(ideas=[], scanned=12, ranked=0, errors=0)
        title, body = format_volatility_scan(res)
        assert "none" in title.lower()
        assert "12 scanned" in body
        assert "full premium" in body

    def test_ranked_ideas_render(self):
        res = VolatilityScanResult(ideas=[self._idea()], scanned=50, ranked=48, errors=0)
        title, body = format_volatility_scan(res)
        assert "RELIANCE" in title
        assert "27-Aug-2026" in body
        assert "1320" in body and "1480" in body      # breakevens
        assert "either way" in body                    # straddle explainer

    def test_estimate_marker_present_when_not_live(self):
        res = VolatilityScanResult(ideas=[self._idea(live=False)], scanned=50, ranked=48, errors=0)
        _, body = format_volatility_scan(res)
        assert "*" in body
        assert "estimated premium" in body

    def test_no_estimate_marker_when_all_live(self):
        res = VolatilityScanResult(ideas=[self._idea(live=True)], scanned=50, ranked=48, errors=0)
        _, body = format_volatility_scan(res)
        assert "estimated premium" not in body
