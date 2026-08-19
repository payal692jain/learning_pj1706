"""Tests for the sector-grouped scan view and its cash-market levels."""

import pytest

from nifty_ai_agent.data.fundamentals import StockFundamentals
from nifty_ai_agent.reports.sector_scan import (
    cash_line,
    format_sector_scan,
    group_by_sector,
    sector_of,
)
from nifty_ai_agent.strategies.stock_scanner import ScanResult, StockIdea


def _idea(symbol="RELIANCE", opt_type="CE", sector="Energy", confidence=70,
          spot=1400.0, cash_target=1500.0, cash_stop=1350.0):
    return StockIdea(
        symbol=symbol, signal=f"BUY_{opt_type}", confidence=confidence,
        conviction="MODERATE", opt_type=opt_type, strike=1400.0,
        expiry="25-Aug-2026", lot_size=500, entry_premium=32.0, is_live=True,
        spot=spot, target=spot * 1.005, stop_loss=spot * 0.997, rr=2.0,
        fundamentals=StockFundamentals(symbol=symbol, sector=sector),
        cash_target=cash_target, cash_stop=cash_stop,
    )


def _result(ideas):
    return ScanResult(ideas=ideas, scanned=50, actionable=len(ideas), errors=0)


class TestSectorGrouping:
    def test_reads_the_sector_from_fundamentals(self):
        assert sector_of(_idea(sector="Technology")) == "Technology"

    def test_missing_fundamentals_fall_back_to_other(self):
        idea = _idea()
        idea.fundamentals = None
        assert sector_of(idea) == "Other"

    def test_groups_by_sector(self):
        groups = group_by_sector([
            _idea("TCS", sector="Technology"), _idea("INFY", sector="Technology"),
            _idea("HDFCBANK", sector="Financial Services"),
        ])
        by_name = {g.name: g for g in groups}
        assert len(by_name["Technology"].ideas) == 2
        assert len(by_name["Financial Services"].ideas) == 1

    def test_lean_follows_the_majority(self):
        bullish = group_by_sector([_idea("A", "CE"), _idea("B", "CE")])[0]
        bearish = group_by_sector([_idea("A", "PE"), _idea("B", "PE")])[0]
        mixed = group_by_sector([_idea("A", "CE"), _idea("B", "PE")])[0]
        assert bullish.lean == "BULLISH"
        assert bearish.lean == "BEARISH"
        assert mixed.lean == "MIXED"

    def test_most_lopsided_sector_ranks_first(self):
        """4-0 is a stronger read than 2-1 and should lead."""
        groups = group_by_sector(
            [_idea(f"T{i}", "CE", "Technology") for i in range(4)]
            + [_idea("F1", "CE", "Financial Services"),
               _idea("F2", "PE", "Financial Services")]
        )
        assert groups[0].name == "Technology"


class TestCashLine:
    def test_bullish_gives_a_buy_with_levels(self):
        line = cash_line(_idea(opt_type="CE", spot=1400, cash_target=1500, cash_stop=1350))
        assert "BUY 1400" in line and "1500" in line and "1350" in line

    def test_bearish_never_shows_a_buy_price(self):
        """A precise, actionable buy price under a bearish call is the worst kind of wrong."""
        line = cash_line(_idea(opt_type="PE"))
        assert "BUY" not in line
        assert "AVOID" in line or "EXIT" in line

    def test_falls_back_to_intraday_levels_and_says_so(self):
        """When daily bars were unavailable the levels are scalp-width — label them."""
        idea = _idea(cash_target=0.0, cash_stop=0.0)
        line = cash_line(idea)
        assert "(intraday)" in line

    def test_swing_levels_are_not_labelled_intraday(self):
        assert "(intraday)" not in cash_line(_idea())


class TestFormatting:
    def test_groups_appear_with_counts(self):
        _, body = format_sector_scan(_result([
            _idea("TCS", "PE", "Technology"), _idea("INFY", "PE", "Technology"),
        ]))
        assert "Technology" in body and "0↑ 2↓" in body

    def test_long_sector_names_are_shortened(self):
        _, body = format_sector_scan(_result([_idea(sector="Financial Services")]))
        assert "Financials" in body

    def test_title_names_the_leading_sector(self):
        title, _ = format_sector_scan(_result([
            _idea("TCS", "PE", "Technology"), _idea("INFY", "PE", "Technology"),
        ]))
        assert "Technology" in title and "bearish" in title

    def test_each_idea_carries_both_the_option_and_the_stock(self):
        _, body = format_sector_scan(_result([_idea()]))
        assert "1400CE" in body      # the option
        assert "stock:" in body      # the cash trade

    def test_empty_result_explains_itself(self):
        title, body = format_sector_scan(_result([]))
        assert "no setups" in title and "No actionable" in body

    def test_body_fits_the_pushover_limit(self):
        """Pushover rejects an over-long body outright — the alert never arrives."""
        from nifty_ai_agent.reports.layout import PUSHOVER_LIMIT
        ideas = [
            _idea(f"SYM{i}", "CE" if i % 2 else "PE",
                  ["Technology", "Energy", "Financial Services", "Healthcare"][i % 4])
            for i in range(16)
        ]
        _, body = format_sector_scan(_result(ideas))
        assert len(body) <= PUSHOVER_LIMIT
        assert "not advice" in body.lower()


class TestCashLevelDerivation:
    """Cash levels must be swing-width, not the intraday scalp the option uses."""

    @staticmethod
    def _daily_history(bars=40, start=1000.0, step=5.0, per_day=12):
        """Intraday bars spanning *bars* DISTINCT sessions.

        A flat pd.date_range of 5-minute bars looks long but covers only a few
        calendar days — 3,000 of them span ~10 days, which resamples to far too
        few daily bars for an ATR. Sessions are therefore built one day at a time.
        """
        import numpy as np
        import pandas as pd

        frames = []
        for day in range(bars):
            stamp = pd.Timestamp("2026-06-01 09:15") + pd.Timedelta(days=day)
            idx = pd.date_range(stamp, periods=per_day, freq="5min")
            base = start + step * day
            close = np.linspace(base, base + step, per_day)
            frames.append(pd.DataFrame(
                {"open": close, "high": close + 4, "low": close - 4,
                 "close": close, "volume": np.full(per_day, 1000.0)},
                index=idx,
            ))
        df = pd.concat(frames)
        df.index.name = "datetime"
        return df

    def test_levels_are_wider_than_an_intraday_stop(self):
        from nifty_ai_agent.strategies.stock_scanner import _cash_levels
        hist = self._daily_history()
        spot = float(hist["close"].iloc[-1])
        target, stop = _cash_levels(hist, spot, bullish=True)
        assert target > spot > stop
        assert abs(stop - spot) / spot * 100 > 0.5   # not a 0.4% scalp

    def test_bearish_inverts_the_levels(self):
        from nifty_ai_agent.strategies.stock_scanner import _cash_levels
        hist = self._daily_history()
        spot = float(hist["close"].iloc[-1])
        target, stop = _cash_levels(hist, spot, bullish=False)
        assert target < spot < stop

    def test_too_few_daily_bars_yields_nothing(self):
        """Better to fall back and say so than to quote an ATR from 2 bars."""
        from nifty_ai_agent.strategies.stock_scanner import _cash_levels
        hist = self._daily_history(bars=3)
        assert _cash_levels(hist, 1000.0, bullish=True) == (0.0, 0.0)
