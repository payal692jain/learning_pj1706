"""Tests for BANKNIFTY constituent option suggestions and the instrument master."""

from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from nifty_ai_agent.data.bank_options import (
    BankMove,
    format_bank_options,
    suggest_bank_options,
)
from nifty_ai_agent.data.instrument_master import InstrumentMaster, OptionContract
from nifty_ai_agent.strategies.base import SignalType


def _epoch_ms(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)


@pytest.fixture
def master(tmp_path):
    """An InstrumentMaster backed by a handful of fake rows — no network."""
    expiry = date.today() + timedelta(days=10)
    rows = [
        {"segment": "NSE_EQ", "trading_symbol": "HDFCBANK",
         "instrument_key": "NSE_EQ|INE040A01034", "isin": "INE040A01034"},
    ]
    for strike in (1700, 1750, 1800):
        for opt in ("CE", "PE"):
            rows.append({
                "segment": "NSE_FO", "asset_symbol": "HDFCBANK", "instrument_type": opt,
                "instrument_key": f"NSE_FO|{strike}{opt}", "trading_symbol": f"HDFCBANK {strike} {opt}",
                "strike_price": float(strike), "expiry": _epoch_ms(expiry), "lot_size": 550,
            })
    # An already-expired contract that must never be suggested.
    rows.append({
        "segment": "NSE_FO", "asset_symbol": "HDFCBANK", "instrument_type": "CE",
        "instrument_key": "NSE_FO|DEAD", "trading_symbol": "HDFCBANK 1750 CE OLD",
        "strike_price": 1750.0, "expiry": _epoch_ms(date.today() - timedelta(days=5)),
        "lot_size": 550,
    })

    im = InstrumentMaster(cache_file=tmp_path / "master.json")
    im._rows = rows  # bypass the download
    return im


class TestInstrumentMaster:
    def test_resolves_an_equity_key(self, master):
        assert master.equity_key("HDFCBANK") == "NSE_EQ|INE040A01034"

    def test_unknown_symbol_returns_none(self, master):
        assert master.equity_key("NOTABANK") is None

    def test_expired_contracts_are_excluded(self, master):
        keys = [c.instrument_key for c in master.option_contracts("HDFCBANK")]
        assert "NSE_FO|DEAD" not in keys

    def test_atm_contract_picks_the_nearest_strike(self, master):
        contract = master.atm_contract("HDFCBANK", spot=1743.0, opt_type="CE")
        assert contract.strike == 1750.0
        assert contract.opt_type == "CE"
        assert contract.lot_size == 550

    def test_atm_contract_respects_option_type(self, master):
        assert master.atm_contract("HDFCBANK", 1743.0, "PE").opt_type == "PE"

    def test_no_contracts_for_an_unknown_underlying(self, master):
        assert master.atm_contract("NOTABANK", 100.0, "CE") is None


def _client(prices: dict[str, float]) -> MagicMock:
    client = MagicMock()
    client.get_ltp.return_value = prices
    return client


class TestSuggestBankOptions:
    def test_bullish_signal_suggests_calls_on_the_leaders(self, master):
        moves = [BankMove("HDFCBANK", 1743.0, +1.4)]
        ideas = suggest_bank_options(
            SignalType.BUY_CE, _client({"NSE_FO|1750CE": 32.0}), master=master, moves=moves,
        )
        assert len(ideas) == 1
        assert ideas[0].opt_type == "CE"
        assert ideas[0].strike == 1750.0
        assert ideas[0].premium == 32.0
        assert ideas[0].cost_per_lot == pytest.approx(32.0 * 550)

    def test_bearish_signal_suggests_puts_on_the_laggards(self, master):
        moves = [BankMove("HDFCBANK", 1743.0, -1.6)]
        ideas = suggest_bank_options(
            SignalType.BUY_PE, _client({"NSE_FO|1750PE": 28.0}), master=master, moves=moves,
        )
        assert ideas[0].opt_type == "PE"

    def test_a_bank_moving_against_the_signal_is_not_suggested(self, master):
        """A call on the one bank falling while the index rises is not confirmation —
        it is a bet against the only constituent that disagrees."""
        moves = [BankMove("HDFCBANK", 1743.0, -1.4)]
        ideas = suggest_bank_options(
            SignalType.BUY_CE, _client({"NSE_FO|1750CE": 32.0}), master=master, moves=moves,
        )
        assert ideas == []

    def test_a_barely_moving_bank_is_not_suggested(self, master):
        moves = [BankMove("HDFCBANK", 1743.0, +0.1)]
        ideas = suggest_bank_options(
            SignalType.BUY_CE, _client({"NSE_FO|1750CE": 32.0}), master=master, moves=moves,
        )
        assert ideas == []

    def test_hold_suggests_nothing(self, master):
        ideas = suggest_bank_options(
            SignalType.HOLD, _client({}), master=master,
            moves=[BankMove("HDFCBANK", 1743.0, +2.0)],
        )
        assert ideas == []

    def test_a_dead_token_yields_no_suggestions_rather_than_invented_premiums(self, master):
        client = MagicMock()
        client.get_ltp.side_effect = RuntimeError("token expired")
        ideas = suggest_bank_options(
            SignalType.BUY_CE, client, master=master,
            moves=[BankMove("HDFCBANK", 1743.0, +1.4)],
        )
        assert ideas == []

    def test_a_contract_with_no_live_premium_is_skipped(self, master):
        ideas = suggest_bank_options(
            SignalType.BUY_CE, _client({"NSE_FO|1750CE": 0.0}), master=master,
            moves=[BankMove("HDFCBANK", 1743.0, +1.4)],
        )
        assert ideas == []

    def test_top_n_caps_the_suggestion_count(self, master):
        moves = [BankMove("HDFCBANK", 1743.0, +1.4), BankMove("HDFCBANK", 1743.0, +1.2)]
        ideas = suggest_bank_options(
            SignalType.BUY_CE, _client({"NSE_FO|1750CE": 32.0}),
            top_n=1, master=master, moves=moves,
        )
        assert len(ideas) <= 1


class TestFormatBankOptions:
    def test_lines_show_the_move_strike_and_affordability(self, master):
        ideas = suggest_bank_options(
            SignalType.BUY_CE, _client({"NSE_FO|1750CE": 32.0}), master=master,
            moves=[BankMove("HDFCBANK", 1743.0, +1.4)],
        )
        lines = "\n".join(format_bank_options(ideas, capital=50_000))
        assert "HDFCBANK" in lines
        assert "1750CE" in lines
        assert "+1.4%" in lines
        assert "₹17,600" in lines  # 32 × 550

    def test_no_ideas_renders_nothing(self):
        assert format_bank_options([], capital=50_000) == []
