"""Tests for the on-demand single-stock analysis report and its command."""

from datetime import date, timedelta

import pytest

from nifty_ai_agent.data.fundamentals import Headline, StockFundamentals
from nifty_ai_agent.notifier.telegram_commands import parse_command
from nifty_ai_agent.reports.analyse_one import normalise_symbol
from nifty_ai_agent.reports.stock_analysis import format_stock_analysis
from nifty_ai_agent.strategies.base import SignalType
from nifty_ai_agent.strategies.stock_scanner import StockIdea


def _fund(**kw):
    base = dict(symbol="RELIANCE", pe=23.8, forward_pe=18.2, price_to_book=1.95,
                eps=54.8, sector="Energy", week52_high=1600.0, week52_low=1200.0,
                avg_volume=14_000_000.0)
    base.update(kw)
    return StockFundamentals(**base)


def _idea(**kw):
    base = dict(symbol="RELIANCE", signal="BUY_CE", confidence=72, conviction="STRONG",
                opt_type="CE", strike=1400.0, expiry="25-Aug-2026", lot_size=500,
                entry_premium=32.0, is_live=True, spot=1400.0, target=1420.0,
                stop_loss=1390.0, rr=2.0, target_premium=42.0, stop_premium=27.0)
    base.update(kw)
    return StockIdea(**base)


class TestNormaliseSymbol:
    @pytest.mark.parametrize("raw", ["reliance", "RELIANCE", " Reliance ", "RELIANCE.NS"])
    def test_variants_resolve_to_one_ticker(self, raw):
        assert normalise_symbol(raw) == "RELIANCE.NS"


class TestDeliveryVerdict:
    def test_uptrend_mid_range_accumulates(self):
        out = format_stock_analysis(
            "RELIANCE", 1400.0, SignalType.BUY_CE, 72, "STRONG", "",
            fund=_fund(), idea=_idea(),
        )
        assert "BUY THE STOCK?  ACCUMULATE" in out

    def test_near_52w_high_is_not_an_accumulate(self):
        """A strong chart at the top of the range is a momentum trade, not a buy-and-hold."""
        out = format_stock_analysis(
            "RELIANCE", 1595.0, SignalType.BUY_CE, 72, "STRONG", "", fund=_fund(),
        )
        assert "ACCUMULATE" not in out
        assert "near 52w high" in out

    def test_imminent_results_downgrade_the_verdict(self):
        soon = (date.today() + timedelta(days=2)).isoformat()
        out = format_stock_analysis(
            "RELIANCE", 1400.0, SignalType.BUY_CE, 72, "STRONG", "",
            fund=_fund(next_earnings=soon),
        )
        assert "ACCUMULATE" not in out
        assert "gap risk" in out

    def test_flat_trend_has_no_edge(self):
        out = format_stock_analysis(
            "RELIANCE", 1400.0, SignalType.HOLD, 0, "NO_TRADE", "", fund=_fund(),
        )
        assert "NO EDGE" in out

    def test_downtrend_avoids(self):
        out = format_stock_analysis(
            "RELIANCE", 1400.0, SignalType.BUY_PE, 65, "MODERATE", "", fund=_fund(),
        )
        assert "AVOID FOR NOW" in out


class TestReportSections:
    def test_both_questions_are_answered_separately(self):
        out = format_stock_analysis(
            "RELIANCE", 1400.0, SignalType.BUY_CE, 72, "STRONG", "",
            fund=_fund(), idea=_idea(),
        )
        assert "BUY THE STOCK?" in out and "INTRADAY OPTION" in out

    def test_option_block_carries_entry_and_exit_prices(self):
        out = format_stock_analysis(
            "RELIANCE", 1400.0, SignalType.BUY_CE, 72, "STRONG", "",
            fund=_fund(), idea=_idea(),
        )
        assert "buy ₹32" in out and "sell ₹42" in out and "exit ₹27" in out
        assert "500 qty = ₹16,000/lot" in out

    def test_blocked_entry_explains_itself(self):
        out = format_stock_analysis(
            "RELIANCE", 1400.0, SignalType.BUY_CE, 72, "STRONG", "",
            fund=_fund(), idea=None, blocked_reason="results in 2d",
        )
        assert "NO TRADE — results in 2d" in out

    def test_no_signal_states_the_conviction(self):
        out = format_stock_analysis(
            "RELIANCE", 1400.0, SignalType.HOLD, 0, "NO_TRADE", "4 of 6 flat",
            fund=_fund(),
        )
        assert "NO TRADE — NO_TRADE (0%)" in out

    def test_headlines_carry_their_event_tags(self):
        out = format_stock_analysis(
            "RELIANCE", 1400.0, SignalType.BUY_CE, 72, "STRONG", "",
            fund=_fund(headlines=[Headline("Board clears merger", "2026-08-01", ["MERGER"])]),
        )
        assert "[MERGER]" in out and "Board clears merger" in out

    def test_unusual_volume_is_flagged(self):
        out = format_stock_analysis(
            "RELIANCE", 1400.0, SignalType.BUY_CE, 72, "STRONG", "",
            fund=_fund(), volume_ratio=2.4,
        )
        assert "2.4× avg" in out and "unusual" in out

    def test_normal_volume_is_not_flagged(self):
        out = format_stock_analysis(
            "RELIANCE", 1400.0, SignalType.BUY_CE, 72, "STRONG", "",
            fund=_fund(), volume_ratio=1.1,
        )
        assert "1.1× avg" in out and "unusual" not in out

    def test_renders_without_fundamentals(self):
        """yfinance can fail; the technical half must still come through."""
        out = format_stock_analysis(
            "RELIANCE", 1400.0, SignalType.BUY_CE, 72, "STRONG", "", fund=None,
        )
        assert "BUY THE STOCK?" in out and "VALUATION" not in out

    def test_carries_a_risk_disclaimer(self):
        out = format_stock_analysis("X", 1.0, SignalType.HOLD, 0, "NO_TRADE", "")
        assert "not advice" in out


class TestAnalyseCommand:
    @pytest.mark.parametrize("text,expected", [
        ("/analyse RELIANCE", ("analyse", "RELIANCE")),
        ("/analyze TCS", ("analyze", "TCS")),
        ("/stock britannia", ("stock", "britannia")),
        ("/analyse@MyBot BSE", ("analyse", "BSE")),
    ])
    def test_command_and_symbol_parse(self, text, expected):
        assert parse_command(text) == expected

    def test_missing_symbol_returns_usable_examples(self):
        """Asked with no symbol, the reply must show what to type — not just complain."""
        from nifty_ai_agent.notifier.telegram_commands import _dispatch
        reply = _dispatch("analyse", "", 1, "u", store=None)
        assert "/analyse NIFTY" in reply and "RELIANCE" in reply

    def test_help_advertises_the_command(self):
        from nifty_ai_agent.notifier.telegram_commands import HELP
        assert "/analyse" in HELP


class TestIndexRouting:
    """/analyse must reach the index path, not look NIFTY up as an equity ticker."""

    @pytest.mark.parametrize("raw,expected", [
        ("NIFTY", "NIFTY"), ("nifty", "NIFTY"), ("Nifty 50", "NIFTY"),
        ("NIFTY50", "NIFTY"), ("BANKNIFTY", "BANKNIFTY"), ("bank nifty", "BANKNIFTY"),
        ("SENSEX", "SENSEX"), ("^NSEI", "NIFTY"),
    ])
    def test_index_aliases_resolve(self, raw, expected):
        from nifty_ai_agent.reports.analyse_index import resolve_index
        assert resolve_index(raw) == expected

    @pytest.mark.parametrize("raw", ["RELIANCE", "TCS", "BSE", "NIFTYBEES1"])
    def test_stocks_are_not_mistaken_for_indices(self, raw):
        from nifty_ai_agent.reports.analyse_index import resolve_index
        assert resolve_index(raw) is None


class TestConversationalEntry:
    """Making people remember a command name is what stops a bot being used."""

    def _reply(self, text, monkeypatch, tradeable=True):
        import nifty_ai_agent.notifier.telegram_commands as tc
        monkeypatch.setattr(tc, "_analyse", lambda s: f"ANALYSED:{s}")
        monkeypatch.setattr(tc, "_is_tradeable", lambda w: tradeable)
        return tc.handle_update({"message": {"chat": {"id": 1}, "text": text}}, None)[1]

    def test_bare_symbol_without_a_slash_is_analysed(self, monkeypatch):
        assert self._reply("NIFTY", monkeypatch) == "ANALYSED:NIFTY"

    def test_bare_symbol_with_a_slash_is_analysed(self, monkeypatch):
        assert self._reply("/reliance", monkeypatch) == "ANALYSED:reliance"

    def test_two_word_index_name_is_analysed(self, monkeypatch):
        assert self._reply("bank nifty", monkeypatch) == "ANALYSED:bank nifty"

    def test_real_commands_still_win(self, monkeypatch):
        """'status' is a command, not a ticker — routing must not swallow it."""
        class _Store:
            def get_subscriber(self, cid): return None
        import nifty_ai_agent.notifier.telegram_commands as tc
        monkeypatch.setattr(tc, "_analyse", lambda s: pytest.fail("should not analyse"))
        reply = tc.handle_update(
            {"message": {"chat": {"id": 1}, "text": "/status"}}, _Store(),
        )[1]
        assert "not subscribed" in reply

    def test_gibberish_is_not_treated_as_a_symbol(self, monkeypatch):
        reply = self._reply("hey there, how much money", monkeypatch, tradeable=False)
        assert "help" in reply.lower()

    def test_mistyped_command_is_not_sent_for_analysis(self, monkeypatch):
        """/frobnicate must not become a slow network call that fails confusingly."""
        reply = self._reply("/frobnicate", monkeypatch, tradeable=False)
        assert "Unknown command" in reply

    def test_help_lists_index_usage(self):
        from nifty_ai_agent.notifier.telegram_commands import HELP
        assert "/analyse NIFTY" in HELP and "BANKNIFTY" in HELP

    def test_bare_analyse_offers_examples(self):
        from nifty_ai_agent.notifier.telegram_commands import _dispatch
        reply = _dispatch("analyse", "", 1, "u", store=None)
        assert "/analyse NIFTY" in reply and "RELIANCE" in reply
