"""Tests for the combined BSE Ltd + NSE currency digest."""

from nifty_ai_agent.reports.bse_currency import format_bse_currency_scan
from nifty_ai_agent.reports.layout import PUSHOVER_LIMIT
from nifty_ai_agent.strategies.stock_scanner import ScanResult, StockIdea


def _idea(symbol="BSE", opt_type="CE", premium_=52.0, lot=200, strike=3400.0):
    return StockIdea(
        symbol=symbol, signal=f"BUY_{opt_type}", confidence=78, conviction="STRONG",
        opt_type=opt_type, strike=strike, expiry="25-Aug-2026", lot_size=lot,
        entry_premium=premium_, is_live=True, spot=strike,
        target=strike * 1.03, stop_loss=strike * 0.985, rr=2.0,
    )


def _result(ideas):
    return ScanResult(ideas=ideas, scanned=len(ideas) or 1,
                      actionable=len(ideas), errors=0)


def _empty():
    return ScanResult(ideas=[], scanned=1, actionable=0, errors=0)


class TestFormatBseCurrencyScan:
    def test_no_setups_still_returns_a_usable_notification(self):
        title, body = format_bse_currency_scan(_empty(), _empty())
        assert "no setups" in title
        assert "No actionable" in body
        assert len(body) <= PUSHOVER_LIMIT

    def test_both_books_are_labelled_separately(self):
        title, body = format_bse_currency_scan(
            _result([_idea()]),
            _result([_idea("USDINR", "PE", 0.2875, 1000, 87.75)]),
        )
        assert "BSE LTD" in body and "CURRENCY (NSE)" in body
        assert "BSE" in title and "USDINR" in title

    def test_currency_premium_keeps_its_paise_precision(self):
        """premium() rounds to rupees, which would show a 0.2875 USDINR option as 0."""
        _, body = format_bse_currency_scan(
            _empty(), _result([_idea("USDINR", "CE", 0.2875, 1000, 87.75)]),
        )
        assert "0.2875" in body
        assert " 0 " not in body

    def test_cost_per_lot_uses_contract_size(self):
        # 1000 units x 0.2875 = Rs 288, NOT 0.2875 x 1.
        _, body = format_bse_currency_scan(
            _empty(), _result([_idea("USDINR", "CE", 0.2875, 1000, 87.75)]),
        )
        assert "1,000 × 0.2875 = ₹288/lot" in body

    def test_equity_cost_per_lot(self):
        _, body = format_bse_currency_scan(_result([_idea()]), _empty())
        assert "200 × 52 = ₹10,400/lot" in body

    def test_estimated_premiums_are_flagged(self):
        idea = _idea()
        idea.is_live = False
        _, body = format_bse_currency_scan(_result([idea]), _empty())
        assert "*" in body and "estimated premium" in body

    def test_rbi_exposure_warning_always_present(self):
        _, body = format_bse_currency_scan(_result([_idea()]), _empty())
        assert "underlying exposure" in body

    def test_body_never_exceeds_the_pushover_limit(self):
        """Pushover REJECTS an over-long body outright — the alert simply never arrives."""
        many = [_idea(f"SYM{i}", "CE", 52.0 + i, 200, 3400.0 + i * 50) for i in range(6)]
        ccy = [_idea(p, "PE", 0.31, 1000, 87.5) for p in
               ("USDINR", "EURINR", "GBPINR", "JPYINR")]
        _, body = format_bse_currency_scan(_result(many), _result(ccy))
        assert len(body) <= PUSHOVER_LIMIT
        assert "Estimates, not advice" in body   # footer survives the trim


class TestHoldsAreAlwaysVisible:
    """A five-name book must never let a symbol vanish just because it is flat."""

    @staticmethod
    def _hold(symbol="BSE", spot=3452.1):
        from nifty_ai_agent.strategies.stock_scanner import HoldRead
        return HoldRead(symbol=symbol, confidence=0, conviction="NO_TRADE", spot=spot)

    def _held(self, holds):
        return ScanResult(ideas=[], scanned=len(holds), actionable=0,
                          errors=0, holds=holds)

    def test_bse_appears_even_when_undecided(self):
        """The reported bug: BSE HOLDs, currency fires, BSE disappears entirely."""
        _, body = format_bse_currency_scan(
            self._held([self._hold()]),
            _result([_idea("GBPINR", "CE", 0.9125, 1000, 129.0)]),
        )
        assert "BSE LTD" in body
        assert "BSE       no trade" in body
        assert "GBPINR" in body

    def test_all_flat_still_lists_every_symbol(self):
        _, body = format_bse_currency_scan(
            self._held([self._hold()]),
            self._held([self._hold(p, 90.0) for p in
                        ("USDINR", "EURINR", "GBPINR", "JPYINR")]),
        )
        for sym in ("BSE", "USDINR", "EURINR", "GBPINR", "JPYINR"):
            assert sym in body

    def test_section_omitted_when_nothing_was_scanned(self):
        _, body = format_bse_currency_scan(
            ScanResult(ideas=[], scanned=0, actionable=0, errors=0, holds=[]),
            _result([_idea("USDINR", "CE", 0.31, 1000, 95.5)]),
        )
        assert "BSE LTD" not in body

    def test_ideas_outrank_watch_lines_when_trimming(self):
        """A real idea must never be dropped to make room for a 'nothing here' line."""
        many = [_idea(f"SYM{i}", "CE", 52.0 + i, 200, 3400.0 + i * 50) for i in range(6)]
        holds = [self._hold(f"H{i}", 90.0 + i) for i in range(6)]
        _, body = format_bse_currency_scan(
            _result(many), self._held(holds),
        )
        assert len(body) <= PUSHOVER_LIMIT
        assert "SYM0" in body        # first idea survives the trim
