"""Tests for the top gainers/losers digest."""

import pytest

from nifty_ai_agent.data.market_movers import Mover, MoversSnapshot, _to_mover
from nifty_ai_agent.reports.layout import PUSHOVER_LIMIT
from nifty_ai_agent.reports.movers import format_movers


def _quote(symbol="RELIANCE", last=1400.0, net=14.0, volume=1_000_000):
    return {"symbol": symbol, "last_price": last, "net_change": net,
            "volume": volume, "ohlc": {"close": last}}


def _mover(symbol="X", pct=1.5, last=1000.0):
    return Mover(symbol=symbol, last_price=last, change_pct=pct,
                 net_change=last * pct / 100, volume=1_000_000)


class TestQuoteParsing:
    def test_percent_is_derived_from_net_change(self):
        """ohlc.close is the CURRENT session close — using it shows every stock flat."""
        m = _to_mover(_quote(last=2280.0, net=-33.2))
        assert m.change_pct == pytest.approx(-1.44, abs=0.01)
        assert m.last_price == 2280.0

    def test_gain_is_positive(self):
        assert _to_mover(_quote(last=1414.0, net=14.0)).change_pct > 0

    def test_untraded_symbol_is_dropped(self):
        """A stale price with no volume today would render as a fake mover."""
        assert _to_mover(_quote(volume=0)) is None

    def test_missing_or_zero_price_is_dropped(self):
        assert _to_mover(_quote(last=0.0)) is None
        assert _to_mover({"symbol": "X"}) is None

    def test_unnamed_quote_is_dropped(self):
        assert _to_mover(_quote(symbol="")) is None

    def test_impossible_previous_close_is_dropped(self):
        """net_change larger than the price implies a bad feed, not a 200% move."""
        assert _to_mover(_quote(last=100.0, net=150.0)) is None

    def test_malformed_types_do_not_raise(self):
        assert _to_mover({"symbol": "X", "last_price": "abc"}) is None

    def test_turnover_is_in_crore(self):
        m = Mover("X", last_price=1000.0, change_pct=1.0, net_change=10.0,
                  volume=1_000_000)
        assert m.turnover_cr == pytest.approx(100.0)


class TestSnapshot:
    def test_advance_decline_ratio(self):
        s = MoversSnapshot(advances=60, declines=120)
        assert s.advance_decline_ratio == pytest.approx(0.5)

    def test_no_declines_does_not_divide_by_zero(self):
        assert MoversSnapshot(advances=10, declines=0).advance_decline_ratio == 0.0


class TestFormatting:
    def _snapshot(self, n=10):
        return MoversSnapshot(
            gainers=[_mover(f"UP{i}", 8.0 - i * 0.5, 1000.0 + i) for i in range(n)],
            losers=[_mover(f"DN{i}", -4.0 + i * 0.2, 500.0 + i) for i in range(n)],
            scanned=207, advances=65, declines=141,
        )

    def test_lists_both_halves(self):
        _, body = format_movers(self._snapshot())
        assert "TOP GAINERS" in body and "TOP LOSERS" in body
        assert "UP0" in body and "DN0" in body

    def test_title_leads_with_the_extremes(self):
        title, _ = format_movers(self._snapshot())
        assert "UP0" in title and "DN0" in title

    def test_breadth_line_is_present(self):
        _, body = format_movers(self._snapshot())
        assert "207 stocks" in body and "65↑ / 141↓" in body

    def test_sign_stays_attached_to_its_number(self):
        """'+  7.94%' — a sign formatted outside the field floats away from it."""
        _, body = format_movers(self._snapshot())
        assert "+  " not in body

    def test_body_fits_the_pushover_limit(self):
        """Pushover rejects an over-long body outright; the alert simply never arrives."""
        _, body = format_movers(self._snapshot(n=10))
        assert len(body) <= PUSHOVER_LIMIT

    def test_empty_snapshot_explains_itself(self):
        title, body = format_movers(MoversSnapshot())
        assert "unavailable" in title
        assert "No quote data" in body

    def test_carries_a_disclaimer(self):
        _, body = format_movers(self._snapshot())
        assert "not advice" in body


class TestCommand:
    def test_help_advertises_movers(self):
        from nifty_ai_agent.notifier.telegram_commands import HELP
        assert "/movers" in HELP

    @pytest.mark.parametrize("cmd", ["movers", "gainers", "losers", "topmovers"])
    def test_aliases_route_to_movers(self, cmd, monkeypatch):
        import nifty_ai_agent.notifier.telegram_commands as tc
        monkeypatch.setattr(tc, "_movers", lambda: "MOVERS")
        assert tc._dispatch(cmd, "", 1, "u", store=None) == "MOVERS"
