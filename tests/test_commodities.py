"""Tests for crude/commodity context and its India-specific interpretation."""

import pytest

from nifty_ai_agent.data.market_context import (
    IndexSnapshot,
    MarketContext,
    crude_impact,
    format_context_for_notification,
)


def _snap(name="Crude (INR)", pct=0.0, price=8147.0):
    return IndexSnapshot(
        name=name, symbol="CRUDEOIL FUT", price=price, change_pct=pct,
        direction="↑" if pct > 0.05 else ("↓" if pct < -0.05 else "→"),
    )


def _ctx(commodities):
    return MarketContext(indices=[], gift_nifty=None, commodities=commodities)


class TestCrudeImpact:
    def test_a_crude_rally_is_a_headwind_for_india(self):
        """India imports ~85% of its crude — rising oil is the opposite of risk-on."""
        label, why = crude_impact(_snap(pct=3.0))
        assert label == "HEADWIND"
        assert "rupee" in why

    def test_falling_crude_is_a_tailwind(self):
        label, why = crude_impact(_snap(pct=-2.5))
        assert label == "TAILWIND"
        assert "deficit" in why or "inflation" in why

    def test_small_moves_are_not_called_either_way(self):
        """Crude moves +/-1% on nothing; calling that a signal is noise."""
        assert crude_impact(_snap(pct=0.8))[0] == "NEUTRAL"
        assert crude_impact(_snap(pct=-1.0))[0] == "NEUTRAL"

    def test_threshold_boundary(self):
        assert crude_impact(_snap(pct=1.5))[0] == "HEADWIND"
        assert crude_impact(_snap(pct=1.4))[0] == "NEUTRAL"

    def test_missing_crude_is_reported_not_guessed(self):
        assert crude_impact(None)[0] == "UNKNOWN"

    def test_explanation_names_both_sides_of_the_trade(self):
        """The effect is not uniform — upstream gains where refiners lose."""
        _, why = crude_impact(_snap(pct=4.0))
        assert "ONGC" in why


class TestCrudeSelection:
    def test_rupee_crude_is_preferred(self):
        ctx = _ctx([_snap("Brent", 2.0), _snap("Crude (INR)", 1.0)])
        assert ctx.crude.name == "Crude (INR)"

    def test_falls_back_to_brent_then_wti(self):
        assert _ctx([_snap("WTI", 1.0), _snap("Brent", 2.0)]).crude.name == "Brent"
        assert _ctx([_snap("WTI", 1.0)]).crude.name == "WTI"

    def test_no_commodities_yields_none(self):
        assert _ctx([]).crude is None

    def test_natural_gas_is_not_mistaken_for_crude(self):
        assert _ctx([_snap("Nat Gas (INR)", 5.0)]).crude is None


class TestBiasIsolation:
    def test_crude_does_not_leak_into_the_global_bias(self):
        """A crude rally must never read as 'another green market'."""
        from nifty_ai_agent.data.market_context import compute_global_bias

        equities = [
            IndexSnapshot("S&P 500", "^GSPC", 5000, -0.5, "↓"),
            IndexSnapshot("Dow Jones", "^DJI", 40000, -0.6, "↓"),
            IndexSnapshot("NASDAQ", "^IXIC", 16000, -0.7, "↓"),
        ]
        bearish = compute_global_bias(equities)
        with_crude = compute_global_bias(equities + [_snap(pct=5.0)])
        assert bearish == with_crude == "BEARISH"


class TestFormatting:
    def test_commodities_are_rendered(self):
        body = format_context_for_notification(
            _ctx([_snap("Crude (INR)", 3.2), _snap("Nat Gas (INR)", 1.0)])
        )
        assert "Crude (INR): +3.20%" in body
        assert "Nat Gas (INR): +1.00%" in body

    def test_impact_line_accompanies_the_number(self):
        body = format_context_for_notification(_ctx([_snap(pct=3.2)]))
        assert "Crude → HEADWIND" in body

    def test_context_without_commodities_still_renders(self):
        assert "Global Bias" in format_context_for_notification(_ctx([]))


class TestCrudeInConfidence:
    """Crude must reach the signal with its sign INVERTED for India."""

    @staticmethod
    def _snapshot(**kw):
        from nifty_ai_agent.strategies.global_analyser import GlobalSnapshot
        base = dict(global_bias="NEUTRAL", gift_nifty_pct=0.0, vix=14.0,
                    is_available=True)
        base.update(kw)
        return GlobalSnapshot(**base)

    def _delta(self, signal, **kw):
        from nifty_ai_agent.strategies.global_analyser import global_confidence_adjustment
        return global_confidence_adjustment(self._snapshot(**kw), signal)

    def test_crude_rally_penalises_a_bullish_signal(self):
        """Rising oil is a headwind for India — it must OPPOSE a call, not confirm it."""
        delta, why = self._delta("BUY_CE", crude_pct=3.0, crude_name="Crude (INR)")
        assert delta < 0
        assert "headwind" in why

    def test_crude_rally_supports_a_bearish_signal(self):
        delta, why = self._delta("BUY_PE", crude_pct=3.0)
        assert delta > 0

    def test_falling_crude_supports_a_bullish_signal(self):
        delta, why = self._delta("BUY_CE", crude_pct=-3.0)
        assert delta > 0
        assert "tailwind" in why

    def test_small_crude_moves_are_ignored(self):
        assert self._delta("BUY_CE", crude_pct=1.0)[0] == 0
        assert self._delta("BUY_CE", crude_pct=-1.4)[0] == 0

    def test_crude_is_not_treated_like_an_equity_index(self):
        """The bug this guards: a +3% crude print reading as 'another green market'."""
        up = self._delta("BUY_CE", crude_pct=3.0)[0]
        down = self._delta("BUY_CE", crude_pct=-3.0)[0]
        assert up < 0 < down


class TestRegionalNewsInConfidence:
    @staticmethod
    def _sentiment(score, headlines=5):
        from nifty_ai_agent.strategies.global_analyser import NewsSentiment
        return NewsSentiment(score=score, bullish_hits=5, bearish_hits=0,
                             headlines=headlines)

    def _delta(self, signal, india=None, world=None):
        from nifty_ai_agent.strategies.global_analyser import (
            GlobalSnapshot, global_confidence_adjustment,
        )
        from nifty_ai_agent.strategies.global_analyser import NewsSentiment
        empty = NewsSentiment(0.0, 0, 0, 0)
        snap = GlobalSnapshot(
            global_bias="NEUTRAL", gift_nifty_pct=0.0, vix=14.0, is_available=True,
            india_news=india or empty, world_news=world or empty,
        )
        return global_confidence_adjustment(snap, signal)

    def test_indian_headlines_outweigh_world_ones(self):
        """A domestic story bears on an Indian index more directly than a US one."""
        india_only = self._delta("BUY_CE", india=self._sentiment(0.8))[0]
        world_only = self._delta("BUY_CE", world=self._sentiment(0.8))[0]
        assert india_only > world_only > 0

    def test_opposing_news_costs_more_than_agreeing_news_pays(self):
        agree = self._delta("BUY_CE", india=self._sentiment(0.8))[0]
        oppose = self._delta("BUY_CE", india=self._sentiment(-0.8))[0]
        assert abs(oppose) > abs(agree)

    def test_regions_can_disagree_and_partially_cancel(self):
        delta, why = self._delta(
            "BUY_CE", india=self._sentiment(0.8), world=self._sentiment(-0.8),
        )
        assert "India" in why and "World" in why

    def test_empty_news_contributes_nothing(self):
        assert self._delta("BUY_CE")[0] == 0

    def test_hold_signals_are_never_adjusted(self):
        assert self._delta("HOLD", india=self._sentiment(0.9))[0] == 0
