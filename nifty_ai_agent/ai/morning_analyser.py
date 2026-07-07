"""Morning AI analyst — feeds the full pre-market brief to Claude and returns
a concise daily trading plan grounded in the user's book library.
"""

import logging
import time
from dataclasses import dataclass

import anthropic

from nifty_ai_agent.data.market_context import MarketContext
from nifty_ai_agent.data.news_fetcher import NewsItem
from nifty_ai_agent.data.nifty50_stocks import Nifty50Summary
from nifty_ai_agent.strategies.option_analyser import ExpiryAnalysis

logger = logging.getLogger(__name__)

_RETRY_COUNT = 3
_RETRY_DELAY = 2

_SYSTEM_PROMPT = """\
You are an experienced NIFTY 50 options trader and analyst with deep knowledge of:
- Al Brooks price action (trend, pullbacks, breakouts)
- Anna Coulling volume-price analysis
- Bulkowski chart pattern statistics
- Martin Pring momentum analysis
- Bob Volman tight-range scalping setups

Your job is to synthesise pre-market data into a concise, actionable daily plan for \
a retail trader who trades NIFTY weekly expiry options (CE/PE only, no intraday futures).

Output format — strictly follow this structure:
1. BIAS (one word + one sentence reason)
2. KEY LEVELS (3 bullet points: resistance, support, max pain)
3. BUY_CE TRIGGER (one sentence: what market condition fires a call entry)
4. BUY_PE TRIGGER (one sentence: what condition fires a put entry)
5. RISK TO WATCH (one sentence: the main thing that could invalidate today's bias)

Total length: under 180 words. No fluff, no hedging. Be direct.
"""


@dataclass
class DailyPlan:
    text: str
    model: str
    input_tokens: int
    output_tokens: int


class MorningAnalyser:
    """Synthesises all pre-market data into a daily trading plan via Claude."""

    def __init__(self, api_key: str, model: str = "claude-opus-4-8") -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def analyse(
        self,
        market_context: MarketContext,
        option_analysis: ExpiryAnalysis | None,
        nifty50_summary: Nifty50Summary | None,
        news_items: list[NewsItem],
        monthly_option_analysis: ExpiryAnalysis | None = None,
    ) -> DailyPlan:
        prompt = self._build_prompt(
            market_context, option_analysis, nifty50_summary, news_items,
            monthly_option_analysis,
        )

        for attempt in range(1, _RETRY_COUNT + 1):
            try:
                return self._call_api(prompt)
            except anthropic.APIError as exc:
                logger.warning(
                    "Morning analyser attempt %d/%d failed: %s",
                    attempt, _RETRY_COUNT, exc,
                )
                if attempt < _RETRY_COUNT:
                    time.sleep(_RETRY_DELAY)

        logger.error("Morning analysis failed after %d attempts.", _RETRY_COUNT)
        return DailyPlan(
            text="Morning analysis unavailable — check API key and credits.",
            model=self._model,
            input_tokens=0,
            output_tokens=0,
        )

    def _call_api(self, prompt: str) -> DailyPlan:
        with self._client.messages.stream(
            model=self._model,
            max_tokens=500,
            thinking={"type": "adaptive"},
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            message = stream.get_final_message()

        text = "".join(
            block.text for block in message.content if hasattr(block, "text")
        )
        return DailyPlan(
            text=text.strip(),
            model=message.model,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        )

    @staticmethod
    def _build_prompt(
        ctx: MarketContext,
        opt: ExpiryAnalysis | None,
        n50: Nifty50Summary | None,
        news: list[NewsItem],
        monthly_opt: ExpiryAnalysis | None = None,
    ) -> str:
        lines = ["PRE-MARKET DATA FOR TODAY:\n"]

        # Global context
        lines.append(f"Global Bias: {ctx.global_bias}")
        if ctx.gift_nifty:
            g = ctx.gift_nifty
            lines.append(
                f"GIFT Nifty: {g.price:,.0f}  ({g.change_pct:+.2f}%)"
            )
        for idx in ctx.indices:
            lines.append(f"  {idx.name}: {idx.change_pct:+.2f}%")

        # NIFTY 50 A/D
        if n50:
            lines.append(
                f"\nNIFTY 50 Advance/Decline: {n50.advances}↑ / {n50.declines}↓  "
                f"(A/D ratio: {n50.advance_decline_ratio})"
            )
            if n50.top_gainers:
                gainers = ", ".join(f"{m.name} +{m.change_pct}%" for m in n50.top_gainers[:3])
                lines.append(f"  Top gainers: {gainers}")
            if n50.top_losers:
                losers = ", ".join(f"{m.name} {m.change_pct}%" for m in n50.top_losers[:3])
                lines.append(f"  Top losers:  {losers}")

        # Option chain
        if opt:
            lines.append(f"\nOption Chain (Expiry: {opt.expiry}, DTE: {opt.days_to_expiry}d):")
            lines.append(f"  Spot: {opt.spot:,.0f}  ATM Strike: {opt.atm_strike:,}")
            lines.append(f"  Max Pain: {opt.max_pain:,.0f}")
            lines.append(f"  PCR: {opt.pcr}  (>1.2 = bullish, <0.8 = bearish)")
            lines.append(f"  CE OI Resistance: {opt.call_oi_resistance:,}")
            lines.append(f"  PE OI Support:    {opt.put_oi_support:,}")
            lines.append(f"  Option chain bias: {opt.bias}")
            if opt.theoretical_ce_atm:
                lines.append(
                    f"  Theoretical ATM CE: ₹{opt.theoretical_ce_atm}  "
                    f"PE: ₹{opt.theoretical_pe_atm}"
                )

        # Monthly option chain (structural reference)
        if monthly_opt:
            lines.append(
                f"\nMonthly Option Chain (Expiry: {monthly_opt.expiry},"
                f" DTE: {monthly_opt.days_to_expiry}d):"
            )
            lines.append(f"  Monthly Max Pain: {monthly_opt.max_pain:,.0f}")
            lines.append(f"  Monthly PCR: {monthly_opt.pcr}  bias: {monthly_opt.bias}")
            lines.append(f"  Monthly CE Wall (resistance): {monthly_opt.call_oi_resistance:,}")
            lines.append(f"  Monthly PE Floor (support):   {monthly_opt.put_oi_support:,}")

        # News
        if news:
            lines.append("\nKey Headlines:")
            for item in news[:4]:
                lines.append(f"  • {item.title}")

        lines.append(
            "\nBased on all of the above, write the daily trading plan in the required format."
        )
        return "\n".join(lines)
