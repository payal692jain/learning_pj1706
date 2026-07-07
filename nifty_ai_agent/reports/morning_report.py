"""Morning report orchestrator — runs at 8 AM, collects all pre-market data,
generates a full brief and sends it via Pushover in focused sections.

Message order:
  1. Global markets + GIFT Nifty
  2. NIFTY 50 Advance/Decline
  3. Weekly expiry option chain
  4. Monthly expiry option chain  (if different from weekly)
  5. Top market news
  6. Claude AI daily trading plan  (only when ANTHROPIC_API_KEY is set)
"""

import logging
from dataclasses import dataclass

from nifty_ai_agent.config import Settings
from nifty_ai_agent.data.market_context import (
    MarketContext,
    fetch_market_context,
    format_context_for_notification,
)
from nifty_ai_agent.data.news_fetcher import (
    NewsItem,
    fetch_news,
    format_news_for_notification,
)
from nifty_ai_agent.data.nifty50_stocks import (
    Nifty50Summary,
    fetch_nifty50_movers,
    format_movers_for_notification,
)
from nifty_ai_agent.data.nse_provider import NSEDataProvider
from nifty_ai_agent.notifier.pushover import PushoverNotifier
from nifty_ai_agent.strategies.option_analyser import (
    ExpiryAnalysis,
    analyse_option_chain,
    format_analysis_for_notification,
    format_monthly_analysis_for_notification,
)

logger = logging.getLogger(__name__)


def run_morning_report(settings: Settings) -> None:
    """Collect all pre-market data and push a Pushover morning brief."""
    logger.info("=== MORNING REPORT START (8 AM) ===")
    notifier = PushoverNotifier(
        user_key=settings.pushover_user_key,
        api_token=settings.pushover_api_token,
    )

    # ── 1. Global markets + GIFT Nifty ─────────────────────────────────────────
    ctx: MarketContext | None = _safe_fetch("global market context", fetch_market_context)
    if ctx:
        notifier.send_text(
            title=f"🌍 Pre-Market | {ctx.global_bias}",
            message=format_context_for_notification(ctx),
            priority=0,
        )
        logger.info("Global context sent: bias=%s", ctx.global_bias)

    # ── 2. NIFTY 50 Advance / Decline ──────────────────────────────────────────
    movers: Nifty50Summary | None = _safe_fetch("NIFTY 50 movers", fetch_nifty50_movers)
    if movers:
        notifier.send_text(
            title="📊 NIFTY 50 A/D Ratio",
            message=format_movers_for_notification(movers),
            priority=0,
        )
        logger.info("Movers sent: %d↑ %d↓", movers.advances, movers.declines)

    # ── 3. Weekly + Monthly expiry option chains ────────────────────────────────
    option_analysis: ExpiryAnalysis | None = None
    monthly_option_analysis: ExpiryAnalysis | None = None
    try:
        data_provider = NSEDataProvider(symbol=settings.nifty_symbol)
        spot_data = data_provider.get_spot_data()
        chain_data = data_provider.get_option_chain()

        if not chain_data.strikes.empty:
            option_analysis = analyse_option_chain(
                option_chain=chain_data.strikes,
                spot=spot_data.price,
                expiry=chain_data.expiry,
            )
            notifier.send_text(
                title=f"📈 Weekly OC | {option_analysis.bias} | PCR {option_analysis.pcr}",
                message=format_analysis_for_notification(option_analysis),
                priority=0,
            )
            logger.info(
                "Weekly OC sent: expiry=%s bias=%s max_pain=%.0f",
                option_analysis.expiry, option_analysis.bias, option_analysis.max_pain,
            )
        else:
            logger.warning("Weekly option chain empty — skipping")

        # Monthly — only if a different expiry exists
        if (
            not chain_data.monthly_strikes.empty
            and chain_data.monthly_expiry
            and chain_data.monthly_expiry != chain_data.expiry
        ):
            monthly_option_analysis = analyse_option_chain(
                option_chain=chain_data.monthly_strikes,
                spot=spot_data.price,
                expiry=chain_data.monthly_expiry,
                strikes_each_side=5,
            )
            notifier.send_text(
                title=f"📅 Monthly OC | {monthly_option_analysis.bias} | PCR {monthly_option_analysis.pcr}",
                message=format_monthly_analysis_for_notification(monthly_option_analysis),
                priority=0,
            )
            logger.info(
                "Monthly OC sent: expiry=%s bias=%s max_pain=%.0f",
                monthly_option_analysis.expiry,
                monthly_option_analysis.bias,
                monthly_option_analysis.max_pain,
            )
    except Exception as exc:
        logger.error("Option chain analysis failed: %s", exc)

    # ── 4. Top market news ──────────────────────────────────────────────────────
    news_items: list[NewsItem] = _safe_fetch("news", fetch_news) or []
    if news_items:
        notifier.send_text(
            title="📰 Market News",
            message=format_news_for_notification(news_items, limit=5),
            priority=-1,
        )
        logger.info("News sent: %d headlines", len(news_items))

    # ── 5. Claude AI daily trading plan ────────────────────────────────────────
    if settings.anthropic_api_key and ctx:
        try:
            from nifty_ai_agent.ai.morning_analyser import MorningAnalyser
            plan = MorningAnalyser(
                api_key=settings.anthropic_api_key,
                model=settings.claude_model,
            ).analyse(
                market_context=ctx,
                option_analysis=option_analysis,
                monthly_option_analysis=monthly_option_analysis,
                nifty50_summary=movers,
                news_items=news_items,
            )
            notifier.send_text(
                title="🤖 Claude Daily Plan",
                message=plan.text,
                priority=1,   # high priority — this is the most actionable message
            )
            logger.info(
                "Claude daily plan sent (%d tokens)", plan.output_tokens
            )
        except Exception as exc:
            logger.error("Claude morning analysis failed: %s", exc)
    else:
        if not settings.anthropic_api_key:
            logger.info("ANTHROPIC_API_KEY not set — skipping Claude daily plan.")

    logger.info("=== MORNING REPORT COMPLETE ===")


def _safe_fetch(label: str, fn, *args, **kwargs):
    """Call *fn* and return None on any exception, logging the error."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        logger.error("Morning report — %s failed: %s", label, exc)
        return None
