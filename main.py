"""NIFTY AI Agent — main entry point.

Scheduled jobs (all IST):
  06:45        — next-session outlook (GIFT Nifty Session 1 has opened; final pre-open read)
  07:45        — overnight market analysis (global markets + GIFT implied open + news)
  08:00        — pre-market morning report (global cues, option chain, news)
  08:05, hourly — Upstox token health check
  09:10        — pre-open trade plan (day's levels, all three indices, before the open)
  every 5 min  — intraday signal pipeline → one consensus trade call per index
  every 30 min — capital-aware trade plan across all three indices
  every 30 min — single-stock CE/PE scan + volatile-stock straddle scan (configurable)
  16:00        — EOD prediction for the next session
  17:00        — next-session outlook (GIFT Nifty Session 2 has opened; first overnight read)

Notifications go to Pushover (single user) and, when TELEGRAM_BOT_TOKEN is set, are
broadcast to every subscriber of the Telegram bot (multi-user delivery).

Start:
    python main.py

Dashboard (separate terminal):
    streamlit run dashboard/app.py
"""

import ctypes
import dataclasses
import logging
import sys
import threading
import time
from typing import Callable, NamedTuple

import pytz
import schedule
from datetime import date, datetime, timedelta

from nifty_ai_agent.ai.explainer import SignalExplainer
from nifty_ai_agent.config import configure_logging, get_settings
from nifty_ai_agent.data.bank_options import BankOptionIdea, suggest_bank_options
from nifty_ai_agent.data.banknifty_breadth import fetch_banknifty_breadth
from nifty_ai_agent.data.gift_nifty import build_outlook, fetch_gift_nifty
from nifty_ai_agent.data.instrument_master import get_instrument_master
from nifty_ai_agent.data.nifty50_stocks import NIFTY50_SYMBOLS
from nifty_ai_agent.data.stock_data import fetch_stock_histories
from nifty_ai_agent.data.token_health import TokenMonitor
from nifty_ai_agent.data.breadth import BreadthSnapshot, fetch_realtime_breadth
from nifty_ai_agent.data.bse_provider import BSEDataProvider
from nifty_ai_agent.data.nse_provider import NSEDataProvider
from nifty_ai_agent.data.sensex_breadth import fetch_sensex_breadth
from nifty_ai_agent.database.repository import DatabaseRepository
from nifty_ai_agent.notifier.pushover import PushoverNotifier
from nifty_ai_agent.reports.morning_report import run_morning_report
from nifty_ai_agent.reports.next_session import format_next_session
from nifty_ai_agent.reports.trade_call import format_trade_call
from nifty_ai_agent.reports.stock_scan import format_stock_scan
from nifty_ai_agent.reports.volatility_scan import format_volatility_scan
from nifty_ai_agent.reports.trade_plan import (
    FALLBACK_LOT_SIZES,
    TradeIdea,
    build_trade_idea,
    format_trade_plan,
)
from nifty_ai_agent.risk.calculator import RiskCalculator, RiskParameters
from nifty_ai_agent.risk.margin import MarginCalculator
from nifty_ai_agent.strategies.base import BaseStrategy, Signal, SignalType
from nifty_ai_agent.strategies.consensus import Consensus, build_consensus
from nifty_ai_agent.strategies.gap_analyser import analyse_gap_history, compute_pivots
from nifty_ai_agent.strategies.global_analyser import (
    GlobalSnapshot,
    fetch_global_snapshot,
    global_confidence_adjustment,
)
from nifty_ai_agent.strategies.pipeline import compute_all_indicators, DEFAULT_STRATEGIES
from nifty_ai_agent.strategies.stock_scanner import scan_stocks
from nifty_ai_agent.strategies.volatility_scanner import scan_volatile_straddles
from nifty_ai_agent.strategies.option_analyser import (
    ExpiryAnalysis,
    analyse_option_chain,
    compute_atm_theoretical_prices,
    monthly_option_chain_note,
    option_chain_confidence_adjustment,
)
from nifty_ai_agent.strategies.rsi_analyser import analyse_rsi, rsi_confidence_adjustment

logger = logging.getLogger(__name__)

_IST = pytz.timezone("Asia/Kolkata")
_MARKET_OPEN_HOUR = 9
_MARKET_CLOSE_HOUR = 15
_MARKET_CLOSE_MINUTE = 30


class IndexConfig(NamedTuple):
    """All index-specific settings needed to run a signal pipeline."""
    name: str                              # "NIFTY", "SENSEX", or "BANKNIFTY"
    symbol: str                            # yfinance symbol
    strike_step: int                       # 50 for NIFTY, 100 for SENSEX/BANKNIFTY
    expiry_weekday: int                    # 1=Tuesday (NIFTY, BANKNIFTY), 3=Thursday (SENSEX)
    make_provider: Callable                # factory → MarketDataProvider
    fetch_breadth: Callable[[], BreadthSnapshot]


# Populated in main() once settings are loaded
_INDEX_CONFIGS: list[IndexConfig] = []

# Every strategy runs independently each cycle — all of their predictions are
# saved and notified, not just one "winning" signal. The strategy book and the
# indicator set both live in strategies/pipeline.py so the stock scanner runs
# on identical machinery.
_STRATEGIES: list[BaseStrategy] = DEFAULT_STRATEGIES
_compute_indicators = compute_all_indicators


def _is_market_hours() -> bool:
    """Return True if current IST time is within NSE trading hours."""
    now_ist = datetime.now(_IST)
    if now_ist.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    open_time = now_ist.replace(hour=_MARKET_OPEN_HOUR, minute=15, second=0, microsecond=0)
    close_time = now_ist.replace(hour=_MARKET_CLOSE_HOUR, minute=_MARKET_CLOSE_MINUTE, second=0, microsecond=0)
    return open_time <= now_ist <= close_time


def _is_trading_weekday() -> bool:
    """True Mon–Fri (no holiday calendar). Used by the pre-open jobs that run
    before market hours, where _is_market_hours() would always be False."""
    return datetime.now(_IST).weekday() < 5


# ── Option chain cache (per index) ──────────────────────────────────────────────
# TTL is tied to the signal-loop interval (default 5 min) so the ATM premium a
# trade call quotes is re-fetched every cycle rather than lingering up to 15 min
# behind the live index. The trade call still reprices the premium onto the live
# spot as a second guard, but a tight TTL keeps that adjustment small. Cost: one
# option-chain API call per index per cycle — the intended trade-off for a "Buy ₹"
# figure that tracks the live value.
_option_caches: dict[str, dict] = {}   # keyed by index name ("NIFTY", "SENSEX")


def _last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    """Return the date of the last *weekday* (0=Mon…6=Sun) in *year*-*month*."""
    if month == 12:
        next_month_first = date(year + 1, 1, 1)
    else:
        next_month_first = date(year, month + 1, 1)
    last_day = next_month_first - timedelta(days=1)
    offset = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=offset)


def _estimated_monthly_expiry(weekday: int) -> str:
    """Estimate the monthly expiry as the last *weekday* of the month, as 'DD-Mon-YYYY'.

    Only used when live NSE/BSE data is unavailable and we've fallen back to a
    VIX-based synthetic chain, which has no real expiry calendar to read from.
    Rolls to next month if this month's last such weekday has already passed.
    """
    today = date.today()
    candidate = _last_weekday_of_month(today.year, today.month, weekday)
    if candidate > today:
        return candidate.strftime("%d-%b-%Y")

    next_month = today.month + 1 if today.month < 12 else 1
    next_year = today.year if today.month < 12 else today.year + 1
    return _last_weekday_of_month(next_year, next_month, weekday).strftime("%d-%b-%Y")


def _get_cached_option_analysis(
    index: IndexConfig,
    data_provider,
    spot: float,
) -> tuple[ExpiryAnalysis | None, ExpiryAnalysis | None]:
    """Return *(weekly_analysis, monthly_analysis)* for *index*, refreshing if stale."""
    cache = _option_caches.setdefault(
        index.name, {"weekly": None, "monthly": None, "fetched_at": None}
    )
    now = datetime.now(_IST)
    fetched_at = cache["fetched_at"]
    ttl_minutes = get_settings().data_fetch_interval_minutes

    if fetched_at is not None:
        age_minutes = (now - fetched_at).total_seconds() / 60
        if age_minutes < ttl_minutes:
            logger.debug("%s option chain cache hit (%.1f min old)", index.name, age_minutes)
            return cache["weekly"], cache["monthly"]

    try:
        chain_data = data_provider.get_option_chain()

        # analyse_option_chain handles an empty DataFrame via _stub_analysis.
        # For VIX-based synthetic chains (strikes empty, pcr set from VIX),
        # override the stub's arbitrary values: correct ATM for the index's
        # strike step, VIX-derived PCR, zero OI walls so only PCR fires,
        # and Black-Scholes theoretical prices so the notification shows a
        # real buy price instead of "@ market price".
        weekly = analyse_option_chain(
            chain_data.strikes, spot, chain_data.expiry, strike_step=index.strike_step,
        )
        monthly: ExpiryAnalysis | None = None

        if chain_data.strikes.empty and chain_data.pcr > 0:
            correct_atm = int(round(spot / index.strike_step) * index.strike_step)
            iv = chain_data.iv_proxy if chain_data.iv_proxy > 0 else 0.15
            theo_ce, theo_pe = compute_atm_theoretical_prices(
                spot, correct_atm, chain_data.expiry, iv
            )
            weekly = dataclasses.replace(
                weekly,
                pcr=chain_data.pcr,
                atm_strike=correct_atm,
                call_oi_resistance=0,
                put_oi_support=0,
                max_pain=0.0,
                spot=spot,
                theoretical_ce_atm=theo_ce,
                theoretical_pe_atm=theo_pe,
                atm_ce_ltp=theo_ce,
                atm_pe_ltp=theo_pe,
                is_live=False,
            )
            logger.info(
                "%s synthetic weekly option prices: CE=%.1f PE=%.1f (VIX IV=%.1f%%)",
                index.name, theo_ce, theo_pe, iv * 100,
            )

            # No live monthly chain in this mode either — synthesize one too so
            # the alert always shows weekly + monthly together instead of just
            # weekly. Still a theoretical estimate, not a traded premium — the
            # notification labels it "(Est.)" via ExpiryAnalysis.is_live.
            monthly_expiry_est = _estimated_monthly_expiry(index.expiry_weekday)
            theo_ce_m, theo_pe_m = compute_atm_theoretical_prices(
                spot, correct_atm, monthly_expiry_est, iv
            )
            monthly_dte = max(
                0,
                (datetime.strptime(monthly_expiry_est, "%d-%b-%Y").date() - date.today()).days,
            )
            monthly = dataclasses.replace(
                weekly,
                expiry=monthly_expiry_est,
                days_to_expiry=monthly_dte,
                theoretical_ce_atm=theo_ce_m,
                theoretical_pe_atm=theo_pe_m,
                atm_ce_ltp=theo_ce_m,
                atm_pe_ltp=theo_pe_m,
                is_live=False,
            )
            logger.info(
                "%s synthetic monthly option prices: CE=%.1f PE=%.1f expiry=%s",
                index.name, theo_ce_m, theo_pe_m, monthly_expiry_est,
            )
        elif (
            not chain_data.monthly_strikes.empty
            and chain_data.monthly_expiry
            and chain_data.monthly_expiry != chain_data.expiry
        ):
            monthly = analyse_option_chain(
                chain_data.monthly_strikes, spot,
                chain_data.monthly_expiry, strikes_each_side=5,
                strike_step=index.strike_step,
            )

        cache["weekly"] = weekly
        cache["monthly"] = monthly
        cache["fetched_at"] = now
        logger.info(
            "%s option chain refreshed: weekly=%s  monthly=%s  pcr=%.2f",
            index.name, chain_data.expiry, chain_data.monthly_expiry or "N/A", weekly.pcr,
        )
        return weekly, monthly
    except Exception as exc:
        logger.warning("%s option chain failed — skipping OC filter: %s", index.name, exc)
        return None, None


def _adjust_for_breadth(signal: Signal, breadth: BreadthSnapshot) -> Signal:
    """Adjust signal confidence based on heavyweight breadth confirmation.

    BUY_CE with majority heavyweights declining → lower confidence (divergence).
    BUY_CE with majority advancing             → higher confidence (confirmation).
    Symmetric for BUY_PE.  HOLD is unchanged.
    """
    if signal.signal == SignalType.HOLD or breadth.total == 0:
        return signal

    bullish_signal = signal.signal == SignalType.BUY_CE
    score = breadth.score  # positive = more advancing, negative = more declining

    # Confirmation: breadth agrees with signal direction
    # Contradiction: breadth opposes signal direction
    confirming_score = score if bullish_signal else -score

    if confirming_score >= 0.4:
        delta = +8
        detail = (
            f" Breadth confirms: {breadth.advancing}/{breadth.total} heavyweights"
            f" advancing ({', '.join(breadth.leaders[:3])})."
        )
    elif confirming_score >= 0.2:
        delta = +4
        detail = (
            f" Mild breadth support: {breadth.advancing}↑/{breadth.declining}↓"
            f" among heavyweights."
        )
    elif confirming_score <= -0.4:
        delta = -12
        detail = (
            f" Breadth diverges: {breadth.declining}/{breadth.total} heavyweights"
            f" moving against signal ({', '.join(breadth.laggards[:3])})."
        )
    elif confirming_score <= -0.2:
        delta = -6
        detail = (
            f" Weak breadth: {breadth.advancing}↑/{breadth.declining}↓ heavyweights"
            f" — mixed confirmation."
        )
    else:
        return signal  # neutral breadth — no change

    new_confidence = max(10, min(95, signal.confidence + delta))
    return dataclasses.replace(
        signal,
        confidence=new_confidence,
        reason=signal.reason + detail,
    )


def _generate_and_adjust_signal(
    strategy: BaseStrategy,
    df,
    rsi_analysis,
    oc_weekly: ExpiryAnalysis | None,
    oc_monthly: ExpiryAnalysis | None,
    breadth: BreadthSnapshot,
    index_name: str,
    global_snapshot: GlobalSnapshot | None = None,
) -> Signal:
    """Run one strategy and apply the shared RSI/option-chain/breadth/global adjustments."""
    signal = strategy.generate_signal(df)
    logger.info(
        "%s %s signal (raw): %s confidence=%d%%  %s",
        index_name, strategy.NAME, signal.signal.value, signal.confidence, signal.reason,
    )

    rsi_delta, rsi_detail = rsi_confidence_adjustment(rsi_analysis, signal.signal.value)

    oc_delta, oc_detail = (
        option_chain_confidence_adjustment(oc_weekly, signal.signal.value)
        if oc_weekly else (0, "")
    )
    m_delta, m_detail = monthly_option_chain_note(oc_monthly, signal.signal.value)
    oc_delta += m_delta
    oc_detail += m_detail

    breadth_signal = _adjust_for_breadth(signal, breadth)
    breadth_delta = breadth_signal.confidence - signal.confidence
    breadth_detail = breadth_signal.reason[len(signal.reason):]

    # Global cues used to be computed at 08:00 and then thrown away — an intraday
    # signal that ignores an overnight risk-off tape is reading half the market.
    global_delta, global_detail = (
        global_confidence_adjustment(global_snapshot, signal.signal.value)
        if global_snapshot else (0, "")
    )

    if signal.signal != SignalType.HOLD:
        total_delta = rsi_delta + oc_delta + breadth_delta + global_delta
        new_confidence = max(10, min(95, signal.confidence + total_delta))
        new_reason = signal.reason + rsi_detail + oc_detail + breadth_detail + global_detail
        signal = dataclasses.replace(signal, confidence=new_confidence, reason=new_reason)
        logger.info(
            "%s %s signal (final): %s confidence=%d%%  Δrsi=%+d Δoc=%+d Δbreadth=%+d Δglobal=%+d",
            index_name, strategy.NAME, signal.signal.value, signal.confidence,
            rsi_delta, oc_delta, breadth_delta, global_delta,
        )

    return signal


# ── Pipeline ───────────────────────────────────────────────────────────────────

def run_pipeline(index: IndexConfig, after_hours: bool = False) -> None:
    """Execute one full signal generation cycle for *index* (NIFTY or SENSEX).

    after_hours=True: skips the market-hours guard, skips live breadth (market
    is closed), and labels the Pushover notification as an EOD prediction.
    """
    if not after_hours and not _is_market_hours():
        logger.info("%s: outside market hours — skipping.", index.name)
        return

    settings = get_settings()
    data_provider = index.make_provider()
    logger.info("=== %s pipeline (after_hours=%s) ===", index.name, after_hours)

    # ── Data ────────────────────────────────────────────────────────────────────
    try:
        spot = data_provider.get_spot_data()
        logger.info("%s spot: %.2f", index.name, spot.price)
    except Exception as exc:
        logger.error("%s: failed to fetch spot data: %s", index.name, exc)
        return

    try:
        hist = data_provider.get_historical_data(
            days=settings.historical_days,
            interval=settings.data_interval,
        )
    except Exception as exc:
        logger.error("%s: failed to fetch historical data: %s", index.name, exc)
        return

    # ── Indicators ───────────────────────────────────────────────────────────────
    df = _compute_indicators(hist)

    latest = df.dropna(subset=["ema_20", "ema_50", "rsi", "atr"]).iloc[-1]
    current_atr = float(latest["atr"])

    # ── RSI analysis ─────────────────────────────────────────────────────────────
    rsi_analysis = analyse_rsi(df)

    # ── Option chain filter (15-min cached) ──────────────────────────────────────
    oc_weekly, oc_monthly = _get_cached_option_analysis(index, data_provider, spot.price)

    # ── Breadth confirmation (live market only) ───────────────────────────────────
    breadth = BreadthSnapshot(0, 0, 0, 0, 0.0, "NEUTRAL", [], [])
    if not after_hours:
        try:
            breadth = index.fetch_breadth()
        except Exception as exc:
            logger.warning("%s breadth fetch failed — skipping: %s", index.name, exc)
    else:
        logger.info("%s: after-hours mode — skipping live breadth.", index.name)

    # ── Global context (30-min cached; shared across indices) ─────────────────────
    global_snapshot = fetch_global_snapshot()

    # ── Indicators snapshot (shared across strategies) ────────────────────────────
    indicators: dict[str, float] = {
        "ema_20":                float(latest["ema_20"]),
        "ema_50":                float(latest["ema_50"]),
        "rsi":                   float(latest["rsi"]),
        "rsi_zone":              rsi_analysis.value,
        "macd":                  float(latest.get("macd", 0)),
        "macd_signal":           float(latest.get("macd_signal", 0)),
        "atr":                   current_atr,
        "vwap":                  float(latest.get("vwap", 0)),
        "breadth_score":         breadth.score,
        "breadth_advances":      float(breadth.advancing),
        "breadth_declines":      float(breadth.declining),
        "pcr":                   float(oc_weekly.pcr) if oc_weekly else 0.0,
        "max_pain":              float(oc_weekly.max_pain) if oc_weekly else 0.0,
        "ce_resistance":         float(oc_weekly.call_oi_resistance) if oc_weekly else 0.0,
        "pe_support":            float(oc_weekly.put_oi_support) if oc_weekly else 0.0,
        "monthly_pcr":           float(oc_monthly.pcr) if oc_monthly else 0.0,
        "monthly_max_pain":      float(oc_monthly.max_pain) if oc_monthly else 0.0,
        "monthly_ce_resistance": float(oc_monthly.call_oi_resistance) if oc_monthly else 0.0,
        "monthly_pe_support":    float(oc_monthly.put_oi_support) if oc_monthly else 0.0,
    }

    # "NIFTY 50" vs "BSE SENSEX" for the Claude prompt
    ai_index_label = "BSE SENSEX" if index.name == "SENSEX" else "NIFTY 50"

    # ── Run every strategy, then fold them into ONE verdict ───────────────────────
    signals = [
        _generate_and_adjust_signal(
            strategy, df, rsi_analysis, oc_weekly, oc_monthly, breadth, index.name,
            global_snapshot,
        )
        for strategy in _STRATEGIES
    ]

    consensus = build_consensus(signals, now=datetime.now(_IST).time())

    risk_calculator = RiskCalculator(
        max_risk_pct=settings.max_risk_per_trade_pct,
        daily_loss_limit_pct=settings.daily_loss_limit_pct,
        min_rr=settings.min_risk_reward_ratio,
        atr_sl_multiplier=settings.atr_sl_multiplier,
    )
    risk = risk_calculator.calculate(consensus.signal, spot.price, current_atr)
    if not risk.is_valid and consensus.signal != SignalType.HOLD:
        logger.warning(
            "%s consensus rejected by risk manager: %s", index.name, risk.rejection_reason,
        )

    # One AI explanation for the verdict, not one per strategy. The old per-strategy
    # loop meant six Claude calls per index per five-minute cycle — the same market,
    # explained six times, at six times the cost.
    ai_explanation = ""
    if settings.anthropic_api_key and consensus.is_actionable:
        try:
            consensus_signal = Signal(
                signal=consensus.signal,
                confidence=consensus.confidence,
                reason=consensus.rationale,
                strategy=f"Consensus({consensus.conviction})",
            )
            explanation = SignalExplainer(
                api_key=settings.anthropic_api_key,
                model=settings.claude_model,
            ).explain(consensus_signal, risk, indicators, index_name=ai_index_label)
            ai_explanation = explanation.text
            logger.info(
                "%s consensus AI explanation generated (%d tokens)",
                index.name, explanation.output_tokens,
            )
        except Exception as exc:
            logger.error("%s AI explanation failed: %s", index.name, exc)

    # ── Bank constituents (BANKNIFTY only) ────────────────────────────────────────
    bank_ideas: list[BankOptionIdea] = []
    if index.name == "BANKNIFTY" and consensus.is_actionable and not after_hours:
        token = get_settings().upstox_access_token
        if token:
            try:
                from nifty_ai_agent.data.upstox_provider import UpstoxClient
                bank_ideas = suggest_bank_options(consensus.signal, UpstoxClient(token))
            except Exception as exc:
                logger.warning("Bank option suggestions failed: %s", exc)

    # ── Database — every strategy's vote is still recorded, for later review ──────
    db = DatabaseRepository(settings.database_url)
    try:
        db.save_market_data(hist)
    except Exception as exc:
        logger.error("%s: failed to save market data: %s", index.name, exc)

    for signal in signals:
        try:
            db.save_signal(signal, risk_calculator.calculate(
                signal.signal, spot.price, current_atr,
            ), "")
        except Exception as exc:
            logger.error("%s: failed to save %s signal: %s", index.name, signal.strategy, exc)

    # ── Notify only high-conviction, actionable calls ─────────────────────────────
    # Every strategy vote is already saved above; the notification itself is gated
    # so a HOLD cycle or a sub-threshold BUY does not buzz anyone.
    if not (consensus.is_actionable and consensus.confidence >= settings.min_signal_confidence):
        logger.info(
            "%s: %s %d%% below notify threshold (%d%%) — recorded, not notified.",
            index.name, consensus.signal.value, consensus.confidence,
            settings.min_signal_confidence,
        )
        return

    # ── Pushover — the call, not the research dump ────────────────────────────────
    margin_calculator = MarginCalculator(
        capital=settings.trading_capital,
        max_risk_per_trade_pct=settings.max_risk_per_trade_pct,
        daily_loss_limit_pct=settings.daily_loss_limit_pct,
        max_margin_utilisation_pct=settings.max_margin_utilisation_pct,
    )
    title, body = format_trade_call(
        index_name=index.name,
        consensus=consensus,
        risk=risk,
        analysis=oc_weekly,
        margin=margin_calculator,
        lot_size=_get_lot_size(index.name),
        global_snapshot=global_snapshot,
        bank_ideas=bank_ideas,
        prediction=after_hours,
    )
    if ai_explanation:
        body = f"{body}\n\n{ai_explanation}"[: 1024]

    try:
        PushoverNotifier(
            user_key=settings.pushover_user_key,
            api_token=settings.pushover_api_token,
            enabled=settings.pushover_enabled,
        ).send_text(
            title=title,
            message=body,
            monospace=True,
        )
        logger.info(
            "%s trade call sent: %s %s (%d%%)",
            index.name, consensus.conviction, consensus.signal.value, consensus.confidence,
        )
    except Exception as exc:
        logger.error("%s Pushover failed: %s", index.name, exc)

    _broadcast_telegram(title, body)


# ── Next-session outlook (GIFT Nifty) ─────────────────────────────────────────

def _run_next_session_outlook() -> None:
    """Send the GIFT Nifty read on where NIFTY opens next session.

    Scheduled for the two moments GIFT actually says something new: 17:00 (Session 2
    has opened at 16:35, pricing tomorrow overnight) and 06:45 (Session 1 has opened
    at 06:30 — the final pre-open read, with Wall Street's full day now in the price).
    """
    settings = get_settings()

    gift = fetch_gift_nifty()
    if gift is None:
        logger.warning("Next-session outlook: GIFT Nifty unavailable — skipping.")
        return

    try:
        daily = NSEDataProvider(
            symbol=settings.nifty_symbol,
            upstox_access_token=settings.upstox_access_token,
        ).get_historical_data(days=settings.gap_history_days, interval="1d")
    except Exception as exc:
        logger.error("Next-session outlook: NIFTY daily history failed: %s", exc)
        return

    clean = daily.dropna(subset=["open", "high", "low", "close"])
    if clean.empty:
        logger.error("Next-session outlook: no usable daily bars.")
        return

    last = clean.iloc[-1]
    outlook = build_outlook(gift, float(last["close"]))
    stats = analyse_gap_history(clean, outlook.bucket)
    pivots = compute_pivots(float(last["high"]), float(last["low"]), float(last["close"]))

    title, body = format_next_session(outlook, stats, pivots)
    try:
        PushoverNotifier(
            user_key=settings.pushover_user_key,
            api_token=settings.pushover_api_token,
            enabled=settings.pushover_enabled,
        ).send_text(title=title, message=body, monospace=True)
        logger.info(
            "Next-session outlook sent: %s %+.0f pts (%s, n=%d)",
            outlook.direction, outlook.gap_points, outlook.bucket, stats.sample,
        )
    except Exception as exc:
        logger.error("Next-session outlook Pushover failed: %s", exc)


# The monitor is stateful — it alerts on token state CHANGES, not on every cycle,
# so a dead token produces one notification per day rather than one every 5 minutes.
_token_monitor: TokenMonitor | None = None


def _check_token() -> None:
    """Probe the Upstox token; alert once if it died, once again when it comes back.

    The agent keeps running either way — it degrades to estimated premiums rather
    than stopping — and resumes live data automatically the moment a fresh token is
    written to .env, because get_settings() re-reads the file on every call.
    """
    if _token_monitor is None:
        return
    _token_monitor.check_and_alert(get_settings().upstox_access_token)


def _run_overnight_analysis() -> None:
    """Send the 07:45 overnight backdrop: global markets + GIFT implied open + news.

    A single consolidated pre-open read (weekdays only), ahead of the fuller 08:00
    morning report. Each piece is best-effort — the digest still goes out if news or
    the implied-open calc is unavailable, as long as *some* data was gathered.
    """
    if not _is_trading_weekday():
        logger.info("Overnight analysis: weekend — skipping.")
        return

    settings = get_settings()
    from nifty_ai_agent.data.market_context import compute_global_bias, fetch_global_indices
    from nifty_ai_agent.data.news_fetcher import fetch_news
    from nifty_ai_agent.reports.overnight import format_overnight_analysis

    indices = []
    try:
        indices = fetch_global_indices()
    except Exception as exc:
        logger.warning("Overnight analysis: global indices failed: %s", exc)
    bias = compute_global_bias(indices)

    # GIFT implied open + gap base rate (best-effort — needs GIFT + NIFTY daily bars).
    outlook = stats = None
    gift = fetch_gift_nifty()
    if gift is not None:
        try:
            daily = NSEDataProvider(
                symbol=settings.nifty_symbol,
                upstox_access_token=settings.upstox_access_token,
            ).get_historical_data(days=settings.gap_history_days, interval="1d")
            clean = daily.dropna(subset=["open", "high", "low", "close"])
            if not clean.empty:
                outlook = build_outlook(gift, float(clean["close"].iloc[-1]))
                stats = analyse_gap_history(clean, outlook.bucket)
                if gift.change_pct > 0.5:
                    bias = "BULLISH"
                elif gift.change_pct < -0.5:
                    bias = "BEARISH"
        except Exception as exc:
            logger.warning("Overnight analysis: implied-open calc failed: %s", exc)

    news = []
    try:
        news = fetch_news()
    except Exception as exc:
        logger.warning("Overnight analysis: news failed: %s", exc)

    if not indices and outlook is None:
        logger.warning("Overnight analysis: no data gathered — skipping notification.")
        return

    title, body = format_overnight_analysis(indices, bias, outlook, stats, news)
    _send_notification(settings, title, body)
    logger.info("Overnight analysis sent (bias=%s, gift=%s)", bias, gift is not None)


def _run_morning_report() -> None:
    """Wrapper for the 8 AM morning report job."""
    settings = get_settings()
    try:
        run_morning_report(settings)
    except Exception as exc:
        logger.error("Morning report crashed: %s", exc)


def _run_all_pipelines(after_hours: bool = False) -> None:
    """Run signal pipeline for every configured index in sequence."""
    for idx in _INDEX_CONFIGS:
        try:
            run_pipeline(idx, after_hours=after_hours)
        except Exception as exc:
            logger.error("%s pipeline crashed: %s", idx.name, exc)


def _run_eod_prediction() -> None:
    """Wrapper for the 4 PM EOD prediction job — runs for all indices."""
    logger.info("=== EOD PREDICTION (after market close) ===")
    _run_all_pipelines(after_hours=True)


# ── Trade plan (capital-aware, all indices in one message) ─────────────────────

def _get_lot_size(index_name: str) -> int:
    """Live lot size from Upstox contract data, falling back to known constants."""
    token = get_settings().upstox_access_token
    if token:
        try:
            from nifty_ai_agent.data.upstox_provider import UpstoxClient
            return UpstoxClient(token).get_lot_size(index_name)
        except Exception as exc:
            logger.warning(
                "Lot size fetch failed for %s (%s) — using fallback", index_name, exc,
            )
    return FALLBACK_LOT_SIZES.get(index_name, 50)


def _build_index_trade_idea(index: IndexConfig, settings) -> TradeIdea | None:
    """Compute the highest-confidence actionable trade for one index, or None (HOLD)."""
    provider = index.make_provider()
    spot = provider.get_spot_data()
    hist = provider.get_historical_data(
        days=settings.historical_days, interval=settings.data_interval,
    )

    df = _compute_indicators(hist)
    latest = df.dropna(subset=["ema_20", "ema_50", "rsi", "atr"]).iloc[-1]

    rsi_analysis = analyse_rsi(df)
    oc_weekly, _ = _get_cached_option_analysis(index, provider, spot.price)
    breadth = BreadthSnapshot(0, 0, 0, 0, 0.0, "NEUTRAL", [], [])

    signals = [
        _generate_and_adjust_signal(s, df, rsi_analysis, oc_weekly, None, breadth, index.name)
        for s in _STRATEGIES
    ]
    actionable = [s for s in signals if s.signal != SignalType.HOLD]
    if not actionable or oc_weekly is None:
        return None
    best = max(actionable, key=lambda s: s.confidence)

    risk = RiskCalculator(
        max_risk_pct=settings.max_risk_per_trade_pct,
        daily_loss_limit_pct=settings.daily_loss_limit_pct,
        min_rr=settings.min_risk_reward_ratio,
        atr_sl_multiplier=settings.atr_sl_multiplier,
    ).calculate(best.signal, spot.price, float(latest["atr"]))

    return build_trade_idea(index.name, best.signal, best.confidence, oc_weekly, risk)


def _run_trade_plan(pre_open: bool = False) -> None:
    """Send one capital-aware trade-plan notification covering all three indices.

    *pre_open* runs the plan before the 09:15 open (the 09:10 job): it swaps the
    market-hours guard for a weekday guard — the levels come from the prior session's
    bars plus the live option chain — and labels the notification accordingly.
    """
    if pre_open:
        if not _is_trading_weekday():
            logger.info("Pre-open trade plan: weekend — skipping.")
            return
    elif not _is_market_hours():
        logger.info("Trade plan: outside market hours — skipping.")
        return

    settings = get_settings()
    ideas: list[TradeIdea] = []
    holds: list[str] = []
    for index in _INDEX_CONFIGS:
        try:
            idea = _build_index_trade_idea(index, settings)
        except Exception as exc:
            logger.error("Trade plan: %s failed: %s", index.name, exc)
            idea = None
        (ideas.append(idea) if idea else holds.append(index.name))

    # Only high-conviction ideas make the plan; skip the whole notification if none.
    ideas = [i for i in ideas if i.confidence >= settings.min_signal_confidence]
    if not ideas:
        logger.info(
            "Trade plan: no idea >= %d%% — not notified.", settings.min_signal_confidence,
        )
        return

    title, body = format_trade_plan(ideas, holds)
    if pre_open:
        title = f"🌅 Pre-Open · {title}"
    try:
        PushoverNotifier(
            user_key=settings.pushover_user_key,
            api_token=settings.pushover_api_token,
            enabled=settings.pushover_enabled,
        ).send_text(title=title, message=body, monospace=True)
        logger.info("Trade plan sent (%d ideas, %d holds)", len(ideas), len(holds))
    except Exception as exc:
        logger.error("Trade plan Pushover failed: %s", exc)

    _broadcast_telegram(title, body)


# ── Stock option scan (NIFTY 50 constituents) ──────────────────────────────────

def _stock_premium_lookup(token: str):
    """Return an instrument_key → LTP lookup, or a no-op estimator when no token.

    Stock strikes/lots resolve from the (tokenless) instrument master, but live
    premiums need an authenticated quote — without a token the scan still runs and
    quotes Black-Scholes estimates instead.
    """
    if not token:
        return None
    from nifty_ai_agent.data.upstox_provider import UpstoxClient

    client = UpstoxClient(token)

    def _lookup(instrument_keys: list[str]) -> dict[str, float]:
        return client.get_ltp(instrument_keys)

    return _lookup


def _run_stock_scan() -> None:
    """Scan the NIFTY 50 constituents for monthly-option ideas and send one digest."""
    settings = get_settings()
    if not settings.stock_scan_enabled:
        logger.info("Stock scan disabled — skipping.")
        return
    if not _is_market_hours():
        logger.info("Stock scan: outside market hours — skipping.")
        return

    logger.info("=== STOCK SCAN (%d symbols) ===", len(NIFTY50_SYMBOLS))
    try:
        histories, spots = fetch_stock_histories(
            NIFTY50_SYMBOLS,
            interval=settings.data_interval,
            upstox_token=settings.upstox_access_token,
            master=get_instrument_master(),
        )
    except Exception as exc:
        logger.error("Stock scan: history fetch failed: %s", exc)
        return

    if not histories:
        logger.warning("Stock scan: no usable histories — skipping.")
        return

    risk_calculator = RiskCalculator(
        max_risk_pct=settings.max_risk_per_trade_pct,
        daily_loss_limit_pct=settings.daily_loss_limit_pct,
        min_rr=settings.min_risk_reward_ratio,
        atr_sl_multiplier=settings.atr_sl_multiplier,
    )
    lookup = _stock_premium_lookup(settings.upstox_access_token)

    result = scan_stocks(
        histories,
        spots,
        master=get_instrument_master(),
        risk_calculator=risk_calculator,
        now=datetime.now(_IST).time(),
        premium_lookup=lookup if lookup is not None else (lambda keys: {}),
        default_iv=settings.stock_default_iv,
        top_n=settings.stock_scan_top_n,
    )

    # Keep only high-conviction ideas — the rest are dropped from the digest.
    result.ideas = [i for i in result.ideas if i.confidence >= settings.min_signal_confidence]

    title, body = format_stock_scan(result)
    # Running every 30 min, a "no setups" digest should not buzz the phone — only an
    # actual CE/PE idea earns a sound; empty scans go out silent (priority -1).
    priority = 0 if result.ideas else -1
    try:
        PushoverNotifier(
            user_key=settings.pushover_user_key,
            api_token=settings.pushover_api_token,
            enabled=settings.pushover_enabled,
        ).send_text(title=title, message=body, priority=priority, monospace=True)
        logger.info(
            "Stock scan sent: %d ideas (%d actionable, %d errors)",
            len(result.ideas), result.actionable, result.errors,
        )
    except Exception as exc:
        logger.error("Stock scan Pushover failed: %s", exc)

    # Only broadcast an actual CE/PE digest to subscribers — skip empty scans.
    if result.ideas:
        _broadcast_telegram(title, body)


# ── Volatile-stock straddle scan (long-volatility CE+PE ideas) ──────────────────

def _run_volatility_scan() -> None:
    """Rank NIFTY 50 by ATR% and send one digest of ATM straddles on the most volatile."""
    settings = get_settings()
    if not settings.volatility_scan_enabled:
        logger.info("Volatility scan disabled — skipping.")
        return
    if not _is_market_hours():
        logger.info("Volatility scan: outside market hours — skipping.")
        return

    logger.info("=== VOLATILITY SCAN (%d symbols) ===", len(NIFTY50_SYMBOLS))
    try:
        histories, spots = fetch_stock_histories(
            NIFTY50_SYMBOLS,
            interval=settings.data_interval,
            upstox_token=settings.upstox_access_token,
            master=get_instrument_master(),
        )
    except Exception as exc:
        logger.error("Volatility scan: history fetch failed: %s", exc)
        return

    if not histories:
        logger.warning("Volatility scan: no usable histories — skipping.")
        return

    lookup = _stock_premium_lookup(settings.upstox_access_token)
    result = scan_volatile_straddles(
        histories,
        spots,
        master=get_instrument_master(),
        premium_lookup=lookup if lookup is not None else (lambda keys: {}),
        default_iv=settings.stock_default_iv,
        top_n=settings.volatility_scan_top_n,
    )

    title, body = format_volatility_scan(result)
    # Silent when there is nothing to trade — only an actual straddle earns a sound.
    priority = 0 if result.ideas else -1
    try:
        PushoverNotifier(
            user_key=settings.pushover_user_key,
            api_token=settings.pushover_api_token,
            enabled=settings.pushover_enabled,
        ).send_text(title=title, message=body, priority=priority, monospace=True)
        logger.info(
            "Volatility scan sent: %d straddles (%d ranked, %d errors)",
            len(result.ideas), result.ranked, result.errors,
        )
    except Exception as exc:
        logger.error("Volatility scan Pushover failed: %s", exc)

    # Only broadcast an actual straddle digest to subscribers — skip empty scans.
    if result.ideas:
        _broadcast_telegram(title, body)


# ── Telegram broadcast bot (multi-user delivery) ────────────────────────────────

def _broadcast_telegram(title: str, body: str, *, monospace: bool = True) -> None:
    """Fan one (title, body) out to every active Telegram subscriber, if the bot is on.

    An index signal is identical for every user, so it is built once (for Pushover)
    and delivered to all subscribers here. Chats that have blocked the bot are pruned
    so a dead subscriber is not retried every cycle.
    """
    settings = get_settings()
    if not settings.telegram_bot_token:
        return
    from nifty_ai_agent.notifier.telegram import TelegramNotifier, broadcast

    repo = DatabaseRepository(settings.database_url)
    chat_ids = [s.chat_id for s in repo.list_active_subscribers()]
    if not chat_ids:
        return

    text = f"{title}\n{body}" if title else body
    try:
        delivered, blocked = broadcast(
            TelegramNotifier(settings.telegram_bot_token), chat_ids, text, monospace=monospace,
        )
        for chat_id in blocked:
            repo.deactivate_subscriber(chat_id)
        logger.info("Telegram broadcast: %d delivered, %d pruned", delivered, len(blocked))
    except Exception as exc:
        logger.error("Telegram broadcast failed: %s", exc)


def _send_notification(settings, title: str, body: str, *, monospace: bool = True) -> None:
    """Deliver one (title, body) to both channels — Pushover (if enabled) and every
    Telegram subscriber. The single place new digests route through so they reach
    whichever channels are configured without repeating the boilerplate."""
    try:
        PushoverNotifier(
            user_key=settings.pushover_user_key,
            api_token=settings.pushover_api_token,
            enabled=settings.pushover_enabled,
        ).send_text(title=title, message=body, monospace=monospace)
    except Exception as exc:
        logger.error("Pushover send failed: %s", exc)
    _broadcast_telegram(title, body, monospace=monospace)


def _run_telegram_bot() -> None:
    """Long-poll Telegram for commands and reply — runs forever in a daemon thread.

    The loop never dies on error: a failed poll sleeps briefly and retries, so a
    transient network blip cannot take the subscribe/unsubscribe surface offline.
    """
    settings = get_settings()
    token = settings.telegram_bot_token
    if not token:
        return
    from nifty_ai_agent.notifier.telegram import TelegramNotifier, TelegramBlockedError
    from nifty_ai_agent.notifier.telegram_commands import handle_update

    notifier = TelegramNotifier(token)
    repo = DatabaseRepository(settings.database_url)
    offset: int | None = None
    logger.info("Telegram bot polling started")

    while True:
        try:
            for update in notifier.get_updates(offset=offset, timeout=25):
                offset = update["update_id"] + 1
                result = handle_update(update, repo)
                if result is None:
                    continue
                chat_id, reply = result
                try:
                    notifier.send_message(chat_id, reply)
                except TelegramBlockedError:
                    repo.deactivate_subscriber(chat_id)
                except Exception as exc:
                    logger.warning("Telegram reply to %s failed: %s", chat_id, exc)
        except Exception as exc:
            logger.error("Telegram poll loop error: %s", exc)
            time.sleep(5)


# ── Entry point ────────────────────────────────────────────────────────────────

_ES_CONTINUOUS      = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def _prevent_sleep() -> None:
    """Tell Windows not to sleep while the agent is running (no-op off Windows).

    Only Windows has this power-management API; in a Linux container (deployment)
    there is nothing to keep awake, so the call is skipped rather than crashing on
    the absent ctypes.windll.
    """
    if sys.platform != "win32":
        return
    ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)


def _allow_sleep() -> None:
    """Restore normal sleep behaviour when the agent exits (no-op off Windows)."""
    if sys.platform != "win32":
        return
    ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)


def main() -> None:
    global _token_monitor

    configure_logging()
    _prevent_sleep()
    settings = get_settings()

    _token_monitor = TokenMonitor(
        notifier=PushoverNotifier(
            user_key=settings.pushover_user_key,
            api_token=settings.pushover_api_token,
            enabled=settings.pushover_enabled,
        )
    )

    # ── Build per-index configurations ──────────────────────────────────────────
    _INDEX_CONFIGS.clear()
    _INDEX_CONFIGS.extend([
        IndexConfig(
            name="NIFTY",
            symbol=settings.nifty_symbol,
            strike_step=50,
            expiry_weekday=1,  # Tuesday — confirmed live via Upstox; NSE moved NIFTY off Thursday
            # Re-reads the token from get_settings() on every call (not captured as a
            # default arg) so a daily scripts/upstox_login.py refresh takes effect
            # without restarting the agent.
            make_provider=lambda s=settings.nifty_symbol: (
                NSEDataProvider(symbol=s, upstox_access_token=get_settings().upstox_access_token)
            ),
            fetch_breadth=fetch_realtime_breadth,
        ),
        IndexConfig(
            name="SENSEX",
            symbol=settings.sensex_symbol,
            strike_step=100,
            expiry_weekday=3,  # Thursday — confirmed live via Upstox; BSE moved SENSEX off Friday
            make_provider=lambda s=settings.sensex_symbol: (
                BSEDataProvider(symbol=s, upstox_access_token=get_settings().upstox_access_token)
            ),
            fetch_breadth=fetch_sensex_breadth,
        ),
        IndexConfig(
            name="BANKNIFTY",
            symbol=settings.banknifty_symbol,
            strike_step=100,
            # Tuesday — confirmed live via Upstox. BANKNIFTY has no true weekly expiry
            # anymore (SEBI's Nov-2024 rules left only one weekly per exchange, which
            # NSE assigned to NIFTY) — BANKNIFTY is monthly-only, so "weekly" here
            # really means "nearest available (monthly) contract."
            expiry_weekday=1,
            make_provider=lambda s=settings.banknifty_symbol: (
                NSEDataProvider(
                    symbol=s, upstox_access_token=get_settings().upstox_access_token,
                    index_name="BANKNIFTY", strike_step=100,
                )
            ),
            fetch_breadth=fetch_banknifty_breadth,
        ),
    ])

    logger.info(
        "NIFTY+SENSEX+BANKNIFTY AI Agent starting — morning report @ 08:00 IST | "
        "intraday signals every %d min | indices: %s",
        settings.data_fetch_interval_minutes,
        ", ".join(c.name for c in _INDEX_CONFIGS),
    )

    # Startup ping
    try:
        PushoverNotifier(
            user_key=settings.pushover_user_key,
            api_token=settings.pushover_api_token,
            enabled=settings.pushover_enabled,
        ).send_text(
            title="Market Agent Started",
            message=(
                f"Tracking: {', '.join(c.name for c in _INDEX_CONFIGS)}\n"
                f"Morning report @ 08:00 IST daily.\n"
                f"Signals every {settings.data_fetch_interval_minutes} min "
                f"(09:15–15:30 IST)."
            ),
        )
    except Exception as exc:
        logger.warning("Startup ping failed: %s", exc)

    # ── Telegram broadcast bot ────────────────────────────────────────────────────
    # When a bot token is set, poll for /start-etc commands in a daemon thread so
    # users can subscribe while the scheduler keeps running; signals fan out to all
    # active subscribers from inside each _run_* via _broadcast_telegram().
    if settings.telegram_bot_token:
        threading.Thread(target=_run_telegram_bot, name="telegram-bot", daemon=True).start()
        logger.info("Telegram broadcast bot enabled")

    # ── Schedule ────────────────────────────────────────────────────────────────
    # GIFT Nifty reads on the next session: 17:00 (Session 2 open, first overnight
    # read) and 06:45 (Session 1 open, final pre-open read before 09:15).
    schedule.every().day.at("06:45").do(_run_next_session_outlook)
    schedule.every().day.at("17:00").do(_run_next_session_outlook)
    schedule.every().day.at("07:45").do(_run_overnight_analysis)
    schedule.every().day.at("08:00").do(_run_morning_report)
    # Pre-open trade plan — 5 min before the 09:15 open, so the day's levels are ready.
    schedule.every().day.at("09:10").do(_run_trade_plan, pre_open=True)
    # Token health runs before the market opens (so a dead overnight token is caught
    # while there is still time to fix it) and hourly through the session.
    schedule.every().day.at("08:05").do(_check_token)
    schedule.every().hour.do(_check_token)
    schedule.every(settings.data_fetch_interval_minutes).minutes.do(_run_all_pipelines)
    schedule.every(30).minutes.do(_run_trade_plan)  # buy/sell prices, all 3 indices
    schedule.every().day.at("16:00").do(_run_eod_prediction)

    # Single-stock CE/PE scan on a fixed interval (market-hours guarded).
    if settings.stock_scan_enabled:
        schedule.every(settings.stock_scan_interval_minutes).minutes.do(_run_stock_scan)
        logger.info("Stock scan scheduled every %d min", settings.stock_scan_interval_minutes)

    # Volatile-stock straddle scan on a fixed interval (market-hours guarded).
    if settings.volatility_scan_enabled:
        schedule.every(settings.volatility_scan_interval_minutes).minutes.do(_run_volatility_scan)
        logger.info("Volatility scan scheduled every %d min", settings.volatility_scan_interval_minutes)

    # ── Immediate startup action ─────────────────────────────────────────────
    _check_token()
    now_ist = datetime.now(_IST)
    market_close = now_ist.replace(hour=_MARKET_CLOSE_HOUR, minute=_MARKET_CLOSE_MINUTE, second=0, microsecond=0)
    if now_ist.hour < _MARKET_OPEN_HOUR:
        logger.info("Pre-market start — running morning report now.")
        _run_morning_report()
    elif now_ist <= market_close:
        logger.info("Market hours — running live pipeline now.")
        _run_all_pipelines()
        _run_trade_plan()
        _run_stock_scan()
        _run_volatility_scan()
    else:
        logger.info("Post-market start — running EOD prediction now.")
        _run_eod_prediction()

    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    finally:
        _allow_sleep()


if __name__ == "__main__":
    main()
