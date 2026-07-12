"""NIFTY AI Agent — main entry point.

Scheduled jobs (all IST):
  06:45        — next-session outlook (GIFT Nifty Session 1 has opened; final pre-open read)
  08:00        — pre-market morning report (global cues, option chain, news)
  08:05, hourly — Upstox token health check
  every 5 min  — intraday signal pipeline → one consensus trade call per index
  every 15 min — risk & margin report (sent whatever the signals say)
  every 30 min — capital-aware trade plan across all three indices
  16:00        — EOD prediction for the next session
  17:00        — next-session outlook (GIFT Nifty Session 2 has opened; first overnight read)

Start:
    python main.py

Dashboard (separate terminal):
    streamlit run dashboard/app.py
"""

import ctypes
import dataclasses
import logging
import time
from typing import Callable, NamedTuple

import pandas as pd
import pytz
import schedule
from datetime import date, datetime, timedelta

from nifty_ai_agent.ai.explainer import SignalExplainer
from nifty_ai_agent.config import configure_logging, get_settings
from nifty_ai_agent.data.bank_options import BankOptionIdea, suggest_bank_options
from nifty_ai_agent.data.banknifty_breadth import fetch_banknifty_breadth
from nifty_ai_agent.data.gift_nifty import build_outlook, fetch_gift_nifty
from nifty_ai_agent.data.token_health import TokenMonitor
from nifty_ai_agent.data.breadth import BreadthSnapshot, fetch_realtime_breadth
from nifty_ai_agent.data.bse_provider import BSEDataProvider
from nifty_ai_agent.data.nse_provider import NSEDataProvider
from nifty_ai_agent.data.sensex_breadth import fetch_sensex_breadth
from nifty_ai_agent.database.repository import DatabaseRepository
from nifty_ai_agent.indicators.atr import compute_atr
from nifty_ai_agent.indicators.bollinger import compute_bollinger
from nifty_ai_agent.indicators.ema import compute_ema
from nifty_ai_agent.indicators.macd import compute_macd
from nifty_ai_agent.indicators.rsi import compute_rsi
from nifty_ai_agent.indicators.supertrend import compute_supertrend
from nifty_ai_agent.indicators.vwap import compute_vwap
from nifty_ai_agent.notifier.pushover import PushoverNotifier
from nifty_ai_agent.reports.margin_report import (
    IndexMarginView,
    build_index_margin_view,
    format_margin_report,
)
from nifty_ai_agent.reports.morning_report import run_morning_report
from nifty_ai_agent.reports.next_session import format_next_session
from nifty_ai_agent.reports.trade_call import format_trade_call
from nifty_ai_agent.reports.trade_plan import (
    FALLBACK_LOT_SIZES,
    TradeIdea,
    build_trade_idea,
    format_trade_plan,
)
from nifty_ai_agent.risk.calculator import RiskCalculator, RiskParameters
from nifty_ai_agent.risk.margin import MarginCalculator
from nifty_ai_agent.strategies.base import BaseStrategy, Signal, SignalType
from nifty_ai_agent.strategies.bollinger_squeeze import BollingerSqueezeStrategy
from nifty_ai_agent.strategies.consensus import Consensus, build_consensus
from nifty_ai_agent.strategies.gap_analyser import analyse_gap_history, compute_pivots
from nifty_ai_agent.strategies.global_analyser import (
    GlobalSnapshot,
    fetch_global_snapshot,
    global_confidence_adjustment,
)
from nifty_ai_agent.strategies.ema_crossover import EMACrossoverStrategy
from nifty_ai_agent.strategies.macd_momentum import MACDMomentumStrategy
from nifty_ai_agent.strategies.orb import OpeningRangeBreakoutStrategy
from nifty_ai_agent.strategies.supertrend import SupertrendStrategy
from nifty_ai_agent.strategies.vwap_breakout import VWAPBreakoutStrategy
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
# saved and notified, not just one "winning" signal.
_STRATEGIES: list[BaseStrategy] = [
    EMACrossoverStrategy(),
    VWAPBreakoutStrategy(),
    SupertrendStrategy(),
    MACDMomentumStrategy(),
    OpeningRangeBreakoutStrategy(),
    BollingerSqueezeStrategy(),
]


def _compute_indicators(hist: pd.DataFrame) -> pd.DataFrame:
    """Attach every indicator column the strategy engine reads to *hist*."""
    df = compute_ema(hist, periods=[20, 50])
    df = compute_rsi(df)
    df = compute_macd(df)
    df = compute_atr(df)
    df = compute_vwap(df)
    df = compute_supertrend(df)
    df = compute_bollinger(df)
    return df


def _is_market_hours() -> bool:
    """Return True if current IST time is within NSE trading hours."""
    now_ist = datetime.now(_IST)
    if now_ist.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    open_time = now_ist.replace(hour=_MARKET_OPEN_HOUR, minute=15, second=0, microsecond=0)
    close_time = now_ist.replace(hour=_MARKET_CLOSE_HOUR, minute=_MARKET_CLOSE_MINUTE, second=0, microsecond=0)
    return open_time <= now_ist <= close_time


# ── Option chain cache (per index, refreshed every 15 min) ──────────────────────
_OPTION_CACHE_TTL = 15  # minutes
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

    if fetched_at is not None:
        age_minutes = (now - fetched_at).total_seconds() / 60
        if age_minutes < _OPTION_CACHE_TTL:
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

    return build_trade_idea(
        index.name, best.signal, best.confidence, oc_weekly, risk,
        _get_lot_size(index.name),
    )


def _run_trade_plan() -> None:
    """Send one capital-aware trade-plan notification covering all three indices."""
    if not _is_market_hours():
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

    title, body = format_trade_plan(
        ideas, holds, settings.trading_capital, settings.daily_profit_target,
    )
    try:
        PushoverNotifier(
            user_key=settings.pushover_user_key,
            api_token=settings.pushover_api_token,
        ).send_text(title=title, message=body, monospace=True)
        logger.info("Trade plan sent (%d ideas, %d holds)", len(ideas), len(holds))
    except Exception as exc:
        logger.error("Trade plan Pushover failed: %s", exc)


# ── Risk & margin report (sent every cycle, regardless of signals) ─────────────

def _build_index_margin_view(index: IndexConfig, settings) -> IndexMarginView | None:
    """Price the future and both ATM option legs for one index, sized against capital."""
    provider = index.make_provider()
    spot = provider.get_spot_data()
    hist = provider.get_historical_data(
        days=settings.historical_days, interval=settings.data_interval,
    )
    df = _compute_indicators(hist)
    latest = df.dropna(subset=["atr"]).iloc[-1]

    oc_weekly, _ = _get_cached_option_analysis(index, provider, spot.price)
    if oc_weekly is None:
        logger.warning("%s: no option chain — margin view unavailable.", index.name)
        return None

    calculator = MarginCalculator(
        capital=settings.trading_capital,
        max_risk_per_trade_pct=settings.max_risk_per_trade_pct,
        daily_loss_limit_pct=settings.daily_loss_limit_pct,
        max_margin_utilisation_pct=settings.max_margin_utilisation_pct,
    )
    return build_index_margin_view(
        index_name=index.name,
        analysis=oc_weekly,
        atr=float(latest["atr"]),
        lot_size=_get_lot_size(index.name),
        calculator=calculator,
        atr_sl_multiplier=settings.atr_sl_multiplier,
    )


def _run_margin_report() -> None:
    """Send the standalone risk & margin notification for every index.

    Sent on its own schedule and independently of any signal — a HOLD cycle still
    needs to tell you what a position would cost and whether it fits the account.
    """
    if not _is_market_hours():
        logger.info("Margin report: outside market hours — skipping.")
        return

    settings = get_settings()
    views: list[IndexMarginView] = []
    for index in _INDEX_CONFIGS:
        try:
            view = _build_index_margin_view(index, settings)
        except Exception as exc:
            logger.error("Margin report: %s failed: %s", index.name, exc)
            continue
        if view:
            views.append(view)

    calculator = MarginCalculator(
        capital=settings.trading_capital,
        max_risk_per_trade_pct=settings.max_risk_per_trade_pct,
        daily_loss_limit_pct=settings.daily_loss_limit_pct,
        max_margin_utilisation_pct=settings.max_margin_utilisation_pct,
    )
    title, body = format_margin_report(views, calculator)

    try:
        PushoverNotifier(
            user_key=settings.pushover_user_key,
            api_token=settings.pushover_api_token,
        ).send_text(title=title, message=body, monospace=True)
        logger.info("Margin report sent (%d indices)", len(views))
    except Exception as exc:
        logger.error("Margin report Pushover failed: %s", exc)


# ── Entry point ────────────────────────────────────────────────────────────────

_ES_CONTINUOUS      = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def _prevent_sleep() -> None:
    """Tell Windows not to sleep while the agent is running."""
    ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)


def _allow_sleep() -> None:
    """Restore normal sleep behaviour when the agent exits."""
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

    # ── Schedule ────────────────────────────────────────────────────────────────
    # GIFT Nifty reads on the next session: 17:00 (Session 2 open, first overnight
    # read) and 06:45 (Session 1 open, final pre-open read before 09:15).
    schedule.every().day.at("06:45").do(_run_next_session_outlook)
    schedule.every().day.at("17:00").do(_run_next_session_outlook)
    schedule.every().day.at("08:00").do(_run_morning_report)
    # Token health runs before the market opens (so a dead overnight token is caught
    # while there is still time to fix it) and hourly through the session.
    schedule.every().day.at("08:05").do(_check_token)
    schedule.every().hour.do(_check_token)
    schedule.every(settings.data_fetch_interval_minutes).minutes.do(_run_all_pipelines)
    schedule.every(30).minutes.do(_run_trade_plan)  # capital-aware plan, all 3 indices
    # Risk & margin goes out on its own clock so it lands whether or not a signal fired.
    schedule.every(settings.margin_report_interval_minutes).minutes.do(_run_margin_report)
    schedule.every().day.at("16:00").do(_run_eod_prediction)

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
        _run_margin_report()
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
