"""NIFTY AI Agent — main entry point.

Two scheduled jobs:
  1. 08:00 AM IST daily  — pre-market morning report (global cues, option chain, news)
  2. Every 5 min         — intraday signal pipeline (during market hours)

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

import pytz
import schedule
from datetime import date, datetime, timedelta

from nifty_ai_agent.ai.explainer import SignalExplainer
from nifty_ai_agent.config import configure_logging, get_settings
from nifty_ai_agent.data.banknifty_breadth import fetch_banknifty_breadth
from nifty_ai_agent.data.breadth import BreadthSnapshot, fetch_realtime_breadth
from nifty_ai_agent.data.bse_provider import BSEDataProvider
from nifty_ai_agent.data.nse_provider import NSEDataProvider
from nifty_ai_agent.data.sensex_breadth import fetch_sensex_breadth
from nifty_ai_agent.database.repository import DatabaseRepository
from nifty_ai_agent.indicators.atr import compute_atr
from nifty_ai_agent.indicators.ema import compute_ema
from nifty_ai_agent.indicators.macd import compute_macd
from nifty_ai_agent.indicators.rsi import compute_rsi
from nifty_ai_agent.indicators.vwap import compute_vwap
from nifty_ai_agent.notifier.pushover import PushoverNotifier
from nifty_ai_agent.reports.morning_report import run_morning_report
from nifty_ai_agent.risk.calculator import RiskCalculator, RiskParameters
from nifty_ai_agent.strategies.base import BaseStrategy, Signal, SignalType
from nifty_ai_agent.strategies.ema_crossover import EMACrossoverStrategy
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
_STRATEGIES: list[BaseStrategy] = [EMACrossoverStrategy(), VWAPBreakoutStrategy()]


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
) -> Signal:
    """Run one strategy and apply the shared RSI/option-chain/breadth adjustments."""
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

    if signal.signal != SignalType.HOLD:
        total_delta = rsi_delta + oc_delta + breadth_delta
        new_confidence = max(10, min(95, signal.confidence + total_delta))
        new_reason = signal.reason + rsi_detail + oc_detail + breadth_detail
        signal = dataclasses.replace(signal, confidence=new_confidence, reason=new_reason)
        logger.info(
            "%s %s signal (final): %s confidence=%d%%  Δrsi=%+d Δoc=%+d Δbreadth=%+d",
            index_name, strategy.NAME, signal.signal.value, signal.confidence,
            rsi_delta, oc_delta, breadth_delta,
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
    df = compute_ema(hist, periods=[20, 50])
    df = compute_rsi(df)
    df = compute_macd(df)
    df = compute_atr(df)
    df = compute_vwap(df)

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

    # ── Run every strategy independently ──────────────────────────────────────────
    results: list[tuple[Signal, RiskParameters, str]] = []
    for strategy in _STRATEGIES:
        signal = _generate_and_adjust_signal(
            strategy, df, rsi_analysis, oc_weekly, oc_monthly, breadth, index.name,
        )

        risk = RiskCalculator(
            max_risk_pct=settings.max_risk_per_trade_pct,
            daily_loss_limit_pct=settings.daily_loss_limit_pct,
            min_rr=settings.min_risk_reward_ratio,
        ).calculate(signal.signal, spot.price, current_atr)

        if not risk.is_valid and signal.signal != SignalType.HOLD:
            logger.warning(
                "%s %s signal rejected by risk manager: %s",
                index.name, strategy.NAME, risk.rejection_reason,
            )

        ai_explanation = ""
        if settings.anthropic_api_key:
            try:
                explanation = SignalExplainer(
                    api_key=settings.anthropic_api_key,
                    model=settings.claude_model,
                ).explain(signal, risk, indicators, index_name=ai_index_label)
                ai_explanation = explanation.text
                logger.info(
                    "%s %s AI explanation generated (%d tokens)",
                    index.name, strategy.NAME, explanation.output_tokens,
                )
            except Exception as exc:
                logger.error("%s %s AI explanation failed: %s", index.name, strategy.NAME, exc)
        else:
            logger.info("ANTHROPIC_API_KEY not set — skipping AI explanation.")

        results.append((signal, risk, ai_explanation))

    # ── Database ──────────────────────────────────────────────────────────────────
    db = DatabaseRepository(settings.database_url)
    try:
        db.save_market_data(hist)
    except Exception as exc:
        logger.error("%s: failed to save market data: %s", index.name, exc)

    signal_ids: list[int] = []
    for signal, risk, ai_explanation in results:
        try:
            signal_ids.append(db.save_signal(signal, risk, ai_explanation))
        except Exception as exc:
            logger.error("%s: failed to save %s signal: %s", index.name, signal.strategy, exc)

    # ── Pushover ──────────────────────────────────────────────────────────────────
    try:
        PushoverNotifier(
            user_key=settings.pushover_user_key,
            api_token=settings.pushover_api_token,
        ).send_multi_signal(
            results,
            option_analysis=oc_weekly,
            monthly_option_analysis=oc_monthly,
            prediction=after_hours,
            index_name=index.name,
            strike_step=index.strike_step,
            expiry_weekday=index.expiry_weekday,
        )
        logger.info("%s Pushover sent (signal ids=%s)", index.name, signal_ids)
    except Exception as exc:
        logger.error("%s Pushover failed: %s", index.name, exc)


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
    configure_logging()
    _prevent_sleep()
    settings = get_settings()

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
    schedule.every().day.at("08:00").do(_run_morning_report)
    schedule.every(settings.data_fetch_interval_minutes).minutes.do(_run_all_pipelines)
    schedule.every().day.at("16:00").do(_run_eod_prediction)

    # ── Immediate startup action ─────────────────────────────────────────────
    now_ist = datetime.now(_IST)
    market_close = now_ist.replace(hour=_MARKET_CLOSE_HOUR, minute=_MARKET_CLOSE_MINUTE, second=0, microsecond=0)
    if now_ist.hour < _MARKET_OPEN_HOUR:
        logger.info("Pre-market start — running morning report now.")
        _run_morning_report()
    elif now_ist <= market_close:
        logger.info("Market hours — running live pipeline now.")
        _run_all_pipelines()
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
