"""Analyse a single stock on demand — the engine behind /analyse <SYMBOL>.

Reuses the scan pipeline rather than reimplementing it, so an on-demand answer
and the scheduled digest can never disagree about the same stock. The only
difference is scope: one symbol instead of fifty, and the full context rendered
rather than a one-line summary.
"""

import logging
from datetime import datetime, time as dt_time

import pytz

from nifty_ai_agent.data.fundamentals import fetch_fundamentals, volume_pace_ratio
from nifty_ai_agent.data.instrument_master import get_instrument_master
from nifty_ai_agent.data.stock_data import fetch_stock_histories
from nifty_ai_agent.reports.stock_analysis import format_stock_analysis
from nifty_ai_agent.risk.calculator import RiskCalculator
from nifty_ai_agent.strategies.consensus import build_consensus
from nifty_ai_agent.strategies.pipeline import DEFAULT_STRATEGIES, compute_all_indicators
from nifty_ai_agent.strategies.stock_scanner import HoldRead, scan_stocks

logger = logging.getLogger(__name__)

_IST = pytz.timezone("Asia/Kolkata")
_NS_SUFFIX = ".NS"


def normalise_symbol(raw: str) -> str:
    """'britannia' / 'BRITANNIA.NS' / ' bse ' → 'BRITANNIA.NS'."""
    symbol = raw.strip().upper().replace(" ", "")
    return symbol if symbol.endswith(_NS_SUFFIX) else f"{symbol}{_NS_SUFFIX}"


def analyse_stock(symbol: str, settings, premium_lookup=None) -> str:
    """Return the rendered analysis for *symbol*, or a plain error line.

    Never raises: this is reached from a user command, and an unknown ticker or a
    dead upstream should come back as a readable message rather than silence.
    """
    ticker = normalise_symbol(symbol)
    bare = ticker[: -len(_NS_SUFFIX)]

    try:
        master = get_instrument_master()
        histories, spots = fetch_stock_histories(
            [ticker], interval=settings.data_interval,
            upstox_token=settings.upstox_access_token, master=master,
        )
        if not histories or spots.get(ticker, 0.0) <= 0:
            # Offer candidates rather than a dead end: the usual causes are a typo
            # and a ticker retired by a corporate action, and both are recoverable
            # if the user is shown what does exist.
            suggestions = master.suggest_symbols(bare)
            if suggestions:
                options = "\n".join(f"  /analyse {s} — {n[:38]}" for s, n in suggestions)
                return f"❌ No price data for {bare}. Did you mean:\n{options}"
            return (
                f"❌ {bare}: no price data and no similar NSE symbol. "
                f"Try a cash-market ticker, e.g. RELIANCE."
            )

        spot = spots[ticker]
        now = datetime.now(_IST).time()
        fundamentals = fetch_fundamentals([ticker])
        fund = fundamentals.get(bare)

        # The consensus is recomputed here (cheaply, from the same bars) because
        # the report shows the read even when it produces no trade — scan_stocks
        # only returns the signal for names that became ideas.
        df = compute_all_indicators(histories[ticker])
        signals = [s.generate_signal(df) for s in DEFAULT_STRATEGIES]
        consensus = build_consensus(signals, now=now, intraday=False)

        risk_calculator = RiskCalculator(
            max_risk_pct=settings.max_risk_per_trade_pct,
            daily_loss_limit_pct=settings.daily_loss_limit_pct,
            min_rr=settings.min_risk_reward_ratio,
            atr_sl_multiplier=settings.atr_sl_multiplier,
        )
        result = scan_stocks(
            histories, spots, master=master, risk_calculator=risk_calculator,
            now=now, premium_lookup=premium_lookup or (lambda keys: {}),
            default_iv=settings.stock_default_iv, top_n=1,
            fundamentals=fundamentals,
            earnings_blackout_days=settings.earnings_blackout_days,
        )
        idea = result.ideas[0] if result.ideas else None
        blocked = next(
            (h.reason for h in result.holds if isinstance(h, HoldRead) and h.reason), "",
        )

        volume_ratio = None
        if fund is not None and "volume" in df.columns:
            today = df[df.index.date == df.index[-1].date()]
            volume_ratio = volume_pace_ratio(
                float(today["volume"].sum()), fund.avg_volume, now,
            )

        return format_stock_analysis(
            symbol=bare, spot=spot, signal=consensus.signal,
            confidence=consensus.confidence, conviction=consensus.conviction,
            reason=consensus.rationale, fund=fund, idea=idea,
            volume_ratio=volume_ratio, blocked_reason=blocked,
        )
    except Exception as exc:
        logger.error("analyse_stock(%s) failed: %s", symbol, exc)
        return f"❌ {bare}: analysis failed ({type(exc).__name__}). Try again shortly."
