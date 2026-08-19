"""On-demand analysis of an index — the /analyse path for NIFTY, BANKNIFTY, SENSEX.

Indices differ from stocks in three ways that matter here, which is why they get
their own path rather than being forced through the stock scanner:

  * The underlying has no fundamentals. There is no P/E or results date for an
    index, so the "buy the shares?" half of the stock report is meaningless and
    is replaced by a directional read on the index itself.
  * The contract comes from a live option chain, not the instrument master, so
    the ATM strike and both CE/PE premiums are already known.
  * Weekly expiries exist (NIFTY), so the near contract is days away rather than
    a month.
"""

import logging
from datetime import datetime

import pytz

from nifty_ai_agent.risk.calculator import RiskCalculator
from nifty_ai_agent.strategies.base import SignalType
from nifty_ai_agent.strategies.consensus import build_consensus
from nifty_ai_agent.strategies.multi_timeframe import (
    INDEX_TIMEFRAMES,
    gated_signal,
    read_timeframes,
)
from nifty_ai_agent.strategies.option_analyser import analyse_option_chain
from nifty_ai_agent.strategies.pipeline import DEFAULT_STRATEGIES, compute_all_indicators

logger = logging.getLogger(__name__)

_IST = pytz.timezone("Asia/Kolkata")

# What a user might type → the canonical index name.
INDEX_ALIASES: dict[str, str] = {
    "NIFTY": "NIFTY", "NIFTY50": "NIFTY", "NIFTY 50": "NIFTY", "^NSEI": "NIFTY",
    "BANKNIFTY": "BANKNIFTY", "BANK NIFTY": "BANKNIFTY", "NIFTYBANK": "BANKNIFTY",
    "SENSEX": "SENSEX", "^BSESN": "SENSEX", "BSESN": "SENSEX",
}


def resolve_index(raw: str) -> str | None:
    """'bank nifty' → 'BANKNIFTY'; anything unrecognised → None."""
    return INDEX_ALIASES.get(" ".join(raw.strip().upper().split()))


def _make_provider(index: str, settings):
    """Build the data provider for *index*, mirroring main's IndexConfig wiring."""
    token = settings.upstox_access_token
    if index == "SENSEX":
        from nifty_ai_agent.data.bse_provider import BSEDataProvider
        return BSEDataProvider(symbol=settings.sensex_symbol, upstox_access_token=token)

    from nifty_ai_agent.data.nse_provider import NSEDataProvider
    if index == "BANKNIFTY":
        return NSEDataProvider(
            symbol=settings.banknifty_symbol, upstox_access_token=token,
            index_name="BANKNIFTY", strike_step=100,
        )
    return NSEDataProvider(symbol=settings.nifty_symbol, upstox_access_token=token)


def analyse_index(index: str, settings) -> str:
    """Return the rendered analysis for *index*, or a readable error line."""
    try:
        provider = _make_provider(index, settings)
        spot = provider.get_spot_data()
        hist = provider.get_historical_data(
            days=settings.historical_days, interval=settings.data_interval,
        )
        df = compute_all_indicators(hist)
        usable = df.dropna(subset=["ema_20", "ema_50", "rsi", "atr"])
        if usable.empty:
            return f"❌ {index}: not enough history to compute indicators yet."

        now = datetime.now(_IST).time()
        signals = [s.generate_signal(df) for s in DEFAULT_STRATEGIES]
        consensus = build_consensus(signals, now=now, intraday=True)

        # Indices are volatile enough that one timeframe is one opinion. The 5m
        # read decides direction; the 60m read decides whether it is worth acting
        # on — that filter took 5m entries from +0.003% to +0.072% expectancy in
        # backtest, mostly by removing entries rather than changing the payoff.
        mtf = read_timeframes(hist, INDEX_TIMEFRAMES, now=now, intraday=True)
        gated, gate_reason = gated_signal(mtf, entry_tf="5m", trend_tf="60m")

        risk = RiskCalculator(
            max_risk_pct=settings.max_risk_per_trade_pct,
            daily_loss_limit_pct=settings.daily_loss_limit_pct,
            min_rr=settings.min_risk_reward_ratio,
            atr_sl_multiplier=settings.atr_sl_multiplier,
        ).calculate(consensus.signal, spot.price, float(usable.iloc[-1]["atr"]))

        chain = None
        try:
            data = provider.get_option_chain()
            if not data.strikes.empty:
                chain = analyse_option_chain(
                    option_chain=data.strikes, spot=spot.price, expiry=data.expiry,
                )
        except Exception as exc:
            logger.warning("%s option chain unavailable: %s", index, exc)

        return _format(index, spot.price, consensus, risk, chain, signals,
                       mtf, gated, gate_reason)
    except Exception as exc:
        logger.error("analyse_index(%s) failed: %s", index, exc)
        return f"❌ {index}: analysis failed ({type(exc).__name__}). Try again shortly."


def _format(index, spot, consensus, risk, chain, signals,
            mtf=None, gated=None, gate_reason="") -> str:
    lines = [f"📊 {index} — {spot:,.2f}", ""]

    # The headline is the GATED call, not the raw 5m read. A 5m signal the 60m
    # trend contradicts is the single most common way this agent lost money in
    # backtest, so it must not be the first thing the eye lands on.
    verdict = gated if gated is not None else consensus.signal
    icon = {"BUY_CE": "📈", "BUY_PE": "📉"}.get(verdict.value, "⏸")
    lines.append(f"DIRECTION  {icon} {verdict.value}")
    lines.append(f"  {consensus.confidence}% · {consensus.conviction}")
    if gate_reason:
        lines.append(f"  {gate_reason}")
    lines.append(f"  {consensus.rationale[:110]}")

    if mtf is not None:
        lines += ["", "TIMEFRAMES"]
        lines.append(f"  {mtf.summary()}")
        if mtf.is_conflicted():
            lines.append("  ⚠ timeframes disagree — noise arguing with trend")

    # Every vote, not just the verdict: on an index the disagreement is the most
    # useful part — a 3-3 split is a different market from a 6-0 one.
    lines += ["", "STRATEGY VOTES"]
    for s in signals:
        mark = {"BUY_CE": "CE", "BUY_PE": "PE"}.get(s.signal.value, "--")
        lines.append(f"  {mark}  {s.strategy:<24}{s.confidence:>3}%")

    lines += ["", "LEVELS"]
    if risk.is_valid:
        lines.append(f"  entry {risk.entry_price:,.0f} · SL {risk.stop_loss:,.0f} "
                     f"· tgt {risk.target:,.0f} (1:{risk.risk_reward_ratio:g})")
    else:
        lines.append(f"  no valid trade — {risk.rejection_reason or 'HOLD'}")

    if chain is not None:
        lines += ["", f"OPTION CHAIN  exp {chain.expiry} ({chain.days_to_expiry}d)"]
        lines.append(f"  ATM {chain.atm_strike:,} · PCR {chain.pcr} · {chain.bias}")
        lines.append(f"  CE ₹{chain.atm_ce_ltp:g} · PE ₹{chain.atm_pe_ltp:g}")
        lines.append(f"  max pain {chain.max_pain:,.0f} · "
                     f"R {chain.call_oi_resistance:,} · S {chain.put_oi_support:,}")
        if not chain.is_live:
            lines.append("  * estimated chain — live data unavailable")

        if verdict != SignalType.HOLD:
            side = "CE" if verdict == SignalType.BUY_CE else "PE"
            premium = chain.atm_ce_ltp if side == "CE" else chain.atm_pe_ltp
            lines += ["", "SUGGESTED CONTRACT"]
            lines.append(f"  BUY {index} {chain.atm_strike:,}{side} @ ₹{premium:g}")
        elif consensus.signal != SignalType.HOLD:
            # There IS a fast signal, it just did not survive the trend filter.
            # Saying so is more useful than showing nothing.
            lines += ["", "NO CONTRACT SUGGESTED"]
            lines.append(f"  5m says {consensus.signal.value}, but {gate_reason}")
    else:
        lines += ["", "OPTION CHAIN  unavailable right now"]

    lines += ["", "⚠️ Analysis, not advice. Options can lose 100%."]
    return "\n".join(lines)
