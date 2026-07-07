"""Option chain analyser — weekly expiry CE/PE values, max pain, support/resistance from OI."""

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

_RISK_FREE_RATE = 0.068  # RBI repo rate approx
_STRIKE_STEP = 50         # NIFTY strikes are in multiples of 50


@dataclass
class OptionLeg:
    strike: int
    ce_ltp: float
    pe_ltp: float
    ce_oi: int
    pe_oi: int
    ce_iv: float
    pe_iv: float
    is_atm: bool = False


@dataclass
class ExpiryAnalysis:
    expiry: str
    spot: float
    atm_strike: int
    max_pain: float
    pcr: float
    legs: list[OptionLeg]               # ATM ± 3 strikes
    call_oi_resistance: int             # strike with highest CE OI (resistance)
    put_oi_support: int                 # strike with highest PE OI (support)
    bias: str                           # "BULLISH", "BEARISH", "NEUTRAL"
    days_to_expiry: int = 0
    theoretical_ce_atm: float = 0.0    # Black-Scholes ATM CE price
    theoretical_pe_atm: float = 0.0    # Black-Scholes ATM PE price
    atm_ce_ltp: float = 0.0            # Live ATM CE last traded price from chain
    atm_pe_ltp: float = 0.0            # Live ATM PE last traded price from chain
    is_live: bool = True               # False when derived from a VIX-based estimate, not a real chain


def analyse_option_chain(
    option_chain: pd.DataFrame,
    spot: float,
    expiry: str,
    strikes_each_side: int = 3,
) -> ExpiryAnalysis:
    """Analyse the option chain and return a structured weekly expiry report.

    Args:
        option_chain: DataFrame with columns: strike, ce_oi, pe_oi, ce_ltp, pe_ltp, ce_iv, pe_iv
        spot: Current NIFTY spot price.
        expiry: Expiry date string (e.g. "27-Jun-2024").
        strikes_each_side: How many strikes above and below ATM to include.
    """
    if option_chain.empty:
        logger.warning("Empty option chain — returning stub analysis")
        return _stub_analysis(spot, expiry)

    df = option_chain.copy()
    df = df.sort_values("strike").reset_index(drop=True)

    # ATM = nearest strike to spot
    atm_strike = _nearest_atm(spot)
    df["is_atm"] = df["strike"] == atm_strike

    # PCR and max pain
    total_ce_oi = df["ce_oi"].sum()
    total_pe_oi = df["pe_oi"].sum()
    pcr = round(total_pe_oi / total_ce_oi, 3) if total_ce_oi > 0 else 0.0
    max_pain = _compute_max_pain(df)

    # CE OI resistance and PE OI support
    ce_resistance_row = df.loc[df["ce_oi"].idxmax()]
    pe_support_row = df.loc[df["pe_oi"].idxmax()]
    call_oi_resistance = int(ce_resistance_row["strike"])
    put_oi_support = int(pe_support_row["strike"])

    # Select ATM ± N strikes
    strikes_window = [
        atm_strike + i * _STRIKE_STEP
        for i in range(-strikes_each_side, strikes_each_side + 1)
    ]
    window_df = df[df["strike"].isin(strikes_window)]

    legs: list[OptionLeg] = []
    for _, row in window_df.iterrows():
        legs.append(OptionLeg(
            strike=int(row["strike"]),
            ce_ltp=float(row.get("ce_ltp", 0)),
            pe_ltp=float(row.get("pe_ltp", 0)),
            ce_oi=int(row.get("ce_oi", 0)),
            pe_oi=int(row.get("pe_oi", 0)),
            ce_iv=float(row.get("ce_iv", 0)),
            pe_iv=float(row.get("pe_iv", 0)),
            is_atm=(int(row["strike"]) == atm_strike),
        ))

    # Bias: PCR > 1.2 bullish, < 0.8 bearish; also check where max pain is vs spot
    bias = _compute_bias(pcr, spot, max_pain)

    # Days to expiry
    dte = _days_to_expiry(expiry)

    # Theoretical pricing (Black-Scholes) for ATM options using average IV
    atm_row = df[df["strike"] == atm_strike]
    theo_ce = theo_pe = 0.0
    if not atm_row.empty and dte > 0:
        iv_ce = float(atm_row.iloc[0].get("ce_iv", 0)) / 100
        iv_pe = float(atm_row.iloc[0].get("pe_iv", 0)) / 100
        iv = (iv_ce + iv_pe) / 2 if iv_ce and iv_pe else (iv_ce or iv_pe or 0.15)
        T = dte / 365
        theo_ce = round(_bs_price(spot, atm_strike, T, _RISK_FREE_RATE, iv, "CE"), 2)
        theo_pe = round(_bs_price(spot, atm_strike, T, _RISK_FREE_RATE, iv, "PE"), 2)

    # Live ATM LTP from chain (fall back to theoretical if chain data is missing)
    atm_leg = next((l for l in legs if l.is_atm), None)
    atm_ce_ltp = (atm_leg.ce_ltp if atm_leg and atm_leg.ce_ltp > 0 else theo_ce)
    atm_pe_ltp = (atm_leg.pe_ltp if atm_leg and atm_leg.pe_ltp > 0 else theo_pe)

    return ExpiryAnalysis(
        expiry=expiry,
        spot=round(spot, 2),
        atm_strike=atm_strike,
        max_pain=max_pain,
        pcr=pcr,
        legs=legs,
        call_oi_resistance=call_oi_resistance,
        put_oi_support=put_oi_support,
        bias=bias,
        days_to_expiry=dte,
        theoretical_ce_atm=theo_ce,
        theoretical_pe_atm=theo_pe,
        atm_ce_ltp=round(atm_ce_ltp, 1),
        atm_pe_ltp=round(atm_pe_ltp, 1),
    )


def format_analysis_for_notification(analysis: ExpiryAnalysis) -> str:
    """Format option analysis as a Pushover notification body."""
    lines = [
        f"📅 Expiry: {analysis.expiry}  ({analysis.days_to_expiry}d)",
        f"Spot: {analysis.spot:,.0f}  ATM: {analysis.atm_strike:,}",
        f"Max Pain: {analysis.max_pain:,.0f}  |  PCR: {analysis.pcr}  |  {analysis.bias}",
        f"🔴 CE Resistance: {analysis.call_oi_resistance:,}",
        f"🟢 PE Support:    {analysis.put_oi_support:,}",
        "",
        f"{'Strike':>7} {'CE LTP':>7} {'CE OI':>8} | {'PE LTP':>7} {'PE OI':>8}",
        "─" * 48,
    ]
    for leg in analysis.legs:
        atm_tag = " ◀ ATM" if leg.is_atm else ""
        lines.append(
            f"{leg.strike:>7,} {leg.ce_ltp:>7.1f} {leg.ce_oi:>8,} | "
            f"{leg.pe_ltp:>7.1f} {leg.pe_oi:>8,}{atm_tag}"
        )
    if analysis.theoretical_ce_atm:
        lines += [
            "",
            f"Theoretical ATM CE: ₹{analysis.theoretical_ce_atm}",
            f"Theoretical ATM PE: ₹{analysis.theoretical_pe_atm}",
        ]
    return "\n".join(lines)


def format_monthly_analysis_for_notification(analysis: ExpiryAnalysis) -> str:
    """Compact Pushover body for the monthly expiry — key levels only, no strike table."""
    lines = [
        f"📅 Monthly Expiry: {analysis.expiry}  ({analysis.days_to_expiry}d to go)",
        f"Spot: {analysis.spot:,.0f}  ATM: {analysis.atm_strike:,}",
        f"Max Pain: {analysis.max_pain:,.0f}  |  PCR: {analysis.pcr}  |  {analysis.bias}",
        f"🔴 CE OI Wall (resistance): {analysis.call_oi_resistance:,}",
        f"🟢 PE OI Floor (support):   {analysis.put_oi_support:,}",
    ]
    if analysis.theoretical_ce_atm:
        lines += [
            "",
            f"Theoretical ATM CE: ₹{analysis.theoretical_ce_atm}",
            f"Theoretical ATM PE: ₹{analysis.theoretical_pe_atm}",
        ]
    lines += [
        "",
        "Key monthly level: market tends to gravitate toward max pain into expiry.",
    ]
    return "\n".join(lines)


def monthly_option_chain_note(
    analysis: ExpiryAnalysis | None,
    signal_type: str,
) -> tuple[int, str]:
    """Lighter confidence adjustment from monthly expiry key levels.

    Monthly OI walls are stronger long-term barriers than weekly ones but
    they matter less for a 5-minute signal, so adjustments are smaller.
    Checks: monthly PCR bias, proximity to monthly CE wall, proximity to
    monthly PE floor, and max pain distance (DTE ≤ 3 for pin risk).
    """
    if analysis is None or signal_type == "HOLD" or analysis.spot <= 0:
        return 0, ""

    bullish = signal_type == "BUY_CE"
    spot = analysis.spot
    delta = 0
    notes: list[str] = []

    # Monthly PCR structural bias
    if analysis.pcr > 1.5:
        if bullish:
            delta += 3
            notes.append(f"Monthly PCR {analysis.pcr} — strong structural bull bias")
        else:
            delta -= 3
            notes.append(f"Monthly PCR {analysis.pcr} — bullish backdrop against PE")
    elif analysis.pcr < 0.7:
        if not bullish:
            delta += 3
            notes.append(f"Monthly PCR {analysis.pcr} — strong structural bear bias")
        else:
            delta -= 3
            notes.append(f"Monthly PCR {analysis.pcr} — bearish backdrop against CE")

    # Monthly CE OI wall — major resistance for BUY_CE
    if bullish and analysis.call_oi_resistance > 0:
        gap_pct = (analysis.call_oi_resistance - spot) / spot * 100
        if 0 < gap_pct < 0.5:
            delta -= 8
            notes.append(
                f"Monthly CE wall at {analysis.call_oi_resistance:,}"
                f" — only {gap_pct:.1f}% above (major barrier)"
            )
        elif 0.5 <= gap_pct < 1.0:
            delta -= 3
            notes.append(
                f"Monthly CE resistance at {analysis.call_oi_resistance:,}"
                f" ({gap_pct:.1f}% above)"
            )

    # Monthly PE OI floor — major support for BUY_PE
    if not bullish and analysis.put_oi_support > 0:
        gap_pct = (spot - analysis.put_oi_support) / spot * 100
        if 0 < gap_pct < 0.5:
            delta -= 6
            notes.append(
                f"Monthly PE floor at {analysis.put_oi_support:,}"
                f" — only {gap_pct:.1f}% below (strong floor)"
            )
        elif 0.5 <= gap_pct < 1.0:
            delta -= 2
            notes.append(
                f"Monthly PE support at {analysis.put_oi_support:,}"
                f" ({gap_pct:.1f}% below)"
            )

    # Max pain pinning near monthly expiry
    if analysis.max_pain > 0 and analysis.days_to_expiry <= 3:
        pin_pct = abs(spot - analysis.max_pain) / spot * 100
        if pin_pct < 0.5:
            delta -= 6
            notes.append(
                f"Near monthly max pain ({analysis.max_pain:,.0f})"
                f" with {analysis.days_to_expiry}d left — pinning risk"
            )

    detail = (
        f" [Monthly {analysis.expiry}: {'; '.join(notes)}]"
        if notes else ""
    )
    return delta, detail


# ── Helpers ────────────────────────────────────────────────────────────────────

def _nearest_atm(spot: float) -> int:
    return int(round(spot / _STRIKE_STEP) * _STRIKE_STEP)


def _compute_max_pain(df: pd.DataFrame) -> float:
    if df.empty or "strike" not in df.columns:
        return 0.0
    strikes = df["strike"].tolist()
    min_pain = float("inf")
    max_pain_strike = 0.0
    for s in strikes:
        pain = sum(
            row["ce_oi"] * max(0, row["strike"] - s)
            + row["pe_oi"] * max(0, s - row["strike"])
            for _, row in df.iterrows()
        )
        if pain < min_pain:
            min_pain = pain
            max_pain_strike = s
    return float(max_pain_strike)


def _compute_bias(pcr: float, spot: float, max_pain: float) -> str:
    score = 0
    if pcr > 1.2:
        score += 1
    elif pcr < 0.8:
        score -= 1
    if max_pain > spot * 1.005:
        score += 1
    elif max_pain < spot * 0.995:
        score -= 1
    return "BULLISH" if score > 0 else ("BEARISH" if score < 0 else "NEUTRAL")


def _days_to_expiry(expiry_str: str) -> int:
    """Parse expiry string like '27-Jun-2024' and return days from today."""
    try:
        expiry_date = datetime.strptime(expiry_str, "%d-%b-%Y").date()
        return max(0, (expiry_date - date.today()).days)
    except Exception:
        return 1


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using math.erf (no scipy needed)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_price(S: float, K: int, T: float, r: float, sigma: float, option_type: str) -> float:
    """Black-Scholes option price."""
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if option_type == "CE" else (K - S))
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "CE":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def option_chain_confidence_adjustment(
    analysis: ExpiryAnalysis,
    signal_type: str,
) -> tuple[int, str]:
    """Return *(confidence_delta, detail_text)* derived from option chain data.

    Checks applied (in order):
    1. PCR direction vs signal
    2. Spot proximity to CE OI resistance (overhead ceiling for BUY_CE)
    3. Spot proximity to PE OI support (floor for BUY_PE)
    4. Max pain pinning risk (within 0.3% of max pain on expiry day or DTE ≤ 1)

    signal_type: "BUY_CE", "BUY_PE", or "HOLD".
    """
    if signal_type == "HOLD" or analysis.spot <= 0:
        return 0, ""

    bullish = signal_type == "BUY_CE"
    spot = analysis.spot
    delta = 0
    notes: list[str] = []

    # ── PCR ────────────────────────────────────────────────────────────────────
    if analysis.pcr > 1.2:
        if bullish:
            delta += 5
            notes.append(f"PCR {analysis.pcr} (high put writing — bullish)")
        else:
            delta -= 5
            notes.append(f"PCR {analysis.pcr} (high put writing — against PE signal)")
    elif analysis.pcr < 0.8:
        if not bullish:
            delta += 5
            notes.append(f"PCR {analysis.pcr} (heavy call writing — bearish)")
        else:
            delta -= 5
            notes.append(f"PCR {analysis.pcr} (heavy call writing — against CE signal)")

    # ── CE OI resistance proximity ─────────────────────────────────────────────
    if bullish and analysis.call_oi_resistance > 0:
        gap_pct = (analysis.call_oi_resistance - spot) / spot * 100
        if 0 < gap_pct < 0.5:
            delta -= 12
            notes.append(
                f"CE OI wall at {analysis.call_oi_resistance:,} only"
                f" {gap_pct:.1f}% above — strong ceiling"
            )
        elif 0.5 <= gap_pct < 1.2:
            delta -= 5
            notes.append(
                f"CE OI resistance at {analysis.call_oi_resistance:,}"
                f" ({gap_pct:.1f}% above)"
            )
        elif gap_pct >= 1.2:
            delta += 4
            notes.append(
                f"Open air to CE resistance at {analysis.call_oi_resistance:,}"
                f" ({gap_pct:.1f}% away)"
            )

    # ── PE OI support proximity ────────────────────────────────────────────────
    if not bullish and analysis.put_oi_support > 0:
        gap_pct = (spot - analysis.put_oi_support) / spot * 100
        if 0 < gap_pct < 0.5:
            delta -= 10
            notes.append(
                f"PE OI support at {analysis.put_oi_support:,} only"
                f" {gap_pct:.1f}% below — limited downside near floor"
            )
        elif 0.5 <= gap_pct < 1.2:
            delta -= 4
            notes.append(
                f"PE support at {analysis.put_oi_support:,} ({gap_pct:.1f}% below)"
            )
        elif gap_pct >= 1.2:
            delta += 4
            notes.append(
                f"Room to PE support at {analysis.put_oi_support:,}"
                f" ({gap_pct:.1f}% below)"
            )

    # ── Max pain pinning (near expiry) ─────────────────────────────────────────
    if analysis.max_pain > 0 and analysis.days_to_expiry <= 1:
        pin_pct = abs(spot - analysis.max_pain) / spot * 100
        if pin_pct < 0.3:
            delta -= 10
            notes.append(
                f"Spot within 0.3% of max pain ({analysis.max_pain:,.0f})"
                f" on expiry day — pinning risk"
            )

    detail = (
        f" OC ({analysis.expiry}, PCR {analysis.pcr}): {'; '.join(notes)}."
        if notes else ""
    )
    return delta, detail


def compute_atm_theoretical_prices(
    spot: float,
    atm_strike: int,
    expiry_str: str,
    iv: float,
) -> tuple[float, float]:
    """Return (CE price, PE price) for the ATM strike using Black-Scholes.

    Args:
        spot: Current index spot price.
        atm_strike: ATM strike price (rounded to the index's strike step).
        expiry_str: Expiry date string, e.g. '10-Jul-2026'.
        iv: Annualised implied volatility as a decimal, e.g. 0.145 for VIX 14.5.

    Returns:
        Tuple of (CE price, PE price) rounded to 1 decimal place.
    """
    dte = _days_to_expiry(expiry_str)
    T = max(dte, 1) / 365
    iv = max(iv, 0.05)  # floor at 5% to avoid degenerate results
    ce = _bs_price(spot, atm_strike, T, _RISK_FREE_RATE, iv, "CE")
    pe = _bs_price(spot, atm_strike, T, _RISK_FREE_RATE, iv, "PE")
    return round(ce, 1), round(pe, 1)


def _stub_analysis(spot: float, expiry: str) -> ExpiryAnalysis:
    return ExpiryAnalysis(
        expiry=expiry, spot=spot,
        atm_strike=_nearest_atm(spot),
        max_pain=spot, pcr=1.0, legs=[],
        call_oi_resistance=_nearest_atm(spot) + 200,
        put_oi_support=_nearest_atm(spot) - 200,
        bias="NEUTRAL",
        is_live=False,
    )
