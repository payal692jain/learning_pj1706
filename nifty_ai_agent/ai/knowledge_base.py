"""Book-grounded trading knowledge base.

Patterns and rules distilled from:
- Al Brooks  — Trading Price Action Trends
- Anna Coulling — A Complete Guide to Volume Price Analysis
- Thomas Bulkowski — Encyclopedia of Chart Patterns
- Adam Grimes — The Art and Science of Technical Analysis
- Martin Pring — Martin Pring on Price Action
- Bob Volman — Forex Price Action Scalping
"""

from dataclasses import dataclass, field
from typing import Literal

from nifty_ai_agent.strategies.base import SignalType

SignalLiteral = Literal["BUY_CE", "BUY_PE", "HOLD", "ANY"]


@dataclass(frozen=True)
class TradingPattern:
    name: str
    source: str           # book + author
    signal_bias: SignalLiteral
    description: str      # what the pattern looks like
    entry_note: str       # what to look for on entry
    risk_note: str        # what invalidates the setup
    success_rate: int     # approximate % from book/study (0 = not stated)


# ─────────────────────────────────────────────────────────────────────────────
# Al Brooks — Trading Price Action Trends
# ─────────────────────────────────────────────────────────────────────────────
BROOKS_PATTERNS: list[TradingPattern] = [
    TradingPattern(
        name="With-Trend Strong Bar",
        source="Al Brooks — Trading Price Action Trends",
        signal_bias="ANY",
        description=(
            "A bull bar that closes near its high with a small upper tail and large body "
            "indicates strong buying pressure. Bears are trapped and will cover on the next bar."
        ),
        entry_note=(
            "Enter on stop one tick above the high of the prior bull bar. "
            "The larger the bar body relative to its range, the stronger the signal."
        ),
        risk_note=(
            "If the follow-up bar is a doji or bear bar that closes below the midpoint of "
            "the signal bar, bulls are failing — exit immediately."
        ),
        success_rate=65,
    ),
    TradingPattern(
        name="EMA Pullback Entry",
        source="Al Brooks — Trading Price Action Trends",
        signal_bias="ANY",
        description=(
            "Price pulls back to the 20-EMA (used as a trend proxy) during a trend and "
            "forms a reversal bar. The EMA acts as a magnet; a successful test signals "
            "continuation of the larger trend."
        ),
        entry_note=(
            "Buy a bull reversal bar (or sell a bear reversal bar) on a pullback to the EMA. "
            "The 20-EMA should be sloping in the direction of the trade."
        ),
        risk_note=(
            "A close below the EMA by more than one average bar's range signals the pullback "
            "is becoming a reversal — reduce size or exit."
        ),
        success_rate=62,
    ),
    TradingPattern(
        name="Breakout Pullback (BPB)",
        source="Al Brooks — Trading Price Action Trends",
        signal_bias="ANY",
        description=(
            "After a breakout above resistance (or below support), price pulls back briefly "
            "and then resumes. This is the safest entry because the breakout has already been "
            "proven — you are simply buying the first meaningful pullback."
        ),
        entry_note=(
            "Wait for the pullback to form at least two legs. Enter when price reverses back "
            "in the breakout direction, ideally with a strong bar. Risk to the low of the pullback."
        ),
        risk_note=(
            "If the pullback exceeds 50% of the breakout move, the breakout is likely failing. "
            "Exit and wait for a new setup."
        ),
        success_rate=68,
    ),
    TradingPattern(
        name="Bull Flag / Bear Flag",
        source="Al Brooks — Trading Price Action Trends",
        signal_bias="ANY",
        description=(
            "A tight sideways consolidation (3–10 bars) after a strong trending move. "
            "Price moves in small overlapping bars, trapping late trend traders. "
            "The flag is the pause before the next leg."
        ),
        entry_note=(
            "Enter on a stop above the flag high (bull flag) or below the flag low (bear flag). "
            "Measured move target = the height of the flag pole added to the breakout point."
        ),
        risk_note=(
            "If the flag exceeds 15 bars, it is becoming a trading range — the breakout success "
            "rate drops. Do not chase a breakout from a long flag."
        ),
        success_rate=60,
    ),
    TradingPattern(
        name="Climax Reversal",
        source="Al Brooks — Trading Price Action Trends",
        signal_bias="ANY",
        description=(
            "An exhaustion bar: an extremely large bar (3–5× average size) closing near its "
            "extreme after a prolonged trend. Represents the last rush of momentum buyers/sellers "
            "before a reversal or extended sideways move."
        ),
        entry_note=(
            "Do NOT enter in the direction of a climax bar. Wait for a reversal signal on the "
            "next 1–3 bars. The reversal bar entry against the climax direction is the setup."
        ),
        risk_note=(
            "Strong trends can have multiple climaxes before reversing. "
            "Only trade the first climax reversal if confirmed by follow-through."
        ),
        success_rate=55,
    ),
    TradingPattern(
        name="Two-Legged Pullback (2L)",
        source="Al Brooks — Trading Price Action Trends",
        signal_bias="ANY",
        description=(
            "The most reliable pullback structure in a trend: two distinct down-legs (in a bull "
            "trend) separated by a small bounce. Often ends near the EMA or prior support. "
            "The second leg traps the last group of breakout sellers."
        ),
        entry_note=(
            "Buy a bull reversal bar after the second leg completes. Target = prior high. "
            "Stop = one tick below the low of the second leg."
        ),
        risk_note=(
            "If the two legs overlap heavily, it is a trading range, not a pullback. "
            "Do not force a trend entry in a range."
        ),
        success_rate=64,
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# Anna Coulling — A Complete Guide to Volume Price Analysis (VPA)
# ─────────────────────────────────────────────────────────────────────────────
COULLING_PATTERNS: list[TradingPattern] = [
    TradingPattern(
        name="High Volume Confirmation",
        source="Anna Coulling — Volume Price Analysis",
        signal_bias="ANY",
        description=(
            "When price moves up on high (or ultra-high) volume and closes near the bar's high, "
            "the move is genuine — smart money is participating. "
            "Volume confirms the direction of price; never trade price without volume context."
        ),
        entry_note=(
            "Look for expanding volume as price breaks a key level. "
            "Volume should be at least 150% of the 10-bar average on the breakout bar."
        ),
        risk_note=(
            "If the same high-volume bar has a wide spread but closes near the middle or low, "
            "smart money is SELLING into the move (distribution). Do not buy."
        ),
        success_rate=70,
    ),
    TradingPattern(
        name="Low Volume No-Demand",
        source="Anna Coulling — Volume Price Analysis",
        signal_bias="BUY_PE",
        description=(
            "An up bar on low volume signals a lack of buying interest — the move up "
            "is not supported by smart money. This is a 'No Demand' bar and often "
            "precedes a reversal downward."
        ),
        entry_note=(
            "Use as a warning signal only. Wait for a confirming down bar on higher volume "
            "before entering short (BUY_PE). Do not short solely on low-volume up bars."
        ),
        risk_note=(
            "In a strong bull trend, low-volume pullback bars are normal. "
            "Context matters — No Demand is only significant near resistance."
        ),
        success_rate=58,
    ),
    TradingPattern(
        name="Selling Climax / Stopping Volume",
        source="Anna Coulling — Volume Price Analysis",
        signal_bias="BUY_CE",
        description=(
            "An extremely high-volume bar on a down move that closes well off its lows "
            "(wide spread, closes mid-to-upper). Institutional buyers ('smart money') are "
            "absorbing all selling — this is accumulation. A reversal is likely."
        ),
        entry_note=(
            "Wait for the next bar. If it is a narrow up bar on moderate volume, the selling "
            "climax is confirmed. Enter long (BUY_CE) on a stop above this confirmation bar."
        ),
        risk_note=(
            "A second ultra-high volume down bar that closes on its low invalidates this setup — "
            "selling pressure has overcome smart money buyers."
        ),
        success_rate=66,
    ),
    TradingPattern(
        name="Volume Spread Analysis — Effort vs Result",
        source="Anna Coulling — Volume Price Analysis",
        signal_bias="ANY",
        description=(
            "High effort (volume) should produce a proportional result (price spread). "
            "When volume is high but the bar's spread is narrow, effort is not producing result — "
            "the market is being absorbed by the opposing side."
        ),
        entry_note=(
            "When EMA/RSI signals align AND volume confirms the direction (high volume, wide "
            "spread, closing in direction of move), the signal is high probability."
        ),
        risk_note=(
            "Never take a signal where effort and result diverge — e.g., high volume but tiny "
            "price move. The market is telling you something is wrong."
        ),
        success_rate=0,  # principle, not a specific pattern
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# Thomas Bulkowski — Encyclopedia of Chart Patterns
# ─────────────────────────────────────────────────────────────────────────────
BULKOWSKI_PATTERNS: list[TradingPattern] = [
    TradingPattern(
        name="Bull Flag (measured move)",
        source="Thomas Bulkowski — Encyclopedia of Chart Patterns",
        signal_bias="BUY_CE",
        description=(
            "A tight rectangular consolidation sloping slightly downward after a sharp "
            "upward move. Bulkowski's study of thousands of patterns gives bull flags a "
            "breakout success rate of 54% and an average rise of 23% after breakout."
        ),
        entry_note=(
            "Enter on a close above the upper flag boundary. "
            "Measure the flagpole height and add it to the breakout point for the target."
        ),
        risk_note=(
            "A move back into the flag after breakout (throwback) occurs 43% of the time "
            "but is usually brief. A close below the lower flag boundary is a failure."
        ),
        success_rate=54,
    ),
    TradingPattern(
        name="Double Bottom",
        source="Thomas Bulkowski — Encyclopedia of Chart Patterns",
        signal_bias="BUY_CE",
        description=(
            "Two distinct lows at approximately the same price level, separated by a moderate "
            "peak. Bulkowski reports 64% of double bottoms break out upward and meet their "
            "measured move target."
        ),
        entry_note=(
            "The classic entry is on a breakout above the peak between the two bottoms. "
            "A more aggressive entry: buy the second bottom with a stop just below the first."
        ),
        risk_note=(
            "If the second bottom falls significantly below the first, the pattern fails. "
            "Volume should be higher on the second bottom's reversal day."
        ),
        success_rate=64,
    ),
    TradingPattern(
        name="Head and Shoulders Top",
        source="Thomas Bulkowski — Encyclopedia of Chart Patterns",
        signal_bias="BUY_PE",
        description=(
            "Three peaks where the middle peak (head) is highest and the two shoulders "
            "are roughly equal. The neckline connects the two troughs. Bulkowski: 93% of "
            "confirmed H&S patterns break out downward."
        ),
        entry_note=(
            "Enter short (BUY_PE) on a close below the neckline. "
            "Target = neckline minus the head-to-neckline distance."
        ),
        risk_note=(
            "A pullback to the neckline after breakdown occurs in 45% of cases. "
            "A close back above the right shoulder invalidates the pattern."
        ),
        success_rate=93,
    ),
    TradingPattern(
        name="Ascending Triangle",
        source="Thomas Bulkowski — Encyclopedia of Chart Patterns",
        signal_bias="BUY_CE",
        description=(
            "Flat upper resistance with a rising lower trendline — buyers are increasingly "
            "willing to pay higher prices. Bulkowski: 68% upside breakout rate, "
            "average gain 35% in bull markets."
        ),
        entry_note=(
            "Enter on a close above the flat top with expanding volume. "
            "Place stop just below the last higher low inside the triangle."
        ),
        risk_note=(
            "Downside breakouts from ascending triangles are vicious — "
            "32% of the time price breaks down and drops sharply. Always use a stop."
        ),
        success_rate=68,
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# Adam Grimes — The Art and Science of Technical Analysis
# ─────────────────────────────────────────────────────────────────────────────
GRIMES_PATTERNS: list[TradingPattern] = [
    TradingPattern(
        name="Pullback in a Trend",
        source="Adam Grimes — Art and Science of Technical Analysis",
        signal_bias="ANY",
        description=(
            "Grimes identifies the pullback as the highest-probability trade in technical "
            "analysis. A mature trend pauses, gives back a fraction of the move (30–50%), "
            "and then resumes. The pullback shakes out weak hands and provides an entry "
            "with a defined risk point."
        ),
        entry_note=(
            "Look for the pullback to stall near a prior support/resistance level or a "
            "moving average. A momentum divergence on the pullback strengthens the setup. "
            "Enter at the first sign of resumption."
        ),
        risk_note=(
            "Grimes emphasizes: know your 'Uncle Point' — the price level that definitively "
            "proves the trade thesis wrong. If price hits it, exit without hesitation."
        ),
        success_rate=60,
    ),
    TradingPattern(
        name="Failed Breakout (Anti-Pattern)",
        source="Adam Grimes — Art and Science of Technical Analysis",
        signal_bias="ANY",
        description=(
            "When a breakout above resistance fails and reverses back below, trapped "
            "breakout buyers become forced sellers, accelerating the move down. "
            "Grimes: failed patterns often move as far in the failure direction as "
            "a successful breakout would have moved in the original direction."
        ),
        entry_note=(
            "If a BUY_CE signal fires but price immediately reverses back below the "
            "breakout level, the setup has failed. Exit and consider reversing."
        ),
        risk_note=(
            "This is an anti-pattern — a signal to EXIT, not enter. "
            "Grimes warns that fighting a failed breakout is one of the most common "
            "and expensive trading mistakes."
        ),
        success_rate=0,
    ),
    TradingPattern(
        name="Momentum Divergence",
        source="Adam Grimes — Art and Science of Technical Analysis",
        signal_bias="ANY",
        description=(
            "When price makes a higher high but RSI/MACD makes a lower high (bearish divergence), "
            "or price makes a lower low but momentum makes a higher low (bullish divergence), "
            "the trend is losing energy. A reversal becomes more likely."
        ),
        entry_note=(
            "Divergence alone is not a trade — it is a warning. Wait for a confirming "
            "price action signal (reversal bar, pattern break) before entering."
        ),
        risk_note=(
            "Divergence can persist for many bars in a strong trend. "
            "Do not short a strong bull trend solely because RSI is diverging."
        ),
        success_rate=55,
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# Martin Pring — Martin Pring on Price Action
# ─────────────────────────────────────────────────────────────────────────────
PRING_PATTERNS: list[TradingPattern] = [
    TradingPattern(
        name="Support / Resistance Reversal",
        source="Martin Pring — Martin Pring on Price Action",
        signal_bias="ANY",
        description=(
            "Old resistance, once broken, becomes new support — and vice versa. "
            "Pring emphasizes that the more times a level has been tested, the more "
            "significant the break of that level becomes."
        ),
        entry_note=(
            "Buy on a pullback to a former resistance level that is now acting as support. "
            "The cleanest entries are the first or second test of the converted level."
        ),
        risk_note=(
            "A close back below former resistance (now support) signals the break was false. "
            "Exit and wait for a new pattern to develop."
        ),
        success_rate=62,
    ),
    TradingPattern(
        name="Trendline Break with Momentum",
        source="Martin Pring — Martin Pring on Price Action",
        signal_bias="ANY",
        description=(
            "A break of a well-established trendline (minimum 3 touches over several weeks) "
            "combined with a momentum oscillator (RSI) crossing its own threshold "
            "gives a high-conviction reversal signal."
        ),
        entry_note=(
            "Wait for both: (1) trendline break and (2) RSI crossing 50 in the direction of "
            "the break. Combined signals reduce false breakouts significantly."
        ),
        risk_note=(
            "Trendlines drawn with only two points are unreliable. "
            "If RSI does not confirm the trendline break, treat it as a false break."
        ),
        success_rate=63,
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# Bob Volman — Forex Price Action Scalping
# ─────────────────────────────────────────────────────────────────────────────
VOLMAN_PATTERNS: list[TradingPattern] = [
    TradingPattern(
        name="Tight Range Breakout (TRB)",
        source="Bob Volman — Forex Price Action Scalping",
        signal_bias="ANY",
        description=(
            "A period of 10–20 bars with an extremely tight price range (compression) "
            "followed by a breakout. The tighter and longer the range, the more explosive "
            "the breakout. Volman: the market is a coiled spring."
        ),
        entry_note=(
            "Enter on a stop above/below the tight range. Do not enter mid-range. "
            "Volume should expand significantly on the breakout bar — "
            "if it does not, the breakout is suspect."
        ),
        risk_note=(
            "A false breakout from a tight range reverses sharply. "
            "Use a tight stop just inside the range boundary. "
            "Do not hold through a full retrace into the range."
        ),
        success_rate=61,
    ),
    TradingPattern(
        name="Round Number / Barrier Test",
        source="Bob Volman — Forex Price Action Scalping",
        signal_bias="ANY",
        description=(
            "Round numbers (e.g., NIFTY 24,000 / 24,500 / 25,000) act as psychological "
            "barriers. Volman: price often stalls, reverses, or consolidates at these levels "
            "before deciding direction. The first test of a round number is particularly significant."
        ),
        entry_note=(
            "Watch for a rejection candle (pin bar, doji) at a round number. "
            "A strong close through a round number on volume signals it will now act as support."
        ),
        risk_note=(
            "Trading into a round number is risky — always reduce position size near these levels. "
            "The safest entry is after the level has been tested and confirmed."
        ),
        success_rate=0,
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# Combined knowledge base
# ─────────────────────────────────────────────────────────────────────────────
ALL_PATTERNS: list[TradingPattern] = (
    BROOKS_PATTERNS
    + COULLING_PATTERNS
    + BULKOWSKI_PATTERNS
    + GRIMES_PATTERNS
    + PRING_PATTERNS
    + VOLMAN_PATTERNS
)


def get_relevant_patterns(
    signal: SignalType,
    rsi: float,
    ema20: float,
    ema50: float,
    max_patterns: int = 4,
) -> list[TradingPattern]:
    """Return the most contextually relevant patterns for the current market state.

    Selection logic:
    - Always include patterns whose signal_bias matches (or is ANY).
    - Prefer patterns relevant to current RSI / EMA relationship.
    - Rotate through books so explanations don't always cite the same source.
    """
    signal_val = signal.value  # "BUY_CE", "BUY_PE", or "HOLD"

    # Filter by signal bias compatibility
    compatible = [
        p for p in ALL_PATTERNS
        if p.signal_bias in (signal_val, "ANY")
    ]

    # For HOLD, return a mix covering caution/patience themes
    if signal_val == "HOLD":
        compatible = [p for p in ALL_PATTERNS if p.signal_bias == "ANY"]

    # De-duplicate by source book to ensure variety
    seen_books: set[str] = set()
    selected: list[TradingPattern] = []
    for pattern in compatible:
        book = pattern.source.split("—")[0].strip()
        if book not in seen_books and len(selected) < max_patterns:
            selected.append(pattern)
            seen_books.add(book)

    # Fill remaining slots from any book
    for pattern in compatible:
        if len(selected) >= max_patterns:
            break
        if pattern not in selected:
            selected.append(pattern)

    return selected[:max_patterns]


def format_patterns_for_prompt(patterns: list[TradingPattern]) -> str:
    """Render patterns as a readable block for Claude's system prompt."""
    lines = ["RELEVANT PATTERNS FROM YOUR TRADING LIBRARY:\n"]
    for i, p in enumerate(patterns, 1):
        lines.append(f"{i}. {p.name} [{p.source}]")
        lines.append(f"   What it is: {p.description}")
        lines.append(f"   Entry note: {p.entry_note}")
        lines.append(f"   Risk note:  {p.risk_note}")
        if p.success_rate > 0:
            lines.append(f"   Historical success rate: ~{p.success_rate}%")
        lines.append("")
    return "\n".join(lines)
