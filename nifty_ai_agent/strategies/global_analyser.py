"""Global context analyser — turns global indices, GIFT Nifty, India VIX, and news
headlines into an intraday confidence adjustment.

This exists because the pre-market picture was previously computed at 08:00, shown
once in the morning report, and then discarded: run_pipeline() never looked at it.
An intraday signal that ignores a -1.5% S&P session and a spiking VIX is reading
half the market.

Nothing here generates or vetoes a signal — it only nudges the confidence of a
signal the strategy engine already produced, exactly like the RSI, option-chain,
and breadth adjusters alongside it.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from nifty_ai_agent.data.market_context import MarketContext, fetch_market_context
from nifty_ai_agent.data.news_fetcher import NewsItem, fetch_news

logger = logging.getLogger(__name__)

# Global cues move on a far slower clock than a 5-minute bar — refetching every
# cycle would burn API calls to re-learn that the S&P closed down 1.2% last night.
_CACHE_TTL_MINUTES = 30

# Headline keyword scoring. Deliberately crude and transparent: it is a tiebreaker
# on confidence, not a trading decision, and a black-box sentiment model would be
# harder to audit when a signal looks wrong after the fact.
#
# Matched against whole TOKENS, never substrings. Substring matching quietly breaks
# the scorer in both directions: "rallies" does not contain "rally" (so a bullish
# headline scores zero), while "war" hides inside "toward" and "miss" inside
# "dismissed" (so neutral headlines score bearish). Inflections are therefore listed
# explicitly rather than being approximated by stems.
_BULLISH_WORDS = {
    "rally", "rallies", "rallied", "surge", "surges", "surged", "gain", "gains",
    "jump", "jumps", "jumped", "soar", "soars", "soared", "climb", "climbs", "climbed",
    "rise", "rises", "rose", "higher", "bull", "bullish", "upbeat", "recovery",
    "rebound", "rebounds", "boost", "boosts", "optimism", "inflow", "inflows",
    "beat", "beats", "upgrade", "upgrades", "outperform", "record",
}
_BEARISH_WORDS = {
    "fall", "falls", "fell", "drop", "drops", "dropped", "plunge", "plunges", "plunged",
    "slump", "slumps", "slumped", "crash", "crashes", "crashed", "sink", "sinks",
    "tumble", "tumbles", "tumbled", "lower", "bear", "bearish", "selloff", "slide",
    "slides", "fear", "fears", "worry", "worries", "outflow", "outflows", "miss",
    "misses", "downgrade", "downgrades", "recession", "slowdown", "tariff", "tariffs",
    "conflict", "underperform",
}

_TOKEN_RE = re.compile(r"[a-z]+")

# India VIX above this is a market pricing in real fear: option premiums are rich,
# which is precisely when a naked option BUYER is paying the most for the same move.
_VIX_ELEVATED = 18.0
_VIX_HIGH = 22.0


@dataclass
class NewsSentiment:
    score: float                 # -1.0 (bearish) … +1.0 (bullish)
    bullish_hits: int
    bearish_hits: int
    headlines: int
    top_headline: str = ""

    @property
    def label(self) -> str:
        if self.score > 0.2:
            return "BULLISH"
        if self.score < -0.2:
            return "BEARISH"
        return "NEUTRAL"


@dataclass
class GlobalSnapshot:
    """Everything outside the price chart that should colour an intraday signal."""
    global_bias: str = "NEUTRAL"      # from the major global indices
    gift_nifty_pct: float = 0.0       # GIFT Nifty % — the closest thing to a pre-open print
    vix: float = 0.0                  # India VIX level
    vix_change_pct: float = 0.0
    news: NewsSentiment = field(
        default_factory=lambda: NewsSentiment(0.0, 0, 0, 0)
    )
    is_available: bool = False        # False when every upstream fetch failed

    @property
    def vix_regime(self) -> str:
        if self.vix >= _VIX_HIGH:
            return "HIGH"
        if self.vix >= _VIX_ELEVATED:
            return "ELEVATED"
        return "CALM"


def analyse_news(items: list[NewsItem]) -> NewsSentiment:
    """Score headlines by bullish/bearish keyword balance."""
    if not items:
        return NewsSentiment(0.0, 0, 0, 0)

    bullish = bearish = 0
    strongest = ""
    best_margin = 0

    for item in items:
        # Hyphens collapse so "sell-off" tokenises as the single word "selloff".
        text = f"{item.title} {item.summary}".lower().replace("-", "")
        tokens = set(_TOKEN_RE.findall(text))
        up = len(tokens & _BULLISH_WORDS)
        down = len(tokens & _BEARISH_WORDS)
        bullish += up
        bearish += down
        if abs(up - down) > best_margin:
            best_margin = abs(up - down)
            strongest = item.title

    total = bullish + bearish
    score = round((bullish - bearish) / total, 2) if total else 0.0
    sentiment = NewsSentiment(
        score=score,
        bullish_hits=bullish,
        bearish_hits=bearish,
        headlines=len(items),
        top_headline=strongest,
    )
    logger.info(
        "News sentiment: %s (score=%.2f, %d↑ / %d↓ across %d headlines)",
        sentiment.label, score, bullish, bearish, len(items),
    )
    return sentiment


def build_snapshot(context: MarketContext, items: list[NewsItem]) -> GlobalSnapshot:
    """Fold a MarketContext and a headline list into one GlobalSnapshot."""
    vix = next((i for i in context.indices if i.name == "India VIX"), None)
    return GlobalSnapshot(
        global_bias=context.global_bias,
        gift_nifty_pct=context.gift_nifty.change_pct if context.gift_nifty else 0.0,
        vix=vix.price if vix else 0.0,
        vix_change_pct=vix.change_pct if vix else 0.0,
        news=analyse_news(items),
        is_available=bool(context.indices or items),
    )


# ── Cached fetch ───────────────────────────────────────────────────────────────

_cache: dict[str, object] = {"snapshot": None, "fetched_at": None}


def fetch_global_snapshot(force: bool = False) -> GlobalSnapshot:
    """Return the global snapshot, refetching at most every 30 minutes.

    Never raises: a failed fetch yields an unavailable snapshot, which the
    adjuster treats as "no opinion" rather than as a bearish signal.
    """
    now = datetime.now()
    fetched_at = _cache.get("fetched_at")
    cached = _cache.get("snapshot")

    if not force and cached is not None and fetched_at is not None:
        age = now - fetched_at
        if age < timedelta(minutes=_CACHE_TTL_MINUTES):
            logger.debug("Global context cache hit (%.0f min old)", age.total_seconds() / 60)
            return cached

    try:
        snapshot = build_snapshot(fetch_market_context(), fetch_news())
    except Exception as exc:
        logger.warning("Global context fetch failed — signals run without it: %s", exc)
        return GlobalSnapshot()

    _cache["snapshot"] = snapshot
    _cache["fetched_at"] = now
    logger.info(
        "Global context: bias=%s GIFT=%+.2f%% VIX=%.1f (%s) news=%s",
        snapshot.global_bias, snapshot.gift_nifty_pct, snapshot.vix,
        snapshot.vix_regime, snapshot.news.label,
    )
    return snapshot


def reset_cache() -> None:
    """Drop the cached snapshot — used by tests and by the daily pre-market refresh."""
    _cache["snapshot"] = None
    _cache["fetched_at"] = None


# ── Confidence adjustment ──────────────────────────────────────────────────────

def global_confidence_adjustment(
    snapshot: GlobalSnapshot, signal_value: str,
) -> tuple[int, str]:
    """Return (confidence_delta, explanation) for *signal_value* given global cues.

    Agreement between the global tape and the signal adds confidence; disagreement
    subtracts more than agreement adds, because a domestic breakout fighting a
    risk-off global tape fails more often than it succeeds.

    A high India VIX penalises BOTH directions: it makes option premiums expensive,
    and this system buys options rather than selling them, so the buyer overpays for
    the same index move regardless of which way it goes.
    """
    if signal_value == "HOLD" or not snapshot.is_available:
        return 0, ""

    bullish_signal = signal_value == "BUY_CE"
    delta = 0
    notes: list[str] = []

    # ── Global index bias ──
    if snapshot.global_bias != "NEUTRAL":
        agrees = (snapshot.global_bias == "BULLISH") == bullish_signal
        delta += 6 if agrees else -10
        verb = "confirms" if agrees else "opposes"
        notes.append(f"Global tape ({snapshot.global_bias}) {verb} the signal.")

    # ── GIFT Nifty ──
    if abs(snapshot.gift_nifty_pct) >= 0.3:
        agrees = (snapshot.gift_nifty_pct > 0) == bullish_signal
        delta += 4 if agrees else -6
        notes.append(f"GIFT Nifty {snapshot.gift_nifty_pct:+.2f}%.")

    # ── News sentiment ──
    if snapshot.news.label != "NEUTRAL":
        agrees = (snapshot.news.label == "BULLISH") == bullish_signal
        delta += 3 if agrees else -5
        verb = "supports" if agrees else "contradicts"
        notes.append(f"Headlines skew {snapshot.news.label.lower()} — {verb} the trade.")

    # ── India VIX — direction-agnostic, hits option buyers either way ──
    regime = snapshot.vix_regime
    if regime == "HIGH":
        delta -= 8
        notes.append(
            f"India VIX {snapshot.vix:.1f} is HIGH — premiums are rich and you are buying."
        )
    elif regime == "ELEVATED":
        delta -= 4
        notes.append(f"India VIX {snapshot.vix:.1f} is elevated — options are pricey.")

    if not notes:
        return 0, ""

    detail = " 🌍 " + " ".join(notes)
    logger.debug("Global adjustment for %s: %+d — %s", signal_value, delta, detail)
    return delta, detail
