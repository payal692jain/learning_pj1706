"""Per-stock fundamentals, event dates and headlines.

The strategy stack is purely technical: it reads price and volume and knows
nothing about why they moved. That is fine most days and badly wrong on a few —
a stock two days from quarterly results, or in the middle of a merger, is not the
same trade its 5-minute chart says it is. This module supplies that missing
context, and the scanner uses it to refuse entries into a known event.

Source is yfinance: Upstox serves prices, not fundamentals. A `.info` call costs
~0.7s, so a 50-name universe is ~35s — far too slow to repeat every 30-minute
scan. Everything here is therefore cached to disk for the day, which matches how
often the underlying data actually changes.
"""

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
_CACHE_FILE = _CACHE_DIR / "stock_fundamentals.json"

# Corporate-action language worth surfacing on an options alert. These move a
# stock in ways no indicator anticipates, so the headline itself is the signal.
_EVENT_PATTERNS = {
    "MERGER": r"\b(merger|merges?|amalgamation|demerger)\b",
    "M&A": r"\b(acquisition|acquires?|acquiring|takeover|buyout)\b",
    # "stake" appears both ways round in practice — "stake sale" and "to sell
    # stake" — and often with a percentage between the verb and the noun. Matching
    # bare "stake" would drag in "raises the stakes", so the verb is required.
    "STAKE": (
        r"\bstake\s+(sale|buy|purchase)\b"
        r"|\b(sell|sells|selling|sold|buy|buys|bought|acquire[sd]?|offload\w*)\s+"
        r"(a\s+)?(\d+(\.\d+)?%\s+)?stake\b"
        r"|\b(block deal|open offer|divest\w*)\b"
    ),
    "IPO": r"\b(ipo|listing|delisting)\b",
    "RESULTS": r"\b(q[1-4]|quarterly|earnings call|results)\b",
    "RATING": r"\b(upgrade[sd]?|downgrade[sd]?|rating)\b",
}

# NSE cash session, used to pace part-day volume against a full-day average.
_SESSION_OPEN = time(9, 15)
_SESSION_CLOSE = time(15, 30)
_SESSION_MINUTES = 375


@dataclass
class Headline:
    title: str
    published: str      # ISO date, "" when unknown
    tags: list[str] = field(default_factory=list)


@dataclass
class StockFundamentals:
    """What is knowable about a stock beyond its chart."""
    symbol: str
    pe: float = 0.0
    forward_pe: float = 0.0
    price_to_book: float = 0.0
    peg: float = 0.0
    eps: float = 0.0
    sector: str = ""
    market_cap: float = 0.0
    week52_high: float = 0.0
    week52_low: float = 0.0
    avg_volume: float = 0.0
    next_earnings: str = ""     # ISO date, "" when unknown
    headlines: list[Headline] = field(default_factory=list)

    @property
    def days_to_earnings(self) -> int | None:
        """Calendar days until the next results date, or None when unknown.

        Negative values are treated as unknown: yfinance sometimes carries a stale
        date from the previous cycle, and a past date must never read as "results
        are imminent" nor as "safely far away".
        """
        if not self.next_earnings:
            return None
        try:
            days = (date.fromisoformat(self.next_earnings) - date.today()).days
        except ValueError:
            return None
        return days if days >= 0 else None

    def position_in_range(self, spot: float) -> float | None:
        """Where *spot* sits in the 52-week range: 0.0 at the low, 1.0 at the high.

        Clamped, because a stock making a new high prints a spot outside the range
        yfinance last published, and an unclamped 1.04 would render as "104%".
        """
        span = self.week52_high - self.week52_low
        if span <= 0 or spot <= 0:
            return None
        return max(0.0, min(1.0, (spot - self.week52_low) / span))

    @property
    def event_tags(self) -> list[str]:
        """Distinct corporate-action tags across this stock's recent headlines."""
        seen: list[str] = []
        for h in self.headlines:
            for tag in h.tags:
                if tag not in seen:
                    seen.append(tag)
        return seen


def volume_pace_ratio(
    intraday_volume: float, avg_daily_volume: float, now: time
) -> float | None:
    """Today's volume projected to a full session, as a multiple of the average.

    Comparing part-day volume straight to a daily average always reads low and
    would call every morning quiet, so the elapsed fraction of the session is
    divided out first. Returns None before the open or without an average to
    compare against.
    """
    if avg_daily_volume <= 0 or intraday_volume <= 0:
        return None
    if now <= _SESSION_OPEN:
        return None
    elapsed = min(
        _SESSION_MINUTES,
        (datetime.combine(date.today(), min(now, _SESSION_CLOSE))
         - datetime.combine(date.today(), _SESSION_OPEN)).total_seconds() / 60,
    )
    if elapsed < 5:   # the first bars are too thin to extrapolate from
        return None
    projected = intraday_volume * (_SESSION_MINUTES / elapsed)
    return round(projected / avg_daily_volume, 2)


def _tag_headline(title: str) -> list[str]:
    lowered = title.lower()
    return [tag for tag, pat in _EVENT_PATTERNS.items() if re.search(pat, lowered)]


def _next_earnings_date(ticker) -> str:
    """Next results date as ISO, or "" when unknown.

    Prefers Ticker.calendar, which carries the forward-looking date. The `.info`
    earningsTimestamp fields are inconsistent — sometimes the *last* reported
    quarter, sometimes a date a year stale — so they are only a fallback, and
    anything already in the past is discarded rather than reported.
    """
    today = date.today()
    try:
        cal = ticker.calendar or {}
        dates = cal.get("Earnings Date") or []
        upcoming = sorted(d for d in dates if isinstance(d, date) and d >= today)
        if upcoming:
            return upcoming[0].isoformat()
    except Exception as exc:
        logger.debug("calendar lookup failed: %s", exc)

    try:
        ts = (ticker.info or {}).get("earningsTimestampStart")
        if ts:
            d = datetime.fromtimestamp(int(ts), tz=timezone.utc).date()
            if d >= today:
                return d.isoformat()
    except Exception as exc:
        logger.debug("earningsTimestampStart lookup failed: %s", exc)
    return ""


def _fetch_one(symbol: str, news_limit: int) -> StockFundamentals:
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    info = ticker.info or {}
    bare = symbol[:-3] if symbol.endswith(".NS") else symbol

    headlines: list[Headline] = []
    try:
        for article in (ticker.news or [])[:news_limit]:
            content = article.get("content", article)
            title = (content.get("title") or "").strip()
            if not title:
                continue
            published = str(content.get("pubDate") or "")[:10]
            headlines.append(Headline(title=title, published=published,
                                      tags=_tag_headline(title)))
    except Exception as exc:
        logger.debug("news lookup failed for %s: %s", symbol, exc)

    def num(key: str) -> float:
        value = info.get(key)
        return float(value) if isinstance(value, (int, float)) else 0.0

    return StockFundamentals(
        symbol=bare,
        pe=num("trailingPE"),
        forward_pe=num("forwardPE"),
        price_to_book=num("priceToBook"),
        peg=num("pegRatio"),
        eps=num("trailingEps"),
        sector=str(info.get("sector") or ""),
        market_cap=num("marketCap"),
        week52_high=num("fiftyTwoWeekHigh"),
        week52_low=num("fiftyTwoWeekLow"),
        avg_volume=num("averageVolume"),
        next_earnings=_next_earnings_date(ticker),
        headlines=headlines,
    )


def fetch_fundamentals(
    symbols: list[str], news_limit: int = 4, use_cache: bool = True
) -> dict[str, StockFundamentals]:
    """Return {bare symbol: StockFundamentals} for *symbols* (yfinance tickers).

    Cached to disk for the calendar day. A symbol yfinance cannot serve is simply
    absent from the result — the scan is more useful over the names that resolved
    than failed wholesale for one bad ticker.
    """
    cached = _read_cache() if use_cache else {}
    missing = [s for s in symbols if (s[:-3] if s.endswith(".NS") else s) not in cached]

    if missing:
        logger.info("Fundamentals: fetching %d symbol(s) (%d cached)",
                    len(missing), len(cached))
        for symbol in missing:
            try:
                data = _fetch_one(symbol, news_limit)
                cached[data.symbol] = data
            except Exception as exc:
                logger.warning("Fundamentals: %s failed — %s", symbol, exc)
        if use_cache:
            _write_cache(cached)

    wanted = {(s[:-3] if s.endswith(".NS") else s) for s in symbols}
    return {k: v for k, v in cached.items() if k in wanted}


def _read_cache() -> dict[str, StockFundamentals]:
    """Today's cached fundamentals, or {} when absent or stale."""
    if not _CACHE_FILE.exists():
        return {}
    modified = datetime.fromtimestamp(_CACHE_FILE.stat().st_mtime, tz=timezone.utc)
    if modified.date() != datetime.now(tz=timezone.utc).date():
        return {}
    try:
        raw = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        return {
            k: StockFundamentals(
                **{**v, "headlines": [Headline(**h) for h in v.get("headlines", [])]}
            )
            for k, v in raw.items()
        }
    except Exception as exc:
        logger.warning("Fundamentals cache unreadable (%s) — refetching", exc)
        return {}


def _write_cache(data: dict[str, StockFundamentals]) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(
            json.dumps({k: asdict(v) for k, v in data.items()}), encoding="utf-8",
        )
    except Exception as exc:
        # A cache we cannot write costs speed, not correctness.
        logger.warning("Could not cache fundamentals: %s", exc)
