"""News fetcher — Indian and global market headlines via RSS feeds."""

import html
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import feedparser
import requests

logger = logging.getLogger(__name__)

_TIMEOUT = 10
_MAX_ITEMS = 5  # headlines per feed

# ── RSS feed sources ── (name → (url, region)) ──────────────────────────────────
# region tags the headline as Indian ("IN") or global ("WORLD") market news so the
# market analysis can score sentiment for each separately. Reuters' RSS was retired
# (feeds.reuters.com now returns nothing), so CNBC + Investing.com carry world news.
_FEEDS: dict[str, tuple[str, str]] = {
    "Economic Times Markets": ("https://economictimes.indiatimes.com/markets/rss.cms", "IN"),
    "Moneycontrol Markets":   ("https://www.moneycontrol.com/rss/marketreports.xml", "IN"),
    "Livemint Markets":       ("https://www.livemint.com/rss/markets", "IN"),
    "CNBC Markets":           ("https://www.cnbc.com/id/10000664/device/rss/rss.html", "WORLD"),
    "Investing.com":          ("https://www.investing.com/rss/news_25.rss", "WORLD"),
}


@dataclass
class NewsItem:
    title: str
    source: str
    published: str
    summary: str = ""
    region: str = "IN"          # "IN" (Indian market news) or "WORLD" (global)


def fetch_news(max_items_per_feed: int = _MAX_ITEMS) -> list[NewsItem]:
    """Fetch top headlines from Indian and global financial RSS feeds.

    Returns a combined list, newest first per feed.
    Silently skips any feed that is unreachable.
    """
    results: list[NewsItem] = []

    for source, (url, region) in _FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_items_per_feed]:
                title = _strip_html(entry.get("title", ""))
                # strip HTML tags + decode entities from summary
                summary = _strip_html(entry.get("summary", ""))[:200]
                published = entry.get("published", "")
                if title:
                    results.append(
                        NewsItem(
                            title=title, source=source, published=published,
                            summary=summary, region=region,
                        )
                    )
            logger.debug("Fetched %d items from %s", len(feed.entries[:max_items_per_feed]), source)
        except Exception as exc:
            logger.warning("News feed '%s' failed: %s", source, exc)

    return results


def format_news_for_prompt(items: list[NewsItem], limit: int = 8) -> str:
    """Format news headlines for Claude's context prompt."""
    if not items:
        return "No news available."
    lines = ["LATEST MARKET HEADLINES:"]
    for item in items[:limit]:
        lines.append(f"• [{item.source}] {item.title}")
    return "\n".join(lines)


def format_news_for_notification(items: list[NewsItem], limit: int = 5) -> str:
    """Format headlines as a compact push notification body."""
    if not items:
        return "No news available."
    lines = []
    for item in items[:limit]:
        lines.append(f"• {item.title}")
    return "\n".join(lines)


_TAG_RE = re.compile(r"<[^>]+>")
# Numeric HTML entities — with OR without the leading '&'. Some feeds (Moneycontrol)
# emit them with the ampersand dropped, e.g. "day#39;s", which html.unescape can't
# repair on its own because there is no '&' to anchor.
_NUM_ENTITY_RE = re.compile(r"&?#(\d+);")
_HEX_ENTITY_RE = re.compile(r"&?#[xX]([0-9a-fA-F]+);")


def _num_to_char(match: re.Match, base: int) -> str:
    try:
        return chr(int(match.group(1), base))
    except (ValueError, OverflowError):
        return match.group(0)  # leave an out-of-range code point untouched


def _strip_html(text: str) -> str:
    """Strip HTML tags and decode HTML entities.

    Handles the malformed numeric entities some RSS feeds emit without the leading
    '&' (Moneycontrol's "day#39;s" → "day's"), then html.unescape() covers the
    well-formed named/numeric ones (&amp;, &quot;, &#39;, …).
    """
    text = _TAG_RE.sub("", text)
    text = _NUM_ENTITY_RE.sub(lambda m: _num_to_char(m, 10), text)
    text = _HEX_ENTITY_RE.sub(lambda m: _num_to_char(m, 16), text)
    return html.unescape(text).strip()
