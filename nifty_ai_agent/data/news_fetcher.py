"""News fetcher — Indian and global market headlines via RSS feeds."""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import feedparser
import requests

logger = logging.getLogger(__name__)

_TIMEOUT = 10
_MAX_ITEMS = 5  # headlines per feed

# ── RSS feed sources ───────────────────────────────────────────────────────────
_FEEDS = {
    "Economic Times Markets": "https://economictimes.indiatimes.com/markets/rss.cms",
    "Moneycontrol Markets":   "https://www.moneycontrol.com/rss/marketreports.xml",
    "Reuters Business":       "https://feeds.reuters.com/reuters/businessNews",
    "Livemint Markets":       "https://www.livemint.com/rss/markets",
}


@dataclass
class NewsItem:
    title: str
    source: str
    published: str
    summary: str = ""


def fetch_news(max_items_per_feed: int = _MAX_ITEMS) -> list[NewsItem]:
    """Fetch top headlines from Indian and global financial RSS feeds.

    Returns a combined list, newest first per feed.
    Silently skips any feed that is unreachable.
    """
    results: list[NewsItem] = []

    for source, url in _FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_items_per_feed]:
                title = entry.get("title", "").strip()
                summary = entry.get("summary", "").strip()
                # strip HTML tags from summary
                summary = _strip_html(summary)[:200]
                published = entry.get("published", "")
                if title:
                    results.append(
                        NewsItem(title=title, source=source, published=published, summary=summary)
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


def _strip_html(text: str) -> str:
    """Minimal HTML tag stripper — avoids adding BeautifulSoup dependency."""
    import re
    return re.sub(r"<[^>]+>", "", text).strip()
