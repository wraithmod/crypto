"""Crypto news ingestion and LLM-based sentiment analysis.

Fetches RSS feeds from tiered sources, scores the top 8 headlines
concurrently via the configured LLM provider, and exposes both global
and per-symbol weighted sentiment for the trading engine.

Weighting model (GEMINI + CODEX design, Feb 2026):
  - Source credibility tier  (1.0 / 0.85 / 0.65 / 0.6 / 0.4)
  - Impact magnitude          (high=2.0 / medium=1.5 / low=1.0)
  - Time decay                (age<1h=1.0 / 1-6h=0.5 / >6h=0.1)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time as _time_module
from dataclasses import dataclass, field
from datetime import timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

import feedparser

from config import config
from src.agents.base import LLMProvider, LLMError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source credibility weights  (keyed on registered domain, no www.)
# Based on GEMINI research — see GEMINI.md
# ---------------------------------------------------------------------------
_SOURCE_WEIGHTS: dict[str, float] = {
    # Tier-1 crypto
    "coindesk.com":        1.0,
    "cointelegraph.com":   1.0,
    "decrypt.co":          1.0,
    "bitcoinmagazine.com": 1.0,
    # Tier-1 macro / newswires
    "reuters.com":         1.0,
    "bloomberg.com":       1.0,
    "finance.yahoo.com":   1.0,
    "wsj.com":             1.0,
    "ft.com":              1.0,
    # Tier-2 DeFi / Web3
    "thedefiant.io":       0.85,
    "blockworks.co":       0.85,
    "theblock.co":         0.85,
    # Tier-2 crypto
    "cryptoslate.com":     0.65,
    "beincrypto.com":      0.65,
    "newsbtc.com":         0.65,
    "ambcrypto.com":       0.65,
    "coingape.com":        0.65,
    "cryptonews.com":      0.65,
    # Tier-2 finance / mainstream
    "cnbc.com":            0.6,
    "marketwatch.com":     0.6,
    "investing.com":       0.6,
}
_DEFAULT_SOURCE_WEIGHT: float = 0.4   # Tier-3 / unknown sources

# Impact magnitude → numeric multiplier
_MAGNITUDE_WEIGHT: dict[str, float] = {
    "high":   2.0,
    "medium": 1.5,
    "low":    1.0,
}

# Known quote currencies for stripping from Binance-style symbols
_QUOTE_CURRENCIES = ("USDT", "USDC", "BUSD", "BTC", "ETH", "BNB")

# Time decay thresholds
_ONE_HOUR:   float = 3_600.0
_SIX_HOURS:  float = 6 * _ONE_HOUR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _source_weight_from_url(url: str) -> float:
    """Return credibility weight for a given article/feed URL."""
    try:
        host = urlparse(url).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return _SOURCE_WEIGHTS.get(host, _DEFAULT_SOURCE_WEIGHT)
    except Exception:
        return _DEFAULT_SOURCE_WEIGHT


def _extract_coin_ticker(symbol: str) -> str:
    """Convert a Binance-style trading pair to a plain ticker.

    Examples: "BTCUSDT" -> "BTC", "SOLUSDT" -> "SOL", "PEPEUSDT" -> "PEPE"
    """
    upper = symbol.upper()
    for quote in _QUOTE_CURRENCIES:
        if upper.endswith(quote) and len(upper) > len(quote):
            return upper[: -len(quote)]
    return upper


def _time_decay_factor(published: str) -> float:
    """Return a recency multiplier based on headline age.

    age < 1 h  -> 1.0  (full weight)
    1 h – 6 h  -> 0.5
    > 6 h      -> 0.1
    Falls back to 1.0 if the date string cannot be parsed.
    """
    if not published:
        return 1.0

    age: float | None = None

    # Try RFC 2822 first (standard feedparser format)
    try:
        dt = parsedate_to_datetime(published)
        age = _time_module.time() - dt.timestamp()
    except Exception:
        pass

    # Fallback: try dateutil (handles ISO 8601 and other variants)
    if age is None:
        try:
            from dateutil import parser as dtparser
            dt = dtparser.parse(published)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age = _time_module.time() - dt.timestamp()
        except Exception:
            return 1.0  # conservative: treat unknown age as fresh

    if age < _ONE_HOUR:
        return 1.0
    if age < _SIX_HOURS:
        return 0.5
    return 0.1


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SentimentResult:
    score: float                          # -1.0 (bearish) to +1.0 (bullish)
    label: str                            # "bearish", "neutral", "bullish"
    summary: str                          # 1-sentence market impact explanation
    affected_coins: list[str] = field(default_factory=list)  # e.g. ["BTC", "ETH"]
    impact_magnitude: str = "low"         # "low", "medium", "high"
    source_weight: float = 1.0            # set after construction from URL


@dataclass
class NewsItem:
    title: str
    url: str
    source: str
    published: str      # raw date string from RSS entry
    feed_url: str = ""  # originating RSS feed URL (used for source weighting)
    sentiment: SentimentResult | None = None


# ---------------------------------------------------------------------------
# NewsFeed
# ---------------------------------------------------------------------------

class NewsFeed:
    """Fetches RSS headlines, scores them via LLM, and exposes weighted sentiment.

    Usage::

        feed = NewsFeed(llm_provider)
        await feed.fetch_and_analyze()
        score = feed.get_symbol_sentiment("BTCUSDT")   # per-coin weighted score
        avg   = feed.get_avg_sentiment()               # global weighted average
    """

    def __init__(self, llm_provider: LLMProvider) -> None:
        self._provider = llm_provider
        self._latest: list[NewsItem] = []

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------

    async def fetch_headlines(self) -> list[NewsItem]:
        """Parse all configured RSS feeds and return deduplicated items."""
        loop = asyncio.get_event_loop()
        fetch_tasks = [
            loop.run_in_executor(None, feedparser.parse, url)
            for url in config.news_feeds
        ]
        results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        seen_titles: set[str] = set()
        items: list[NewsItem] = []

        for feed_url, result in zip(config.news_feeds, results):
            if isinstance(result, Exception):
                logger.warning("Failed to fetch feed %s: %s", feed_url, result)
                continue

            feed_source = getattr(result.feed, "title", feed_url)

            for entry in result.entries:
                title: str = getattr(entry, "title", "").strip()
                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)

                url: str = getattr(entry, "link", "")
                published: str = (
                    getattr(entry, "published", "") or getattr(entry, "updated", "")
                )

                items.append(NewsItem(
                    title=title,
                    url=url,
                    source=feed_source,
                    published=published,
                    feed_url=feed_url,
                ))

        logger.debug("fetch_headlines: collected %d items across %d feeds",
                     len(items), len(config.news_feeds))
        return items[:30]   # cap before scoring; LLM only sees top 8

    # ------------------------------------------------------------------
    # Sentiment analysis
    # ------------------------------------------------------------------

    async def analyze_sentiment(self, headline: str) -> SentimentResult:
        """Score a single headline for market impact via LLM."""
        try:
            raw = await self._provider.complete(
                system=(
                    "You are a crypto market impact analyst. "
                    "Return only valid JSON with no other text."
                ),
                user=(
                    f"Analyze the crypto market impact of this headline: '{headline}'. "
                    "Return a JSON object with exactly these keys: "
                    '"score" (float -1.0 to 1.0, negative=bearish, positive=bullish), '
                    '"label" (must be exactly one of: bearish, neutral, bullish), '
                    '"summary" (one sentence describing market impact), '
                    '"affected_coins" (JSON array of uppercase coin tickers directly '
                    'impacted, e.g. ["BTC","ETH"]; empty array if general market), '
                    '"impact_magnitude" (must be exactly one of: low, medium, high). '
                    'Example: {"score": 0.8, "label": "bullish", '
                    '"summary": "ETF approval drives institutional demand.", '
                    '"affected_coins": ["BTC"], "impact_magnitude": "high"}'
                ),
                max_tokens=300,
            )

            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            data = json.loads(raw)
            score = max(-1.0, min(1.0, float(data.get("score", 0.0))))
            label = str(data.get("label", "neutral")).lower()
            if label not in ("bearish", "neutral", "bullish"):
                label = "neutral"
            summary = str(data.get("summary", "Analysis unavailable."))

            raw_coins = data.get("affected_coins", [])
            affected_coins = (
                [str(c).upper() for c in raw_coins]
                if isinstance(raw_coins, list) else []
            )
            magnitude = str(data.get("impact_magnitude", "low")).lower()
            if magnitude not in ("low", "medium", "high"):
                magnitude = "low"

            return SentimentResult(
                score=score,
                label=label,
                summary=summary,
                affected_coins=affected_coins,
                impact_magnitude=magnitude,
                # source_weight assigned by caller from the item's URL
            )

        except Exception as exc:
            logger.error("Sentiment analysis failed for '%s': %s", headline, exc)
            return SentimentResult(
                score=0.0,
                label="neutral",
                summary="Analysis unavailable",
                affected_coins=[],
                impact_magnitude="low",
            )

    # ------------------------------------------------------------------
    # Fetch + analyze pipeline
    # ------------------------------------------------------------------

    async def fetch_and_analyze(self) -> list[NewsItem]:
        """Fetch headlines then concurrently score the top 8."""
        items = await self.fetch_headlines()
        candidates = items[:8]

        # Score all 8 concurrently; return_exceptions prevents one failure
        # from cancelling the others.
        sentiment_results = await asyncio.gather(
            *[self.analyze_sentiment(item.title) for item in candidates],
            return_exceptions=True,
        )

        for item, result in zip(candidates, sentiment_results):
            if isinstance(result, Exception):
                logger.warning(
                    "Concurrent sentiment failed for '%s': %s", item.title, result
                )
                item.sentiment = SentimentResult(
                    score=0.0, label="neutral", summary="Analysis unavailable"
                )
            else:
                item.sentiment = result

            # Assign source credibility weight from the article URL
            if item.sentiment is not None:
                item.sentiment.source_weight = _source_weight_from_url(
                    item.url or item.feed_url
                )

        self._latest = items
        logger.info(
            "fetch_and_analyze: %d items fetched, %d scored", len(items), len(candidates)
        )
        return items

    # ------------------------------------------------------------------
    # Sentiment accessors
    # ------------------------------------------------------------------

    def get_latest(self) -> list[NewsItem]:
        """Return the cached latest news items."""
        return self._latest

    def get_avg_sentiment(self) -> float:
        """Global weighted average sentiment (source credibility × time decay)."""
        weighted_sum = 0.0
        total_weight = 0.0

        for item in self._latest:
            if item.sentiment is None:
                continue
            decay = _time_decay_factor(item.published)
            weight = item.sentiment.source_weight * decay
            weighted_sum += item.sentiment.score * weight
            total_weight += weight

        if total_weight == 0.0:
            return 0.0
        return weighted_sum / total_weight

    def get_symbol_sentiment(self, symbol: str) -> float | None:
        """Return a weighted sentiment score specific to *symbol*.

        Filters items where the coin ticker appears in ``affected_coins``
        **or** in the headline title.  Each matching item is weighted by
        ``source_weight × magnitude_weight × time_decay``.

        Returns ``None`` if no relevant items are found — the caller
        should fall back to :meth:`get_avg_sentiment`.
        """
        coin = _extract_coin_ticker(symbol)
        weighted_sum = 0.0
        total_weight = 0.0

        for item in self._latest:
            if item.sentiment is None:
                continue

            in_coins = coin in [c.upper() for c in item.sentiment.affected_coins]
            in_title = coin.lower() in item.title.lower()
            if not in_coins and not in_title:
                continue

            decay = _time_decay_factor(item.published)
            magnitude = _MAGNITUDE_WEIGHT.get(item.sentiment.impact_magnitude, 1.0)
            weight = item.sentiment.source_weight * magnitude * decay

            weighted_sum += item.sentiment.score * weight
            total_weight += weight

        if total_weight == 0.0:
            return None
        return weighted_sum / total_weight
