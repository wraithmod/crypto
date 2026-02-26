"""Tests for src/news/feed.py."""
import json
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.base import LLMProvider, LLMError
from src.news.feed import NewsFeed, NewsItem, SentimentResult


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

class MockProvider(LLMProvider):
    """Minimal LLMProvider implementation for testing.

    Construct with the text you want complete() to return.  Pass
    side_effect=SomeException(...) to simulate API failures instead.
    """

    def __init__(self, response_text: str = "", side_effect: Exception | None = None) -> None:
        self._response_text = response_text
        self._side_effect = side_effect

    @property
    def name(self) -> str:
        return "mock-provider"

    async def complete(self, system: str, user: str, max_tokens: int = 150) -> str:
        if self._side_effect is not None:
            raise self._side_effect
        return self._response_text


def _make_feed(response_json: dict | None = None, side_effect: Exception | None = None) -> NewsFeed:
    """Create a NewsFeed backed by a MockProvider.

    Args:
        response_json: Dict that MockProvider will serialise and return from complete().
                       Defaults to a neutral payload when not supplied.
        side_effect:   If set, MockProvider.complete() raises this exception.
    """
    if response_json is None and side_effect is None:
        response_json = {"score": 0.0, "label": "neutral", "summary": "Analysis unavailable."}
    text = json.dumps(response_json) if response_json is not None else ""
    provider = MockProvider(response_text=text, side_effect=side_effect)
    return NewsFeed(llm_provider=provider)


def _fake_feedparser_result(entries: list[dict]) -> MagicMock:
    """Return a mock feedparser result with .feed.title and .entries."""
    result = MagicMock()
    result.feed.title = "Test Feed"

    mock_entries = []
    for e in entries:
        entry = MagicMock()
        entry.title = e.get("title", "")
        entry.link = e.get("link", "https://example.com")
        entry.published = e.get("published", "")
        entry.updated = e.get("updated", "")
        mock_entries.append(entry)

    result.entries = mock_entries
    return result


# ---------------------------------------------------------------------------
# SentimentResult dataclass
# ---------------------------------------------------------------------------

class TestSentimentResultDataclass:
    def test_create_sentiment_result(self):
        sr = SentimentResult(score=0.8, label="bullish", summary="Very positive.")
        assert sr.score == pytest.approx(0.8)
        assert sr.label == "bullish"
        assert sr.summary == "Very positive."

    def test_bearish_sentiment(self):
        sr = SentimentResult(score=-0.5, label="bearish", summary="Market looks grim.")
        assert sr.score == pytest.approx(-0.5)
        assert sr.label == "bearish"

    def test_neutral_sentiment(self):
        sr = SentimentResult(score=0.0, label="neutral", summary="No clear direction.")
        assert sr.score == pytest.approx(0.0)
        assert sr.label == "neutral"


# ---------------------------------------------------------------------------
# NewsItem dataclass
# ---------------------------------------------------------------------------

class TestNewsItemDataclass:
    def test_create_news_item_without_sentiment(self):
        item = NewsItem(
            title="BTC breaks $50k",
            url="https://example.com/btc",
            source="CoinDesk",
            published="2026-02-26",
        )
        assert item.title == "BTC breaks $50k"
        assert item.url == "https://example.com/btc"
        assert item.source == "CoinDesk"
        assert item.published == "2026-02-26"
        assert item.sentiment is None

    def test_create_news_item_with_sentiment(self):
        sr = SentimentResult(score=0.7, label="bullish", summary="Positive outlook.")
        item = NewsItem(
            title="BTC rallies",
            url="https://example.com/rally",
            source="Reuters",
            published="2026-02-26",
            sentiment=sr,
        )
        assert item.sentiment is sr
        assert item.sentiment.score == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_get_latest_empty_initially(self):
        feed = _make_feed()
        assert feed.get_latest() == []

    def test_get_avg_sentiment_empty_returns_zero(self):
        feed = _make_feed()
        assert feed.get_avg_sentiment() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# get_avg_sentiment
# ---------------------------------------------------------------------------

class TestGetAvgSentiment:
    def test_avg_sentiment_with_scored_items(self):
        feed = _make_feed()
        feed._latest = [
            NewsItem("A", "", "", "", SentimentResult(0.6, "bullish", "")),
            NewsItem("B", "", "", "", SentimentResult(-0.4, "bearish", "")),
            NewsItem("C", "", "", "", SentimentResult(0.2, "neutral", "")),
        ]
        expected = (0.6 + (-0.4) + 0.2) / 3
        assert feed.get_avg_sentiment() == pytest.approx(expected)

    def test_avg_sentiment_ignores_items_without_sentiment(self):
        feed = _make_feed()
        feed._latest = [
            NewsItem("A", "", "", "", SentimentResult(0.8, "bullish", "")),
            NewsItem("B", "", "", "", None),  # no sentiment
        ]
        assert feed.get_avg_sentiment() == pytest.approx(0.8)

    def test_avg_sentiment_all_no_sentiment_returns_zero(self):
        feed = _make_feed()
        feed._latest = [
            NewsItem("A", "", "", "", None),
            NewsItem("B", "", "", "", None),
        ]
        assert feed.get_avg_sentiment() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# fetch_headlines (mocked feedparser)
# ---------------------------------------------------------------------------

class TestFetchHeadlinesMocked:
    @pytest.mark.asyncio
    async def test_fetch_headlines_returns_news_items(self):
        """feedparser.parse returns two entries; fetch_headlines returns both."""
        fake_entries = [
            {"title": "BTC hits ATH", "link": "https://ex.com/1", "published": "2026-02-26"},
            {"title": "ETH upgrade live", "link": "https://ex.com/2", "published": "2026-02-25"},
        ]
        fake_result = _fake_feedparser_result(fake_entries)

        with patch("src.news.feed.feedparser.parse", return_value=fake_result):
            feed = _make_feed()
            items = await feed.fetch_headlines()

        assert len(items) >= 1
        titles = [item.title for item in items]
        assert "BTC hits ATH" in titles
        assert "ETH upgrade live" in titles

    @pytest.mark.asyncio
    async def test_fetch_headlines_deduplicates_titles(self):
        """Duplicate titles across feeds are included only once."""
        duplicate_entries = [
            {"title": "Same headline", "link": "https://ex.com/1", "published": "2026-02-26"},
            {"title": "Same headline", "link": "https://ex.com/2", "published": "2026-02-26"},
        ]
        fake_result = _fake_feedparser_result(duplicate_entries)

        with patch("src.news.feed.feedparser.parse", return_value=fake_result):
            feed = _make_feed()
            items = await feed.fetch_headlines()

        titles = [item.title for item in items]
        assert titles.count("Same headline") == 1

    @pytest.mark.asyncio
    async def test_fetch_headlines_skips_blank_titles(self):
        """Entries with empty titles are silently skipped."""
        entries = [
            {"title": "", "link": "https://ex.com/1", "published": ""},
            {"title": "Valid headline", "link": "https://ex.com/2", "published": ""},
        ]
        fake_result = _fake_feedparser_result(entries)

        with patch("src.news.feed.feedparser.parse", return_value=fake_result):
            feed = _make_feed()
            items = await feed.fetch_headlines()

        titles = [item.title for item in items]
        assert "" not in titles
        assert "Valid headline" in titles

    @pytest.mark.asyncio
    async def test_fetch_headlines_limits_to_twenty(self):
        """At most 20 items are returned regardless of feed size."""
        entries = [
            {"title": f"Headline {i}", "link": f"https://ex.com/{i}", "published": ""}
            for i in range(30)
        ]
        fake_result = _fake_feedparser_result(entries)

        with patch("src.news.feed.feedparser.parse", return_value=fake_result):
            feed = _make_feed()
            items = await feed.fetch_headlines()

        assert len(items) <= 20

    @pytest.mark.asyncio
    async def test_fetch_headlines_handles_feedparser_exception(self):
        """If feedparser raises, fetch_headlines returns an empty list gracefully."""
        with patch(
            "src.news.feed.feedparser.parse",
            side_effect=Exception("network error"),
        ):
            feed = _make_feed()
            # Should not raise; bad feeds are logged and skipped.
            items = await feed.fetch_headlines()

        assert isinstance(items, list)

    @pytest.mark.asyncio
    async def test_fetch_headlines_news_item_fields(self):
        """Returned NewsItem has correctly mapped fields."""
        entries = [
            {
                "title": "BTC breaks resistance",
                "link": "https://coindesk.com/btc",
                "published": "2026-02-26",
            }
        ]
        fake_result = _fake_feedparser_result(entries)

        with patch("src.news.feed.feedparser.parse", return_value=fake_result):
            feed = _make_feed()
            items = await feed.fetch_headlines()

        assert len(items) >= 1
        item = items[0]
        assert isinstance(item, NewsItem)
        assert item.title == "BTC breaks resistance"
        assert item.url == "https://coindesk.com/btc"
        assert item.sentiment is None


# ---------------------------------------------------------------------------
# analyze_sentiment (mocked LLMProvider)
# ---------------------------------------------------------------------------

class TestAnalyzeSentimentMocked:
    @pytest.mark.asyncio
    async def test_analyze_sentiment_parses_bullish(self):
        payload = {"score": 0.8, "label": "bullish", "summary": "BTC outlook strong."}
        feed = _make_feed(response_json=payload)

        result = await feed.analyze_sentiment("BTC hits new all-time high")

        assert isinstance(result, SentimentResult)
        assert result.score == pytest.approx(0.8)
        assert result.label == "bullish"
        assert result.summary == "BTC outlook strong."

    @pytest.mark.asyncio
    async def test_analyze_sentiment_parses_bearish(self):
        payload = {"score": -0.7, "label": "bearish", "summary": "Crypto market down."}
        feed = _make_feed(response_json=payload)

        result = await feed.analyze_sentiment("BTC crashes 20%")

        assert result.score == pytest.approx(-0.7)
        assert result.label == "bearish"

    @pytest.mark.asyncio
    async def test_analyze_sentiment_clamps_score_above_one(self):
        """Score above 1.0 should be clamped to 1.0."""
        payload = {"score": 1.5, "label": "bullish", "summary": "Off the charts."}
        feed = _make_feed(response_json=payload)

        result = await feed.analyze_sentiment("Extreme bull run")
        assert result.score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_analyze_sentiment_clamps_score_below_minus_one(self):
        """Score below -1.0 should be clamped to -1.0."""
        payload = {"score": -2.0, "label": "bearish", "summary": "Apocalyptic."}
        feed = _make_feed(response_json=payload)

        result = await feed.analyze_sentiment("Crypto apocalypse")
        assert result.score == pytest.approx(-1.0)

    @pytest.mark.asyncio
    async def test_analyze_sentiment_unknown_label_defaults_to_neutral(self):
        payload = {"score": 0.1, "label": "unknown_label", "summary": "Unclear."}
        feed = _make_feed(response_json=payload)

        result = await feed.analyze_sentiment("Something happened")
        assert result.label == "neutral"

    @pytest.mark.asyncio
    async def test_analyze_sentiment_strips_markdown_fences(self):
        """The LLM response sometimes wraps JSON in ```json ... ``` fences."""
        raw_text = "```json\n{\"score\": 0.5, \"label\": \"bullish\", \"summary\": \"Positive.\"}\n```"
        provider = MockProvider(response_text=raw_text)
        feed = NewsFeed(llm_provider=provider)

        result = await feed.analyze_sentiment("BTC is looking good")
        assert result.score == pytest.approx(0.5)
        assert result.label == "bullish"

    @pytest.mark.asyncio
    async def test_analyze_sentiment_returns_neutral_on_api_error(self):
        """Any exception from the LLM provider returns a neutral fallback."""
        feed = _make_feed(side_effect=LLMError("API error"))

        result = await feed.analyze_sentiment("BTC moves sideways")
        assert isinstance(result, SentimentResult)
        assert result.score == pytest.approx(0.0)
        assert result.label == "neutral"

    @pytest.mark.asyncio
    async def test_analyze_sentiment_returns_neutral_on_invalid_json(self):
        """Malformed JSON from the provider returns a neutral fallback."""
        provider = MockProvider(response_text="not valid json !!!")
        feed = NewsFeed(llm_provider=provider)

        result = await feed.analyze_sentiment("Broken response")
        assert result.score == pytest.approx(0.0)
        assert result.label == "neutral"


# ---------------------------------------------------------------------------
# fetch_and_analyze integration (mocked)
# ---------------------------------------------------------------------------

class TestFetchAndAnalyze:
    @pytest.mark.asyncio
    async def test_fetch_and_analyze_attaches_sentiments(self):
        """fetch_and_analyze should annotate the top-3 headlines with sentiment."""
        entries = [
            {"title": f"Headline {i}", "link": f"https://ex.com/{i}", "published": ""}
            for i in range(6)
        ]
        fake_result = _fake_feedparser_result(entries)

        sentiment_payload = {"score": 0.5, "label": "bullish", "summary": "Good news."}

        with patch("src.news.feed.feedparser.parse", return_value=fake_result):
            feed = _make_feed(response_json=sentiment_payload)
            items = await feed.fetch_and_analyze()

        # Top 3 items should have sentiment; the rest may not.
        for item in items[:3]:
            assert item.sentiment is not None

    @pytest.mark.asyncio
    async def test_fetch_and_analyze_updates_latest_cache(self):
        entries = [
            {"title": "Latest news", "link": "https://ex.com/1", "published": ""}
        ]
        fake_result = _fake_feedparser_result(entries)

        sentiment_payload = {"score": 0.3, "label": "neutral", "summary": "OK."}

        with patch("src.news.feed.feedparser.parse", return_value=fake_result):
            feed = _make_feed(response_json=sentiment_payload)
            await feed.fetch_and_analyze()

        assert len(feed.get_latest()) >= 1
        assert feed.get_latest()[0].title == "Latest news"
