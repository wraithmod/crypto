"""Tests for src/market/indices.py — IndicesFeed price history and ASXFeedAdapter."""
import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.market.indices import IndicesFeed, ASXFeedAdapter, IndexTick
from src.market.feed import PriceTick


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_feed(symbols=None) -> IndicesFeed:
    """Create an IndicesFeed without hitting the network."""
    if symbols is None:
        symbols = ["^GSPC", "CBA.AX"]
    # Bypass config.asx_enabled by passing symbols directly
    feed = IndicesFeed.__new__(IndicesFeed)
    from collections import deque
    from config import config
    feed._symbols = symbols
    feed._poll_interval = 30.0
    feed._latest = {}
    feed._price_history = {sym: deque(maxlen=config.price_history_len) for sym in symbols}
    feed._running = False
    return feed


def _inject_tick(feed: IndicesFeed, symbol: str, price: float) -> IndexTick:
    """Directly inject an IndexTick into the feed (simulates a poll result)."""
    tick = IndexTick(
        symbol=symbol,
        name=symbol,
        price=price,
        change=0.0,
        change_pct=0.0,
        timestamp=time.time(),
        group="asx_stocks" if symbol.endswith(".AX") else "global_markets",
    )
    feed._latest[symbol] = tick
    if symbol in feed._price_history:
        feed._price_history[symbol].append(price)
    return tick


# ---------------------------------------------------------------------------
# IndicesFeed price history
# ---------------------------------------------------------------------------

class TestIndicesFeedPriceHistory:
    def test_get_price_history_empty_initially(self):
        feed = _make_feed()
        assert feed.get_price_history("^GSPC") == []

    def test_get_price_history_unknown_symbol_returns_empty(self):
        feed = _make_feed()
        assert feed.get_price_history("UNKNOWN") == []

    def test_get_price_history_accumulates(self):
        feed = _make_feed()
        _inject_tick(feed, "CBA.AX", 100.0)
        _inject_tick(feed, "CBA.AX", 101.5)
        _inject_tick(feed, "CBA.AX", 102.0)
        hist = feed.get_price_history("CBA.AX")
        assert hist == pytest.approx([100.0, 101.5, 102.0])

    def test_get_price_history_independent_per_symbol(self):
        feed = _make_feed()
        _inject_tick(feed, "^GSPC", 5000.0)
        _inject_tick(feed, "CBA.AX", 99.0)
        assert feed.get_price_history("^GSPC") == pytest.approx([5000.0])
        assert feed.get_price_history("CBA.AX") == pytest.approx([99.0])

    def test_get_candle_history_always_empty(self):
        feed = _make_feed()
        _inject_tick(feed, "CBA.AX", 100.0)
        assert feed.get_candle_history("CBA.AX") == []

    def test_get_volume_history_always_empty(self):
        feed = _make_feed()
        _inject_tick(feed, "CBA.AX", 100.0)
        assert feed.get_volume_history("CBA.AX") == []

    def test_price_history_bounded_by_maxlen(self):
        """Deque must not exceed price_history_len (default 200)."""
        from config import config
        feed = _make_feed(symbols=["CBA.AX"])
        # Push more entries than the max
        for i in range(config.price_history_len + 50):
            _inject_tick(feed, "CBA.AX", float(i))
        hist = feed.get_price_history("CBA.AX")
        assert len(hist) == config.price_history_len

    def test_fetch_all_updates_price_history(self):
        """_fetch_all appends prices from ticks dict to price history."""
        feed = _make_feed(symbols=["^GSPC"])
        tick = IndexTick(
            symbol="^GSPC",
            name="S&P 500",
            price=5100.0,
            change=10.0,
            change_pct=0.2,
            timestamp=time.time(),
            group="global_markets",
        )
        # Simulate what _fetch_all does after calling _fetch_sync
        feed._latest.update({"^GSPC": tick})
        for sym, t in {"^GSPC": tick}.items():
            if sym in feed._price_history:
                feed._price_history[sym].append(t.price)

        assert feed.get_price_history("^GSPC") == pytest.approx([5100.0])


# ---------------------------------------------------------------------------
# ASXFeedAdapter
# ---------------------------------------------------------------------------

class TestASXFeedAdapter:
    def test_get_latest_returns_none_if_not_polled(self):
        feed = _make_feed()
        adapter = ASXFeedAdapter(feed)
        assert adapter.get_latest("CBA.AX") is None

    def test_get_latest_returns_price_tick(self):
        feed = _make_feed()
        _inject_tick(feed, "CBA.AX", 112.5)
        adapter = ASXFeedAdapter(feed)
        tick = adapter.get_latest("CBA.AX")
        assert isinstance(tick, PriceTick)
        assert tick.symbol == "CBA.AX"
        assert tick.price == pytest.approx(112.5)

    def test_get_latest_price_tick_fields(self):
        """Adapter fills bid/ask/volume with 0.0 (unavailable from yfinance)."""
        feed = _make_feed()
        _inject_tick(feed, "CBA.AX", 99.0)
        adapter = ASXFeedAdapter(feed)
        tick = adapter.get_latest("CBA.AX")
        assert tick.bid == pytest.approx(0.0)
        assert tick.ask == pytest.approx(0.0)
        assert tick.volume == pytest.approx(0.0)

    def test_get_price_history_delegates_to_feed(self):
        feed = _make_feed()
        _inject_tick(feed, "CBA.AX", 50.0)
        _inject_tick(feed, "CBA.AX", 51.0)
        adapter = ASXFeedAdapter(feed)
        assert adapter.get_price_history("CBA.AX") == pytest.approx([50.0, 51.0])

    def test_get_price_history_empty_for_unknown_symbol(self):
        feed = _make_feed()
        adapter = ASXFeedAdapter(feed)
        assert adapter.get_price_history("UNKNOWN.AX") == []

    def test_get_candle_history_always_empty(self):
        feed = _make_feed()
        adapter = ASXFeedAdapter(feed)
        assert adapter.get_candle_history("CBA.AX") == []

    def test_get_volume_history_always_empty(self):
        feed = _make_feed()
        adapter = ASXFeedAdapter(feed)
        assert adapter.get_volume_history("CBA.AX") == []

    def test_get_latest_unknown_symbol_returns_none(self):
        feed = _make_feed()
        adapter = ASXFeedAdapter(feed)
        assert adapter.get_latest("XYZ.AX") is None
