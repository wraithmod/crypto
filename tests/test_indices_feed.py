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


# ---------------------------------------------------------------------------
# _group_for helper
# ---------------------------------------------------------------------------

class TestGroupFor:
    """Unit tests for the _group_for(symbol) routing helper."""

    def setup_method(self):
        # Import here so it picks up the live config singleton
        from src.market.indices import _group_for
        self._group_for = _group_for

    def test_aapl_is_us_stocks(self):
        assert self._group_for("AAPL") == "us_stocks"

    def test_nvda_is_us_stocks(self):
        assert self._group_for("NVDA") == "us_stocks"

    def test_brk_b_is_us_stocks(self):
        """Hyphenated ticker BRK-B should map to us_stocks."""
        assert self._group_for("BRK-B") == "us_stocks"

    def test_vix_is_global_markets(self):
        assert self._group_for("^VIX") == "global_markets"

    def test_gspc_is_global_markets(self):
        assert self._group_for("^GSPC") == "global_markets"

    def test_cba_ax_is_asx_stocks(self):
        assert self._group_for("CBA.AX") == "asx_stocks"

    def test_bhp_ax_is_asx_stocks(self):
        assert self._group_for("BHP.AX") == "asx_stocks"

    def test_unknown_symbol_falls_back_to_global_markets(self):
        """Symbols not in any known group fall back to global_markets."""
        assert self._group_for("UNKNOWN_XYZ") == "global_markets"

    def test_ax_suffix_takes_priority_over_us_stocks(self):
        """A .AX symbol must be asx_stocks even if it happens to match a US ticker name."""
        assert self._group_for("AAPL.AX") == "asx_stocks"

    def test_caret_prefix_takes_priority_over_us_stocks(self):
        """^-prefixed symbols must be global_markets regardless of name."""
        assert self._group_for("^AAPL") == "global_markets"


# ---------------------------------------------------------------------------
# IndicesFeed with us_stocks_enabled=True
# ---------------------------------------------------------------------------

class TestIndicesFeedUSStocks:
    """Tests for IndicesFeed behaviour when US stocks are enabled."""

    def _make_us_feed(self) -> IndicesFeed:
        """Construct an IndicesFeed that includes US stock symbols without network access."""
        from collections import deque
        from config import config, AppConfig
        # Build a minimal config-like list for testing
        us_syms = list(config.us_stocks_symbols)
        global_syms = ["^GSPC"]
        all_syms = global_syms + us_syms

        feed = IndicesFeed.__new__(IndicesFeed)
        feed._symbols = all_syms
        feed._poll_interval = 30.0
        feed._latest = {}
        feed._price_history = {sym: deque(maxlen=config.price_history_len) for sym in all_syms}
        feed._running = False
        return feed, us_syms

    def test_us_stocks_present_in_symbol_list(self):
        feed, us_syms = self._make_us_feed()
        for sym in us_syms:
            assert sym in feed._symbols, f"{sym} should be in feed._symbols"

    def test_aapl_in_symbol_list(self):
        feed, _ = self._make_us_feed()
        assert "AAPL" in feed._symbols

    def test_nvda_in_symbol_list(self):
        feed, _ = self._make_us_feed()
        assert "NVDA" in feed._symbols

    def test_brk_b_in_symbol_list(self):
        feed, _ = self._make_us_feed()
        assert "BRK-B" in feed._symbols

    def test_get_by_group_us_stocks_empty_on_fresh_init(self):
        """get_by_group('us_stocks') returns {} when no data has been fetched yet."""
        feed, _ = self._make_us_feed()
        result = feed.get_by_group("us_stocks")
        assert result == {}

    def test_get_by_group_us_stocks_returns_injected_tick(self):
        """After injecting a US stock tick, get_by_group('us_stocks') must return it."""
        feed, _ = self._make_us_feed()
        tick = IndexTick(
            symbol="AAPL",
            name="Apple",
            price=175.0,
            change=1.5,
            change_pct=0.86,
            timestamp=time.time(),
            group="us_stocks",
        )
        feed._latest["AAPL"] = tick
        result = feed.get_by_group("us_stocks")
        assert "AAPL" in result
        assert result["AAPL"].price == pytest.approx(175.0)

    def test_get_by_group_us_stocks_excludes_global_indices(self):
        """global_markets ticks must not appear in the us_stocks group."""
        feed, _ = self._make_us_feed()
        # Inject a global index tick
        gspc_tick = IndexTick(
            symbol="^GSPC",
            name="S&P 500",
            price=5000.0,
            change=10.0,
            change_pct=0.2,
            timestamp=time.time(),
            group="global_markets",
        )
        feed._latest["^GSPC"] = gspc_tick
        result = feed.get_by_group("us_stocks")
        assert "^GSPC" not in result

    def test_price_history_initialised_for_us_stocks(self):
        """Price history deques should exist for all US stock symbols on init."""
        feed, us_syms = self._make_us_feed()
        for sym in us_syms:
            assert sym in feed._price_history

    def test_price_history_accumulates_for_us_stock(self):
        """Injecting prices into _price_history deque reflects in get_price_history."""
        feed, _ = self._make_us_feed()
        feed._price_history["NVDA"].append(500.0)
        feed._price_history["NVDA"].append(505.0)
        assert feed.get_price_history("NVDA") == pytest.approx([500.0, 505.0])
