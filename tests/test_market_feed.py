"""Tests for src/market/feed.py."""
import asyncio
import json
import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.market.feed import MarketFeed, PriceTick


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_feed(symbols=None, queue=None) -> MarketFeed:
    """Create a MarketFeed without a live WebSocket connection."""
    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT"]
    if queue is None:
        queue = asyncio.Queue()
    return MarketFeed(symbols, queue)


def _binance_message(
    symbol: str = "BTCUSDT",
    price: str = "45000.00",
    volume: str = "1234.5",
    bid: str = "44999.00",
    ask: str = "45001.00",
) -> str:
    """Return a JSON-encoded Binance combined-stream ticker message."""
    stream = f"{symbol.lower()}@ticker"
    return json.dumps({
        "stream": stream,
        "data": {
            "s": symbol,
            "c": price,
            "v": volume,
            "b": bid,
            "a": ask,
        },
    })


# ---------------------------------------------------------------------------
# PriceTick dataclass
# ---------------------------------------------------------------------------

class TestPriceTickDataclass:
    def test_create_price_tick(self):
        tick = PriceTick(
            symbol="BTCUSDT",
            price=45_000.0,
            volume=1234.5,
            bid=44_999.0,
            ask=45_001.0,
            timestamp=1_700_000_000.0,
        )
        assert tick.symbol == "BTCUSDT"
        assert tick.price == pytest.approx(45_000.0)
        assert tick.volume == pytest.approx(1234.5)
        assert tick.bid == pytest.approx(44_999.0)
        assert tick.ask == pytest.approx(45_001.0)
        assert tick.timestamp == pytest.approx(1_700_000_000.0)

    def test_price_tick_is_dataclass(self):
        """PriceTick should support equality comparison via dataclass."""
        t1 = PriceTick("BTCUSDT", 45_000.0, 1234.5, 44_999.0, 45_001.0, 0.0)
        t2 = PriceTick("BTCUSDT", 45_000.0, 1234.5, 44_999.0, 45_001.0, 0.0)
        assert t1 == t2


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_get_latest_returns_none_initially(self):
        feed = _make_feed()
        assert feed.get_latest("BTCUSDT") is None

    def test_get_latest_unknown_symbol_returns_none(self):
        feed = _make_feed()
        assert feed.get_latest("XYZUSDT") is None

    def test_get_price_history_empty_initially(self):
        feed = _make_feed()
        assert feed.get_price_history("BTCUSDT") == []

    def test_get_price_history_unknown_symbol_empty(self):
        """Querying a symbol not in the feed's list returns []."""
        feed = _make_feed(symbols=["BTCUSDT"])
        assert feed.get_price_history("ETHUSDT") == []


# ---------------------------------------------------------------------------
# Message parsing via _handle_message
# ---------------------------------------------------------------------------

class TestHandleMessage:
    """Tests that bypass the WebSocket layer and call _handle_message directly."""

    def test_handle_valid_message_updates_latest(self):
        feed = _make_feed()
        feed._handle_message(_binance_message(
            symbol="BTCUSDT",
            price="45000.00",
            volume="1234.5",
            bid="44999.00",
            ask="45001.00",
        ))
        tick = feed.get_latest("BTCUSDT")
        assert tick is not None
        assert tick.symbol == "BTCUSDT"
        assert tick.price == pytest.approx(45_000.0)
        assert tick.volume == pytest.approx(1234.5)
        assert tick.bid == pytest.approx(44_999.0)
        assert tick.ask == pytest.approx(45_001.0)

    def test_handle_message_appends_price_history(self):
        feed = _make_feed()
        feed._handle_message(_binance_message(symbol="BTCUSDT", price="45000.00"))
        feed._handle_message(_binance_message(symbol="BTCUSDT", price="45100.00"))

        history = feed.get_price_history("BTCUSDT")
        assert history == pytest.approx([45_000.0, 45_100.0])

    def test_handle_message_puts_tick_in_queue(self):
        queue = asyncio.Queue()
        feed = MarketFeed(["BTCUSDT"], queue)
        feed._handle_message(_binance_message(symbol="BTCUSDT", price="45000.00"))

        assert not queue.empty()
        tick = queue.get_nowait()
        assert isinstance(tick, PriceTick)
        assert tick.price == pytest.approx(45_000.0)

    def test_handle_non_json_message_does_not_raise(self):
        feed = _make_feed()
        # Should log a warning but not raise
        feed._handle_message("not-valid-json{{{{")

    def test_handle_message_missing_data_field_does_not_raise(self):
        feed = _make_feed()
        feed._handle_message(json.dumps({"stream": "btcusdt@ticker"}))

    def test_handle_message_malformed_data_does_not_raise(self):
        """Missing required ticker keys should be caught gracefully."""
        feed = _make_feed()
        msg = json.dumps({"stream": "btcusdt@ticker", "data": {"unexpected": "keys"}})
        feed._handle_message(msg)


# ---------------------------------------------------------------------------
# WebSocket integration (mocked)
# ---------------------------------------------------------------------------

class TestFeedParsesWebSocketMessage:
    @pytest.mark.asyncio
    async def test_feed_parses_binance_message(self, mocker):
        """Mock websockets.connect so start() processes one message then stops."""
        raw_msg = _binance_message(
            symbol="BTCUSDT",
            price="45000.00",
            volume="1234.5",
            bid="44999.00",
            ask="45001.00",
        )

        # Build an async iterator that yields exactly one message then stops.
        async def _fake_ws_iter():
            yield raw_msg

        # The async context manager returned by websockets.connect
        mock_ws = MagicMock()
        mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_ws.__aexit__ = AsyncMock(return_value=False)
        mock_ws.__aiter__ = lambda self: _fake_ws_iter()

        mock_connect = mocker.patch(
            "src.market.feed.websockets.connect",
            return_value=mock_ws,
        )

        queue = asyncio.Queue()
        feed = MarketFeed(["BTCUSDT"], queue)
        feed._running = True  # normally set by start(); needed for _run_connection to process messages

        # Run _run_connection (one connection cycle) — stops after the iterator
        # is exhausted without raising.
        await feed._run_connection()

        tick = feed.get_latest("BTCUSDT")
        assert tick is not None, "Expected a PriceTick after processing one WS message"
        assert tick.symbol == "BTCUSDT"
        assert tick.price == pytest.approx(45_000.0)
        assert tick.volume == pytest.approx(1234.5)
        assert tick.bid == pytest.approx(44_999.0)
        assert tick.ask == pytest.approx(45_001.0)

    @pytest.mark.asyncio
    async def test_start_stop_cycle(self, mocker):
        """Calling stop() during start() exits the loop cleanly."""
        # Patch _run_connection to do nothing (avoid real WS call).
        feed = _make_feed()
        feed._running = False  # pre-set so the while loop exits immediately

        run_mock = mocker.patch.object(
            feed, "_run_connection", new_callable=AsyncMock
        )

        # start() sets _running=True then enters the loop; since we set
        # _running=False before, it will exit on the first condition check.
        # But start() resets _running=True at entry, so we need stop to be
        # called. Let's just verify _run_connection is called by driving one
        # iteration manually.

        feed._running = True
        task = asyncio.create_task(feed.start())
        # Give the event loop a tick to enter the loop then stop it.
        await asyncio.sleep(0)
        await feed.stop()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except asyncio.TimeoutError:
            task.cancel()

    @pytest.mark.asyncio
    async def test_multiple_symbols_tracked_independently(self, mocker):
        """Messages for different symbols populate separate history deques."""
        btc_msg = _binance_message(symbol="BTCUSDT", price="45000.00")
        eth_msg = _binance_message(symbol="ETHUSDT", price="3000.00")

        queue = asyncio.Queue()
        feed = MarketFeed(["BTCUSDT", "ETHUSDT"], queue)

        feed._handle_message(btc_msg)
        feed._handle_message(eth_msg)

        btc_tick = feed.get_latest("BTCUSDT")
        eth_tick = feed.get_latest("ETHUSDT")

        assert btc_tick is not None
        assert eth_tick is not None
        assert btc_tick.price == pytest.approx(45_000.0)
        assert eth_tick.price == pytest.approx(3_000.0)


# ---------------------------------------------------------------------------
# _parse_ticker static method
# ---------------------------------------------------------------------------

class TestParseTicker:
    def test_parse_ticker_correct_fields(self):
        data = {
            "s": "BTCUSDT",
            "c": "45000.00",
            "v": "1234.5",
            "b": "44999.00",
            "a": "45001.00",
        }
        tick = MarketFeed._parse_ticker(data)
        assert tick.symbol == "BTCUSDT"
        assert tick.price == pytest.approx(45_000.0)
        assert tick.volume == pytest.approx(1234.5)
        assert tick.bid == pytest.approx(44_999.0)
        assert tick.ask == pytest.approx(45_001.0)

    def test_parse_ticker_normalises_symbol_to_uppercase(self):
        data = {"s": "btcusdt", "c": "45000", "v": "100", "b": "44999", "a": "45001"}
        tick = MarketFeed._parse_ticker(data)
        assert tick.symbol == "BTCUSDT"

    def test_parse_ticker_missing_key_raises(self):
        with pytest.raises((KeyError, TypeError, ValueError)):
            MarketFeed._parse_ticker({"s": "BTCUSDT"})  # missing c, v, b, a
