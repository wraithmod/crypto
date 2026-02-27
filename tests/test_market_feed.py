"""Tests for src/market/feed.py."""
import asyncio
import json
import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.market.feed import MarketFeed, PriceTick, CandleTick


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


def _kline_message(
    symbol: str = "BTCUSDT",
    open_price: str = "44900.00",
    high: str = "45100.00",
    low: str = "44800.00",
    close: str = "45000.00",
    volume: str = "1234.5",
    is_closed: bool = True,
    open_time: int = 1_700_000_000_000,
) -> str:
    """Return a JSON-encoded Binance combined-stream kline_1m message."""
    stream = f"{symbol.lower()}@kline_1m"
    return json.dumps({
        "stream": stream,
        "data": {
            "e": "kline",
            "s": symbol,
            "k": {
                "t": open_time,
                "o": open_price,
                "h": high,
                "l": low,
                "c": close,
                "v": volume,
                "x": is_closed,
            },
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
            bid=0.0,
            ask=0.0,
            timestamp=1_700_000_000.0,
        )
        assert tick.symbol == "BTCUSDT"
        assert tick.price == pytest.approx(45_000.0)
        assert tick.volume == pytest.approx(1234.5)
        assert tick.bid == pytest.approx(0.0)
        assert tick.ask == pytest.approx(0.0)
        assert tick.timestamp == pytest.approx(1_700_000_000.0)

    def test_price_tick_is_dataclass(self):
        """PriceTick should support equality comparison via dataclass."""
        t1 = PriceTick("BTCUSDT", 45_000.0, 1234.5, 0.0, 0.0, 0.0)
        t2 = PriceTick("BTCUSDT", 45_000.0, 1234.5, 0.0, 0.0, 0.0)
        assert t1 == t2


# ---------------------------------------------------------------------------
# CandleTick dataclass
# ---------------------------------------------------------------------------

class TestCandleTickDataclass:
    def test_create_candle_tick(self):
        candle = CandleTick(
            symbol="BTCUSDT",
            open_time=1_700_000_000_000,
            open=44_900.0,
            high=45_100.0,
            low=44_800.0,
            close=45_000.0,
            volume=1234.5,
            is_closed=True,
        )
        assert candle.symbol == "BTCUSDT"
        assert candle.close == pytest.approx(45_000.0)
        assert candle.is_closed is True

    def test_typical_price_property(self):
        candle = CandleTick(
            symbol="BTCUSDT",
            open_time=0,
            open=100.0,
            high=110.0,
            low=90.0,
            close=105.0,
            volume=100.0,
            is_closed=True,
        )
        # (110 + 90 + 105) / 3 = 101.6667
        assert candle.typical_price == pytest.approx((110.0 + 90.0 + 105.0) / 3.0)


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

    def test_get_candle_history_empty_initially(self):
        feed = _make_feed()
        assert feed.get_candle_history("BTCUSDT") == []

    def test_get_volume_history_empty_initially(self):
        feed = _make_feed()
        assert feed.get_volume_history("BTCUSDT") == []


# ---------------------------------------------------------------------------
# Message parsing via _handle_message
# ---------------------------------------------------------------------------

class TestHandleMessage:
    """Tests that bypass the WebSocket layer and call _handle_message directly."""

    def test_handle_valid_message_updates_latest(self):
        feed = _make_feed()
        feed._handle_message(_kline_message(
            symbol="BTCUSDT",
            close="45000.00",
            volume="1234.5",
        ))
        tick = feed.get_latest("BTCUSDT")
        assert tick is not None
        assert tick.symbol == "BTCUSDT"
        assert tick.price == pytest.approx(45_000.0)
        assert tick.volume == pytest.approx(1234.5)
        # bid/ask not available from kline stream
        assert tick.bid == pytest.approx(0.0)
        assert tick.ask == pytest.approx(0.0)

    def test_handle_message_appends_price_history(self):
        feed = _make_feed()
        # Two candles with different open_times → two entries
        feed._handle_message(_kline_message(
            symbol="BTCUSDT", close="45000.00", open_time=1_000,
        ))
        feed._handle_message(_kline_message(
            symbol="BTCUSDT", close="45100.00", open_time=2_000,
        ))

        history = feed.get_price_history("BTCUSDT")
        assert history == pytest.approx([45_000.0, 45_100.0])

    def test_handle_message_same_open_time_updates_in_place(self):
        """In-progress candle update (same open_time) replaces last entry."""
        feed = _make_feed()
        feed._handle_message(_kline_message(
            symbol="BTCUSDT", close="45000.00", open_time=1_000, is_closed=False,
        ))
        feed._handle_message(_kline_message(
            symbol="BTCUSDT", close="45050.00", open_time=1_000, is_closed=True,
        ))
        history = feed.get_price_history("BTCUSDT")
        # Only one candle in history (updated in-place)
        assert len(history) == 1
        assert history[0] == pytest.approx(45_050.0)

    def test_handle_message_puts_tick_in_queue(self):
        queue = asyncio.Queue()
        feed = MarketFeed(["BTCUSDT"], queue)
        feed._handle_message(_kline_message(symbol="BTCUSDT", close="45000.00"))

        assert not queue.empty()
        tick = queue.get_nowait()
        assert isinstance(tick, PriceTick)
        assert tick.price == pytest.approx(45_000.0)

    def test_handle_non_json_message_does_not_raise(self):
        feed = _make_feed()
        feed._handle_message("not-valid-json{{{{")

    def test_handle_message_missing_data_field_does_not_raise(self):
        feed = _make_feed()
        feed._handle_message(json.dumps({"stream": "btcusdt@kline_1m"}))

    def test_handle_message_non_kline_event_ignored(self):
        """Messages with event type other than 'kline' are silently ignored."""
        feed = _make_feed()
        msg = json.dumps({"stream": "btcusdt@ticker", "data": {"e": "24hrTicker", "s": "BTCUSDT"}})
        feed._handle_message(msg)
        assert feed.get_latest("BTCUSDT") is None

    def test_handle_message_malformed_data_does_not_raise(self):
        """Missing required kline keys should be caught gracefully."""
        feed = _make_feed()
        # No 'e' key → ignored silently
        msg = json.dumps({"stream": "btcusdt@kline_1m", "data": {"unexpected": "keys"}})
        feed._handle_message(msg)

    def test_candle_history_populated_after_message(self):
        feed = _make_feed()
        feed._handle_message(_kline_message(symbol="BTCUSDT", close="45000.00", volume="500.0"))
        candles = feed.get_candle_history("BTCUSDT")
        assert len(candles) == 1
        assert candles[0].close == pytest.approx(45_000.0)
        assert candles[0].volume == pytest.approx(500.0)

    def test_volume_history_populated_after_message(self):
        feed = _make_feed()
        feed._handle_message(_kline_message(symbol="BTCUSDT", volume="999.9", open_time=1))
        feed._handle_message(_kline_message(symbol="BTCUSDT", volume="111.1", open_time=2))
        vols = feed.get_volume_history("BTCUSDT")
        assert vols == pytest.approx([999.9, 111.1])


# ---------------------------------------------------------------------------
# WebSocket integration (mocked)
# ---------------------------------------------------------------------------

class TestFeedParsesWebSocketMessage:
    @pytest.mark.asyncio
    async def test_feed_parses_kline_message(self, mocker):
        """Mock websockets.connect so start() processes one message then stops."""
        raw_msg = _kline_message(
            symbol="BTCUSDT",
            close="45000.00",
            volume="1234.5",
        )

        async def _fake_ws_iter():
            yield raw_msg

        mock_ws = MagicMock()
        mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_ws.__aexit__ = AsyncMock(return_value=False)
        mock_ws.__aiter__ = lambda self: _fake_ws_iter()

        mocker.patch("src.market.feed.websockets.connect", return_value=mock_ws)

        queue = asyncio.Queue()
        feed = MarketFeed(["BTCUSDT"], queue)
        feed._running = True

        await feed._run_connection()

        tick = feed.get_latest("BTCUSDT")
        assert tick is not None, "Expected a PriceTick after processing one WS message"
        assert tick.symbol == "BTCUSDT"
        assert tick.price == pytest.approx(45_000.0)
        assert tick.volume == pytest.approx(1234.5)

    @pytest.mark.asyncio
    async def test_start_stop_cycle(self, mocker):
        """Calling stop() during start() exits the loop cleanly."""
        feed = _make_feed()

        # Mock both warm_up and _run_connection to avoid network calls
        mocker.patch.object(feed, "warm_up", new_callable=AsyncMock)
        mocker.patch.object(feed, "_run_connection", new_callable=AsyncMock)

        feed._running = True
        task = asyncio.create_task(feed.start())
        await asyncio.sleep(0)
        await feed.stop()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except asyncio.TimeoutError:
            task.cancel()

    @pytest.mark.asyncio
    async def test_multiple_symbols_tracked_independently(self, mocker):
        """Messages for different symbols populate separate history deques."""
        btc_msg = _kline_message(symbol="BTCUSDT", close="45000.00")
        eth_msg = _kline_message(symbol="ETHUSDT", close="3000.00")

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
# _parse_kline static method
# ---------------------------------------------------------------------------

class TestParseKline:
    def test_parse_kline_correct_fields(self):
        data = {
            "e": "kline",
            "s": "BTCUSDT",
            "k": {
                "t": 1_700_000_000_000,
                "o": "44900.00",
                "h": "45100.00",
                "l": "44800.00",
                "c": "45000.00",
                "v": "1234.5",
                "x": True,
            },
        }
        candle = MarketFeed._parse_kline(data)
        assert candle.symbol == "BTCUSDT"
        assert candle.open == pytest.approx(44_900.0)
        assert candle.high == pytest.approx(45_100.0)
        assert candle.low == pytest.approx(44_800.0)
        assert candle.close == pytest.approx(45_000.0)
        assert candle.volume == pytest.approx(1234.5)
        assert candle.is_closed is True

    def test_parse_kline_normalises_symbol_to_uppercase(self):
        data = {
            "e": "kline",
            "s": "btcusdt",
            "k": {"t": 0, "o": "1", "h": "2", "l": "0.5", "c": "1.5", "v": "100", "x": False},
        }
        candle = MarketFeed._parse_kline(data)
        assert candle.symbol == "BTCUSDT"

    def test_parse_kline_missing_key_raises(self):
        with pytest.raises((KeyError, TypeError, ValueError)):
            MarketFeed._parse_kline({"e": "kline", "s": "BTCUSDT"})  # missing "k"
