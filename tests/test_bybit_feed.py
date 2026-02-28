"""Tests for src/market/bybit_feed.py.

All external I/O (WebSocket connections, REST calls) is fully mocked.
No real network traffic is produced by this test suite.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections import deque
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.market.bybit_feed import BybitFeed
from src.market.feed import CandleTick, PriceTick


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_feed(
    symbols: list[str] | None = None,
    queue: asyncio.Queue | None = None,
) -> BybitFeed:
    """Construct a BybitFeed without opening any network connection."""
    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT"]
    if queue is None:
        queue = asyncio.Queue()
    return BybitFeed(symbols, queue)


def _kline_envelope(
    symbol: str = "BTCUSDT",
    open_price: str = "44900.00",
    high: str = "45100.00",
    low: str = "44800.00",
    close: str = "45000.00",
    volume: str = "1234.5",
    confirm: bool = True,
    start: int = 1_700_000_000_000,
) -> str:
    """Return a JSON-encoded Bybit v5 kline WebSocket message."""
    topic = f"kline.1.{symbol}"
    return json.dumps({
        "topic": topic,
        "type": "snapshot",
        "ts": start + 30_000,
        "data": [{
            "start": start,
            "end": start + 59_999,
            "interval": "1",
            "open": open_price,
            "close": close,
            "high": high,
            "low": low,
            "volume": volume,
            "turnover": "0",
            "confirm": confirm,
            "timestamp": start + 30_000,
        }],
    })


def _make_candle(
    symbol: str = "BTCUSDT",
    open_time: int = 1_700_000_000_000,
    close: float = 45_000.0,
    volume: float = 1234.5,
    is_closed: bool = True,
) -> CandleTick:
    return CandleTick(
        symbol=symbol,
        open_time=open_time,
        open=44_900.0,
        high=45_100.0,
        low=44_800.0,
        close=close,
        volume=volume,
        is_closed=is_closed,
    )


# ---------------------------------------------------------------------------
# TestBybitFeedInit
# ---------------------------------------------------------------------------

class TestBybitFeedInit:
    def test_symbols_are_uppercased(self):
        feed = BybitFeed(["btcusdt", "ethusdt"], asyncio.Queue())
        assert feed.symbols == ["BTCUSDT", "ETHUSDT"]

    def test_mixed_case_symbols_uppercased(self):
        feed = BybitFeed(["BtCuSdT"], asyncio.Queue())
        assert feed.symbols == ["BTCUSDT"]

    def test_price_queue_is_stored(self):
        q = asyncio.Queue()
        feed = BybitFeed(["BTCUSDT"], q)
        assert feed._queue is q

    def test_candle_history_deque_empty_initially(self):
        feed = _make_feed(["BTCUSDT", "ETHUSDT"])
        assert len(feed._candle_history["BTCUSDT"]) == 0
        assert len(feed._candle_history["ETHUSDT"]) == 0

    def test_price_cache_empty_initially(self):
        feed = _make_feed()
        assert feed._price_cache == {}

    def test_running_false_initially(self):
        feed = _make_feed()
        assert feed._running is False

    def test_symbols_property_returns_list(self):
        feed = _make_feed(["BTCUSDT"])
        assert isinstance(feed.symbols, list)


# ---------------------------------------------------------------------------
# TestBybitFeedKlineParsing
# ---------------------------------------------------------------------------

class TestBybitFeedKlineParsing:
    """Unit tests for _parse_kline (static method) and _handle_message."""

    def test_parse_valid_kline_produces_candle_tick(self):
        topic = "kline.1.BTCUSDT"
        data = {
            "start": 1_700_000_000_000,
            "end": 1_700_000_059_999,
            "open": "44900.00",
            "high": "45100.00",
            "low": "44800.00",
            "close": "45000.00",
            "volume": "1234.5",
            "confirm": True,
        }
        candle = BybitFeed._parse_kline(topic, data)
        assert isinstance(candle, CandleTick)
        assert candle.symbol == "BTCUSDT"
        assert candle.open_time == 1_700_000_000_000
        assert candle.open == pytest.approx(44_900.0)
        assert candle.high == pytest.approx(45_100.0)
        assert candle.low == pytest.approx(44_800.0)
        assert candle.close == pytest.approx(45_000.0)
        assert candle.volume == pytest.approx(1234.5)
        assert candle.is_closed is True

    def test_confirm_true_sets_is_closed_true(self):
        topic = "kline.1.BTCUSDT"
        data = {
            "start": 0,
            "open": "1", "high": "2", "low": "0.5", "close": "1.5",
            "volume": "10", "confirm": True,
        }
        candle = BybitFeed._parse_kline(topic, data)
        assert candle.is_closed is True

    def test_confirm_false_sets_is_closed_false(self):
        topic = "kline.1.BTCUSDT"
        data = {
            "start": 0,
            "open": "1", "high": "2", "low": "0.5", "close": "1.5",
            "volume": "10", "confirm": False,
        }
        candle = BybitFeed._parse_kline(topic, data)
        assert candle.is_closed is False

    def test_symbol_extracted_from_topic_string(self):
        topic = "kline.1.SOLUSDT"
        data = {
            "start": 0,
            "open": "1", "high": "2", "low": "0.5", "close": "1.5",
            "volume": "10", "confirm": False,
        }
        candle = BybitFeed._parse_kline(topic, data)
        assert candle.symbol == "SOLUSDT"

    def test_symbol_normalised_to_uppercase(self):
        """Even if topic symbol is lowercase (defensive), we uppercase it."""
        topic = "kline.1.btcusdt"
        data = {
            "start": 0,
            "open": "1", "high": "2", "low": "0.5", "close": "1.5",
            "volume": "10", "confirm": True,
        }
        candle = BybitFeed._parse_kline(topic, data)
        assert candle.symbol == "BTCUSDT"

    def test_subscription_confirm_message_ignored(self):
        """The initial subscription ACK has no 'topic' key — must be silently dropped."""
        feed = _make_feed()
        ack = json.dumps({
            "success": True,
            "ret_msg": "",
            "op": "subscribe",
            "conn_id": "abc123",
        })
        feed._handle_message(ack)
        assert feed.get_latest("BTCUSDT") is None

    def test_ping_response_message_ignored(self):
        """Bybit ping response has no 'topic' key — must be silently dropped."""
        feed = _make_feed()
        pong = json.dumps({"success": True, "ret_msg": "pong", "op": "pong"})
        feed._handle_message(pong)
        assert feed.get_latest("BTCUSDT") is None

    def test_non_json_message_does_not_raise(self):
        feed = _make_feed()
        feed._handle_message("not-valid-json{{{{")

    def test_handle_valid_kline_message_updates_cache(self):
        feed = _make_feed()
        feed._handle_message(_kline_envelope(symbol="BTCUSDT", close="45000.00"))
        tick = feed.get_latest("BTCUSDT")
        assert tick is not None
        assert tick.price == pytest.approx(45_000.0)


# ---------------------------------------------------------------------------
# TestBybitFeedUpsertCandle
# ---------------------------------------------------------------------------

class TestBybitFeedUpsertCandle:
    def test_same_open_time_replaces_last_element(self):
        feed = _make_feed(["BTCUSDT"])
        c1 = _make_candle(open_time=1_000, close=100.0, is_closed=False)
        c2 = _make_candle(open_time=1_000, close=105.0, is_closed=True)
        feed._upsert_candle(c1)
        feed._upsert_candle(c2)
        hist = feed._candle_history["BTCUSDT"]
        assert len(hist) == 1
        assert hist[-1].close == pytest.approx(105.0)
        assert hist[-1].is_closed is True

    def test_different_open_time_appends_new_candle(self):
        feed = _make_feed(["BTCUSDT"])
        c1 = _make_candle(open_time=1_000, close=100.0)
        c2 = _make_candle(open_time=2_000, close=200.0)
        feed._upsert_candle(c1)
        feed._upsert_candle(c2)
        hist = feed._candle_history["BTCUSDT"]
        assert len(hist) == 2
        assert hist[0].close == pytest.approx(100.0)
        assert hist[1].close == pytest.approx(200.0)

    def test_deque_does_not_exceed_maxlen(self):
        feed = _make_feed(["BTCUSDT"])
        maxlen = feed._candle_history["BTCUSDT"].maxlen
        assert maxlen is not None
        # Insert maxlen + 10 candles with distinct open_times.
        for i in range(maxlen + 10):
            c = _make_candle(open_time=i * 60_000, close=float(i))
            feed._upsert_candle(c)
        assert len(feed._candle_history["BTCUSDT"]) == maxlen

    def test_upsert_unknown_symbol_does_not_raise(self):
        """Candles for symbols not in the feed are silently ignored."""
        feed = _make_feed(["BTCUSDT"])
        c = _make_candle(symbol="XYZUSDT")
        feed._upsert_candle(c)  # should not raise

    def test_first_candle_appended_to_empty_history(self):
        feed = _make_feed(["BTCUSDT"])
        c = _make_candle(open_time=5_000, close=50.0)
        feed._upsert_candle(c)
        assert len(feed._candle_history["BTCUSDT"]) == 1
        assert feed._candle_history["BTCUSDT"][-1].close == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# TestBybitFeedPublicInterface
# ---------------------------------------------------------------------------

class TestBybitFeedPublicInterface:
    def test_get_latest_returns_none_when_no_data(self):
        feed = _make_feed()
        assert feed.get_latest("BTCUSDT") is None

    def test_get_latest_unknown_symbol_returns_none(self):
        feed = _make_feed(["BTCUSDT"])
        assert feed.get_latest("XYZUSDT") is None

    def test_get_latest_returns_price_tick_after_candle_processed(self):
        feed = _make_feed()
        feed._handle_message(
            _kline_envelope(symbol="BTCUSDT", close="45000.00", volume="500.0")
        )
        tick = feed.get_latest("BTCUSDT")
        assert isinstance(tick, PriceTick)
        assert tick.symbol == "BTCUSDT"
        assert tick.price == pytest.approx(45_000.0)
        assert tick.volume == pytest.approx(500.0)
        assert tick.bid == pytest.approx(0.0)
        assert tick.ask == pytest.approx(0.0)

    def test_get_latest_candle_returns_none_when_empty(self):
        feed = _make_feed()
        assert feed.get_latest_candle("BTCUSDT") is None

    def test_get_latest_candle_returns_most_recent(self):
        feed = _make_feed(["BTCUSDT"])
        c1 = _make_candle(open_time=1_000, close=100.0)
        c2 = _make_candle(open_time=2_000, close=200.0)
        feed._upsert_candle(c1)
        feed._upsert_candle(c2)
        candle = feed.get_latest_candle("BTCUSDT")
        assert candle is not None
        assert candle.close == pytest.approx(200.0)

    def test_get_price_history_empty_initially(self):
        feed = _make_feed()
        assert feed.get_price_history("BTCUSDT") == []

    def test_get_price_history_returns_close_prices_oldest_first(self):
        feed = _make_feed()
        feed._handle_message(
            _kline_envelope(symbol="BTCUSDT", close="45000.00", start=1_000)
        )
        feed._handle_message(
            _kline_envelope(symbol="BTCUSDT", close="45100.00", start=2_000)
        )
        history = feed.get_price_history("BTCUSDT")
        assert history == pytest.approx([45_000.0, 45_100.0])

    def test_get_volume_history_empty_initially(self):
        feed = _make_feed()
        assert feed.get_volume_history("BTCUSDT") == []

    def test_get_volume_history_returns_volumes_oldest_first(self):
        feed = _make_feed()
        feed._handle_message(
            _kline_envelope(symbol="BTCUSDT", volume="100.0", start=1_000)
        )
        feed._handle_message(
            _kline_envelope(symbol="BTCUSDT", volume="200.0", start=2_000)
        )
        vols = feed.get_volume_history("BTCUSDT")
        assert vols == pytest.approx([100.0, 200.0])

    def test_get_candle_history_empty_initially(self):
        feed = _make_feed()
        assert feed.get_candle_history("BTCUSDT") == []

    def test_get_candle_history_returns_candle_tick_list(self):
        feed = _make_feed(["BTCUSDT"])
        c = _make_candle(open_time=1_000, close=99.0)
        feed._upsert_candle(c)
        result = feed.get_candle_history("BTCUSDT")
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], CandleTick)
        assert result[0].close == pytest.approx(99.0)

    def test_get_candle_history_returns_copy_not_deque(self):
        """Mutating the returned list must not corrupt internal state."""
        feed = _make_feed(["BTCUSDT"])
        feed._upsert_candle(_make_candle(open_time=1_000, close=50.0))
        result = feed.get_candle_history("BTCUSDT")
        result.clear()
        assert len(feed._candle_history["BTCUSDT"]) == 1

    def test_tick_emitted_to_queue_on_handle_message(self):
        q = asyncio.Queue()
        feed = BybitFeed(["BTCUSDT"], q)
        feed._handle_message(_kline_envelope(symbol="BTCUSDT", close="50000.00"))
        assert not q.empty()
        tick = q.get_nowait()
        assert isinstance(tick, PriceTick)
        assert tick.price == pytest.approx(50_000.0)

    def test_multiple_symbols_tracked_independently(self):
        feed = _make_feed(["BTCUSDT", "ETHUSDT"])
        feed._handle_message(_kline_envelope(symbol="BTCUSDT", close="45000.00"))
        feed._handle_message(_kline_envelope(symbol="ETHUSDT", close="3000.00"))
        btc = feed.get_latest("BTCUSDT")
        eth = feed.get_latest("ETHUSDT")
        assert btc is not None and btc.price == pytest.approx(45_000.0)
        assert eth is not None and eth.price == pytest.approx(3_000.0)


# ---------------------------------------------------------------------------
# TestBybitFeedWarmUp
# ---------------------------------------------------------------------------

class TestBybitFeedWarmUp:
    @pytest.mark.asyncio
    async def test_rest_response_parsed_and_inserted(self):
        """A successful REST response populates the candle history."""
        feed = _make_feed(["BTCUSDT"])

        # Bybit REST returns rows newest-first; provide two rows.
        rest_payload = {
            "retCode": 0,
            "result": {
                "symbol": "BTCUSDT",
                "category": "linear",
                "list": [
                    # newest first
                    ["1700000060000", "45100", "45200", "45050", "45150", "0.5", "22567"],
                    ["1700000000000", "45000", "45100", "44900", "45100", "1.0", "45100"],
                ],
            },
        }

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=rest_payload)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        await feed._warm_up_symbol("BTCUSDT", mock_session)

        history = feed.get_candle_history("BTCUSDT")
        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_newest_first_list_reversed_correctly(self):
        """After reversal the oldest candle must be first in the deque."""
        feed = _make_feed(["BTCUSDT"])

        rest_payload = {
            "retCode": 0,
            "result": {
                "symbol": "BTCUSDT",
                "category": "linear",
                "list": [
                    # newest-first from Bybit
                    ["1700000060000", "45100", "45200", "45050", "45150", "0.5", "0"],
                    ["1700000000000", "45000", "45100", "44900", "45100", "1.0", "0"],
                ],
            },
        }

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=rest_payload)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        await feed._warm_up_symbol("BTCUSDT", mock_session)

        history = feed.get_candle_history("BTCUSDT")
        # Oldest open_time should come first after reversal.
        assert history[0].open_time == 1_700_000_000_000
        assert history[1].open_time == 1_700_000_060_000
        # Price cache holds the most recent close price.
        assert feed._price_cache["BTCUSDT"] == pytest.approx(45_150.0)

    @pytest.mark.asyncio
    async def test_non_200_response_logs_warning_and_continues(self):
        """HTTP error must not raise — only log a warning."""
        feed = _make_feed(["BTCUSDT"])

        mock_resp = AsyncMock()
        mock_resp.status = 503
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        # Must not raise.
        await feed._warm_up_symbol("BTCUSDT", mock_session)
        assert feed.get_candle_history("BTCUSDT") == []

    @pytest.mark.asyncio
    async def test_network_exception_logs_warning_and_continues(self):
        """Network exception during warm-up must be caught and logged."""
        feed = _make_feed(["BTCUSDT"])

        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=OSError("connection refused"))

        await feed._warm_up_symbol("BTCUSDT", mock_session)
        assert feed.get_candle_history("BTCUSDT") == []

    @pytest.mark.asyncio
    async def test_nonzero_ret_code_logs_warning_and_continues(self):
        """A Bybit API error (retCode != 0) must not raise."""
        feed = _make_feed(["BTCUSDT"])

        rest_payload = {"retCode": 10001, "retMsg": "Invalid symbol", "result": {}}

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=rest_payload)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        await feed._warm_up_symbol("BTCUSDT", mock_session)
        assert feed.get_candle_history("BTCUSDT") == []

    @pytest.mark.asyncio
    async def test_warm_up_calls_all_symbols(self):
        """_warm_up should call _warm_up_symbol for every tracked symbol."""
        feed = _make_feed(["BTCUSDT", "ETHUSDT"])

        called_symbols: list[str] = []

        async def fake_warm_up_symbol(sym: str, session: object) -> None:
            called_symbols.append(sym)

        with patch.object(feed, "_warm_up_symbol", side_effect=fake_warm_up_symbol):
            mock_session = MagicMock()
            await feed._warm_up(mock_session)

        assert sorted(called_symbols) == ["BTCUSDT", "ETHUSDT"]


# ---------------------------------------------------------------------------
# TestBybitFeedStop
# ---------------------------------------------------------------------------

class TestBybitFeedStop:
    @pytest.mark.asyncio
    async def test_stop_sets_running_false(self):
        feed = _make_feed()
        feed._running = True
        await feed.stop()
        assert feed._running is False

    @pytest.mark.asyncio
    async def test_stop_while_already_stopped_does_not_raise(self):
        feed = _make_feed()
        feed._running = False
        await feed.stop()
        assert feed._running is False

    @pytest.mark.asyncio
    async def test_main_loop_exits_after_stop(self, mocker):
        """start() should return cleanly when stop() is called."""
        feed = _make_feed()
        mocker.patch.object(feed, "_warm_up", new_callable=AsyncMock)
        mocker.patch.object(feed, "_run_connection", new_callable=AsyncMock)

        task = asyncio.create_task(feed.start())
        await asyncio.sleep(0)
        await feed.stop()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except asyncio.TimeoutError:
            task.cancel()
            pytest.fail("start() did not exit after stop() was called")
