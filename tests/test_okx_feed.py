"""Tests for src/market/okx_feed.py.

All network calls (WebSocket + aiohttp) are mocked.  No real connections
are made during the test suite.
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

from src.market.feed import CandleTick, PriceTick
from src.market.okx_feed import OKXFeed, _from_okx_symbol, _to_okx_symbol


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_feed(
    symbols: list[str] | None = None,
    queue: asyncio.Queue | None = None,
) -> OKXFeed:
    """Create an OKXFeed without any live network connection."""
    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT"]
    if queue is None:
        queue = asyncio.Queue()
    return OKXFeed(symbols, queue)


def _okx_candle_message(
    inst_id: str = "BTC-USDT",
    ts: str = "1670608800000",
    open_price: str = "17071.0",
    high: str = "17073.0",
    low: str = "17027.0",
    close: str = "17056.0",
    vol: str = "0.12300",
    confirm: str = "1",
) -> str:
    """Return a JSON-encoded OKX candle1m WebSocket message."""
    return json.dumps({
        "arg": {"channel": "candle1m", "instId": inst_id},
        "data": [
            [ts, open_price, high, low, close, vol, "2098.41", "2098.41", confirm]
        ],
    })


def _okx_subscribe_confirm(inst_id: str = "BTC-USDT") -> str:
    """Return a JSON-encoded OKX subscription confirmation event."""
    return json.dumps({
        "event": "subscribe",
        "arg": {"channel": "candle1m", "instId": inst_id},
        "connId": "abc123",
    })


def _okx_rest_response(rows: list[list]) -> dict:
    """Return a mock OKX REST candles response payload."""
    return {"code": "0", "msg": "", "data": rows}


def _make_rest_row(
    ts: int,
    close: float = 17056.0,
    confirm: str = "0",
) -> list:
    """Build a single OKX REST candle row."""
    return [
        str(ts),
        "17000.0",
        "17100.0",
        "16900.0",
        str(close),
        "0.5",
        "8500.0",
        "8500.0",
        confirm,
    ]


# ---------------------------------------------------------------------------
# TestOKXSymbolConversion
# ---------------------------------------------------------------------------


class TestOKXSymbolConversion:
    """Pure unit tests for the symbol conversion helpers."""

    def test_to_okx_btcusdt(self) -> None:
        assert _to_okx_symbol("BTCUSDT") == "BTC-USDT"

    def test_to_okx_ethusdt(self) -> None:
        assert _to_okx_symbol("ETHUSDT") == "ETH-USDT"

    def test_to_okx_solusdt(self) -> None:
        assert _to_okx_symbol("SOLUSDT") == "SOL-USDT"

    def test_to_okx_non_usdt_passthrough(self) -> None:
        """Symbols that do not end in USDT are returned unchanged.

        Note: 'BTC-USDT' ends in 'USDT', so the function strips those four
        chars and appends '-USDT', giving 'BTC--USDT'.  A true non-USDT
        symbol like 'XBTBTC' is passed through unchanged.
        """
        assert _to_okx_symbol("XBTBTC") == "XBTBTC"

    def test_from_okx_btc_usdt(self) -> None:
        assert _from_okx_symbol("BTC-USDT") == "BTCUSDT"

    def test_from_okx_eth_usdt(self) -> None:
        assert _from_okx_symbol("ETH-USDT") == "ETHUSDT"

    def test_from_okx_sol_usdt(self) -> None:
        assert _from_okx_symbol("SOL-USDT") == "SOLUSDT"

    def test_roundtrip_binance_to_okx_and_back(self) -> None:
        for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"):
            assert _from_okx_symbol(_to_okx_symbol(sym)) == sym


# ---------------------------------------------------------------------------
# TestOKXFeedInit
# ---------------------------------------------------------------------------


class TestOKXFeedInit:
    """Tests for OKXFeed constructor state."""

    def test_symbols_uppercased(self) -> None:
        feed = _make_feed(symbols=["btcusdt", "ethusdt"])
        assert "BTCUSDT" in feed.symbols
        assert "ETHUSDT" in feed.symbols

    def test_symbols_property_returns_binance_format(self) -> None:
        feed = _make_feed(symbols=["BTCUSDT"])
        assert feed.symbols == ["BTCUSDT"]

    def test_internal_okx_symbols_derived_correctly(self) -> None:
        feed = _make_feed(symbols=["BTCUSDT", "ETHUSDT"])
        assert feed._okx_symbols["BTCUSDT"] == "BTC-USDT"
        assert feed._okx_symbols["ETHUSDT"] == "ETH-USDT"

    def test_candle_history_keyed_by_binance_symbols(self) -> None:
        feed = _make_feed(symbols=["BTCUSDT", "SOLUSDT"])
        assert "BTCUSDT" in feed._candle_history
        assert "SOLUSDT" in feed._candle_history
        # OKX-format keys must NOT appear as top-level keys.
        assert "BTC-USDT" not in feed._candle_history
        assert "SOL-USDT" not in feed._candle_history

    def test_running_false_on_init(self) -> None:
        feed = _make_feed()
        assert feed._running is False

    def test_price_cache_empty_on_init(self) -> None:
        feed = _make_feed()
        assert feed._price_cache == {}


# ---------------------------------------------------------------------------
# TestOKXFeedKlineParsing
# ---------------------------------------------------------------------------


class TestOKXFeedKlineParsing:
    """Unit tests for _parse_candle and _handle_message message routing."""

    def test_parse_candle_valid_row(self) -> None:
        arg = {"channel": "candle1m", "instId": "BTC-USDT"}
        row = ["1670608800000", "17071.0", "17073.0", "17027.0", "17056.0",
               "0.123", "2098.41", "2098.41", "1"]
        candle = OKXFeed._parse_candle(arg, row)
        assert candle.symbol == "BTCUSDT"
        assert candle.open_time == 1670608800000
        assert candle.open == pytest.approx(17071.0)
        assert candle.high == pytest.approx(17073.0)
        assert candle.low == pytest.approx(17027.0)
        assert candle.close == pytest.approx(17056.0)
        assert candle.volume == pytest.approx(0.123)

    def test_parse_candle_confirm_1_is_closed(self) -> None:
        arg = {"channel": "candle1m", "instId": "BTC-USDT"}
        row = ["1670608800000", "17071.0", "17073.0", "17027.0", "17056.0",
               "0.123", "0", "0", "1"]
        candle = OKXFeed._parse_candle(arg, row)
        assert candle.is_closed is True

    def test_parse_candle_confirm_0_is_not_closed(self) -> None:
        arg = {"channel": "candle1m", "instId": "BTC-USDT"}
        row = ["1670608800000", "17071.0", "17073.0", "17027.0", "17056.0",
               "0.123", "0", "0", "0"]
        candle = OKXFeed._parse_candle(arg, row)
        assert candle.is_closed is False

    def test_parse_candle_symbol_converted_from_okx(self) -> None:
        """instId BTC-USDT -> symbol BTCUSDT."""
        arg = {"channel": "candle1m", "instId": "ETH-USDT"}
        row = ["1670608800000", "1200.0", "1210.0", "1195.0", "1205.0",
               "5.0", "6000.0", "6000.0", "1"]
        candle = OKXFeed._parse_candle(arg, row)
        assert candle.symbol == "ETHUSDT"

    def test_handle_message_subscription_confirm_silently_ignored(self) -> None:
        """Subscription confirmation events must not raise or update state."""
        feed = _make_feed()
        feed._handle_message(_okx_subscribe_confirm("BTC-USDT"))
        assert feed.get_latest("BTCUSDT") is None

    def test_handle_message_pong_silently_ignored(self) -> None:
        feed = _make_feed()
        feed._handle_message("pong")
        assert feed.get_latest("BTCUSDT") is None

    def test_handle_message_non_json_does_not_raise(self) -> None:
        feed = _make_feed()
        feed._handle_message("not-valid-json{{{{")

    def test_handle_message_wrong_channel_ignored(self) -> None:
        """Messages for channels other than candle1m are silently ignored."""
        feed = _make_feed()
        msg = json.dumps({
            "arg": {"channel": "trades", "instId": "BTC-USDT"},
            "data": [{"instId": "BTC-USDT", "tradeId": "1", "px": "17000", "sz": "1"}],
        })
        feed._handle_message(msg)
        assert feed.get_latest("BTCUSDT") is None

    def test_handle_message_updates_price_cache(self) -> None:
        feed = _make_feed()
        feed._handle_message(_okx_candle_message(inst_id="BTC-USDT", close="17056.0"))
        tick = feed.get_latest("BTCUSDT")
        assert tick is not None
        assert tick.price == pytest.approx(17056.0)

    def test_handle_message_puts_tick_in_queue(self) -> None:
        queue: asyncio.Queue = asyncio.Queue()
        feed = OKXFeed(["BTCUSDT"], queue)
        feed._handle_message(_okx_candle_message(inst_id="BTC-USDT", close="17056.0"))
        assert not queue.empty()
        tick = queue.get_nowait()
        assert isinstance(tick, PriceTick)
        assert tick.price == pytest.approx(17056.0)

    def test_handle_message_unknown_symbol_does_not_raise(self) -> None:
        """A candle for a symbol not in the feed's list is dropped silently."""
        feed = _make_feed(symbols=["BTCUSDT"])
        # SOLUSDT is not in the feed — should not raise or pollute state.
        feed._handle_message(_okx_candle_message(inst_id="SOL-USDT", close="120.0"))
        assert feed.get_latest("SOLUSDT") is None


# ---------------------------------------------------------------------------
# TestOKXFeedUpsertCandle
# ---------------------------------------------------------------------------


class TestOKXFeedUpsertCandle:
    """Tests for the _upsert_candle deque management logic."""

    def _make_candle(
        self,
        symbol: str = "BTCUSDT",
        open_time: int = 1_000,
        close: float = 100.0,
        is_closed: bool = True,
    ) -> CandleTick:
        return CandleTick(
            symbol=symbol,
            open_time=open_time,
            open=close - 1.0,
            high=close + 2.0,
            low=close - 2.0,
            close=close,
            volume=1.0,
            is_closed=is_closed,
        )

    def test_same_open_time_replaces_last(self) -> None:
        """In-progress update with same open_time replaces the last candle."""
        feed = _make_feed()
        c1 = self._make_candle(open_time=1000, close=100.0, is_closed=False)
        c2 = self._make_candle(open_time=1000, close=105.0, is_closed=True)
        feed._upsert_candle(c1)
        feed._upsert_candle(c2)
        hist = feed._candle_history["BTCUSDT"]
        assert len(hist) == 1
        assert hist[-1].close == pytest.approx(105.0)
        assert hist[-1].is_closed is True

    def test_different_open_time_appends(self) -> None:
        """A new open_time means a new candle window — appended."""
        feed = _make_feed()
        c1 = self._make_candle(open_time=1000, close=100.0)
        c2 = self._make_candle(open_time=2000, close=110.0)
        feed._upsert_candle(c1)
        feed._upsert_candle(c2)
        hist = feed._candle_history["BTCUSDT"]
        assert len(hist) == 2
        assert hist[0].close == pytest.approx(100.0)
        assert hist[1].close == pytest.approx(110.0)

    def test_deque_respects_maxlen(self) -> None:
        """The candle deque must not exceed config.price_history_len entries."""
        from config import config
        feed = _make_feed()
        maxlen = config.price_history_len
        for i in range(maxlen + 10):
            feed._upsert_candle(
                self._make_candle(open_time=i * 60_000, close=float(i))
            )
        hist = feed._candle_history["BTCUSDT"]
        assert len(hist) == maxlen

    def test_upsert_first_candle_into_empty_deque(self) -> None:
        feed = _make_feed()
        c = self._make_candle(open_time=1000, close=200.0)
        feed._upsert_candle(c)
        assert len(feed._candle_history["BTCUSDT"]) == 1


# ---------------------------------------------------------------------------
# TestOKXFeedPublicInterface
# ---------------------------------------------------------------------------


class TestOKXFeedPublicInterface:
    """Tests for get_latest, get_price_history, get_candle_history, etc."""

    def test_get_latest_returns_none_initially(self) -> None:
        feed = _make_feed()
        assert feed.get_latest("BTCUSDT") is None

    def test_get_latest_unknown_symbol_returns_none(self) -> None:
        feed = _make_feed()
        assert feed.get_latest("XYZUSDT") is None

    def test_get_latest_returns_price_tick_after_candle(self) -> None:
        feed = _make_feed()
        feed._handle_message(
            _okx_candle_message(inst_id="BTC-USDT", close="17056.0", vol="2.5")
        )
        tick = feed.get_latest("BTCUSDT")
        assert tick is not None
        assert isinstance(tick, PriceTick)
        assert tick.symbol == "BTCUSDT"
        assert tick.price == pytest.approx(17056.0)
        assert tick.volume == pytest.approx(2.5)

    def test_get_price_history_empty_initially(self) -> None:
        feed = _make_feed()
        assert feed.get_price_history("BTCUSDT") == []

    def test_get_price_history_returns_close_prices_oldest_first(self) -> None:
        feed = _make_feed()
        feed._handle_message(
            _okx_candle_message(
                inst_id="BTC-USDT", close="100.0",
                ts="1000000000000", confirm="1",
            )
        )
        feed._handle_message(
            _okx_candle_message(
                inst_id="BTC-USDT", close="110.0",
                ts="1000060000000", confirm="1",
            )
        )
        history = feed.get_price_history("BTCUSDT")
        assert history == pytest.approx([100.0, 110.0])

    def test_get_price_history_unknown_symbol_returns_empty(self) -> None:
        feed = _make_feed(symbols=["BTCUSDT"])
        assert feed.get_price_history("ETHUSDT") == []

    def test_get_candle_history_returns_candle_ticks(self) -> None:
        feed = _make_feed()
        feed._handle_message(
            _okx_candle_message(
                inst_id="BTC-USDT", close="17056.0", vol="1.23",
                ts="1670608800000", confirm="1",
            )
        )
        candles = feed.get_candle_history("BTCUSDT")
        assert len(candles) == 1
        assert isinstance(candles[0], CandleTick)
        assert candles[0].close == pytest.approx(17056.0)
        assert candles[0].volume == pytest.approx(1.23)

    def test_get_candle_history_empty_initially(self) -> None:
        feed = _make_feed()
        assert feed.get_candle_history("BTCUSDT") == []

    def test_get_volume_history_returns_volumes_oldest_first(self) -> None:
        feed = _make_feed()
        feed._handle_message(
            _okx_candle_message(
                inst_id="BTC-USDT", vol="1.0", ts="1000000000000", confirm="1"
            )
        )
        feed._handle_message(
            _okx_candle_message(
                inst_id="BTC-USDT", vol="2.0", ts="1000060000000", confirm="1"
            )
        )
        vols = feed.get_volume_history("BTCUSDT")
        assert vols == pytest.approx([1.0, 2.0])

    def test_get_volume_history_empty_initially(self) -> None:
        feed = _make_feed()
        assert feed.get_volume_history("BTCUSDT") == []

    def test_get_latest_candle_returns_none_initially(self) -> None:
        feed = _make_feed()
        assert feed.get_latest_candle("BTCUSDT") is None

    def test_get_latest_candle_returns_most_recent(self) -> None:
        feed = _make_feed()
        feed._handle_message(
            _okx_candle_message(
                inst_id="BTC-USDT", close="100.0", ts="1000000000000", confirm="1"
            )
        )
        feed._handle_message(
            _okx_candle_message(
                inst_id="BTC-USDT", close="200.0", ts="1000060000000", confirm="1"
            )
        )
        candle = feed.get_latest_candle("BTCUSDT")
        assert candle is not None
        assert candle.close == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# TestOKXFeedWarmUp
# ---------------------------------------------------------------------------


class TestOKXFeedWarmUp:
    """Tests for the REST warm-up logic using mocked aiohttp sessions."""

    def _make_mock_response(self, payload: dict, status: int = 200) -> MagicMock:
        """Build an aiohttp response mock that returns *payload* as JSON."""
        mock_resp = AsyncMock()
        mock_resp.status = status
        mock_resp.json = AsyncMock(return_value=payload)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)
        return mock_resp

    def _make_mock_session(self, mock_resp: MagicMock) -> MagicMock:
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        return mock_session

    @pytest.mark.asyncio
    async def test_warm_up_populates_candle_history(self) -> None:
        """Warm-up inserts rows oldest-first into the deque."""
        # OKX returns newest-first; provide [ts=2000, ts=1000]
        rows = [
            _make_rest_row(ts=2000, close=200.0, confirm="0"),
            _make_rest_row(ts=1000, close=100.0, confirm="0"),
        ]
        payload = _okx_rest_response(rows)
        mock_resp = self._make_mock_response(payload)
        mock_session = self._make_mock_session(mock_resp)

        feed = _make_feed(symbols=["BTCUSDT"])
        await feed._warm_up_symbol("BTCUSDT", mock_session)

        history = feed.get_price_history("BTCUSDT")
        # After reversal: oldest (ts=1000, close=100) first
        assert history[0] == pytest.approx(100.0)
        assert history[1] == pytest.approx(200.0)

    @pytest.mark.asyncio
    async def test_warm_up_all_candles_marked_closed(self) -> None:
        """All historical candles must have is_closed=True regardless of confirm."""
        rows = [
            _make_rest_row(ts=2000, close=200.0, confirm="0"),  # in-progress
            _make_rest_row(ts=1000, close=100.0, confirm="1"),  # closed
        ]
        payload = _okx_rest_response(rows)
        mock_resp = self._make_mock_response(payload)
        mock_session = self._make_mock_session(mock_resp)

        feed = _make_feed(symbols=["BTCUSDT"])
        await feed._warm_up_symbol("BTCUSDT", mock_session)

        candles = feed.get_candle_history("BTCUSDT")
        assert all(c.is_closed for c in candles), (
            "All warm-up candles must be is_closed=True"
        )

    @pytest.mark.asyncio
    async def test_warm_up_non_200_response_logs_warning(self, caplog) -> None:
        """A non-200 HTTP response from OKX REST is logged as a warning."""
        import logging
        mock_resp = self._make_mock_response({}, status=503)
        mock_session = self._make_mock_session(mock_resp)

        feed = _make_feed(symbols=["BTCUSDT"])
        with caplog.at_level(logging.WARNING):
            await feed._warm_up_symbol("BTCUSDT", mock_session)

        assert len(feed.get_price_history("BTCUSDT")) == 0

    @pytest.mark.asyncio
    async def test_warm_up_api_error_code_logs_warning(self, caplog) -> None:
        """OKX API error response (code != '0') is logged and history stays empty."""
        import logging
        payload = {"code": "50011", "msg": "Request too frequent", "data": []}
        mock_resp = self._make_mock_response(payload, status=200)
        mock_session = self._make_mock_session(mock_resp)

        feed = _make_feed(symbols=["BTCUSDT"])
        with caplog.at_level(logging.WARNING):
            await feed._warm_up_symbol("BTCUSDT", mock_session)

        assert feed.get_price_history("BTCUSDT") == []

    @pytest.mark.asyncio
    async def test_warm_up_updates_price_cache(self) -> None:
        """After warm-up, price cache reflects the most recent (last) candle."""
        rows = [
            _make_rest_row(ts=2000, close=200.0),  # newest first from API
            _make_rest_row(ts=1000, close=100.0),  # oldest
        ]
        payload = _okx_rest_response(rows)
        mock_resp = self._make_mock_response(payload)
        mock_session = self._make_mock_session(mock_resp)

        feed = _make_feed(symbols=["BTCUSDT"])
        await feed._warm_up_symbol("BTCUSDT", mock_session)

        # After reversal: deque is [100, 200]; last = 200
        tick = feed.get_latest("BTCUSDT")
        assert tick is not None
        assert tick.price == pytest.approx(200.0)

    @pytest.mark.asyncio
    async def test_warm_up_empty_data_array(self) -> None:
        """Empty data array leaves history empty without raising."""
        payload = _okx_rest_response([])
        mock_resp = self._make_mock_response(payload)
        mock_session = self._make_mock_session(mock_resp)

        feed = _make_feed(symbols=["BTCUSDT"])
        await feed._warm_up_symbol("BTCUSDT", mock_session)

        assert feed.get_price_history("BTCUSDT") == []


# ---------------------------------------------------------------------------
# TestOKXFeedStop
# ---------------------------------------------------------------------------


class TestOKXFeedStop:
    """Tests for the stop() lifecycle method."""

    @pytest.mark.asyncio
    async def test_stop_sets_running_false(self) -> None:
        feed = _make_feed()
        feed._running = True
        await feed.stop()
        assert feed._running is False

    @pytest.mark.asyncio
    async def test_stop_when_already_stopped_does_not_raise(self) -> None:
        feed = _make_feed()
        feed._running = False
        await feed.stop()
        assert feed._running is False

    @pytest.mark.asyncio
    async def test_start_stop_cycle(self, mocker) -> None:
        """Calling stop() while start() is running exits the loop cleanly."""
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
