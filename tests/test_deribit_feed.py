"""Tests for src/market/deribit_feed.py — DeribitVolFeed.

All network calls are mocked; no real WebSocket connections are made.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.market.deribit_feed import DeribitVolFeed
from src.market.indices import IndexTick


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_feed() -> DeribitVolFeed:
    """Return a fresh DeribitVolFeed with no network calls."""
    return DeribitVolFeed()


def _vol_push(
    channel: str = "deribit_volatility_index.btc_usd",
    volatility: float = 75.52,
    timestamp_ms: int = 1_670_608_800_000,
) -> dict[str, Any]:
    """Build a simulated Deribit volatility subscription push message."""
    return {
        "jsonrpc": "2.0",
        "method": "subscription",
        "params": {
            "channel": channel,
            "data": {
                "index_name": channel.split(".", 1)[1] if "." in channel else channel,
                "volatility": volatility,
                "timestamp": timestamp_ms,
            },
        },
    }


def _sub_confirm(channels: list[str] | None = None) -> dict[str, Any]:
    """Build a subscription confirmation response (result key present)."""
    if channels is None:
        channels = [
            "deribit_volatility_index.btc_usd",
            "deribit_volatility_index.eth_usd",
        ]
    return {"jsonrpc": "2.0", "id": 1, "result": channels}


def _heartbeat_test_request() -> dict[str, Any]:
    """Build a Deribit heartbeat test_request message."""
    return {
        "jsonrpc": "2.0",
        "method": "heartbeat",
        "params": {"type": "test_request"},
    }


def _inject_tick(
    feed: DeribitVolFeed,
    symbol: str,
    price: float,
    group: str = "crypto_vol",
) -> IndexTick:
    """Directly place an IndexTick into feed._latest (bypasses network)."""
    tick = IndexTick(
        symbol=symbol,
        name=feed._SYMBOL_NAMES.get(symbol, symbol),
        price=price,
        change=0.0,
        change_pct=0.0,
        timestamp=time.time(),
        group=group,
    )
    feed._latest[symbol] = tick
    return tick


# ---------------------------------------------------------------------------
# TestDeribitVolFeedInit
# ---------------------------------------------------------------------------

class TestDeribitVolFeedInit:
    def test_latest_is_empty_dict(self):
        """_latest starts as an empty dict — no ticks before connection."""
        feed = _make_feed()
        assert feed._latest == {}

    def test_running_is_false(self):
        """Feed is not running before start() is called."""
        feed = _make_feed()
        assert feed._running is False

    def test_symbols_contains_dvol_btc(self):
        """SYMBOLS must include DVOL_BTC."""
        assert "DVOL_BTC" in DeribitVolFeed.SYMBOLS

    def test_symbols_contains_dvol_eth(self):
        """SYMBOLS must include DVOL_ETH."""
        assert "DVOL_ETH" in DeribitVolFeed.SYMBOLS

    def test_channel_map_has_both_currencies(self):
        """_CHANNEL_MAP must cover both BTC and ETH DVOL channels."""
        assert "deribit_volatility_index.btc_usd" in DeribitVolFeed._CHANNEL_MAP
        assert "deribit_volatility_index.eth_usd" in DeribitVolFeed._CHANNEL_MAP


# ---------------------------------------------------------------------------
# TestDeribitVolFeedMessageParsing
# ---------------------------------------------------------------------------

class TestDeribitVolFeedMessageParsing:
    """Unit-level tests driving _handle_message / _process_volatility_update."""

    @pytest.mark.asyncio
    async def test_btc_vol_push_creates_dvol_btc_tick(self):
        """A BTC volatility push creates an IndexTick with symbol DVOL_BTC."""
        feed = _make_feed()
        ws = AsyncMock()
        msg = _vol_push(channel="deribit_volatility_index.btc_usd", volatility=75.52)
        await feed._handle_message(ws, msg)
        tick = feed._latest.get("DVOL_BTC")
        assert tick is not None
        assert tick.symbol == "DVOL_BTC"

    @pytest.mark.asyncio
    async def test_eth_vol_push_creates_dvol_eth_tick(self):
        """An ETH volatility push creates an IndexTick with symbol DVOL_ETH."""
        feed = _make_feed()
        ws = AsyncMock()
        msg = _vol_push(channel="deribit_volatility_index.eth_usd", volatility=62.10)
        await feed._handle_message(ws, msg)
        tick = feed._latest.get("DVOL_ETH")
        assert tick is not None
        assert tick.symbol == "DVOL_ETH"

    @pytest.mark.asyncio
    async def test_price_equals_volatility_field(self):
        """tick.price is exactly the value from data['volatility']."""
        feed = _make_feed()
        ws = AsyncMock()
        msg = _vol_push(volatility=88.0)
        await feed._handle_message(ws, msg)
        assert feed._latest["DVOL_BTC"].price == pytest.approx(88.0)

    @pytest.mark.asyncio
    async def test_group_is_crypto_vol(self):
        """All ticks produced by this feed have group='crypto_vol'."""
        feed = _make_feed()
        ws = AsyncMock()
        await feed._handle_message(ws, _vol_push(channel="deribit_volatility_index.btc_usd"))
        await feed._handle_message(ws, _vol_push(channel="deribit_volatility_index.eth_usd"))
        for sym in ("DVOL_BTC", "DVOL_ETH"):
            assert feed._latest[sym].group == "crypto_vol"

    @pytest.mark.asyncio
    async def test_btc_name_is_btc_dvol(self):
        """BTC tick has human-readable name 'BTC DVOL'."""
        feed = _make_feed()
        ws = AsyncMock()
        await feed._handle_message(ws, _vol_push(channel="deribit_volatility_index.btc_usd"))
        assert feed._latest["DVOL_BTC"].name == "BTC DVOL"

    @pytest.mark.asyncio
    async def test_eth_name_is_eth_dvol(self):
        """ETH tick has human-readable name 'ETH DVOL'."""
        feed = _make_feed()
        ws = AsyncMock()
        await feed._handle_message(ws, _vol_push(channel="deribit_volatility_index.eth_usd"))
        assert feed._latest["DVOL_ETH"].name == "ETH DVOL"

    @pytest.mark.asyncio
    async def test_subscription_confirmation_is_ignored(self):
        """A message with a 'result' key does not modify _latest."""
        feed = _make_feed()
        ws = AsyncMock()
        await feed._handle_message(ws, _sub_confirm())
        assert feed._latest == {}

    @pytest.mark.asyncio
    async def test_heartbeat_test_request_sends_public_test(self):
        """On heartbeat/test_request the feed sends a public/test response."""
        feed = _make_feed()
        ws = AsyncMock()
        await feed._handle_message(ws, _heartbeat_test_request())
        ws.send.assert_awaited_once()
        sent = json.loads(ws.send.call_args[0][0])
        assert sent["method"] == "public/test"

    @pytest.mark.asyncio
    async def test_unrecognised_channel_is_ignored(self):
        """A push on an unknown channel does not update _latest."""
        feed = _make_feed()
        ws = AsyncMock()
        msg = {
            "jsonrpc": "2.0",
            "method": "subscription",
            "params": {
                "channel": "some_unknown_channel.xyz",
                "data": {"volatility": 50.0, "timestamp": 1_670_608_800_000},
            },
        }
        await feed._handle_message(ws, msg)
        assert feed._latest == {}

    @pytest.mark.asyncio
    async def test_change_and_change_pct_first_update_are_zero(self):
        """On the very first update change and change_pct must both be 0.0."""
        feed = _make_feed()
        ws = AsyncMock()
        await feed._handle_message(ws, _vol_push(volatility=80.0))
        tick = feed._latest["DVOL_BTC"]
        assert tick.change == pytest.approx(0.0)
        assert tick.change_pct == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_change_computed_correctly_on_second_update(self):
        """change = new_price - prev_price on the second push."""
        feed = _make_feed()
        ws = AsyncMock()
        # First update: DVOL = 80.0
        await feed._handle_message(ws, _vol_push(volatility=80.0, timestamp_ms=1_000_000_000))
        # Second update: DVOL = 84.0 → change = +4.0
        await feed._handle_message(ws, _vol_push(volatility=84.0, timestamp_ms=1_000_060_000))
        tick = feed._latest["DVOL_BTC"]
        assert tick.change == pytest.approx(4.0)

    @pytest.mark.asyncio
    async def test_change_pct_computed_correctly_on_second_update(self):
        """change_pct = (change / prev_price) * 100 on the second push."""
        feed = _make_feed()
        ws = AsyncMock()
        await feed._handle_message(ws, _vol_push(volatility=80.0, timestamp_ms=1_000_000_000))
        await feed._handle_message(ws, _vol_push(volatility=84.0, timestamp_ms=1_000_060_000))
        tick = feed._latest["DVOL_BTC"]
        # (84 - 80) / 80 * 100 = 5.0
        assert tick.change_pct == pytest.approx(5.0)

    @pytest.mark.asyncio
    async def test_timestamp_converted_from_ms_to_seconds(self):
        """tick.timestamp must be the Deribit ms timestamp divided by 1000."""
        feed = _make_feed()
        ws = AsyncMock()
        ts_ms = 1_670_608_800_000
        await feed._handle_message(ws, _vol_push(timestamp_ms=ts_ms))
        tick = feed._latest["DVOL_BTC"]
        assert tick.timestamp == pytest.approx(ts_ms / 1000.0)


# ---------------------------------------------------------------------------
# TestDeribitVolFeedPublicInterface
# ---------------------------------------------------------------------------

class TestDeribitVolFeedPublicInterface:
    def test_get_latest_returns_none_initially(self):
        """get_latest returns None before any messages have been processed."""
        feed = _make_feed()
        assert feed.get_latest("DVOL_BTC") is None
        assert feed.get_latest("DVOL_ETH") is None

    @pytest.mark.asyncio
    async def test_get_latest_returns_index_tick_after_push(self):
        """get_latest returns an IndexTick after a volatility push is processed."""
        feed = _make_feed()
        ws = AsyncMock()
        await feed._handle_message(ws, _vol_push(volatility=70.0))
        tick = feed.get_latest("DVOL_BTC")
        assert isinstance(tick, IndexTick)
        assert tick.price == pytest.approx(70.0)

    @pytest.mark.asyncio
    async def test_get_all_returns_dict_of_all_tracked_symbols(self):
        """get_all returns a dict containing every symbol that has been updated."""
        feed = _make_feed()
        ws = AsyncMock()
        await feed._handle_message(ws, _vol_push(channel="deribit_volatility_index.btc_usd"))
        await feed._handle_message(ws, _vol_push(channel="deribit_volatility_index.eth_usd"))
        result = feed.get_all()
        assert "DVOL_BTC" in result
        assert "DVOL_ETH" in result

    @pytest.mark.asyncio
    async def test_get_all_returns_copy(self):
        """Mutating the returned dict does not affect internal state."""
        feed = _make_feed()
        ws = AsyncMock()
        await feed._handle_message(ws, _vol_push())
        result = feed.get_all()
        result["DVOL_BTC"] = None  # type: ignore[assignment]
        assert feed._latest.get("DVOL_BTC") is not None

    @pytest.mark.asyncio
    async def test_get_by_group_crypto_vol_returns_dvol_entries(self):
        """get_by_group('crypto_vol') returns exactly the DVOL ticks."""
        feed = _make_feed()
        ws = AsyncMock()
        await feed._handle_message(ws, _vol_push(channel="deribit_volatility_index.btc_usd"))
        await feed._handle_message(ws, _vol_push(channel="deribit_volatility_index.eth_usd"))
        group = feed.get_by_group("crypto_vol")
        assert set(group.keys()) == {"DVOL_BTC", "DVOL_ETH"}

    def test_get_by_group_global_markets_returns_empty(self):
        """get_by_group('global_markets') returns {} — this feed owns only crypto_vol."""
        feed = _make_feed()
        _inject_tick(feed, "DVOL_BTC", 75.0, group="crypto_vol")
        assert feed.get_by_group("global_markets") == {}

    def test_get_by_group_unknown_group_returns_empty(self):
        """An unrecognised group name returns an empty dict."""
        feed = _make_feed()
        _inject_tick(feed, "DVOL_BTC", 75.0)
        assert feed.get_by_group("asx_stocks") == {}


# ---------------------------------------------------------------------------
# TestDeribitVolFeedStop
# ---------------------------------------------------------------------------

class TestDeribitVolFeedStop:
    @pytest.mark.asyncio
    async def test_stop_sets_running_false(self):
        """stop() must flip _running to False."""
        feed = _make_feed()
        feed._running = True
        await feed.stop()
        assert feed._running is False

    @pytest.mark.asyncio
    async def test_stop_idempotent_when_already_stopped(self):
        """Calling stop() on an already-stopped feed does not raise."""
        feed = _make_feed()
        await feed.stop()
        await feed.stop()
        assert feed._running is False


# ---------------------------------------------------------------------------
# TestDeribitVolFeedReconnect
# ---------------------------------------------------------------------------

class TestDeribitVolFeedReconnect:
    @pytest.mark.asyncio
    async def test_connection_error_triggers_retry(self, mocker):
        """When _run_connection raises, start() sleeps and retries.

        We allow exactly two attempts: the first raises ConnectionError, the
        second sets _running=False so the loop terminates.
        """
        feed = _make_feed()
        call_count = 0

        async def fake_run_connection():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("simulated drop")
            # Second call: stop the feed so start() exits cleanly.
            feed._running = False

        mocker.patch.object(feed, "_run_connection", side_effect=fake_run_connection)
        sleep_mock = mocker.patch("src.market.deribit_feed.asyncio.sleep", new_callable=AsyncMock)

        await feed.start()

        assert call_count == 2
        # sleep must have been called at least once for the back-off
        sleep_mock.assert_awaited()

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self, mocker):
        """CancelledError raised inside _run_connection must not be swallowed."""
        feed = _make_feed()

        async def fake_run_connection():
            raise asyncio.CancelledError()

        mocker.patch.object(feed, "_run_connection", side_effect=fake_run_connection)

        with pytest.raises(asyncio.CancelledError):
            await feed.start()

    @pytest.mark.asyncio
    async def test_backoff_increases_on_repeated_failures(self, mocker):
        """Back-off delay doubles on successive failures (1 → 2 → 4…)."""
        feed = _make_feed()
        attempt = 0

        async def fake_run_connection():
            nonlocal attempt
            attempt += 1
            if attempt < 4:
                raise ConnectionError("drop")
            feed._running = False

        mocker.patch.object(feed, "_run_connection", side_effect=fake_run_connection)
        sleep_calls: list[float] = []

        async def fake_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        mocker.patch("src.market.deribit_feed.asyncio.sleep", side_effect=fake_sleep)

        await feed.start()

        # Delays should be increasing: 1.0, 2.0, 4.0
        assert sleep_calls == pytest.approx([1.0, 2.0, 4.0])
