"""Async Bybit WebSocket price feed — kline (OHLCV) stream with REST warm-up.

Connects to the Bybit v5 linear perpetual kline stream (1-minute candles),
parses candle events, and puts PriceTick objects into a shared asyncio.Queue
for downstream consumers.

On startup, a REST warm-up is performed first: it fetches up to 200 historical
1-minute candles per symbol via the Bybit v5 REST API so that technical
indicators (RSI/MACD/Bollinger/ADX) can be computed immediately without
waiting for live data to accumulate.

A periodic ping task (every 20 s) keeps the Bybit WebSocket connection alive.

Reconnects automatically with exponential backoff on any connection loss.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from typing import Any

import aiohttp
import websockets
import websockets.exceptions

from config import config
from src.market.feed import CandleTick, PriceTick

logger = logging.getLogger(__name__)

_BYBIT_REST_BASE: str = "https://api.bybit.com"
_BYBIT_WS_URL: str = "wss://stream.bybit.com/v5/public/linear"
_PING_INTERVAL: float = 20.0


class BybitFeed:
    """Streams live 1-minute kline data from Bybit for a list of symbols.

    Replicates the full public interface of :class:`src.market.feed.MarketFeed`
    so that downstream components (dashboard, engine, predictor) can use either
    feed interchangeably.

    Usage::

        queue = asyncio.Queue()
        feed = BybitFeed(["BTCUSDT", "ETHUSDT"], queue)
        asyncio.create_task(feed.start())   # warm-up + WebSocket loop
        tick: PriceTick = await queue.get()
    """

    _BACKOFF_BASE: float = 1.0
    _BACKOFF_MAX: float = 30.0

    def __init__(self, symbols: list[str], price_queue: asyncio.Queue) -> None:
        self._symbols: list[str] = [s.upper() for s in symbols]
        self._queue: asyncio.Queue = price_queue

        # OHLCV candle history — ordered oldest first, most recent last.
        self._candle_history: dict[str, deque[CandleTick]] = {
            s: deque(maxlen=config.price_history_len) for s in self._symbols
        }
        # Fast-access latest close price per symbol.
        self._price_cache: dict[str, float] = {}

        self._running: bool = False

    # ------------------------------------------------------------------
    # Public interface (mirrors MarketFeed exactly)
    # ------------------------------------------------------------------

    @property
    def symbols(self) -> list[str]:
        """Return the list of tracked symbols (uppercase)."""
        return self._symbols

    async def start(self) -> None:
        """Warm up from REST, then stream live klines until stopped."""
        self._running = True
        logger.info("BybitFeed starting for symbols: %s", self._symbols)

        try:
            async with aiohttp.ClientSession() as session:
                await self._warm_up(session)
        except Exception as exc:
            logger.warning("REST warm-up failed (will fill from live stream): %s", exc)

        backoff: float = self._BACKOFF_BASE
        while self._running:
            try:
                await self._run_connection()
                if not self._running:
                    break
                logger.warning(
                    "WebSocket stream ended; reconnecting in %.1f s", backoff
                )
            except websockets.exceptions.ConnectionClosed as exc:
                logger.warning(
                    "WebSocket closed (code=%s reason=%r); reconnecting in %.1f s",
                    exc.code,
                    exc.reason,
                    backoff,
                )
            except OSError as exc:
                logger.error(
                    "Network error: %s; reconnecting in %.1f s", exc, backoff
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception(
                    "Unexpected error in BybitFeed: %s; reconnecting in %.1f s",
                    exc,
                    backoff,
                )

            if not self._running:
                break

            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self._BACKOFF_MAX)

        logger.info("BybitFeed stopped.")

    async def stop(self) -> None:
        """Signal the feed to stop after the current reconnect cycle."""
        logger.info("BybitFeed stop requested.")
        self._running = False

    def get_latest(self, symbol: str) -> PriceTick | None:
        """Return the most recent price as a PriceTick, or None if unavailable."""
        sym = symbol.upper()
        price = self._price_cache.get(sym)
        if price is None:
            return None
        hist = self._candle_history.get(sym)
        vol = hist[-1].volume if hist else 0.0
        return PriceTick(symbol=sym, price=price, volume=vol)

    def get_latest_candle(self, symbol: str) -> CandleTick | None:
        """Return the most recent CandleTick for *symbol*, or None."""
        hist = self._candle_history.get(symbol.upper())
        if not hist:
            return None
        return hist[-1]

    def get_price_history(self, symbol: str) -> list[float]:
        """Return close prices oldest-first as a plain list."""
        hist = self._candle_history.get(symbol.upper())
        if not hist:
            return []
        return [c.close for c in hist]

    def get_volume_history(self, symbol: str) -> list[float]:
        """Return base-asset volumes per candle, oldest-first."""
        hist = self._candle_history.get(symbol.upper())
        if not hist:
            return []
        return [c.volume for c in hist]

    def get_candle_history(self, symbol: str) -> list[CandleTick]:
        """Return the full CandleTick history oldest-first."""
        hist = self._candle_history.get(symbol.upper())
        if not hist:
            return []
        return list(hist)

    # ------------------------------------------------------------------
    # REST warm-up
    # ------------------------------------------------------------------

    async def _warm_up(self, session: aiohttp.ClientSession) -> None:
        """Fetch historical candles for all symbols in parallel."""
        logger.info(
            "BybitFeed: starting REST warm-up for %d symbols", len(self._symbols)
        )
        await asyncio.gather(
            *[self._warm_up_symbol(s, session) for s in self._symbols],
            return_exceptions=True,
        )
        logger.info(
            "BybitFeed: REST warm-up complete — history lengths: %s",
            {s: len(self._candle_history[s]) for s in self._symbols},
        )

    async def _warm_up_symbol(self, symbol: str, session: aiohttp.ClientSession) -> None:
        """Fetch historical klines for one symbol and pre-populate its history."""
        url = (
            f"{_BYBIT_REST_BASE}/v5/market/kline"
            f"?category=linear&symbol={symbol}&interval=1&limit={config.price_history_len}"
        )
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    logger.warning(
                        "warm_up: HTTP %d fetching klines for %s", resp.status, symbol
                    )
                    return
                payload: dict[str, Any] = await resp.json()
        except Exception as exc:
            logger.warning("warm_up: failed to fetch klines for %s: %s", symbol, exc)
            return

        ret_code = payload.get("retCode", -1)
        if ret_code != 0:
            logger.warning(
                "warm_up: Bybit retCode=%s for %s (msg=%s)",
                ret_code,
                symbol,
                payload.get("retMsg", ""),
            )
            return

        result = payload.get("result", {})
        rows: list[list[str]] = result.get("list", [])

        # Bybit returns newest-first — reverse so we insert oldest-first.
        for row in reversed(rows):
            # Row: [startTime(ms), open, high, low, close, volume, turnover]
            try:
                candle = CandleTick(
                    symbol=symbol,
                    open_time=int(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                    is_closed=True,
                )
                self._candle_history[symbol].append(candle)
            except (IndexError, ValueError, TypeError) as exc:
                logger.debug("warm_up: bad kline row for %s: %s", symbol, exc)

        if self._candle_history[symbol]:
            self._price_cache[symbol] = self._candle_history[symbol][-1].close
            logger.debug(
                "warm_up: %s pre-populated with %d candles (latest close=%.4f)",
                symbol,
                len(self._candle_history[symbol]),
                self._price_cache[symbol],
            )

    # ------------------------------------------------------------------
    # WebSocket connection
    # ------------------------------------------------------------------

    async def _run_connection(self) -> None:
        """Open a single WebSocket connection and receive messages until closed."""
        logger.info("BybitFeed: connecting to %s", _BYBIT_WS_URL)
        async with websockets.connect(_BYBIT_WS_URL) as ws:
            logger.info("BybitFeed: WebSocket connected.")

            # Send the subscription request for all symbols.
            sub_args = [f"kline.1.{sym}" for sym in self._symbols]
            sub_msg = json.dumps({"op": "subscribe", "args": sub_args})
            await ws.send(sub_msg)
            logger.debug("BybitFeed: subscribed to %s", sub_args)

            # Start background ping task to keep the connection alive.
            ping_task = asyncio.create_task(self._ping_loop(ws))
            try:
                async for raw_message in ws:
                    if not self._running:
                        break
                    self._handle_message(raw_message)
            except asyncio.CancelledError:
                raise
            finally:
                ping_task.cancel()
                try:
                    await ping_task
                except (asyncio.CancelledError, Exception):
                    pass

    async def _ping_loop(self, ws: Any) -> None:
        """Send a Bybit ping every 20 seconds to keep the connection alive."""
        ping_msg = json.dumps({"op": "ping"})
        while True:
            try:
                await asyncio.sleep(_PING_INTERVAL)
                await ws.send(ping_msg)
                logger.debug("BybitFeed: ping sent")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("BybitFeed: ping error: %s", exc)
                break

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    def _handle_message(self, raw: str) -> None:
        """Parse a Bybit kline stream message and update candle history."""
        try:
            envelope: dict = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("BybitFeed: received non-JSON message: %.120r", raw)
            return

        # Subscription confirmations and ping responses do not have a "topic" key.
        if "topic" not in envelope:
            logger.debug("BybitFeed: non-kline message (no topic): %s", envelope)
            return

        topic: str = envelope.get("topic", "")
        if not topic.startswith("kline."):
            logger.debug("BybitFeed: unexpected topic %r — ignoring", topic)
            return

        data_list: list[dict] = envelope.get("data", [])
        if not data_list:
            logger.debug("BybitFeed: empty data list in topic %r", topic)
            return

        try:
            candle = self._parse_kline(topic, data_list[0])
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            logger.warning(
                "BybitFeed: failed to parse kline (topic=%r, error=%s)", topic, exc
            )
            return

        self._upsert_candle(candle)

        # Update the price cache and emit a PriceTick for downstream consumers.
        self._price_cache[candle.symbol] = candle.close
        tick = PriceTick(
            symbol=candle.symbol,
            price=candle.close,
            volume=candle.volume,
        )
        try:
            self._queue.put_nowait(tick)
        except asyncio.QueueFull:
            logger.debug(
                "BybitFeed: price queue full; dropping tick for %s", candle.symbol
            )

    def _upsert_candle(self, candle: CandleTick) -> None:
        """Insert or update the candle in history.

        In-progress candles (is_closed=False) replace the last element when
        they share the same open_time (same 1-minute window).  Closed candles
        finalise the window and become a new entry.
        """
        hist = self._candle_history.get(candle.symbol)
        if hist is None:
            # Symbol not tracked — ignore.
            return
        if hist and hist[-1].open_time == candle.open_time:
            hist[-1] = candle
        else:
            hist.append(candle)

    @staticmethod
    def _parse_kline(topic: str, data: dict) -> CandleTick:
        """Convert a Bybit kline data payload into a CandleTick.

        Args:
            topic: The Bybit topic string, e.g. ``"kline.1.BTCUSDT"``.
            data:  The first element of the ``"data"`` list from the message.

        Bybit kline data fields used:
            ``start``   — kline open time in Unix ms
            ``open``    — open price (string)
            ``high``    — high price (string)
            ``low``     — low price (string)
            ``close``   — close price (string)
            ``volume``  — base-asset volume (string)
            ``confirm`` — True when the candle is finalised (closed)
        """
        # Symbol is the third component of "kline.1.BTCUSDT"
        symbol = topic.split(".")[2].upper()
        return CandleTick(
            symbol=symbol,
            open_time=int(data["start"]),
            open=float(data["open"]),
            high=float(data["high"]),
            low=float(data["low"]),
            close=float(data["close"]),
            volume=float(data["volume"]),
            is_closed=bool(data["confirm"]),
        )
