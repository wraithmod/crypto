"""Async Binance WebSocket price feed.

Connects to the Binance combined stream endpoint, parses ticker events, and
puts PriceTick objects into a shared asyncio.Queue for downstream consumers.
Reconnects automatically with exponential backoff on any connection loss.
"""

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass

import websockets
import websockets.exceptions

from config import config

logger = logging.getLogger(__name__)


@dataclass
class PriceTick:
    """A single price snapshot from the exchange."""
    symbol: str
    price: float
    volume: float    # 24h volume
    bid: float
    ask: float
    timestamp: float  # unix timestamp


class MarketFeed:
    """Streams live ticker data from Binance for a list of symbols.

    Usage::

        queue = asyncio.Queue()
        feed = MarketFeed(["BTCUSDT", "ETHUSDT"], queue)
        asyncio.create_task(feed.start())
        tick: PriceTick = await queue.get()
    """

    _BACKOFF_BASE: float = 1.0
    _BACKOFF_MAX: float = 30.0

    def __init__(self, symbols: list[str], price_queue: asyncio.Queue) -> None:
        """Initialise the feed.

        Args:
            symbols: List of Binance trading pairs, e.g. ``["BTCUSDT", "ETHUSDT"]``.
                     Case-insensitive; normalised to upper-case internally.
            price_queue: Shared queue into which :class:`PriceTick` objects are
                         placed on every received tick.
        """
        self._symbols: list[str] = [s.upper() for s in symbols]
        self._queue: asyncio.Queue = price_queue
        self._latest: dict[str, PriceTick] = {}
        self._history: dict[str, deque[float]] = {
            s: deque(maxlen=config.price_history_len) for s in self._symbols
        }
        self._running: bool = False
        self._ws_url: str = self._build_url()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def symbols(self) -> list[str]:
        return self._symbols

    async def start(self) -> None:
        """Connect to Binance WebSocket and stream ticks indefinitely.

        Reconnects with exponential backoff (1 s → 2 s → 4 s … ≤ 30 s) whenever
        the connection drops or an error occurs.  Call :meth:`stop` to exit.
        """
        self._running = True
        backoff: float = self._BACKOFF_BASE
        logger.info("MarketFeed starting for symbols: %s", self._symbols)

        while self._running:
            try:
                await self._run_connection()
                # Clean exit (stop() was called); don't reconnect.
                if not self._running:
                    break
                # Stream ended unexpectedly — reconnect immediately.
                logger.warning("WebSocket stream ended; reconnecting in %.1f s", backoff)
            except websockets.exceptions.ConnectionClosed as exc:
                logger.warning(
                    "WebSocket connection closed (code=%s reason=%r); "
                    "reconnecting in %.1f s",
                    exc.code, exc.reason, backoff,
                )
            except OSError as exc:
                logger.error("Network error: %s; reconnecting in %.1f s", exc, backoff)
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("Unexpected error in MarketFeed: %s; reconnecting in %.1f s", exc, backoff)

            if not self._running:
                break

            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self._BACKOFF_MAX)

        logger.info("MarketFeed stopped.")

    async def stop(self) -> None:
        """Signal the feed to stop after the current reconnect cycle."""
        logger.info("MarketFeed stop requested.")
        self._running = False

    def get_latest(self, symbol: str) -> PriceTick | None:
        """Return the most recently received tick for *symbol*, or ``None``."""
        return self._latest.get(symbol.upper())

    def get_price_history(self, symbol: str) -> list[float]:
        """Return a list of up to ``config.price_history_len`` closing prices.

        The oldest price is first; the most recent is last.
        """
        hist = self._history.get(symbol.upper())
        if hist is None:
            return []
        return list(hist)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_url(self) -> str:
        """Construct the Binance combined-stream URL for all tracked symbols."""
        streams = "/".join(f"{s.lower()}@ticker" for s in self._symbols)
        return f"{config.binance_ws_url}?streams={streams}"

    async def _run_connection(self) -> None:
        """Open a single WebSocket connection and receive messages until closed."""
        logger.info("Connecting to %s", self._ws_url)
        async with websockets.connect(self._ws_url) as ws:
            logger.info("WebSocket connected.")
            async for raw_message in ws:
                if not self._running:
                    break
                self._handle_message(raw_message)

    def _handle_message(self, raw: str) -> None:
        """Parse a raw Binance combined-stream message and dispatch the tick."""
        try:
            envelope: dict = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Received non-JSON message: %.120r", raw)
            return

        # Combined stream format: {"stream": "btcusdt@ticker", "data": {...}}
        data: dict | None = envelope.get("data")
        if data is None:
            logger.debug("Message without 'data' field: %s", envelope)
            return

        try:
            tick = self._parse_ticker(data)
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Failed to parse ticker data (%s): %r", exc, data)
            return

        symbol = tick.symbol
        self._latest[symbol] = tick
        self._history[symbol].append(tick.price)

        try:
            self._queue.put_nowait(tick)
        except asyncio.QueueFull:
            logger.debug("Price queue full; dropping tick for %s", symbol)

    @staticmethod
    def _parse_ticker(data: dict) -> PriceTick:
        """Convert a Binance 24-h ticker payload into a :class:`PriceTick`.

        Relevant fields (from Binance docs):
            ``s`` – symbol
            ``c`` – last / close price
            ``v`` – 24-h volume
            ``b`` – best bid price
            ``a`` – best ask price
        """
        return PriceTick(
            symbol=data["s"].upper(),
            price=float(data["c"]),
            volume=float(data["v"]),
            bid=float(data["b"]),
            ask=float(data["a"]),
            timestamp=time.time(),
        )
