"""Async Binance WebSocket price feed — kline (OHLCV) stream with REST warm-up.

Connects to the Binance combined kline_1m stream, parses candle events, and
puts PriceTick objects into a shared asyncio.Queue for downstream consumers.

On startup, warm_up() is called first: it fetches 200 historical 1-minute
candles per symbol via the Binance REST API so that technical indicators
(RSI/MACD/Bollinger/ADX) can be computed immediately without waiting for
200 live candles to accumulate.

Reconnects automatically with exponential backoff on any connection loss.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field

import aiohttp
import websockets
import websockets.exceptions

from config import config

logger = logging.getLogger(__name__)

_BINANCE_REST = "https://api.binance.com"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CandleTick:
    """One 1-minute OHLCV candle from the Binance kline stream."""
    symbol: str
    open_time: int       # Unix ms (candle start)
    open: float
    high: float
    low: float
    close: float
    volume: float        # base asset volume (e.g. BTC, not USDT)
    is_closed: bool      # True when this candle is finalised

    @property
    def typical_price(self) -> float:
        """(high + low + close) / 3 — used for VWAP."""
        return (self.high + self.low + self.close) / 3


@dataclass
class PriceTick:
    """Lightweight price snapshot, backward-compatible with older consumers."""
    symbol: str
    price: float         # = CandleTick.close
    volume: float        # base asset volume of this candle
    bid: float = 0.0     # not available from kline stream
    ask: float = 0.0     # not available from kline stream
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Market feed
# ---------------------------------------------------------------------------

class MarketFeed:
    """Streams live 1-minute kline data from Binance for a list of symbols.

    Usage::

        queue = asyncio.Queue()
        feed = MarketFeed(["BTCUSDT", "ETHUSDT"], queue)
        asyncio.create_task(feed.start())   # warm-up + WebSocket loop
        tick: PriceTick = await queue.get()
    """

    _BACKOFF_BASE: float = 1.0
    _BACKOFF_MAX: float = 30.0

    def __init__(self, symbols: list[str], price_queue: asyncio.Queue) -> None:
        self._symbols: list[str] = [s.upper() for s in symbols]
        self._queue: asyncio.Queue = price_queue

        # OHLCV candle history — ordered oldest first, most recent last
        self._candle_history: dict[str, deque[CandleTick]] = {
            s: deque(maxlen=config.price_history_len) for s in self._symbols
        }
        # Fast-access latest closed-candle close price per symbol
        self._price_cache: dict[str, float] = {}

        self._running: bool = False
        self._ws_url: str = self._build_url()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def symbols(self) -> list[str]:
        return self._symbols

    async def start(self) -> None:
        """Warm up from REST, then stream live klines until stopped."""
        self._running = True
        logger.info("MarketFeed starting for symbols: %s", self._symbols)

        # Pre-populate history before opening the WebSocket connection.
        try:
            async with aiohttp.ClientSession() as session:
                await self.warm_up(session)
        except Exception as exc:
            logger.warning("REST warm-up failed (will fill from live stream): %s", exc)

        backoff: float = self._BACKOFF_BASE
        while self._running:
            try:
                await self._run_connection()
                if not self._running:
                    break
                logger.warning("WebSocket stream ended; reconnecting in %.1f s", backoff)
            except websockets.exceptions.ConnectionClosed as exc:
                logger.warning(
                    "WebSocket closed (code=%s reason=%r); reconnecting in %.1f s",
                    exc.code, exc.reason, backoff,
                )
            except OSError as exc:
                logger.error("Network error: %s; reconnecting in %.1f s", exc, backoff)
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception(
                    "Unexpected error in MarketFeed: %s; reconnecting in %.1f s", exc, backoff
                )

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
        """Return the most recent price as a PriceTick (backward-compatible)."""
        sym = symbol.upper()
        price = self._price_cache.get(sym)
        if price is None:
            return None
        hist = self._candle_history.get(sym)
        vol = hist[-1].volume if hist else 0.0
        return PriceTick(symbol=sym, price=price, volume=vol)

    def get_latest_candle(self, symbol: str) -> CandleTick | None:
        """Return the most recent CandleTick for *symbol*."""
        hist = self._candle_history.get(symbol.upper())
        if not hist:
            return None
        return hist[-1]

    def get_price_history(self, symbol: str) -> list[float]:
        """Return close prices oldest-first (backward-compatible)."""
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

    async def warm_up(self, session: aiohttp.ClientSession) -> None:
        """Fetch 200 historical 1-minute candles per symbol in parallel."""
        logger.info("MarketFeed: starting REST warm-up for %d symbols", len(self._symbols))
        await asyncio.gather(
            *[self._warm_up_symbol(s, session) for s in self._symbols],
            return_exceptions=True,
        )
        logger.info(
            "MarketFeed: REST warm-up complete — history lengths: %s",
            {s: len(self._candle_history[s]) for s in self._symbols},
        )

    async def _warm_up_symbol(self, symbol: str, session: aiohttp.ClientSession) -> None:
        """Fetch historical klines for one symbol and pre-populate its history."""
        url = (
            f"{_BINANCE_REST}/api/v3/klines"
            f"?symbol={symbol}&interval=1m&limit={config.price_history_len}"
        )
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                resp.raise_for_status()
                rows = await resp.json()
        except Exception as exc:
            logger.warning("warm_up: failed to fetch klines for %s: %s", symbol, exc)
            return

        for row in rows:
            # Row format: [openTime, open, high, low, close, volume, closeTime, ...]
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

    def _build_url(self) -> str:
        """Construct the Binance combined-stream URL (1m klines for all symbols)."""
        streams = "/".join(f"{s.lower()}@kline_1m" for s in self._symbols)
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
        """Parse a Binance kline stream message and update candle history."""
        try:
            envelope: dict = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Received non-JSON message: %.120r", raw)
            return

        data: dict | None = envelope.get("data")
        if data is None:
            logger.debug("Message without 'data' field: %s", envelope)
            return

        if data.get("e") != "kline":
            return

        try:
            candle = self._parse_kline(data)
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Failed to parse kline data (%s): %r", exc, data)
            return

        self._upsert_candle(candle)

        # Emit a PriceTick whenever the close price changes, for downstream consumers.
        self._price_cache[candle.symbol] = candle.close
        tick = PriceTick(
            symbol=candle.symbol,
            price=candle.close,
            volume=candle.volume,
        )
        try:
            self._queue.put_nowait(tick)
        except asyncio.QueueFull:
            logger.debug("Price queue full; dropping tick for %s", candle.symbol)

    def _upsert_candle(self, candle: CandleTick) -> None:
        """Insert or update the candle in history.

        In-progress candles (is_closed=False) replace the last element if it
        shares the same open_time (same 1m window).  Closed candles finalise
        the current window.
        """
        hist = self._candle_history[candle.symbol]
        if hist and hist[-1].open_time == candle.open_time:
            # Same candle window — update the last element in-place.
            hist[-1] = candle
        else:
            hist.append(candle)

    @staticmethod
    def _parse_kline(data: dict) -> CandleTick:
        """Convert a Binance kline event payload into a CandleTick.

        Relevant fields from Binance kline event (``data["k"]``):
            ``s`` – symbol (on outer data dict)
            ``t`` – kline open time (ms)
            ``o`` – open price
            ``h`` – high price
            ``l`` – low price
            ``c`` – close price
            ``v`` – base asset volume
            ``x`` – is kline closed
        """
        k = data["k"]
        return CandleTick(
            symbol=data["s"].upper(),
            open_time=int(k["t"]),
            open=float(k["o"]),
            high=float(k["h"]),
            low=float(k["l"]),
            close=float(k["c"]),
            volume=float(k["v"]),
            is_closed=bool(k["x"]),
        )
