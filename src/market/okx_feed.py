"""Async OKX WebSocket price feed — candle1m stream with REST warm-up.

Connects to the OKX public WebSocket (wss://ws.okx.com:8443/ws/v5/public),
subscribes to candle1m channels for each requested symbol, parses OHLCV
candle events, and puts PriceTick objects into a shared asyncio.Queue for
downstream consumers.

On startup, a REST warm-up call fetches up to 300 historical 1-minute
candles per symbol from the OKX REST API so that technical indicators can
be computed immediately without waiting for live candles to accumulate.

OKX uses a different symbol format than Binance (BTC-USDT vs BTCUSDT).
This feed accepts Binance-style symbols and converts them internally so
that all public-facing methods (get_latest, get_price_history, etc.) use
the Binance format that the rest of the platform expects.

Reconnects automatically with exponential backoff on any connection loss.
OKX requires a ping/pong heartbeat: the string "ping" is sent every 25
seconds; the server replies with "pong".
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
from src.market.feed import CandleTick, PriceTick

logger = logging.getLogger(__name__)

_OKX_WS_URL: str = "wss://ws.okx.com:8443/ws/v5/public"
_OKX_REST_BASE: str = "https://www.okx.com"
_PING_INTERVAL: float = 25.0


# ---------------------------------------------------------------------------
# Symbol conversion helpers
# ---------------------------------------------------------------------------


def _to_okx_symbol(sym: str) -> str:
    """Convert BTCUSDT -> BTC-USDT (OKX spot format)."""
    if sym.endswith("USDT"):
        return sym[:-4] + "-USDT"
    return sym  # fallback: return as-is


def _from_okx_symbol(okx_sym: str) -> str:
    """Convert BTC-USDT -> BTCUSDT (Binance format)."""
    return okx_sym.replace("-", "")


# ---------------------------------------------------------------------------
# OKX Market feed
# ---------------------------------------------------------------------------


class OKXFeed:
    """Streams live 1-minute candle data from OKX for a list of symbols.

    Accepts Binance-style symbols (e.g. ``BTCUSDT``) and converts them
    internally to OKX format (``BTC-USDT``).  All public-facing methods
    use the original Binance-style symbol so the rest of the platform
    does not need to know about OKX naming conventions.

    Usage::

        queue = asyncio.Queue()
        feed = OKXFeed(["BTCUSDT", "ETHUSDT"], queue)
        asyncio.create_task(feed.start())   # warm-up + WebSocket loop
        tick: PriceTick = await queue.get()
    """

    _BACKOFF_BASE: float = 1.0
    _BACKOFF_MAX: float = 30.0

    def __init__(self, symbols: list[str], price_queue: asyncio.Queue) -> None:
        # Store symbols in Binance format (uppercased) as the canonical key.
        self._symbols: list[str] = [s.upper() for s in symbols]
        self._queue: asyncio.Queue = price_queue

        # Internal OKX symbol mapping: Binance -> OKX
        self._okx_symbols: dict[str, str] = {
            s: _to_okx_symbol(s) for s in self._symbols
        }

        # OHLCV candle history keyed by Binance-format symbol, oldest first.
        self._candle_history: dict[str, deque[CandleTick]] = {
            s: deque(maxlen=config.price_history_len) for s in self._symbols
        }
        # Fast-access latest close price per Binance-format symbol.
        self._price_cache: dict[str, float] = {}

        self._running: bool = False

    # ------------------------------------------------------------------
    # Public interface (mirrors MarketFeed exactly)
    # ------------------------------------------------------------------

    @property
    def symbols(self) -> list[str]:
        """Return tracked symbols in Binance format."""
        return self._symbols

    async def start(self) -> None:
        """Warm up from REST, then stream live candles until stopped."""
        self._running = True
        logger.info("OKXFeed starting for symbols: %s", self._symbols)

        try:
            async with aiohttp.ClientSession() as session:
                await self._warm_up(session)
        except Exception as exc:
            logger.warning(
                "OKXFeed REST warm-up failed (will fill from live stream): %s", exc
            )

        backoff: float = self._BACKOFF_BASE
        while self._running:
            try:
                await self._run_connection()
                if not self._running:
                    break
                logger.warning(
                    "OKXFeed WebSocket stream ended; reconnecting in %.1f s", backoff
                )
            except websockets.exceptions.ConnectionClosed as exc:
                logger.warning(
                    "OKXFeed WebSocket closed (code=%s reason=%r); reconnecting in %.1f s",
                    exc.code,
                    exc.reason,
                    backoff,
                )
            except OSError as exc:
                logger.error(
                    "OKXFeed network error: %s; reconnecting in %.1f s", exc, backoff
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception(
                    "OKXFeed unexpected error: %s; reconnecting in %.1f s", exc, backoff
                )

            if not self._running:
                break

            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self._BACKOFF_MAX)

        logger.info("OKXFeed stopped.")

    async def stop(self) -> None:
        """Signal the feed to stop after the current reconnect cycle."""
        logger.info("OKXFeed stop requested.")
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
        """Return close prices oldest-first."""
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
        """Fetch historical 1-minute candles per symbol in parallel."""
        logger.info(
            "OKXFeed: starting REST warm-up for %d symbols", len(self._symbols)
        )
        await asyncio.gather(
            *[self._warm_up_symbol(s, session) for s in self._symbols],
            return_exceptions=True,
        )
        logger.info(
            "OKXFeed: REST warm-up complete — history lengths: %s",
            {s: len(self._candle_history[s]) for s in self._symbols},
        )

    async def _warm_up_symbol(
        self, symbol: str, session: aiohttp.ClientSession
    ) -> None:
        """Fetch historical candles for one symbol and populate its deque."""
        okx_sym = self._okx_symbols[symbol]
        url = (
            f"{_OKX_REST_BASE}/api/v5/market/candles"
            f"?instId={okx_sym}&bar=1m&limit={config.price_history_len}"
        )
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    logger.warning(
                        "OKXFeed warm_up: non-200 response for %s: HTTP %d",
                        symbol,
                        resp.status,
                    )
                    return
                payload = await resp.json()
        except Exception as exc:
            logger.warning(
                "OKXFeed warm_up: failed to fetch candles for %s: %s", symbol, exc
            )
            return

        if payload.get("code") != "0":
            logger.warning(
                "OKXFeed warm_up: API error for %s: %s",
                symbol,
                payload.get("msg", "unknown"),
            )
            return

        rows: list[list] = payload.get("data", [])
        # OKX returns data newest-first; reverse so deque is oldest-first.
        for row in reversed(rows):
            try:
                candle = CandleTick(
                    symbol=symbol,
                    open_time=int(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                    # Historical candles are always treated as closed.
                    is_closed=True,
                )
                self._candle_history[symbol].append(candle)
            except (IndexError, ValueError, TypeError) as exc:
                logger.debug(
                    "OKXFeed warm_up: bad candle row for %s: %s", symbol, exc
                )

        if self._candle_history[symbol]:
            self._price_cache[symbol] = self._candle_history[symbol][-1].close
            logger.debug(
                "OKXFeed warm_up: %s pre-populated with %d candles (latest close=%.4f)",
                symbol,
                len(self._candle_history[symbol]),
                self._price_cache[symbol],
            )

    # ------------------------------------------------------------------
    # WebSocket connection
    # ------------------------------------------------------------------

    async def _run_connection(self) -> None:
        """Open a single WebSocket connection and stream messages until closed."""
        logger.info("OKXFeed: connecting to %s", _OKX_WS_URL)
        async with websockets.connect(_OKX_WS_URL) as ws:
            logger.info("OKXFeed: WebSocket connected.")

            # Subscribe to candle1m for all symbols.
            sub_msg = json.dumps({
                "op": "subscribe",
                "args": [
                    {"channel": "candle1m", "instId": okx_sym}
                    for okx_sym in self._okx_symbols.values()
                ],
            })
            await ws.send(sub_msg)
            logger.debug("OKXFeed: sent subscription: %s", sub_msg)

            # Start the ping heartbeat as a background task.
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

    async def _ping_loop(self, ws: websockets.WebSocketClientProtocol) -> None:
        """Send the OKX heartbeat string 'ping' every 25 seconds."""
        try:
            while True:
                await asyncio.sleep(_PING_INTERVAL)
                try:
                    await ws.send("ping")
                    logger.debug("OKXFeed: sent ping")
                except Exception as exc:
                    logger.debug("OKXFeed: ping failed: %s", exc)
                    break
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    def _handle_message(self, raw: str) -> None:
        """Parse an OKX WebSocket message and update candle history."""
        # OKX pong response is just the string "pong".
        if raw == "pong":
            logger.debug("OKXFeed: received pong")
            return

        try:
            envelope: dict = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("OKXFeed: received non-JSON message: %.120r", raw)
            return

        # Subscription confirmation messages have an "event" field — ignore them.
        if "event" in envelope:
            logger.debug("OKXFeed: ignoring event message: %s", envelope.get("event"))
            return

        # Expect {"arg": {"channel": "candle1m", "instId": "BTC-USDT"}, "data": [...]}
        arg = envelope.get("arg", {})
        if arg.get("channel") != "candle1m":
            return

        data_rows: list | None = envelope.get("data")
        if not data_rows:
            return

        try:
            candle = self._parse_candle(arg, data_rows[0])
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            logger.warning(
                "OKXFeed: failed to parse candle data (%s): %r", exc, envelope
            )
            return

        # Guard: only track symbols we subscribed to.
        if candle.symbol not in self._candle_history:
            return

        self._upsert_candle(candle)

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
                "OKXFeed: price queue full; dropping tick for %s", candle.symbol
            )

    def _upsert_candle(self, candle: CandleTick) -> None:
        """Insert or update the candle in the history deque.

        In-progress candles (is_closed=False) replace the last element if
        it shares the same open_time (same 1-minute window).  Closed candles
        finalise the current window and are replaced in-place as well.
        A new open_time triggers an append (new candle window).
        """
        hist = self._candle_history[candle.symbol]
        if hist and hist[-1].open_time == candle.open_time:
            hist[-1] = candle
        else:
            hist.append(candle)

    @staticmethod
    def _parse_candle(arg: dict, row: list) -> CandleTick:
        """Convert an OKX candle1m data row into a CandleTick.

        OKX data array format (index):
            0  ts          — candle open timestamp (Unix ms, string)
            1  o           — open price
            2  h           — high price
            3  l           — low price
            4  c           — close price
            5  vol         — base asset volume
            6  volCcy      — quote asset volume
            7  volCcyQuote — quote asset volume (alternative)
            8  confirm     — "1" = closed candle, "0" = in-progress
        """
        okx_inst_id: str = arg["instId"]
        binance_symbol: str = _from_okx_symbol(okx_inst_id).upper()
        is_closed: bool = row[8] == "1"
        return CandleTick(
            symbol=binance_symbol,
            open_time=int(row[0]),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
            is_closed=is_closed,
        )
