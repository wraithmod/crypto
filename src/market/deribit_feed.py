"""Real-time Deribit Volatility Index (DVOL) feed via WebSocket.

Connects to the Deribit JSON-RPC 2.0 WebSocket API and subscribes to
``deribit_volatility_index.btc_usd`` and ``deribit_volatility_index.eth_usd``
channels, storing results as :class:`~src.market.indices.IndexTick` objects
with ``group="crypto_vol"``.

DVOL is Deribit's implied volatility index for BTC and ETH, analogous to the
VIX for equities.  Values typically range from ~40 (low vol) to ~200+ (extreme
vol) and update continuously during market hours.

Usage::

    feed = DeribitVolFeed()
    asyncio.create_task(feed.start())
    tick = feed.get_latest("DVOL_BTC")
    all_vol = feed.get_by_group("crypto_vol")
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import websockets

from src.market.indices import IndexTick

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WS_URL = "wss://www.deribit.com/ws/api/v2"
_HEARTBEAT_INTERVAL = 30  # seconds — Deribit-required
_RECONNECT_BASE = 1.0     # initial back-off seconds
_RECONNECT_MAX = 30.0     # cap back-off at 30 s


# ---------------------------------------------------------------------------
# Feed
# ---------------------------------------------------------------------------


class DeribitVolFeed:
    """Streams BTC and ETH DVOL from Deribit over a persistent WebSocket.

    Implements the same public interface as
    :class:`~src.market.indices.IndicesFeed` where applicable so it can slot
    into the same dashboard panels or be composed alongside other feeds.

    Reconnect strategy
    ------------------
    On any connection error or unexpected close the feed sleeps for an
    exponentially increasing interval (1 s → 2 s → 4 s … capped at 30 s)
    before attempting a new connection.  The back-off resets to 1 s after a
    successful connection that delivers at least one message.

    Heartbeat
    ---------
    Deribit drops idle connections after ~60 s.  The feed sets a 30-second
    server-side heartbeat interval immediately after connecting.  When Deribit
    sends a ``heartbeat`` / ``test_request`` the feed responds with
    ``public/test`` to keep the session alive.
    """

    # Public symbol list
    SYMBOLS: list[str] = ["DVOL_BTC", "DVOL_ETH"]

    # Map Deribit channel names → internal symbol keys
    _CHANNEL_MAP: dict[str, str] = {
        "deribit_volatility_index.btc_usd": "DVOL_BTC",
        "deribit_volatility_index.eth_usd": "DVOL_ETH",
    }

    # Human-readable display names
    _SYMBOL_NAMES: dict[str, str] = {
        "DVOL_BTC": "BTC DVOL",
        "DVOL_ETH": "ETH DVOL",
    }

    def __init__(self) -> None:
        self._latest: dict[str, IndexTick] = {}
        self._running: bool = False
        self._msg_id: int = 0  # auto-increment for JSON-RPC request IDs

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Connect to Deribit and stream DVOL until :meth:`stop` is called.

        Reconnects automatically on connection failures using exponential
        back-off.  Raises :exc:`asyncio.CancelledError` on task cancellation.
        """
        self._running = True
        logger.info("DeribitVolFeed starting — tracking %s", self.SYMBOLS)

        delay = _RECONNECT_BASE
        while self._running:
            try:
                await self._run_connection()
                # If _run_connection returns normally (feed stopped), exit.
                if not self._running:
                    break
                # Connection closed unexpectedly — reset back-off on clean exit.
                delay = _RECONNECT_BASE
            except asyncio.CancelledError:
                logger.info("DeribitVolFeed cancelled.")
                raise
            except Exception as exc:
                logger.warning(
                    "DeribitVolFeed connection error: %s — retrying in %.0f s",
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2.0, _RECONNECT_MAX)

        logger.info("DeribitVolFeed stopped.")

    async def stop(self) -> None:
        """Signal the feed to stop after the current receive completes."""
        logger.info("DeribitVolFeed stop requested.")
        self._running = False

    def get_latest(self, symbol: str) -> IndexTick | None:
        """Return the most recent :class:`IndexTick` for *symbol*, or ``None``."""
        return self._latest.get(symbol)

    def get_all(self) -> dict[str, IndexTick]:
        """Return a copy of the full latest-tick mapping."""
        return dict(self._latest)

    def get_by_group(self, group: str) -> dict[str, IndexTick]:
        """Return only ticks whose ``group`` matches *group*.

        For example ``get_by_group("crypto_vol")`` returns both DVOL ticks;
        ``get_by_group("global_markets")`` returns an empty dict because this
        feed only produces ``"crypto_vol"`` ticks.
        """
        return {
            sym: tick
            for sym, tick in self._latest.items()
            if tick.group == group
        }

    # ------------------------------------------------------------------
    # Internal — connection lifecycle
    # ------------------------------------------------------------------

    def _next_id(self) -> int:
        """Return the next auto-incremented JSON-RPC request ID."""
        self._msg_id += 1
        return self._msg_id

    async def _run_connection(self) -> None:
        """Open one WebSocket session, subscribe, and receive messages.

        Returns when ``_running`` is ``False`` or the connection closes.
        Raises any unhandled exceptions so the caller can apply back-off.
        """
        async with websockets.connect(_WS_URL) as ws:
            logger.info("DeribitVolFeed connected to %s", _WS_URL)

            await self._setup_heartbeat(ws)
            await self._subscribe(ws)

            async for raw in ws:
                if not self._running:
                    break
                try:
                    msg: dict[str, Any] = json.loads(raw)
                    await self._handle_message(ws, msg)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning("DeribitVolFeed: error handling message: %s", exc)

    async def _setup_heartbeat(self, ws: Any) -> None:
        """Enable server-side heartbeats so the connection is not silently dropped."""
        req = {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "public/set_heartbeat",
            "params": {"interval": _HEARTBEAT_INTERVAL},
        }
        await ws.send(json.dumps(req))
        logger.debug("DeribitVolFeed: heartbeat configured (%d s)", _HEARTBEAT_INTERVAL)

    async def _subscribe(self, ws: Any) -> None:
        """Send the channel subscription request."""
        channels = list(self._CHANNEL_MAP.keys())
        req = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "public/subscribe",
            "params": {"channels": channels},
        }
        await ws.send(json.dumps(req))
        logger.debug("DeribitVolFeed: subscribed to %s", channels)

    async def _handle_message(self, ws: Any, msg: dict[str, Any]) -> None:
        """Dispatch an incoming JSON-RPC message.

        Three cases are handled:

        1. **Subscription confirmation** — ``result`` key present; logged and ignored.
        2. **Heartbeat test_request** — respond with ``public/test``.
        3. **Volatility push** — ``method == "subscription"``; update ``_latest``.
        """
        # Case 1: RPC response (subscription confirmation or heartbeat ack)
        if "result" in msg:
            logger.debug("DeribitVolFeed: RPC response id=%s result=%s",
                         msg.get("id"), msg.get("result"))
            return

        method = msg.get("method")

        # Case 2: Heartbeat keep-alive
        if method == "heartbeat":
            params = msg.get("params", {})
            if params.get("type") == "test_request":
                await self._send_heartbeat_response(ws)
            return

        # Case 3: Subscription push
        if method == "subscription":
            params = msg.get("params", {})
            channel: str = params.get("channel", "")
            data: dict[str, Any] = params.get("data", {})
            self._process_volatility_update(channel, data)
            return

        logger.debug("DeribitVolFeed: unrecognised message method=%s", method)

    async def _send_heartbeat_response(self, ws: Any) -> None:
        """Respond to a Deribit heartbeat test_request."""
        resp = {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "public/test",
            "params": {},
        }
        await ws.send(json.dumps(resp))
        logger.debug("DeribitVolFeed: heartbeat test_request answered.")

    def _process_volatility_update(
        self, channel: str, data: dict[str, Any]
    ) -> None:
        """Parse a DVOL push and update the internal tick store."""
        symbol = self._CHANNEL_MAP.get(channel)
        if symbol is None:
            logger.debug(
                "DeribitVolFeed: ignoring unknown channel '%s'", channel
            )
            return

        try:
            volatility: float = float(data["volatility"])
            ts_ms: int = int(data["timestamp"])
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning(
                "DeribitVolFeed: malformed data on %s: %s — %s",
                channel, data, exc,
            )
            return

        timestamp: float = ts_ms / 1000.0

        # Compute change / change_pct from previous tick
        prev: IndexTick | None = self._latest.get(symbol)
        if prev is not None and prev.price:
            change = volatility - prev.price
            change_pct = change / prev.price * 100.0
        else:
            change = 0.0
            change_pct = 0.0

        tick = IndexTick(
            symbol=symbol,
            name=self._SYMBOL_NAMES[symbol],
            price=volatility,
            change=change,
            change_pct=change_pct,
            timestamp=timestamp,
            group="crypto_vol",
        )
        self._latest[symbol] = tick
        logger.debug(
            "DeribitVolFeed: %s = %.2f (Δ%+.2f%%)",
            symbol, volatility, change_pct,
        )
