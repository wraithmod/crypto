"""Async poller for global financial market indices and ASX stocks via Yahoo Finance.

Wraps the synchronous ``yfinance`` library in ``asyncio``'s thread-pool
executor so the feed integrates cleanly with the rest of the async platform.

Groups:
  global_markets — ^VIX, ^GSPC (S&P 500), ^IXIC, ^DJI, ^N225, ^GDAXI, ^FTSE, ^HSI
  asx_stocks     — CBA.AX, BHP.AX, CSL.AX … (top 50 ASX, feature-flagged)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass

import yfinance as yf

from config import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Display names
# ---------------------------------------------------------------------------

_INDEX_NAMES: dict[str, str] = {
    # Global indices
    "^VIX":   "VIX",
    "^GSPC":  "S&P 500",
    "^IXIC":  "NASDAQ",
    "^DJI":   "Dow Jones",
    "^N225":  "Nikkei 225",
    "^GDAXI": "DAX",
    "^FTSE":  "FTSE 100",
    "^HSI":   "Hang Seng",
    # ASX stocks — stripped ticker is used as name (CBA.AX -> "CBA")
    # Overrides for clarity where needed:
    "CBA.AX":  "CBA",   "NAB.AX":  "NAB",   "WBC.AX":  "WBC",
    "ANZ.AX":  "ANZ",   "MQG.AX":  "MQG",   "QBE.AX":  "QBE",
    "SUN.AX":  "SUN",   "IAG.AX":  "IAG",   "CPU.AX":  "CPU",
    "ASX.AX":  "ASX",   "BHP.AX":  "BHP",   "RIO.AX":  "RIO",
    "FMG.AX":  "FMG",   "S32.AX":  "S32",   "MIN.AX":  "MIN",
    "PLS.AX":  "PLS",   "IGO.AX":  "IGO",   "LYC.AX":  "LYC",
    "WDS.AX":  "WDS",   "STO.AX":  "STO",   "ORG.AX":  "ORG",
    "AGL.AX":  "AGL",   "CSL.AX":  "CSL",   "COH.AX":  "COH",
    "RMD.AX":  "RMD",   "RHC.AX":  "RHC",   "PME.AX":  "PME",
    "SHL.AX":  "SHL",   "WES.AX":  "WES",   "WOW.AX":  "WOW",
    "COL.AX":  "COL",   "JBH.AX":  "JBH",   "HVN.AX":  "HVN",
    "TWE.AX":  "TWE",   "WTC.AX":  "WTC",   "XRO.AX":  "XRO",
    "REA.AX":  "REA",   "CAR.AX":  "CAR",   "SEK.AX":  "SEK",
    "NXT.AX":  "NXT",   "TLS.AX":  "TLS",   "TCL.AX":  "TCL",
    "QAN.AX":  "QAN",   "TPG.AX":  "TPG",   "GMG.AX":  "GMG",
    "SCG.AX":  "SCG",   "DXS.AX":  "DXS",   "MGR.AX":  "MGR",
    "NST.AX":  "NST",   "EVN.AX":  "EVN",   "ALL.AX":  "ALL",
}

# ---------------------------------------------------------------------------
# Group assignment
# ---------------------------------------------------------------------------

def _group_for(symbol: str) -> str:
    """Return the display group for a given symbol."""
    if symbol.endswith(".AX"):
        return "asx_stocks"
    return "global_markets"


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class IndexTick:
    """A single market-index or stock snapshot."""
    symbol: str       # Yahoo Finance symbol, e.g. "^GSPC" or "CBA.AX"
    name: str         # Human-readable name, e.g. "S&P 500" or "CBA"
    price: float      # Latest / last price
    change: float     # Absolute day change  (price - previous_close)
    change_pct: float # Percentage day change
    timestamp: float  # Unix timestamp of fetch
    group: str = "global_markets"  # "global_markets" | "asx_stocks"


# ---------------------------------------------------------------------------
# Feed
# ---------------------------------------------------------------------------

class IndicesFeed:
    """Polls Yahoo Finance for global equity index and ASX stock data.

    Combines ``config.tracked_indices`` (always) with ``config.asx_symbols``
    (when ``config.asx_enabled`` is True) into a single thread-pool fetch
    call per cycle.

    Usage::

        feed = IndicesFeed()
        asyncio.create_task(feed.start())
        tick = feed.get_latest("^GSPC")
        asx  = feed.get_by_group("asx_stocks")
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        poll_interval: float | None = None,
    ) -> None:
        # Build combined symbol list
        base = symbols if symbols is not None else list(config.tracked_indices)
        if config.asx_enabled:
            base = base + list(config.asx_symbols)

        self._symbols: list[str] = base
        self._poll_interval: float = (
            poll_interval if poll_interval is not None else config.indices_poll_interval
        )
        self._latest: dict[str, IndexTick] = {}
        self._price_history: dict[str, deque] = {
            sym: deque(maxlen=config.price_history_len) for sym in base
        }
        self._running: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def symbols(self) -> list[str]:
        return self._symbols

    async def start(self) -> None:
        """Fetch indices in a loop until :meth:`stop` is called."""
        self._running = True
        logger.info(
            "IndicesFeed starting — %d global + %d ASX symbols (poll=%.0f s)",
            len(config.tracked_indices),
            len(config.asx_symbols) if config.asx_enabled else 0,
            self._poll_interval,
        )

        # Immediate first fetch so dashboard has data on first render.
        await self._fetch_all()

        while self._running:
            await asyncio.sleep(self._poll_interval)
            if not self._running:
                break
            await self._fetch_all()

        logger.info("IndicesFeed stopped.")

    async def stop(self) -> None:
        """Signal the feed to exit after the current sleep."""
        logger.info("IndicesFeed stop requested.")
        self._running = False

    def get_latest(self, symbol: str) -> IndexTick | None:
        """Return the most recent snapshot for *symbol*, or ``None``."""
        return self._latest.get(symbol)

    def get_all(self) -> dict[str, IndexTick]:
        """Return a copy of the full latest-tick mapping."""
        return dict(self._latest)

    def get_by_group(self, group: str) -> dict[str, IndexTick]:
        """Return only ticks belonging to *group* (e.g. 'asx_stocks')."""
        return {sym: tick for sym, tick in self._latest.items()
                if tick.group == group}

    def get_price_history(self, symbol: str) -> list[float]:
        """Return the recorded close-price history for *symbol* (most recent last)."""
        hist = self._price_history.get(symbol)
        return list(hist) if hist else []

    def get_candle_history(self, symbol: str) -> list:
        """Not available from yfinance fast_info polls — always returns []."""
        return []

    def get_volume_history(self, symbol: str) -> list[float]:
        """Not available from yfinance fast_info polls — always returns []."""
        return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _fetch_all(self) -> None:
        """Offload the synchronous yfinance calls to the thread-pool."""
        loop = asyncio.get_event_loop()
        try:
            ticks: dict[str, IndexTick] = await loop.run_in_executor(
                None, self._fetch_sync
            )
            self._latest.update(ticks)
            for sym, tick in ticks.items():
                if sym in self._price_history:
                    self._price_history[sym].append(tick.price)
            logger.debug("IndicesFeed updated %d ticks.", len(ticks))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("IndicesFeed _fetch_all error: %s", exc)

    def _fetch_sync(self) -> dict[str, IndexTick]:
        """Fetch all configured symbols synchronously (runs in thread pool)."""
        ticks: dict[str, IndexTick] = {}
        now = time.time()

        for sym in self._symbols:
            try:
                ticker = yf.Ticker(sym)
                fi = ticker.fast_info

                price: float = float(fi.last_price)
                prev_close: float = float(fi.previous_close) if fi.previous_close else price
                change: float = price - prev_close
                change_pct: float = (change / prev_close * 100.0) if prev_close else 0.0

                # Derive display name: use lookup, else strip .AX suffix
                if sym in _INDEX_NAMES:
                    name = _INDEX_NAMES[sym]
                elif sym.endswith(".AX"):
                    name = sym[:-3]
                else:
                    name = sym

                ticks[sym] = IndexTick(
                    symbol=sym,
                    name=name,
                    price=price,
                    change=change,
                    change_pct=change_pct,
                    timestamp=now,
                    group=_group_for(sym),
                )
                logger.debug("  %s (%s): %.4f (%+.2f%%)", sym, name, price, change_pct)

            except Exception as exc:
                logger.warning("IndicesFeed: failed to fetch %s: %s", sym, exc)

        return ticks


# ---------------------------------------------------------------------------
# ASX feed adapter — presents IndicesFeed as a MarketFeed-compatible object
# ---------------------------------------------------------------------------

class ASXFeedAdapter:
    """Wraps :class:`IndicesFeed` to expose the same interface as
    :class:`~src.market.feed.MarketFeed` so that :class:`~src.trading.engine.TradeEngine`
    can trade ASX symbols without modification.

    ``get_latest`` converts :class:`IndexTick` → :class:`~src.market.feed.PriceTick`.
    Price history comes from ``IndicesFeed``'s accumulated poll snapshots.
    Candle and volume history are unavailable from yfinance fast_info — both
    return empty lists, so the predictor falls back to pure RSI/MACD/BB signals.
    """

    def __init__(self, indices_feed: IndicesFeed) -> None:
        self._feed = indices_feed

    def get_latest(self, symbol: str):
        """Return a PriceTick for *symbol*, or None if not yet polled."""
        from src.market.feed import PriceTick  # local import avoids circular dep

        tick = self._feed.get_latest(symbol)
        if tick is None:
            return None
        return PriceTick(
            symbol=tick.symbol,
            price=tick.price,
            volume=0.0,
            bid=0.0,
            ask=0.0,
            timestamp=tick.timestamp,
        )

    def get_price_history(self, symbol: str) -> list[float]:
        return self._feed.get_price_history(symbol)

    def get_candle_history(self, symbol: str) -> list:
        return []

    def get_volume_history(self, symbol: str) -> list[float]:
        return []
