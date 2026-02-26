"""Async poller for global financial market indices via Yahoo Finance.

Wraps the synchronous ``yfinance`` library in ``asyncio``'s thread-pool
executor so the feed integrates cleanly with the rest of the async platform.

Tracked examples:  ^VIX, ^GSPC (S&P 500), ^IXIC (NASDAQ), ^DJI (Dow Jones),
                   ^N225 (Nikkei), ^GDAXI (DAX), ^FTSE (FTSE 100), HSI.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import yfinance as yf

from config import config

logger = logging.getLogger(__name__)

# Human-readable display name for each Yahoo Finance ticker symbol.
_INDEX_NAMES: dict[str, str] = {
    "^VIX":   "VIX",
    "^GSPC":  "S&P 500",
    "^IXIC":  "NASDAQ",
    "^DJI":   "Dow Jones",
    "^N225":  "Nikkei 225",
    "^GDAXI": "DAX",
    "^FTSE":  "FTSE 100",
    "^HSI":   "Hang Seng",
}


@dataclass
class IndexTick:
    """A single market-index snapshot."""
    symbol: str       # Yahoo Finance symbol, e.g. "^GSPC"
    name: str         # Human-readable name, e.g. "S&P 500"
    price: float      # Latest / last price
    change: float     # Absolute day change  (price - previous_close)
    change_pct: float # Percentage day change
    timestamp: float  # Unix timestamp of fetch


class IndicesFeed:
    """Polls Yahoo Finance for global equity index data.

    Runs as an async task; fetches all configured symbols in a single
    thread-pool call every ``poll_interval`` seconds.

    Usage::

        feed = IndicesFeed(config.tracked_indices, config.indices_poll_interval)
        asyncio.create_task(feed.start())
        tick = feed.get_latest("^GSPC")
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        poll_interval: float | None = None,
    ) -> None:
        self._symbols: list[str] = symbols if symbols is not None else config.tracked_indices
        self._poll_interval: float = poll_interval if poll_interval is not None else config.indices_poll_interval
        self._latest: dict[str, IndexTick] = {}
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
        logger.info("IndicesFeed starting for %d symbols (poll=%.0f s)", len(self._symbols), self._poll_interval)

        # Do an immediate first fetch so the dashboard has data right away.
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
                name: str = _INDEX_NAMES.get(sym, sym)

                ticks[sym] = IndexTick(
                    symbol=sym,
                    name=name,
                    price=price,
                    change=change,
                    change_pct=change_pct,
                    timestamp=now,
                )
                logger.debug("  %s (%s): %.2f (%+.2f%%)", sym, name, price, change_pct)

            except Exception as exc:
                logger.warning("IndicesFeed: failed to fetch %s: %s", sym, exc)

        return ticks
