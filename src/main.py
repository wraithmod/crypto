"""
Crypto Trading Platform - Main Entry Point
Run with: python src/main.py
"""
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path so that `config` and `src.*` imports resolve correctly
# regardless of the working directory the user runs from.
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from src.agents.factory import create_provider
from src.market.feed import MarketFeed
from src.market.indices import IndicesFeed
from src.portfolio.portfolio import Portfolio
from src.news.feed import NewsFeed
from src.prediction.predictor import Predictor
from src.trading.engine import TradeEngine
from src.dashboard.dashboard import Dashboard

# ---------------------------------------------------------------------------
# Logging: write everything to a rotating file so the console stays clean.
# The live dashboard owns stdout/stderr; log records must not appear there.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.FileHandler("trading.log")],
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Coroutines
# ---------------------------------------------------------------------------

async def news_loop(news_feed: NewsFeed, interval: float) -> None:
    """Periodically fetch fresh headlines and run LLM sentiment analysis.

    The first fetch is performed immediately so the dashboard has data before
    the first refresh cycle.  Subsequent fetches are spaced ``interval``
    seconds apart.  Any exception raised inside :meth:`NewsFeed.fetch_and_analyze`
    is caught, logged, and does not terminate the loop.

    Args:
        news_feed: The :class:`~src.news.feed.NewsFeed` instance to drive.
        interval:  Seconds to wait between fetches (from ``config.news_interval``).
    """
    logger.info("News loop started (interval=%.1f s)", interval)
    while True:
        try:
            items = await news_feed.fetch_and_analyze()
            logger.info("News loop: fetched %d headlines", len(items))
        except asyncio.CancelledError:
            logger.info("News loop cancelled.")
            raise
        except Exception as exc:
            logger.error("News fetch error: %s", exc)
        await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    """Wire all subsystems together and run the platform.

    Execution order
    ---------------
    1. Load configuration from the ``AppConfig`` singleton (already populated
       by ``config.py`` at import time; API keys are read from ``*.key`` files
       in the project root).
    2. Instantiate all subsystem objects.
    3. Create a shared :class:`asyncio.Queue` for price ticks.
    4. Launch four concurrent tasks via :func:`asyncio.gather`:

       - :meth:`MarketFeed.start` — Binance WebSocket price feed.
       - :meth:`TradeEngine.run_hft_loop` — signal evaluation and paper trading.
       - :func:`news_loop` — periodic headline ingestion and sentiment scoring.
       - :meth:`Dashboard.run` — live console UI.

    5. On :exc:`KeyboardInterrupt`, signal the market feed to stop gracefully,
       cancel all remaining tasks, and log the shutdown event.
    """
    logger.info("=== Crypto Trading Platform starting ===")
    logger.info(
        "Symbols: %s | Initial cash: %.2f",
        config.symbols,
        config.initial_cash,
    )

    # ------------------------------------------------------------------
    # 1. Instantiate subsystems
    # ------------------------------------------------------------------
    price_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)

    llm_provider = create_provider(config)
    logger.info("LLM provider: %s", llm_provider.name)

    market_feed = MarketFeed(symbols=config.symbols, price_queue=price_queue)
    indices_feed = IndicesFeed(
        symbols=config.tracked_indices,
        poll_interval=config.indices_poll_interval,
    )
    portfolio = Portfolio(initial_cash=config.initial_cash)
    news_feed = NewsFeed(llm_provider)
    predictor = Predictor()
    engine = TradeEngine(portfolio=portfolio, predictor=predictor, config=config)
    dashboard = Dashboard()

    logger.info("All subsystems instantiated.")

    # ------------------------------------------------------------------
    # 2. Build concurrent tasks
    # ------------------------------------------------------------------
    tasks = [
        asyncio.create_task(
            market_feed.start(),
            name="market_feed",
        ),
        asyncio.create_task(
            indices_feed.start(),
            name="indices_feed",
        ),
        asyncio.create_task(
            engine.run_hft_loop(
                symbols=config.symbols,
                market_feed=market_feed,
                news_feed=news_feed,
            ),
            name="hft_loop",
        ),
        asyncio.create_task(
            news_loop(news_feed=news_feed, interval=config.news_interval),
            name="news_loop",
        ),
        asyncio.create_task(
            dashboard.run(
                market_feed=market_feed,
                portfolio=portfolio,
                news_feed=news_feed,
                engine=engine,
                indices_feed=indices_feed,
            ),
            name="dashboard",
        ),
    ]

    logger.info(
        "Launching %d concurrent tasks: %s",
        len(tasks),
        [t.get_name() for t in tasks],
    )

    # ------------------------------------------------------------------
    # 3. Run until something stops (cancellation, exception, or Ctrl-C)
    # ------------------------------------------------------------------
    try:
        # return_exceptions=True ensures that a failure in one task does not
        # silently swallow exceptions from others; we inspect results below.
        results = await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        logger.info("Main gather cancelled — shutting down tasks.")
    finally:
        # Signal feeds so their loops exit cleanly.
        await market_feed.stop()
        await indices_feed.stop()

        # Cancel any tasks that are still running.
        for task in tasks:
            if not task.done():
                task.cancel()

        # Wait for all cancellations to propagate.
        await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("=== Crypto Trading Platform stopped ===")

    # Log any unexpected task failures (not CancelledError).
    for task, result in zip(tasks, results if isinstance(results, (list, tuple)) else []):
        if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
            logger.error("Task '%s' raised an unexpected exception: %s", task.get_name(), result)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down...")
