"""
Crypto Trading Platform - Main Entry Point
Run with: python src/main.py [--cash AMOUNT]
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path so that `config` and `src.*` imports resolve correctly
# regardless of the working directory the user runs from.
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from src.trading.risk import PROFILES
from src.trading.strategy import STRATEGIES
from src.agents.factory import create_provider
from src.market.feed import MarketFeed
from src.market.indices import IndicesFeed, ASXFeedAdapter
from src.market.bybit_feed import BybitFeed
from src.market.okx_feed import OKXFeed
from src.market.deribit_feed import DeribitVolFeed
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

async def main(risk_profile=None, trade_groups: set | None = None, strategy=None, feeds: set | None = None) -> None:
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
    if risk_profile is None:
        risk_profile = PROFILES["medium"]
    if strategy is None:
        strategy = STRATEGIES["classic"]

    # Resolve trade groups — "all" expands to crypto + asx
    if trade_groups is None:
        trade_groups = {"crypto"}
    if "all" in trade_groups:
        trade_groups = {"crypto", "asx"}

    # Resolve extra feeds — default to binance only
    if feeds is None:
        feeds = {"binance"}
    if "all" in feeds:
        feeds = {"binance", "bybit", "okx", "deribit"}

    logger.info("=== Crypto Trading Platform starting ===")
    logger.info(
        "Trade groups: %s | Feeds: %s | Initial cash: %.2f | Risk: %s | Strategy: %s "
        "(confidence≥%.0f%% trade=%.0f%% stop=%.1f%%)",
        sorted(trade_groups),
        sorted(feeds),
        config.initial_cash,
        risk_profile.name,
        strategy.name,
        risk_profile.confidence_threshold * 100,
        risk_profile.trade_fraction * 100,
        risk_profile.stop_loss_pct * 100,
    )
    if "global" in trade_groups:
        logger.info(
            "Note: 'global' indices (%s) are display-only — index futures cannot be paper-traded.",
            [s for s in config.tracked_indices],
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
    engine = TradeEngine(
        portfolio=portfolio, predictor=predictor, config=config,
        risk=risk_profile, strategy=strategy,
    )
    dashboard = Dashboard()

    logger.info("All subsystems instantiated.")

    # ------------------------------------------------------------------
    # 2. Build concurrent tasks
    # ------------------------------------------------------------------
    tasks = [
        asyncio.create_task(market_feed.start(), name="market_feed"),
        asyncio.create_task(indices_feed.start(), name="indices_feed"),
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

    # HFT loop for crypto (Binance WebSocket)
    if "crypto" in trade_groups:
        tasks.append(asyncio.create_task(
            engine.run_hft_loop(
                symbols=config.symbols,
                market_feed=market_feed,
                news_feed=news_feed,
            ),
            name="hft_loop_crypto",
        ))
        logger.info("HFT loop active for %d crypto symbols.", len(config.symbols))

    # HFT loop for ASX (yfinance 15-20 min delayed via IndicesFeed)
    if "asx" in trade_groups and config.asx_enabled:
        asx_adapter = ASXFeedAdapter(indices_feed)
        tasks.append(asyncio.create_task(
            engine.run_hft_loop(
                symbols=config.asx_symbols,
                market_feed=asx_adapter,
                news_feed=news_feed,
            ),
            name="hft_loop_asx",
        ))
        logger.info(
            "HFT loop active for %d ASX symbols (15-20 min delayed via yfinance).",
            len(config.asx_symbols),
        )
    elif "asx" in trade_groups and not config.asx_enabled:
        logger.warning("--trade asx requested but config.asx_enabled=False — ASX HFT skipped.")

    # ------------------------------------------------------------------
    # Extra exchange feeds (Bybit, OKX, Deribit) — controlled by --feeds
    # ------------------------------------------------------------------

    # Bybit linear perpetuals feed + HFT loop
    if "bybit" in feeds and config.bybit_enabled:
        bybit_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        bybit_feed = BybitFeed(symbols=config.bybit_symbols, price_queue=bybit_queue)
        tasks.append(asyncio.create_task(bybit_feed.start(), name="bybit_feed"))
        tasks.append(asyncio.create_task(
            engine.run_hft_loop(
                symbols=config.bybit_symbols,
                market_feed=bybit_feed,
                news_feed=news_feed,
            ),
            name="hft_loop_bybit",
        ))
        logger.info("Bybit feed + HFT loop active for %d symbols.", len(config.bybit_symbols))
    elif "bybit" in feeds and not config.bybit_enabled:
        logger.warning("--feeds bybit requested but config.bybit_enabled=False — Bybit feed skipped.")

    # OKX spot feed + HFT loop
    if "okx" in feeds and config.okx_enabled:
        okx_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        okx_feed = OKXFeed(symbols=config.okx_symbols, price_queue=okx_queue)
        tasks.append(asyncio.create_task(okx_feed.start(), name="okx_feed"))
        tasks.append(asyncio.create_task(
            engine.run_hft_loop(
                symbols=config.okx_symbols,
                market_feed=okx_feed,
                news_feed=news_feed,
            ),
            name="hft_loop_okx",
        ))
        logger.info("OKX feed + HFT loop active for %d symbols.", len(config.okx_symbols))
    elif "okx" in feeds and not config.okx_enabled:
        logger.warning("--feeds okx requested but config.okx_enabled=False — OKX feed skipped.")

    # Deribit DVOL implied-volatility feed (display/signal only — no HFT trading)
    if "deribit" in feeds and config.deribit_enabled:
        deribit_feed = DeribitVolFeed()
        tasks.append(asyncio.create_task(deribit_feed.start(), name="deribit_feed"))
        logger.info("Deribit DVOL feed active (BTC + ETH implied volatility).")
    elif "deribit" in feeds and not config.deribit_enabled:
        logger.warning("--feeds deribit requested but config.deribit_enabled=False — Deribit feed skipped.")

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
    parser = argparse.ArgumentParser(description="Crypto Trading Platform")
    parser.add_argument(
        "--cash",
        type=float,
        default=config.initial_cash,
        metavar="AMOUNT",
        help=f"Starting cash (default: {config.initial_cash:,.0f})",
    )
    parser.add_argument(
        "--risk",
        choices=["low", "medium", "high", "extreme"],
        default="medium",
        help="Trading risk profile: low | medium | high | extreme (default: medium)",
    )
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES),
        default="classic",
        help="Signal strategy: classic|trend|breakout|scalp|sentiment (default: classic)",
    )
    parser.add_argument(
        "--trade",
        nargs="+",
        choices=["crypto", "asx", "global", "all"],
        default=["crypto"],
        metavar="GROUP",
        help=(
            "Which groups to actively trade (one or more): "
            "crypto | asx | global | all  "
            "(global is display-only — indices can't be paper-traded; default: crypto)"
        ),
    )
    parser.add_argument(
        "--feeds",
        nargs="+",
        choices=["binance", "bybit", "okx", "deribit", "all"],
        default=["binance"],
        metavar="FEED",
        help=(
            "Which exchange feeds to enable (one or more): "
            "binance | bybit | okx | deribit | all  "
            "(bybit/okx/deribit require config.*_enabled=True; default: binance)"
        ),
    )
    args = parser.parse_args()
    config.initial_cash = args.cash
    risk_profile = PROFILES[args.risk]
    strategy = STRATEGIES[args.strategy]
    trade_groups = set(args.trade)
    feeds = set(args.feeds)

    try:
        asyncio.run(main(risk_profile=risk_profile, trade_groups=trade_groups, strategy=strategy, feeds=feeds))
    except KeyboardInterrupt:
        print("\nShutting down...")
