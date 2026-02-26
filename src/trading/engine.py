"""Trading engine: evaluates signals and executes paper trades.

TradeEngine ties together the portfolio, predictor, market feed, and news feed.
It runs an async HFT loop that continuously evaluates each tracked symbol,
generates signals via the predictor, and places paper trades when confidence
exceeds the configured threshold.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass

from config import AppConfig, config as default_config
from src.portfolio.portfolio import Portfolio, Trade
from src.prediction.predictor import Predictor, Signal
from src.market.feed import MarketFeed, PriceTick
from src.news.feed import NewsFeed

logger = logging.getLogger(__name__)

# Minimum signal confidence required before a trade is placed.
# Uses strict less-than so confidence=0.5 (one triggered condition) passes.
_CONFIDENCE_THRESHOLD: float = 0.45


class TradeEngine:
    """Orchestrates signal evaluation and paper trade execution.

    Args:
        portfolio:  The live portfolio that tracks cash and holdings.
        predictor:  The signal predictor that computes RSI/MACD/BB indicators
                    and blends them with news sentiment.
        config:     Application-level configuration (position sizing, intervals,
                    etc.).  Defaults to the shared singleton.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        predictor: Predictor,
        config: AppConfig = default_config,
    ) -> None:
        self._portfolio: Portfolio = portfolio
        self._predictor: Predictor = predictor
        self._config: AppConfig = config
        self._last_action: str = "No action yet"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def place_paper_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
    ) -> Trade:
        """Create and record a paper trade in the portfolio.

        Builds a :class:`~src.portfolio.portfolio.Trade` dataclass with a
        freshly generated UUID, submits it to the portfolio, logs the
        outcome, and updates the last-action summary string.

        Args:
            symbol:   Trading pair, e.g. ``"BTCUSDT"``.
            side:     Either ``"buy"`` or ``"sell"`` (case-insensitive).
            quantity: Number of units to trade.
            price:    Execution price per unit.

        Returns:
            The recorded :class:`~src.portfolio.portfolio.Trade` object.

        Raises:
            ValueError: Propagated from :class:`Portfolio` if the trade is
                        invalid (insufficient cash, no position to sell, etc.).
        """
        trade = Trade(
            symbol=symbol,
            side=side.lower(),
            quantity=quantity,
            price=price,
            timestamp=time.time(),
            trade_id=str(uuid.uuid4()),
        )

        self._portfolio.add_trade(trade)

        action_str = (
            f"{side.upper()} {quantity:.6f} {symbol} @ {price:.2f}"
        )
        self._last_action = action_str

        logger.info(
            "Paper trade executed: %s (id=%s)",
            action_str,
            trade.trade_id,
        )

        return trade

    async def evaluate_symbol(
        self,
        symbol: str,
        market_feed: MarketFeed,
        news_feed: NewsFeed,
    ) -> None:
        """Evaluate one symbol and conditionally place a paper trade.

        Steps:
            1. Pull price history from the market feed.
            2. Fetch the latest news sentiment from the news feed.
            3. Ask the predictor for a directional signal.
            4. Retrieve the current spot price.
            5. Apply position-sizing rules:
               - ``BUY``:  Use ``trade_fraction * available_cash / price``
                 as quantity.  Only execute if there is cash available *and*
                 the current position is below ``max_position_fraction`` of
                 total portfolio value.
               - ``SELL``: Liquidate the entire holding for the symbol.
            6. Place the trade if ``signal.confidence > 0.5``.

        Missing price data (no history, no latest tick) causes the method to
        return early without placing a trade.

        Args:
            symbol:      Trading pair to evaluate, e.g. ``"BTCUSDT"``.
            market_feed: Source of price history and current tick.
            news_feed:   Source of aggregated news sentiment.
        """
        # 1. Price history
        price_history: list[float] = market_feed.get_price_history(symbol)
        if not price_history:
            logger.debug(
                "evaluate_symbol: no price history for %s — skipping", symbol
            )
            return

        # 2. News sentiment — prefer coin-specific score; fall back to global average.
        symbol_sentiment: float | None = news_feed.get_symbol_sentiment(symbol)
        news_sentiment: float = (
            symbol_sentiment if symbol_sentiment is not None
            else news_feed.get_avg_sentiment()
        )
        logger.debug(
            "Sentiment for %s: %.4f (%s)",
            symbol, news_sentiment,
            "symbol-specific" if symbol_sentiment is not None else "global-avg",
        )

        # 3. Predictor signal
        signal: Signal = await self._predictor.get_signal(
            symbol, price_history, news_sentiment
        )

        logger.debug(
            "Signal for %s: direction=%s confidence=%.4f",
            symbol,
            signal.direction,
            signal.confidence,
        )

        # 4. Current price
        tick: PriceTick | None = market_feed.get_latest(symbol)
        if tick is None:
            logger.debug(
                "evaluate_symbol: no current tick for %s — skipping", symbol
            )
            return

        current_price: float = tick.price

        # Stop-loss check independent of signal direction — runs before confidence gate
        holdings_check: dict = self._portfolio.get_holdings()
        if symbol in holdings_check:
            h = holdings_check[symbol]
            if h.avg_cost > 0:
                pct_change = (current_price - h.avg_cost) / h.avg_cost
                if pct_change <= -self._config.stop_loss_pct:
                    logger.info(
                        "Pre-signal STOP-LOSS for %s: %.2f%% down from avg cost %.4f",
                        symbol, pct_change * 100, h.avg_cost,
                    )
                    try:
                        await self.place_paper_trade(symbol, "sell", h.quantity, current_price)
                    except ValueError as exc:
                        logger.warning("Pre-signal stop-loss rejected for %s: %s", symbol, exc)
                    return

        # Gate on confidence
        if signal.confidence < _CONFIDENCE_THRESHOLD:
            logger.debug(
                "Signal confidence %.4f <= %.2f for %s — no trade",
                signal.confidence,
                _CONFIDENCE_THRESHOLD,
                symbol,
            )
            return

        direction: str = signal.direction.upper()

        # 5 & 6. Position sizing and execution
        if direction == "BUY":
            cash: float = self._portfolio.get_cash()
            if cash <= 0.0:
                logger.debug(
                    "evaluate_symbol: no cash available for BUY %s", symbol
                )
                return

            # Build a prices dict using current tick so position_value works.
            prices_snapshot: dict[str, float] = {symbol: current_price}
            total_value: float = self._portfolio.get_total_value(prices_snapshot)
            position_value: float = self._portfolio.get_position_value(
                symbol, prices_snapshot
            )

            max_position_value: float = (
                self._config.max_position_fraction * total_value
            )

            if position_value >= max_position_value:
                logger.debug(
                    "evaluate_symbol: position %.2f >= max %.2f for %s — no BUY",
                    position_value,
                    max_position_value,
                    symbol,
                )
                return

            # How much cash we're willing to deploy this trade
            trade_cash: float = self._config.trade_fraction * cash

            # Don't exceed the remaining allowed position room
            remaining_room: float = max_position_value - position_value
            trade_cash = min(trade_cash, remaining_room)

            if trade_cash <= 0.0:
                logger.debug(
                    "evaluate_symbol: trade_cash %.6f <= 0 for %s — no BUY",
                    trade_cash,
                    symbol,
                )
                return

            quantity: float = trade_cash / current_price

            try:
                await self.place_paper_trade(symbol, "buy", quantity, current_price)
            except ValueError as exc:
                logger.warning(
                    "BUY paper trade rejected for %s: %s", symbol, exc
                )

        elif direction == "SELL":
            holdings: dict = self._portfolio.get_holdings()
            if symbol not in holdings:
                logger.debug(
                    "evaluate_symbol: no holding to SELL for %s", symbol
                )
                return

            holding = holdings[symbol]
            sell_quantity: float = holding.quantity

            if sell_quantity <= 0.0:
                logger.debug(
                    "evaluate_symbol: zero quantity for SELL %s — skipping",
                    symbol,
                )
                return

            avg_cost: float = holding.avg_cost
            price_change_pct: float = (current_price - avg_cost) / avg_cost if avg_cost > 0 else 0.0
            stop_loss_threshold: float = -self._config.stop_loss_pct

            # Stop-loss: always sell if down beyond the stop-loss threshold
            if price_change_pct <= stop_loss_threshold:
                logger.info(
                    "STOP-LOSS triggered for %s: price_change=%.2f%% <= -%.2f%%",
                    symbol, price_change_pct * 100, self._config.stop_loss_pct * 100,
                )
                try:
                    await self.place_paper_trade(symbol, "sell", sell_quantity, current_price)
                except ValueError as exc:
                    logger.warning("STOP-LOSS sell rejected for %s: %s", symbol, exc)
                return

            # Fee-aware gate: only sell on signal if profit exceeds round-trip fees + margin
            round_trip_fee = 2 * self._config.binance_fee_rate
            min_required = round_trip_fee + self._config.min_profit_to_sell
            if price_change_pct < min_required:
                logger.debug(
                    "evaluate_symbol: SELL gated for %s — profit %.4f%% < required %.4f%%",
                    symbol, price_change_pct * 100, min_required * 100,
                )
                return

            try:
                await self.place_paper_trade(
                    symbol, "sell", sell_quantity, current_price
                )
            except ValueError as exc:
                logger.warning(
                    "SELL paper trade rejected for %s: %s", symbol, exc
                )

        else:
            # direction is "HOLD" or unknown — do nothing
            logger.debug(
                "evaluate_symbol: direction=%s for %s — no trade",
                direction,
                symbol,
            )

    async def run_hft_loop(
        self,
        symbols: list[str],
        market_feed: MarketFeed,
        news_feed: NewsFeed,
    ) -> None:
        """Run the high-frequency trading loop indefinitely.

        For every iteration the engine evaluates each tracked symbol in
        sequence, then sleeps for ``config.hft_interval`` seconds before
        the next cycle.  The loop runs until cancelled (e.g. via
        ``asyncio.CancelledError``).

        Args:
            symbols:     List of trading pairs to monitor, e.g.
                         ``["BTCUSDT", "ETHUSDT"]``.
            market_feed: Live price data source.
            news_feed:   Live news sentiment source.
        """
        logger.info(
            "HFT loop starting for symbols=%s interval=%.2fs",
            symbols,
            self._config.hft_interval,
        )

        while True:
            for symbol in symbols:
                try:
                    await self.evaluate_symbol(symbol, market_feed, news_feed)
                except asyncio.CancelledError:
                    logger.info("HFT loop cancelled during evaluate_symbol(%s)", symbol)
                    raise
                except Exception as exc:  # pylint: disable=broad-except
                    # Log but keep the loop alive for other symbols / next cycle.
                    logger.exception(
                        "Unhandled error evaluating %s: %s", symbol, exc
                    )

            try:
                await asyncio.sleep(self._config.hft_interval)
            except asyncio.CancelledError:
                logger.info("HFT loop cancelled during sleep.")
                raise

    def get_last_action(self) -> str:
        """Return a human-readable summary of the most recent trade action.

        Format: ``"BUY 0.001234 BTCUSDT @ 45000.00"`` or the initial
        ``"No action yet"`` string if no trade has been placed.
        """
        return self._last_action

    def get_active_strategy(self) -> str:
        """Return the name of the currently active trading strategy."""
        return "RSI+MACD+BB with News Sentiment Blend"
