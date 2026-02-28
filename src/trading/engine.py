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
from src.prediction.predictor import Predictor, Signal, Indicators
from src.market.feed import MarketFeed, PriceTick
from src.news.feed import NewsFeed
from src.trading.risk import RiskProfile, MEDIUM
from src.trading.strategy import TradingStrategy, DEFAULT_STRATEGY

logger = logging.getLogger(__name__)


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
        risk: RiskProfile = MEDIUM,
        strategy: TradingStrategy | None = None,
    ) -> None:
        self._portfolio: Portfolio = portfolio
        self._predictor: Predictor = predictor
        self._config: AppConfig = config
        self._risk: RiskProfile = risk
        self._strategy: TradingStrategy = strategy if strategy is not None else DEFAULT_STRATEGY
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

        # 3. Predictor signal — pass candle history for momentum indicators
        candle_history = market_feed.get_candle_history(symbol)
        signal: Signal = await self._predictor.get_signal(
            symbol, price_history, news_sentiment,
            risk=self._risk,
            candles=candle_history if candle_history else None,
            strategy=self._strategy,
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
                if pct_change <= -self._risk.stop_loss_pct:
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
        if signal.confidence < self._risk.confidence_threshold:
            logger.debug(
                "Signal confidence %.4f < %.2f for %s — no trade",
                signal.confidence,
                self._risk.confidence_threshold,
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
                self._risk.max_position_fraction * total_value
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
            trade_cash: float = self._risk.trade_fraction * cash

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
            if quantity < 1e-8:
                logger.debug(
                    "evaluate_symbol: quantity %.2e too small for BUY %s — skipping",
                    quantity, symbol,
                )
                return

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
            stop_loss_threshold: float = -self._risk.stop_loss_pct

            # Stop-loss: always sell if down beyond the stop-loss threshold
            if price_change_pct <= stop_loss_threshold:
                logger.info(
                    "STOP-LOSS triggered for %s: price_change=%.2f%% <= -%.2f%%",
                    symbol, price_change_pct * 100, self._risk.stop_loss_pct * 100,
                )
                try:
                    await self.place_paper_trade(symbol, "sell", sell_quantity, current_price)
                except ValueError as exc:
                    logger.warning("STOP-LOSS sell rejected for %s: %s", symbol, exc)
                return

            # Fee-aware gate: only sell on signal if profit exceeds round-trip fees + margin
            round_trip_fee = 2 * self._config.binance_fee_rate
            min_required = round_trip_fee + self._risk.min_profit_to_sell
            if price_change_pct < min_required:
                logger.debug(
                    "evaluate_symbol: SELL gated for %s — profit %.4f%% < required %.4f%%",
                    symbol, price_change_pct * 100, min_required * 100,
                )
                return

            # Momentum hold: if the position is profitable and upward momentum
            # is still strong, suppress the sell signal and let it run.
            # Stop-loss bypasses this check (handled above).
            if signal.indicators is not None and self._momentum_bullish_for_hold(
                signal.indicators
            ):
                hold_msg = (
                    f"HOLD (momentum) {symbol} @ {current_price:.2f} "
                    f"+{price_change_pct * 100:.2f}%"
                )
                self._last_action = hold_msg
                logger.info(
                    "MOMENTUM HOLD: sell suppressed for %s — profit=%.2f%% "
                    "momentum still bullish (fraction≥%.2f)",
                    symbol,
                    price_change_pct * 100,
                    self._risk.momentum_hold_fraction,
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
            "HFT loop starting for symbols=%s interval=%.2fs risk=%s",
            symbols,
            self._risk.hft_interval,
            self._risk.name,
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
                await asyncio.sleep(self._risk.hft_interval)
            except asyncio.CancelledError:
                logger.info("HFT loop cancelled during sleep.")
                raise

    def _momentum_bullish_for_hold(self, ind: "Indicators") -> bool:
        """Return True when momentum indicators collectively justify holding a profitable position.

        Evaluates up to four signals (only those with data):
          1. MACD bullish AND histogram positive        (always available)
          2. ROC above ``risk.roc_momentum_threshold``  (requires candle history)
          3. ADX strong AND +DI > -DI                   (requires candle history)
          4. Price above VWAP                           (requires candle history)

        At least **two** signals must have data; otherwise returns False
        (insufficient evidence to override a sell).

        Returns True when ``bullish_count / available_count >= risk.momentum_hold_fraction``.
        """
        bullish: int = 0
        available: int = 0

        # 1. MACD — always available as a core indicator
        available += 1
        if ind.macd_bullish and ind.macd_hist_rising:
            bullish += 1

        # 2. Rate of Change
        if ind.roc is not None:
            available += 1
            if ind.roc > self._risk.roc_momentum_threshold:
                bullish += 1

        # 3. ADX directional confirmation
        if ind.adx is not None:
            available += 1
            if ind.adx > self._risk.adx_trend_threshold and ind.trend_bullish:
                bullish += 1

        # 4. VWAP position
        if ind.vwap is not None:
            available += 1
            if ind.price_above_vwap:
                bullish += 1

        # Need at least two sources for a meaningful hold decision
        if available < 2:
            return False

        return (bullish / available) >= self._risk.momentum_hold_fraction

    def get_last_action(self) -> str:
        """Return a human-readable summary of the most recent trade action.

        Format: ``"BUY 0.001234 BTCUSDT @ 45000.00"`` or the initial
        ``"No action yet"`` string if no trade has been placed.
        """
        return self._last_action

    def get_active_strategy(self) -> str:
        """Return the name of the currently active trading strategy."""
        return f"{self._strategy.name.upper()} [{self._risk.name.upper()} risk]"
