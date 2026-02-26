"""Tests for src/trading/engine.py."""
import asyncio
import pytest
import sys
import time
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import AppConfig
from src.portfolio.portfolio import Portfolio, Trade
from src.prediction.predictor import Predictor, Signal, Indicators
from src.market.feed import MarketFeed, PriceTick
from src.news.feed import NewsFeed
from src.trading.engine import TradeEngine


# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> AppConfig:
    defaults = dict(
        symbols=["BTCUSDT", "ETHUSDT"],
        initial_cash=10_000.0,
        hft_interval=0.1,
        trade_fraction=0.05,
        max_position_fraction=0.2,
        claude_api_key="test-key",
    )
    defaults.update(overrides)
    return AppConfig(**defaults)


def _make_portfolio(initial_cash: float = 10_000.0) -> Portfolio:
    return Portfolio(initial_cash=initial_cash)


def _make_predictor() -> Predictor:
    return Predictor()


def _make_engine(
    portfolio: Portfolio = None,
    predictor: Predictor = None,
    config: AppConfig = None,
) -> TradeEngine:
    return TradeEngine(
        portfolio=portfolio or _make_portfolio(),
        predictor=predictor or _make_predictor(),
        config=config or _make_config(),
    )


def _make_news_feed() -> MagicMock:
    """Return a mock NewsFeed that reports 0.0 sentiment."""
    feed = MagicMock(spec=NewsFeed)
    feed.get_avg_sentiment.return_value = 0.0
    return feed


def _make_market_feed(
    symbols: list[str] = None,
    price_history: list[float] = None,
    latest_price: float = 45_000.0,
) -> MagicMock:
    """Return a mock MarketFeed with configurable price history and latest tick."""
    if symbols is None:
        symbols = ["BTCUSDT"]
    if price_history is None:
        price_history = []

    feed = MagicMock(spec=MarketFeed)
    feed.get_price_history.return_value = price_history
    feed.get_latest.return_value = PriceTick(
        symbol="BTCUSDT",
        price=latest_price,
        volume=1234.5,
        bid=latest_price - 1,
        ask=latest_price + 1,
        timestamp=time.time(),
    )
    return feed


def _make_signal(
    direction: str = "hold",
    confidence: float = 0.0,
    reasoning: str = "Test signal.",
) -> Signal:
    return Signal(direction=direction, confidence=confidence, reasoning=reasoning)


# ---------------------------------------------------------------------------
# place_paper_trade
# ---------------------------------------------------------------------------

class TestPlacePaperTrade:
    @pytest.mark.asyncio
    async def test_place_paper_trade_creates_trade(self):
        portfolio = _make_portfolio(initial_cash=100_000.0)
        engine = _make_engine(portfolio=portfolio)

        trade = await engine.place_paper_trade("BTCUSDT", "buy", 0.1, 45_000.0)

        assert isinstance(trade, Trade)
        assert trade.symbol == "BTCUSDT"
        assert trade.side == "buy"
        assert trade.quantity == pytest.approx(0.1)
        assert trade.price == pytest.approx(45_000.0)

    @pytest.mark.asyncio
    async def test_place_paper_trade_updates_portfolio(self):
        portfolio = _make_portfolio(initial_cash=100_000.0)
        engine = _make_engine(portfolio=portfolio)

        await engine.place_paper_trade("BTCUSDT", "buy", 0.1, 45_000.0)

        holdings = portfolio.get_holdings()
        assert "BTCUSDT" in holdings
        assert holdings["BTCUSDT"].quantity == pytest.approx(0.1)

    @pytest.mark.asyncio
    async def test_place_paper_trade_deducts_cash(self):
        portfolio = _make_portfolio(initial_cash=100_000.0)
        engine = _make_engine(portfolio=portfolio)

        await engine.place_paper_trade("BTCUSDT", "buy", 1.0, 45_000.0)

        assert portfolio.get_cash() == pytest.approx(100_000.0 - 45_000.0)

    @pytest.mark.asyncio
    async def test_place_paper_trade_updates_last_action(self):
        engine = _make_engine(portfolio=_make_portfolio(initial_cash=100_000.0))

        await engine.place_paper_trade("BTCUSDT", "buy", 0.1, 45_000.0)

        action = engine.get_last_action()
        assert "BTCUSDT" in action
        assert "BUY" in action.upper() or "buy" in action.lower()

    @pytest.mark.asyncio
    async def test_place_paper_trade_returns_trade_with_uuid(self):
        engine = _make_engine(portfolio=_make_portfolio(initial_cash=100_000.0))
        trade = await engine.place_paper_trade("BTCUSDT", "buy", 0.01, 45_000.0)

        # Verify trade_id is a valid UUID string
        try:
            uuid.UUID(trade.trade_id)
        except ValueError:
            pytest.fail(f"trade_id is not a valid UUID: {trade.trade_id!r}")

    @pytest.mark.asyncio
    async def test_place_paper_trade_sell_updates_realized_pnl(self):
        portfolio = _make_portfolio(initial_cash=100_000.0)
        engine = _make_engine(portfolio=portfolio)

        await engine.place_paper_trade("BTCUSDT", "buy", 1.0, 45_000.0)
        await engine.place_paper_trade("BTCUSDT", "sell", 1.0, 50_000.0)

        assert portfolio.get_realized_pnl() == pytest.approx(5_000.0)

    @pytest.mark.asyncio
    async def test_place_paper_trade_propagates_value_error(self):
        """Insufficient cash should raise ValueError from the portfolio layer."""
        portfolio = _make_portfolio(initial_cash=100.0)
        engine = _make_engine(portfolio=portfolio)

        with pytest.raises(ValueError, match="Insufficient cash"):
            await engine.place_paper_trade("BTCUSDT", "buy", 1.0, 45_000.0)


# ---------------------------------------------------------------------------
# get_last_action
# ---------------------------------------------------------------------------

class TestGetLastAction:
    def test_initial_last_action_is_string(self):
        engine = _make_engine()
        result = engine.get_last_action()
        assert isinstance(result, str)

    def test_initial_last_action_non_empty(self):
        engine = _make_engine()
        assert engine.get_last_action() != ""

    def test_initial_last_action_default_message(self):
        engine = _make_engine()
        assert engine.get_last_action() == "No action yet"


# ---------------------------------------------------------------------------
# get_active_strategy
# ---------------------------------------------------------------------------

class TestGetActiveStrategy:
    def test_get_active_strategy_returns_string(self):
        engine = _make_engine()
        result = engine.get_active_strategy()
        assert isinstance(result, str)

    def test_get_active_strategy_non_empty(self):
        engine = _make_engine()
        assert engine.get_active_strategy() != ""

    def test_get_active_strategy_contains_known_terms(self):
        """Strategy name should mention the indicators we actually use."""
        engine = _make_engine()
        strategy = engine.get_active_strategy()
        # At least one recognisable indicator or sentiment keyword
        keywords = {"RSI", "MACD", "BB", "Sentiment", "News", "Blend"}
        assert any(kw in strategy for kw in keywords), (
            f"Strategy string '{strategy}' doesn't mention any expected keywords"
        )


# ---------------------------------------------------------------------------
# evaluate_symbol — hold on short history
# ---------------------------------------------------------------------------

class TestEvaluateSymbolShortHistory:
    @pytest.mark.asyncio
    async def test_no_trade_when_price_history_empty(self):
        """evaluate_symbol exits immediately if there is no price history."""
        portfolio = _make_portfolio(initial_cash=10_000.0)
        engine = _make_engine(portfolio=portfolio)

        market_feed = _make_market_feed(price_history=[])
        news_feed = _make_news_feed()

        await engine.evaluate_symbol("BTCUSDT", market_feed, news_feed)

        # No trade should have been placed
        assert portfolio.get_holdings() == {}
        assert engine.get_last_action() == "No action yet"

    @pytest.mark.asyncio
    async def test_no_trade_when_history_below_26(self):
        """Less than 26 prices → predictor returns hold → no trade placed."""
        portfolio = _make_portfolio(initial_cash=10_000.0)
        predictor = _make_predictor()
        engine = _make_engine(portfolio=portfolio, predictor=predictor)

        short_prices = [45_000.0 + i * 10 for i in range(20)]  # only 20 prices
        market_feed = _make_market_feed(price_history=short_prices)
        news_feed = _make_news_feed()

        await engine.evaluate_symbol("BTCUSDT", market_feed, news_feed)

        assert portfolio.get_holdings() == {}
        assert engine.get_last_action() == "No action yet"

    @pytest.mark.asyncio
    async def test_no_trade_placed_when_signal_is_hold(self):
        """A hold signal with any confidence should never produce a trade."""
        portfolio = _make_portfolio(initial_cash=10_000.0)
        predictor = _make_predictor()
        engine = _make_engine(portfolio=portfolio, predictor=predictor)

        # 200 prices so indicators can be computed, but we force hold via mock.
        prices = [45_000.0 + i * 5 for i in range(200)]
        market_feed = _make_market_feed(price_history=prices, latest_price=45_000.0)
        news_feed = _make_news_feed()

        hold_signal = _make_signal(direction="hold", confidence=0.0)
        with patch.object(
            predictor, "get_signal", new=AsyncMock(return_value=hold_signal)
        ):
            await engine.evaluate_symbol("BTCUSDT", market_feed, news_feed)

        assert portfolio.get_holdings() == {}
        assert engine.get_last_action() == "No action yet"


# ---------------------------------------------------------------------------
# evaluate_symbol — buy signal places a trade
# ---------------------------------------------------------------------------

class TestEvaluateSymbolBuySignal:
    @pytest.mark.asyncio
    async def test_buy_trade_placed_on_high_confidence_signal(self):
        """A buy signal with confidence > 0.5 should place a paper trade."""
        portfolio = _make_portfolio(initial_cash=10_000.0)
        predictor = _make_predictor()
        config = _make_config(trade_fraction=0.05, max_position_fraction=0.2)
        engine = _make_engine(portfolio=portfolio, predictor=predictor, config=config)

        prices = [45_000.0] * 200
        market_feed = _make_market_feed(price_history=prices, latest_price=45_000.0)
        news_feed = _make_news_feed()

        buy_signal = _make_signal(direction="buy", confidence=0.8)
        with patch.object(
            predictor, "get_signal", new=AsyncMock(return_value=buy_signal)
        ):
            await engine.evaluate_symbol("BTCUSDT", market_feed, news_feed)

        holdings = portfolio.get_holdings()
        assert "BTCUSDT" in holdings
        assert holdings["BTCUSDT"].quantity > 0.0

    @pytest.mark.asyncio
    async def test_buy_trade_quantity_matches_trade_fraction(self):
        """
        trade_fraction=0.05, initial_cash=10_000 → trade_cash=500 → qty≈500/45000.
        """
        initial_cash = 10_000.0
        trade_fraction = 0.05
        price = 45_000.0

        portfolio = _make_portfolio(initial_cash=initial_cash)
        predictor = _make_predictor()
        config = _make_config(
            trade_fraction=trade_fraction,
            max_position_fraction=0.5,  # generous cap so it doesn't block us
        )
        engine = _make_engine(portfolio=portfolio, predictor=predictor, config=config)

        prices = [price] * 200
        market_feed = _make_market_feed(price_history=prices, latest_price=price)
        news_feed = _make_news_feed()

        buy_signal = _make_signal(direction="buy", confidence=1.0)
        with patch.object(
            predictor, "get_signal", new=AsyncMock(return_value=buy_signal)
        ):
            await engine.evaluate_symbol("BTCUSDT", market_feed, news_feed)

        expected_qty = (trade_fraction * initial_cash) / price
        holdings = portfolio.get_holdings()
        assert "BTCUSDT" in holdings
        assert holdings["BTCUSDT"].quantity == pytest.approx(expected_qty, rel=1e-4)

    @pytest.mark.asyncio
    async def test_no_buy_when_confidence_at_threshold(self):
        """Confidence exactly equal to 0.5 must NOT trigger a trade (strict >)."""
        portfolio = _make_portfolio(initial_cash=10_000.0)
        predictor = _make_predictor()
        engine = _make_engine(portfolio=portfolio, predictor=predictor)

        prices = [45_000.0] * 200
        market_feed = _make_market_feed(price_history=prices, latest_price=45_000.0)
        news_feed = _make_news_feed()

        borderline_signal = _make_signal(direction="buy", confidence=0.5)
        with patch.object(
            predictor, "get_signal", new=AsyncMock(return_value=borderline_signal)
        ):
            await engine.evaluate_symbol("BTCUSDT", market_feed, news_feed)

        assert portfolio.get_holdings() == {}

    @pytest.mark.asyncio
    async def test_no_buy_when_no_cash(self):
        """Engine should silently skip buying when cash is 0."""
        portfolio = _make_portfolio(initial_cash=0.0)
        predictor = _make_predictor()
        engine = _make_engine(portfolio=portfolio, predictor=predictor)

        prices = [45_000.0] * 200
        market_feed = _make_market_feed(price_history=prices, latest_price=45_000.0)
        news_feed = _make_news_feed()

        buy_signal = _make_signal(direction="buy", confidence=1.0)
        with patch.object(
            predictor, "get_signal", new=AsyncMock(return_value=buy_signal)
        ):
            await engine.evaluate_symbol("BTCUSDT", market_feed, news_feed)

        assert portfolio.get_holdings() == {}


# ---------------------------------------------------------------------------
# evaluate_symbol — sell signal
# ---------------------------------------------------------------------------

class TestEvaluateSymbolSellSignal:
    @pytest.mark.asyncio
    async def test_sell_trade_placed_when_holding_exists(self):
        """A sell signal liquidates an existing position."""
        portfolio = _make_portfolio(initial_cash=100_000.0)
        predictor = _make_predictor()
        engine = _make_engine(portfolio=portfolio, predictor=predictor)

        # First, manually buy some BTC so there's something to sell.
        await engine.place_paper_trade("BTCUSDT", "buy", 0.5, 45_000.0)
        assert "BTCUSDT" in portfolio.get_holdings()

        prices = [45_000.0] * 200
        market_feed = _make_market_feed(price_history=prices, latest_price=45_000.0)
        news_feed = _make_news_feed()

        sell_signal = _make_signal(direction="sell", confidence=1.0)
        with patch.object(
            predictor, "get_signal", new=AsyncMock(return_value=sell_signal)
        ):
            await engine.evaluate_symbol("BTCUSDT", market_feed, news_feed)

        # Position should be liquidated
        assert "BTCUSDT" not in portfolio.get_holdings()

    @pytest.mark.asyncio
    async def test_sell_skipped_when_no_holding(self):
        """A sell signal with no existing position should be ignored gracefully."""
        portfolio = _make_portfolio(initial_cash=10_000.0)
        predictor = _make_predictor()
        engine = _make_engine(portfolio=portfolio, predictor=predictor)

        prices = [45_000.0] * 200
        market_feed = _make_market_feed(price_history=prices, latest_price=45_000.0)
        news_feed = _make_news_feed()

        sell_signal = _make_signal(direction="sell", confidence=1.0)
        with patch.object(
            predictor, "get_signal", new=AsyncMock(return_value=sell_signal)
        ):
            # Should not raise
            await engine.evaluate_symbol("BTCUSDT", market_feed, news_feed)

        assert portfolio.get_holdings() == {}


# ---------------------------------------------------------------------------
# evaluate_symbol — no latest tick
# ---------------------------------------------------------------------------

class TestEvaluateSymbolNoTick:
    @pytest.mark.asyncio
    async def test_no_trade_when_no_latest_tick(self):
        """If get_latest() returns None, the engine skips trade execution."""
        portfolio = _make_portfolio(initial_cash=10_000.0)
        predictor = _make_predictor()
        engine = _make_engine(portfolio=portfolio, predictor=predictor)

        prices = [45_000.0] * 200
        market_feed = _make_market_feed(price_history=prices, latest_price=45_000.0)
        market_feed.get_latest.return_value = None  # no current tick

        news_feed = _make_news_feed()

        buy_signal = _make_signal(direction="buy", confidence=1.0)
        with patch.object(
            predictor, "get_signal", new=AsyncMock(return_value=buy_signal)
        ):
            await engine.evaluate_symbol("BTCUSDT", market_feed, news_feed)

        assert portfolio.get_holdings() == {}


# ---------------------------------------------------------------------------
# HFT loop smoke test
# ---------------------------------------------------------------------------

class TestRunHftLoop:
    @pytest.mark.asyncio
    async def test_hft_loop_cancels_cleanly(self):
        """run_hft_loop should stop when CancelledError is raised."""
        engine = _make_engine()
        market_feed = _make_market_feed(price_history=[])
        news_feed = _make_news_feed()

        task = asyncio.create_task(
            engine.run_hft_loop(["BTCUSDT"], market_feed, news_feed)
        )
        await asyncio.sleep(0)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_hft_loop_calls_evaluate_symbol(self):
        """run_hft_loop should call evaluate_symbol for each tracked symbol."""
        engine = _make_engine()
        market_feed = _make_market_feed(price_history=[])
        news_feed = _make_news_feed()

        call_count = 0

        async def counting_evaluate(symbol, mf, nf):
            nonlocal call_count
            call_count += 1

        with patch.object(engine, "evaluate_symbol", side_effect=counting_evaluate):
            task = asyncio.create_task(
                engine.run_hft_loop(["BTCUSDT", "ETHUSDT"], market_feed, news_feed)
            )
            # Let the loop run for at least one full cycle (hft_interval=0.1 s).
            await asyncio.sleep(0.15)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # At least both symbols should have been evaluated once.
        assert call_count >= 2
