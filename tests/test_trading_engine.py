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
from src.trading.risk import MEDIUM, RiskProfile


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
    risk: RiskProfile = None,
) -> TradeEngine:
    return TradeEngine(
        portfolio=portfolio or _make_portfolio(),
        predictor=predictor or _make_predictor(),
        config=config or _make_config(),
        risk=risk if risk is not None else MEDIUM,
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
    feed.get_candle_history.return_value = []   # no candle data by default
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
    indicators: Indicators = None,
) -> Signal:
    return Signal(direction=direction, confidence=confidence, reasoning=reasoning,
                  indicators=indicators)


def _make_indicators(
    price: float = 45_000.0,
    rsi: float = 50.0,
    macd: float = 10.0,
    macd_signal: float = 5.0,
    macd_hist: float = 5.0,
    bb_upper: float = 46_000.0,
    bb_mid: float = 45_000.0,
    bb_lower: float = 44_000.0,
    vwap: float = None,
    volume_surge: float = None,
    roc: float = None,
    adx: float = None,
    adx_plus_di: float = None,
    adx_minus_di: float = None,
) -> Indicators:
    """Build an Indicators instance with sensible defaults for testing."""
    return Indicators(
        rsi=rsi,
        macd=macd,
        macd_signal=macd_signal,
        macd_hist=macd_hist,
        bb_upper=bb_upper,
        bb_mid=bb_mid,
        bb_lower=bb_lower,
        price=price,
        vwap=vwap,
        volume_surge=volume_surge,
        roc=roc,
        adx=adx,
        adx_plus_di=adx_plus_di,
        adx_minus_di=adx_minus_di,
    )


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
        trade_fraction=0.05 via custom risk profile, initial_cash=10_000
        → trade_cash=500 → qty≈500/45000.
        """
        initial_cash = 10_000.0
        trade_fraction = 0.05
        price = 45_000.0

        # Custom risk with specific trade sizing; other fields from MEDIUM
        custom_risk = RiskProfile(
            **{**MEDIUM.__dict__, "trade_fraction": trade_fraction, "max_position_fraction": 0.5}
        )

        portfolio = _make_portfolio(initial_cash=initial_cash)
        predictor = _make_predictor()
        engine = _make_engine(portfolio=portfolio, predictor=predictor, risk=custom_risk)

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
    async def test_no_buy_when_confidence_below_threshold(self):
        """Confidence below 0.45 must NOT trigger a trade."""
        portfolio = _make_portfolio(initial_cash=10_000.0)
        predictor = _make_predictor()
        engine = _make_engine(portfolio=portfolio, predictor=predictor)

        prices = [45_000.0] * 200
        market_feed = _make_market_feed(price_history=prices, latest_price=45_000.0)
        news_feed = _make_news_feed()

        borderline_signal = _make_signal(direction="buy", confidence=0.44)
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
        """A sell signal liquidates a profitable position (above fee+margin gate)."""
        portfolio = _make_portfolio(initial_cash=100_000.0)
        predictor = _make_predictor()
        engine = _make_engine(portfolio=portfolio, predictor=predictor)

        # Buy at 45 000; sell at 45 300 = +0.667% which clears the
        # fee-aware gate (round-trip 0.2% + min margin 0.35% = 0.55%).
        buy_price = 45_000.0
        sell_price = 45_300.0
        await engine.place_paper_trade("BTCUSDT", "buy", 0.5, buy_price)
        assert "BTCUSDT" in portfolio.get_holdings()

        prices = [buy_price] * 200
        market_feed = _make_market_feed(price_history=prices, latest_price=sell_price)
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


# ---------------------------------------------------------------------------
# Momentum hold — sell suppression on profitable positions
# ---------------------------------------------------------------------------

class TestMomentumHold:
    """Tests for _momentum_bullish_for_hold and its integration in evaluate_symbol."""

    # -- Unit tests for _momentum_bullish_for_hold ---------------------------

    def test_hold_suppressed_when_only_one_indicator_available(self):
        """With only MACD available (no candle data), always return False."""
        engine = _make_engine()
        ind = _make_indicators(macd=10.0, macd_signal=5.0, macd_hist=5.0)
        # Only MACD available (roc/adx/vwap all None) → available=1 < 2 → False
        assert engine._momentum_bullish_for_hold(ind) is False

    def test_hold_true_when_all_four_indicators_bullish(self):
        """All four indicators bullish at MEDIUM fraction=0.75 → hold."""
        engine = _make_engine()  # MEDIUM: momentum_hold_fraction=0.75
        ind = _make_indicators(
            price=46_000.0,
            macd=10.0, macd_signal=5.0, macd_hist=5.0,    # MACD bullish
            roc=3.0,                                        # ROC > 2.0 (MEDIUM threshold)
            adx=30.0, adx_plus_di=25.0, adx_minus_di=10.0, # ADX strong bullish
            vwap=45_000.0,                                  # price(46k) > vwap(45k)
        )
        # 4/4 = 1.0 >= 0.75 → True
        assert engine._momentum_bullish_for_hold(ind) is True

    def test_hold_false_when_most_indicators_bearish(self):
        """Majority bearish → no hold at MEDIUM fraction=0.75."""
        engine = _make_engine()
        ind = _make_indicators(
            price=44_000.0,
            macd=5.0, macd_signal=10.0, macd_hist=-5.0,   # MACD bearish
            roc=-3.0,                                       # ROC negative
            adx=30.0, adx_plus_di=10.0, adx_minus_di=25.0, # ADX bearish direction
            vwap=45_000.0,                                  # price(44k) < vwap(45k)
        )
        # 0/4 = 0.0 < 0.75 → False
        assert engine._momentum_bullish_for_hold(ind) is False

    def test_hold_at_exact_fraction_boundary(self):
        """3 of 4 bullish = 0.75 which equals MEDIUM threshold exactly → hold."""
        engine = _make_engine()  # MEDIUM: momentum_hold_fraction=0.75
        ind = _make_indicators(
            price=46_000.0,
            macd=10.0, macd_signal=5.0, macd_hist=5.0,    # bullish
            roc=3.0,                                        # bullish
            adx=30.0, adx_plus_di=25.0, adx_minus_di=10.0, # bullish
            vwap=47_000.0,                                  # price(46k) < vwap(47k) → bearish
        )
        # 3/4 = 0.75 >= 0.75 → True
        assert engine._momentum_bullish_for_hold(ind) is True

    def test_hold_false_at_low_risk_with_partial_bullish(self):
        """LOW profile requires fraction=1.0 — 3/4 bullish is not enough."""
        low_risk = RiskProfile(**{**MEDIUM.__dict__,
                                  "name": "low_test",
                                  "momentum_hold_fraction": 1.0,
                                  "roc_momentum_threshold": 2.0,
                                  "adx_trend_threshold": 25.0})
        engine = _make_engine(risk=low_risk)
        ind = _make_indicators(
            price=46_000.0,
            macd=10.0, macd_signal=5.0, macd_hist=5.0,
            roc=3.0,
            adx=30.0, adx_plus_di=25.0, adx_minus_di=10.0,
            vwap=47_000.0,   # price below VWAP → bearish
        )
        # 3/4 = 0.75 < 1.0 → False
        assert engine._momentum_bullish_for_hold(ind) is False

    def test_hold_true_at_extreme_risk_with_two_of_four_bullish(self):
        """EXTREME fraction=0.40: 2/4 = 0.5 >= 0.40 → hold."""
        extreme_risk = RiskProfile(**{**MEDIUM.__dict__,
                                      "name": "extreme_test",
                                      "momentum_hold_fraction": 0.40,
                                      "roc_momentum_threshold": 2.0,
                                      "adx_trend_threshold": 25.0})
        engine = _make_engine(risk=extreme_risk)
        ind = _make_indicators(
            price=44_000.0,
            macd=10.0, macd_signal=5.0, macd_hist=5.0,    # bullish
            roc=3.0,                                        # bullish
            adx=30.0, adx_plus_di=10.0, adx_minus_di=25.0, # bearish direction
            vwap=45_000.0,                                  # price below VWAP → bearish
        )
        # 2/4 = 0.5 >= 0.40 → True
        assert engine._momentum_bullish_for_hold(ind) is True

    # -- Integration tests: evaluate_symbol sell suppression -----------------

    @pytest.mark.asyncio
    async def test_sell_suppressed_when_momentum_bullish(self):
        """Sell signal with bullish momentum and profitable position → no sale."""
        portfolio = _make_portfolio(initial_cash=100_000.0)
        predictor = _make_predictor()
        engine = _make_engine(portfolio=portfolio, predictor=predictor)

        buy_price = 45_000.0
        sell_price = 45_500.0  # +1.1% — above fee gate
        await engine.place_paper_trade("BTCUSDT", "buy", 0.5, buy_price)

        prices = [buy_price] * 200
        market_feed = _make_market_feed(price_history=prices, latest_price=sell_price)
        news_feed = _make_news_feed()

        # Indicators with fully bullish momentum (all 4 available)
        bullish_ind = _make_indicators(
            price=sell_price,
            macd=10.0, macd_signal=5.0, macd_hist=5.0,
            roc=3.0,
            adx=30.0, adx_plus_di=25.0, adx_minus_di=10.0,
            vwap=44_000.0,   # price(45.5k) above vwap(44k)
        )
        sell_signal = _make_signal(direction="sell", confidence=1.0, indicators=bullish_ind)

        with patch.object(predictor, "get_signal", new=AsyncMock(return_value=sell_signal)):
            await engine.evaluate_symbol("BTCUSDT", market_feed, news_feed)

        # Position should NOT have been sold
        assert "BTCUSDT" in portfolio.get_holdings(), "Sell was NOT suppressed by momentum hold"
        assert "HOLD" in engine.get_last_action()

    @pytest.mark.asyncio
    async def test_sell_proceeds_when_momentum_bearish(self):
        """Sell signal with bearish indicators → sell executes normally."""
        portfolio = _make_portfolio(initial_cash=100_000.0)
        predictor = _make_predictor()
        engine = _make_engine(portfolio=portfolio, predictor=predictor)

        buy_price = 45_000.0
        sell_price = 45_300.0  # +0.67% — above fee gate
        await engine.place_paper_trade("BTCUSDT", "buy", 0.5, buy_price)

        prices = [buy_price] * 200
        market_feed = _make_market_feed(price_history=prices, latest_price=sell_price)
        news_feed = _make_news_feed()

        # Indicators with fully bearish momentum → hold should NOT trigger
        bearish_ind = _make_indicators(
            price=sell_price,
            macd=5.0, macd_signal=10.0, macd_hist=-5.0,   # MACD bearish
            roc=-3.0,                                       # ROC negative
            adx=30.0, adx_plus_di=10.0, adx_minus_di=25.0, # ADX bearish
            vwap=46_000.0,                                  # price below VWAP
        )
        sell_signal = _make_signal(direction="sell", confidence=1.0, indicators=bearish_ind)

        with patch.object(predictor, "get_signal", new=AsyncMock(return_value=sell_signal)):
            await engine.evaluate_symbol("BTCUSDT", market_feed, news_feed)

        # Position should be liquidated
        assert "BTCUSDT" not in portfolio.get_holdings()

    @pytest.mark.asyncio
    async def test_stop_loss_not_suppressed_by_momentum(self):
        """Stop-loss fires even when momentum is bullish."""
        portfolio = _make_portfolio(initial_cash=100_000.0)
        predictor = _make_predictor()
        engine = _make_engine(portfolio=portfolio, predictor=predictor)

        buy_price = 45_000.0
        # Price drops 3% — exceeds MEDIUM stop_loss=2.5%
        stop_loss_price = buy_price * (1 - 0.03)
        await engine.place_paper_trade("BTCUSDT", "buy", 0.5, buy_price)

        prices = [buy_price] * 200
        market_feed = _make_market_feed(price_history=prices, latest_price=stop_loss_price)
        news_feed = _make_news_feed()

        # Even bullish indicators should NOT prevent stop-loss
        bullish_ind = _make_indicators(
            price=stop_loss_price,
            macd=10.0, macd_signal=5.0, macd_hist=5.0,
            roc=3.0,
            adx=30.0, adx_plus_di=25.0, adx_minus_di=10.0,
            vwap=44_000.0,
        )
        # signal direction doesn't matter here — stop-loss runs before signal check
        hold_signal = _make_signal(direction="hold", confidence=0.0, indicators=bullish_ind)

        with patch.object(predictor, "get_signal", new=AsyncMock(return_value=hold_signal)):
            await engine.evaluate_symbol("BTCUSDT", market_feed, news_feed)

        # Stop-loss should have liquidated the position
        assert "BTCUSDT" not in portfolio.get_holdings()

    @pytest.mark.asyncio
    async def test_sell_proceeds_when_signal_has_no_indicators(self):
        """Sell signal without indicators (indicators=None) → momentum hold skipped → sell fires."""
        portfolio = _make_portfolio(initial_cash=100_000.0)
        predictor = _make_predictor()
        engine = _make_engine(portfolio=portfolio, predictor=predictor)

        buy_price = 45_000.0
        sell_price = 45_300.0
        await engine.place_paper_trade("BTCUSDT", "buy", 0.5, buy_price)

        prices = [buy_price] * 200
        market_feed = _make_market_feed(price_history=prices, latest_price=sell_price)
        news_feed = _make_news_feed()

        # Signal without indicators (indicators=None) → no hold check possible
        sell_signal = _make_signal(direction="sell", confidence=1.0, indicators=None)

        with patch.object(predictor, "get_signal", new=AsyncMock(return_value=sell_signal)):
            await engine.evaluate_symbol("BTCUSDT", market_feed, news_feed)

        assert "BTCUSDT" not in portfolio.get_holdings()
