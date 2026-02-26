"""Tests for src/prediction/predictor.py."""
import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prediction.predictor import Predictor, Indicators, Signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_indicators(
    rsi: float = 50.0,
    macd: float = 0.0,
    macd_signal: float = 0.0,
    macd_hist: float = 0.0,
    bb_upper: float = 46_000.0,
    bb_mid: float = 45_000.0,
    bb_lower: float = 44_000.0,
    price: float = 45_000.0,
) -> Indicators:
    """Return an Indicators dataclass with sensible neutral defaults."""
    return Indicators(
        rsi=rsi,
        macd=macd,
        macd_signal=macd_signal,
        macd_hist=macd_hist,
        bb_upper=bb_upper,
        bb_mid=bb_mid,
        bb_lower=bb_lower,
        price=price,
    )


# ---------------------------------------------------------------------------
# compute_indicators
# ---------------------------------------------------------------------------

class TestComputeIndicators:
    def test_sufficient_data_returns_indicators(self, sample_prices):
        """200 prices should produce a non-None Indicators object."""
        predictor = Predictor()
        result = predictor.compute_indicators(sample_prices)
        assert result is not None
        assert isinstance(result, Indicators)

    def test_insufficient_data_returns_none(self):
        """Fewer than 26 prices must return None."""
        predictor = Predictor()
        short_prices = [45_000.0 + i for i in range(25)]  # exactly 25
        result = predictor.compute_indicators(short_prices)
        assert result is None

    def test_exactly_zero_prices_returns_none(self):
        predictor = Predictor()
        assert predictor.compute_indicators([]) is None

    def test_exactly_26_prices_attempts_computation(self):
        """26 prices is the minimum required; result may be None due to NaN
        from MACD needing more warm-up, but must not raise an exception."""
        predictor = Predictor()
        prices = [45_000.0 + i * 10 for i in range(26)]
        # Just verify it doesn't throw; value may be None or Indicators.
        result = predictor.compute_indicators(prices)
        assert result is None or isinstance(result, Indicators)

    def test_returned_price_matches_last_element(self, sample_prices):
        predictor = Predictor()
        result = predictor.compute_indicators(sample_prices)
        assert result is not None
        assert result.price == pytest.approx(sample_prices[-1])

    def test_rsi_in_valid_range(self, sample_prices):
        predictor = Predictor()
        result = predictor.compute_indicators(sample_prices)
        assert result is not None
        assert 0.0 <= result.rsi <= 100.0


# ---------------------------------------------------------------------------
# Indicators properties
# ---------------------------------------------------------------------------

class TestIndicatorProperties:
    def test_rsi_oversold_when_below_30(self):
        ind = _make_indicators(rsi=25.0)
        assert ind.rsi_oversold is True
        assert ind.rsi_overbought is False

    def test_rsi_overbought_when_above_70(self):
        ind = _make_indicators(rsi=75.0)
        assert ind.rsi_overbought is True
        assert ind.rsi_oversold is False

    def test_rsi_neutral_range(self):
        ind = _make_indicators(rsi=50.0)
        assert ind.rsi_oversold is False
        assert ind.rsi_overbought is False

    def test_rsi_oversold_boundary_exactly_30(self):
        """rsi_oversold is rsi < 30, so 30.0 must be False."""
        ind = _make_indicators(rsi=30.0)
        assert ind.rsi_oversold is False

    def test_rsi_overbought_boundary_exactly_70(self):
        """rsi_overbought is rsi > 70, so 70.0 must be False."""
        ind = _make_indicators(rsi=70.0)
        assert ind.rsi_overbought is False

    def test_macd_bullish_when_macd_above_signal(self):
        ind = _make_indicators(macd=10.0, macd_signal=5.0)
        assert ind.macd_bullish is True

    def test_macd_not_bullish_when_below_signal(self):
        ind = _make_indicators(macd=5.0, macd_signal=10.0)
        assert ind.macd_bullish is False

    def test_price_near_lower_bb(self):
        # price <= bb_lower * 1.01 → True
        ind = _make_indicators(price=44_000.0, bb_lower=44_000.0)
        assert ind.price_near_lower_bb is True

    def test_price_not_near_lower_bb(self):
        ind = _make_indicators(price=45_000.0, bb_lower=40_000.0)
        assert ind.price_near_lower_bb is False

    def test_price_near_upper_bb(self):
        # price >= bb_upper * 0.99 → True
        ind = _make_indicators(price=46_000.0, bb_upper=46_000.0)
        assert ind.price_near_upper_bb is True

    def test_price_not_near_upper_bb(self):
        ind = _make_indicators(price=44_000.0, bb_upper=50_000.0)
        assert ind.price_near_upper_bb is False


# ---------------------------------------------------------------------------
# rule_based_signal
# ---------------------------------------------------------------------------

class TestRuleBasedSignal:
    def test_returns_signal_object(self):
        predictor = Predictor()
        ind = _make_indicators()
        result = predictor.rule_based_signal(ind)
        assert isinstance(result, Signal)

    def test_direction_is_valid(self):
        predictor = Predictor()
        ind = _make_indicators()
        result = predictor.rule_based_signal(ind)
        assert result.direction in ("buy", "sell", "hold")

    def test_confidence_between_zero_and_one(self, sample_prices):
        predictor = Predictor()
        ind = predictor.compute_indicators(sample_prices)
        assert ind is not None
        result = predictor.rule_based_signal(ind)
        assert 0.0 <= result.confidence <= 1.0

    def test_hold_when_no_conditions_triggered(self):
        """Neutral RSI + neutral MACD + price mid-band → hold."""
        predictor = Predictor()
        # Price is not near either band, RSI is neutral, MACD is neutral.
        ind = _make_indicators(
            rsi=50.0,
            macd=0.0,
            macd_signal=1.0,   # macd < signal → not bullish
            price=45_000.0,
            bb_upper=46_000.0,
            bb_lower=44_000.0,
        )
        result = predictor.rule_based_signal(ind)
        assert result.direction == "hold"
        assert result.confidence == pytest.approx(0.0)

    def test_buy_signal_on_oversold_rsi(self):
        """RSI < 30 should trigger a buy signal."""
        predictor = Predictor()
        ind = _make_indicators(rsi=20.0)
        result = predictor.rule_based_signal(ind)
        assert result.direction == "buy"
        assert result.confidence > 0.0

    def test_sell_signal_on_overbought_rsi(self):
        """RSI > 70 should trigger a sell signal."""
        predictor = Predictor()
        ind = _make_indicators(rsi=80.0)
        result = predictor.rule_based_signal(ind)
        assert result.direction == "sell"
        assert result.confidence > 0.0

    def test_reasoning_non_empty(self):
        predictor = Predictor()
        ind = _make_indicators()
        result = predictor.rule_based_signal(ind)
        assert result.reasoning != ""

    def test_indicators_attached_to_signal(self):
        predictor = Predictor()
        ind = _make_indicators()
        result = predictor.rule_based_signal(ind)
        assert result.indicators is ind


# ---------------------------------------------------------------------------
# get_signal (async)
# ---------------------------------------------------------------------------

class TestGetSignal:
    @pytest.mark.asyncio
    async def test_get_signal_is_async(self, sample_prices):
        """Calling get_signal with await must not raise a TypeError."""
        predictor = Predictor()
        result = await predictor.get_signal("BTCUSDT", sample_prices)
        assert isinstance(result, Signal)

    @pytest.mark.asyncio
    async def test_get_signal_direction_valid(self, sample_prices):
        predictor = Predictor()
        result = await predictor.get_signal("BTCUSDT", sample_prices)
        assert result.direction in ("buy", "sell", "hold")

    @pytest.mark.asyncio
    async def test_get_signal_insufficient_data_returns_hold(self):
        predictor = Predictor()
        result = await predictor.get_signal("BTCUSDT", [45_000.0] * 5)
        assert result.direction == "hold"
        assert result.confidence == pytest.approx(0.0)
        assert result.indicators is None

    @pytest.mark.asyncio
    async def test_news_sentiment_nudges_hold_to_buy(self):
        """A "hold" technical signal with positive sentiment > 0.3 → "buy"."""
        predictor = Predictor()

        # Patch compute_indicators to return neutral indicators that yield hold
        neutral_ind = _make_indicators(
            rsi=50.0,
            macd=0.0,
            macd_signal=1.0,
            price=45_000.0,
            bb_upper=46_000.0,
            bb_lower=44_000.0,
        )

        with patch.object(predictor, "compute_indicators", return_value=neutral_ind):
            # rule_based_signal on neutral_ind returns hold
            hold_signal = predictor.rule_based_signal(neutral_ind)
            assert hold_signal.direction == "hold", (
                "Precondition failed: neutral indicators should yield hold"
            )

            result = await predictor.get_signal(
                "BTCUSDT",
                [45_000.0] * 200,   # list length doesn't matter; compute_indicators is mocked
                news_sentiment=0.8,  # strongly positive
            )

        assert result.direction == "buy"
        assert result.confidence > 0.0

    @pytest.mark.asyncio
    async def test_news_sentiment_nudges_hold_to_sell(self):
        """A "hold" technical signal with negative sentiment < -0.3 → "sell"."""
        predictor = Predictor()
        neutral_ind = _make_indicators(
            rsi=50.0,
            macd=0.0,
            macd_signal=1.0,
            price=45_000.0,
            bb_upper=46_000.0,
            bb_lower=44_000.0,
        )

        with patch.object(predictor, "compute_indicators", return_value=neutral_ind):
            result = await predictor.get_signal(
                "BTCUSDT",
                [45_000.0] * 200,
                news_sentiment=-0.9,
            )

        assert result.direction == "sell"
        assert result.confidence > 0.0

    @pytest.mark.asyncio
    async def test_sentiment_below_threshold_does_not_nudge(self):
        """Sentiment within ±0.3 must not override a "hold" technical signal."""
        predictor = Predictor()
        neutral_ind = _make_indicators(
            rsi=50.0,
            macd=0.0,
            macd_signal=1.0,
            price=45_000.0,
            bb_upper=46_000.0,
            bb_lower=44_000.0,
        )

        with patch.object(predictor, "compute_indicators", return_value=neutral_ind):
            result = await predictor.get_signal(
                "BTCUSDT",
                [45_000.0] * 200,
                news_sentiment=0.2,  # below the 0.3 threshold
            )

        assert result.direction == "hold"

    @pytest.mark.asyncio
    async def test_strong_technical_signal_not_overridden_by_sentiment(
        self, sample_prices
    ):
        """Sentiment nudge only applies when the technical signal is "hold"."""
        predictor = Predictor()
        # Build a strongly bearish indicator set (RSI overbought → sell)
        sell_ind = _make_indicators(rsi=85.0)

        with patch.object(predictor, "compute_indicators", return_value=sell_ind):
            result = await predictor.get_signal(
                "BTCUSDT",
                sample_prices,
                news_sentiment=0.9,  # bullish sentiment — should NOT override sell
            )

        # The rule-based signal is sell; sentiment nudge applies only on hold.
        assert result.direction == "sell"
