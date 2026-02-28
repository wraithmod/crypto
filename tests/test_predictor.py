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
    # New optional fields — default None for backward compatibility
    ema9: float | None = None,
    ema21: float | None = None,
    ema55: float | None = None,
    stoch_k: float | None = None,
    stoch_d: float | None = None,
    donchian_high: float | None = None,
    donchian_low: float | None = None,
    atr: float | None = None,
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
        ema9=ema9,
        ema21=ema21,
        ema55=ema55,
        stoch_k=stoch_k,
        stoch_d=stoch_d,
        donchian_high=donchian_high,
        donchian_low=donchian_low,
        atr=atr,
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

    def test_rsi_oversold_boundary_exactly_35(self):
        """rsi_oversold is rsi < 35, so 35.0 must be False."""
        ind = _make_indicators(rsi=35.0)
        assert ind.rsi_oversold is False

    def test_rsi_overbought_boundary_exactly_65(self):
        """rsi_overbought is rsi > 65, so 65.0 must be False."""
        ind = _make_indicators(rsi=65.0)
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
        # macd_hist > 0 prevents the MACD bearish sell condition from firing
        # (requires not_bullish AND not_hist_rising simultaneously).
        ind = _make_indicators(
            rsi=50.0,
            macd=0.0,
            macd_signal=1.0,   # macd < signal → not bullish
            macd_hist=0.1,     # positive hist → hist_rising=True → no MACD sell
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

        # macd_hist=0.1 prevents MACD bearish sell trigger so rule_based yields hold
        neutral_ind = _make_indicators(
            rsi=50.0,
            macd=0.0,
            macd_signal=1.0,
            macd_hist=0.1,
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
            macd_hist=0.1,
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
            macd_hist=0.1,
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


# ---------------------------------------------------------------------------
# New indicators: EMA, Donchian, strategy routing
# ---------------------------------------------------------------------------

class TestNewIndicators:
    def test_ema9_and_ema21_populated_on_50_prices(self):
        """50 prices is enough for EMA9 and EMA21 but not EMA55."""
        predictor = Predictor()
        prices = [45_000.0 + i * 10 for i in range(50)]
        result = predictor.compute_indicators(prices)
        assert result is not None
        assert result.ema9 is not None
        assert result.ema21 is not None

    def test_ema55_none_on_26_prices(self):
        """26 prices is below the 55-bar warm-up — EMA55 must be None."""
        predictor = Predictor()
        prices = [45_000.0 + i * 10 for i in range(26)]
        result = predictor.compute_indicators(prices)
        # result may be None if MACD NaN, but if we get one, ema55 must be None
        if result is not None:
            assert result.ema55 is None

    def test_ema55_populated_on_60_prices(self):
        """60 prices provides enough warm-up for EMA55."""
        predictor = Predictor()
        prices = [45_000.0 + i * 10 for i in range(60)]
        result = predictor.compute_indicators(prices)
        assert result is not None
        assert result.ema55 is not None

    def test_donchian_populated_on_sufficient_prices(self):
        """donchian_high/low set correctly from the 20 bars before current."""
        predictor = Predictor()
        # Build a price series where the last 20 bars (excl. current) are 44k-45k.
        prices = [45_000.0] * 200
        result = predictor.compute_indicators(prices)
        assert result is not None
        assert result.donchian_high is not None
        assert result.donchian_low is not None
        # All prices equal → high == low == 45_000
        assert result.donchian_high == pytest.approx(45_000.0)
        assert result.donchian_low == pytest.approx(45_000.0)

    def test_donchian_high_is_max_of_prior_20(self):
        """donchian_high should equal the max of prices[-21:-1]."""
        predictor = Predictor()
        prices = [44_000.0] * 180 + [46_000.0] * 19 + [43_000.0]
        result = predictor.compute_indicators(prices)
        assert result is not None
        assert result.donchian_high == pytest.approx(46_000.0)

    def test_new_indicator_fields_default_none_without_candles(self, sample_prices):
        """Without candle data, stoch_k/stoch_d/atr remain None."""
        predictor = Predictor()
        result = predictor.compute_indicators(sample_prices)
        assert result is not None
        assert result.stoch_k is None
        assert result.stoch_d is None
        assert result.atr is None


class TestRuleBasedSignalWithStrategy:
    def test_trend_strategy_hold_when_all_ema_none(self):
        """TrendStrategy with no EMA/ADX/ROC/VWAP data → hold."""
        from src.trading.strategy import TrendStrategy
        predictor = Predictor()
        ind = _make_indicators()  # no EMA, no ADX, neutral RSI
        result = predictor.rule_based_signal(ind, strategy=TrendStrategy())
        assert result.direction == "hold"

    def test_trend_strategy_buy_on_full_ema_alignment(self):
        """TrendStrategy: EMA9>EMA21>EMA55 + ADX bullish → buy."""
        from src.trading.strategy import TrendStrategy
        predictor = Predictor()
        ind = _make_indicators(
            ema9=105.0, ema21=100.0, ema55=95.0,
            price=101.0,
        )
        result = predictor.rule_based_signal(ind, strategy=TrendStrategy())
        assert result.direction == "buy"
        assert result.confidence >= 0.5  # 2.0 / 4.0

    def test_classic_strategy_default_when_none(self):
        """rule_based_signal with strategy=None uses ClassicStrategy (default)."""
        predictor = Predictor()
        ind = _make_indicators(rsi=20.0)  # strongly oversold → buy
        result = predictor.rule_based_signal(ind, strategy=None)
        assert result.direction == "buy"

    def test_sentiment_strategy_skip_nudge_flag(self):
        """get_signal with SentimentStrategy: skip_sentiment_nudge=True suppresses nudge."""
        import asyncio
        from src.trading.strategy import SentimentStrategy
        predictor = Predictor()
        # Neutral technical indicator → would normally get sentiment nudge
        neutral_ind = _make_indicators(
            rsi=50.0, macd=0.0, macd_signal=1.0, macd_hist=0.1,
        )

        async def _run():
            with patch.object(predictor, "compute_indicators", return_value=neutral_ind):
                return await predictor.get_signal(
                    "BTCUSDT",
                    [45_000.0] * 200,
                    news_sentiment=0.9,   # strong positive — but strategy owns sentiment
                    strategy=SentimentStrategy(),
                )

        result = asyncio.get_event_loop().run_until_complete(_run())
        # SentimentStrategy scores sentiment=0.9 → 2.0 points → buy
        assert result.direction == "buy"
