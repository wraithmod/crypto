"""Tests for src/trading/strategy.py — all five strategy implementations."""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prediction.predictor import Indicators
from src.trading.risk import MEDIUM
from src.trading.strategy import (
    STRATEGIES,
    DEFAULT_STRATEGY,
    ClassicStrategy,
    TrendStrategy,
    BreakoutStrategy,
    ScalpStrategy,
    SentimentStrategy,
    TradingStrategy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ind(
    rsi: float = 50.0,
    macd: float = 0.0,
    macd_signal: float = 0.0,
    macd_hist: float = 0.0,
    bb_upper: float = 46_000.0,
    bb_mid: float = 45_000.0,
    bb_lower: float = 44_000.0,
    price: float = 45_000.0,
    vwap: float | None = None,
    volume_surge: float | None = None,
    roc: float | None = None,
    adx: float | None = None,
    adx_plus_di: float | None = None,
    adx_minus_di: float | None = None,
    ema9: float | None = None,
    ema21: float | None = None,
    ema55: float | None = None,
    stoch_k: float | None = None,
    stoch_d: float | None = None,
    donchian_high: float | None = None,
    donchian_low: float | None = None,
    atr: float | None = None,
) -> Indicators:
    return Indicators(
        rsi=rsi, macd=macd, macd_signal=macd_signal, macd_hist=macd_hist,
        bb_upper=bb_upper, bb_mid=bb_mid, bb_lower=bb_lower, price=price,
        vwap=vwap, volume_surge=volume_surge, roc=roc,
        adx=adx, adx_plus_di=adx_plus_di, adx_minus_di=adx_minus_di,
        ema9=ema9, ema21=ema21, ema55=ema55,
        stoch_k=stoch_k, stoch_d=stoch_d,
        donchian_high=donchian_high, donchian_low=donchian_low,
        atr=atr,
    )


# ---------------------------------------------------------------------------
# STRATEGIES registry
# ---------------------------------------------------------------------------

class TestStrategiesRegistry:
    def test_has_exactly_five_keys(self):
        assert set(STRATEGIES.keys()) == {"classic", "trend", "breakout", "scalp", "sentiment"}

    def test_all_values_are_trading_strategy_instances(self):
        for name, strat in STRATEGIES.items():
            assert isinstance(strat, TradingStrategy), f"{name} is not a TradingStrategy"

    def test_default_strategy_is_classic(self):
        assert DEFAULT_STRATEGY is STRATEGIES["classic"]

    def test_each_strategy_has_name_matching_key(self):
        for key, strat in STRATEGIES.items():
            assert strat.name == key

    def test_each_strategy_has_positive_max_score(self):
        for strat in STRATEGIES.values():
            assert strat.max_score > 0


# ---------------------------------------------------------------------------
# ClassicStrategy
# ---------------------------------------------------------------------------

class TestClassicStrategy:
    def setup_method(self):
        self.strategy = ClassicStrategy()

    def test_buy_score_on_oversold_rsi_and_macd_bullish(self):
        """RSI oversold + MACD bullish should produce positive buy score."""
        ind = _make_ind(rsi=25.0, macd=10.0, macd_signal=5.0, macd_hist=5.0)
        buy, sell, b_trig, s_trig = self.strategy.score(ind, MEDIUM)
        assert buy > 0.0
        assert len(b_trig) >= 1

    def test_sell_score_on_overbought_rsi(self):
        """RSI overbought should produce positive sell score."""
        ind = _make_ind(rsi=80.0)
        buy, sell, b_trig, s_trig = self.strategy.score(ind, MEDIUM)
        assert sell > 0.0

    def test_neutral_produces_zero_scores(self):
        """Neutral indicators: mid RSI, MACD neutral, price mid-band → both 0."""
        ind = _make_ind(
            rsi=50.0,
            macd=0.0, macd_signal=1.0, macd_hist=0.1,  # not bullish but hist positive → no MACD sell
            price=45_000.0, bb_mid=45_000.0,
        )
        buy, sell, _, _ = self.strategy.score(ind, MEDIUM)
        assert buy == pytest.approx(0.0)
        assert sell == pytest.approx(0.0)

    def test_skip_sentiment_nudge_is_false(self):
        assert self.strategy.skip_sentiment_nudge is False

    def test_returns_four_tuple(self):
        ind = _make_ind()
        result = self.strategy.score(ind, MEDIUM)
        assert len(result) == 4


# ---------------------------------------------------------------------------
# TrendStrategy
# ---------------------------------------------------------------------------

class TestTrendStrategy:
    def setup_method(self):
        self.strategy = TrendStrategy()

    def test_full_ema_bullish_alignment_gives_two_buy_score(self):
        """EMA9 > EMA21 > EMA55 full alignment → buy score += 2.0."""
        ind = _make_ind(ema9=105.0, ema21=100.0, ema55=95.0)
        buy, sell, b_trig, _ = self.strategy.score(ind, MEDIUM)
        assert buy >= 2.0
        assert any("Full EMA bullish" in t for t in b_trig)

    def test_partial_ema_bullish_alignment_gives_one_buy_score(self):
        """EMA9 > EMA21 but EMA55=None → buy score += 1.0."""
        ind = _make_ind(ema9=105.0, ema21=100.0, ema55=None)
        buy, sell, b_trig, _ = self.strategy.score(ind, MEDIUM)
        assert buy >= 1.0
        assert any("Partial EMA bullish" in t for t in b_trig)

    def test_full_ema_bearish_alignment_gives_two_sell_score(self):
        """EMA9 < EMA21 < EMA55 → sell score += 2.0."""
        ind = _make_ind(ema9=90.0, ema21=95.0, ema55=100.0)
        buy, sell, _, s_trig = self.strategy.score(ind, MEDIUM)
        assert sell >= 2.0
        assert any("Full EMA bearish" in t for t in s_trig)

    def test_ema_none_graceful_hold(self):
        """No EMA data and no ADX/ROC/VWAP → both scores 0."""
        ind = _make_ind()  # all strategy-specific fields None
        buy, sell, _, _ = self.strategy.score(ind, MEDIUM)
        assert buy == pytest.approx(0.0)
        assert sell == pytest.approx(0.0)

    def test_adx_confirmed_bullish_adds_score(self):
        """ADX > threshold + trend_bullish adds 1.0 to buy score."""
        ind = _make_ind(
            adx=30.0, adx_plus_di=25.0, adx_minus_di=10.0,
            ema9=105.0, ema21=100.0, ema55=95.0,
        )
        buy, sell, b_trig, _ = self.strategy.score(ind, MEDIUM)
        assert buy >= 3.0
        assert any("ADX" in t for t in b_trig)


# ---------------------------------------------------------------------------
# BreakoutStrategy
# ---------------------------------------------------------------------------

class TestBreakoutStrategy:
    def setup_method(self):
        self.strategy = BreakoutStrategy()

    def test_price_above_donchian_high_gives_two_buy_score(self):
        """Price > donchian_high → buy score += 2.0."""
        ind = _make_ind(price=50_001.0, donchian_high=50_000.0)
        buy, sell, b_trig, _ = self.strategy.score(ind, MEDIUM)
        assert buy >= 2.0
        assert any("Donchian high" in t for t in b_trig)

    def test_price_below_donchian_low_gives_two_sell_score(self):
        """Price < donchian_low → sell score += 2.0."""
        ind = _make_ind(price=39_999.0, donchian_low=40_000.0)
        buy, sell, _, s_trig = self.strategy.score(ind, MEDIUM)
        assert sell >= 2.0
        assert any("Donchian low" in t for t in s_trig)

    def test_volume_surge_adds_to_buy_score(self):
        """Breakout + volume surge → buy score >= 3.0."""
        ind = _make_ind(
            price=50_001.0, donchian_high=50_000.0,
            volume_surge=3.0,  # > MEDIUM threshold of 2.0
        )
        buy, sell, _, _ = self.strategy.score(ind, MEDIUM)
        assert buy >= 3.0

    def test_no_donchian_data_low_scores(self):
        """Without Donchian data, buy and sell scores stay low."""
        ind = _make_ind(rsi=50.0, macd=0.0, macd_hist=0.1)
        buy, sell, _, _ = self.strategy.score(ind, MEDIUM)
        assert buy < 2.0
        assert sell < 2.0


# ---------------------------------------------------------------------------
# ScalpStrategy
# ---------------------------------------------------------------------------

class TestScalpStrategy:
    def setup_method(self):
        self.strategy = ScalpStrategy()

    def test_strong_stoch_oversold_and_macd_rising_high_buy_score(self):
        """stoch_k=15 (<20) + MACD hist rising → buy score >= 2.5."""
        ind = _make_ind(
            stoch_k=15.0, stoch_d=20.0,  # K oversold and K < D → wait, K < D is bearish
            macd=5.0, macd_signal=3.0, macd_hist=2.0,  # MACD bullish
        )
        # stoch_k=15 < 20 → +1.5; stoch_k(15) < stoch_d(20) → K below D = bearish cross, no +1.0
        # MACD bullish + hist rising → +1.0
        buy, sell, b_trig, _ = self.strategy.score(ind, MEDIUM)
        assert buy >= 2.0

    def test_stoch_oversold_k_above_d_bullish_cross(self):
        """stoch_k=15 < 20 + K > D (bullish cross) → buy score += 1.5 + 1.0 = 2.5."""
        ind = _make_ind(
            stoch_k=15.0, stoch_d=10.0,  # K > D = bullish cross
            macd=0.0, macd_signal=1.0, macd_hist=0.1,  # MACD not bullish
        )
        buy, sell, _, _ = self.strategy.score(ind, MEDIUM)
        assert buy >= 2.5

    def test_stoch_overbought_gives_sell_score(self):
        """stoch_k=85 > 80 → sell score += 1.5."""
        ind = _make_ind(stoch_k=85.0, stoch_d=80.0)
        buy, sell, _, s_trig = self.strategy.score(ind, MEDIUM)
        assert sell >= 1.5
        assert any("overbought" in t for t in s_trig)

    def test_fallback_to_macd_when_stoch_none(self):
        """No stochastic data → falls back to MACD + VWAP only."""
        # MACD bullish fallback
        ind = _make_ind(macd=5.0, macd_signal=3.0, macd_hist=2.0)
        buy, sell, b_trig, _ = self.strategy.score(ind, MEDIUM)
        assert buy >= 1.5
        assert any("fallback" in t for t in b_trig)

    def test_fallback_macd_bearish_gives_sell(self):
        """No stochastic + MACD bearish → sell via fallback."""
        ind = _make_ind(macd=3.0, macd_signal=5.0, macd_hist=-2.0)
        buy, sell, _, s_trig = self.strategy.score(ind, MEDIUM)
        assert sell >= 1.5
        assert any("fallback" in t for t in s_trig)

    def test_skip_sentiment_nudge_is_false(self):
        assert self.strategy.skip_sentiment_nudge is False


# ---------------------------------------------------------------------------
# SentimentStrategy
# ---------------------------------------------------------------------------

class TestSentimentStrategy:
    def setup_method(self):
        self.strategy = SentimentStrategy()

    def test_strong_positive_sentiment_gives_two_buy_score(self):
        """sentiment=0.8 > 0.3 → buy score += 2.0."""
        ind = _make_ind(rsi=50.0)
        buy, sell, b_trig, _ = self.strategy.score(ind, MEDIUM, sentiment=0.8)
        assert buy >= 2.0
        assert any("Strong positive" in t for t in b_trig)

    def test_mild_positive_sentiment_gives_one_buy_score(self):
        """sentiment=0.2 > 0.1 but not > 0.3 → buy score += 1.0."""
        ind = _make_ind(rsi=50.0)
        buy, sell, b_trig, _ = self.strategy.score(ind, MEDIUM, sentiment=0.2)
        assert buy >= 1.0
        # Should not trigger "Strong positive"
        assert not any("Strong positive" in t for t in b_trig)

    def test_strong_negative_sentiment_gives_two_sell_score(self):
        """sentiment=-0.8 < -0.3 → sell score += 2.0."""
        ind = _make_ind(rsi=50.0)
        buy, sell, _, s_trig = self.strategy.score(ind, MEDIUM, sentiment=-0.8)
        assert sell >= 2.0
        assert any("Strong negative" in t for t in s_trig)

    def test_neutral_sentiment_zero_sentiment_score(self):
        """sentiment=0.0 → neither buy nor sell sentiment condition fires."""
        ind = _make_ind(rsi=50.0, macd=0.0, macd_signal=1.0, macd_hist=0.1)
        buy, sell, b_trig, s_trig = self.strategy.score(ind, MEDIUM, sentiment=0.0)
        assert not any("sentiment" in t.lower() for t in b_trig)
        assert not any("sentiment" in t.lower() for t in s_trig)

    def test_skip_sentiment_nudge_is_true(self):
        """SentimentStrategy handles sentiment itself — no nudge needed."""
        assert self.strategy.skip_sentiment_nudge is True

    def test_rsi_not_overbought_adds_to_buy(self):
        """RSI < rsi_overbought gives +1.0 as technical confirmation."""
        ind = _make_ind(rsi=40.0)  # well below MEDIUM overbought=65
        buy, sell, b_trig, _ = self.strategy.score(ind, MEDIUM, sentiment=0.5)
        # 2.0 (sentiment) + 1.0 (RSI confirm) = 3.0 minimum
        assert buy >= 3.0
        assert any("not overbought" in t for t in b_trig)
