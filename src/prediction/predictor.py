from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import ta

if TYPE_CHECKING:
    from src.trading.risk import RiskProfile
    from src.market.feed import CandleTick
    from src.trading.strategy import TradingStrategy

logger = logging.getLogger(__name__)


@dataclass
class Indicators:
    # Core technical indicators
    rsi: float          # 0-100
    macd: float         # MACD line value
    macd_signal: float  # Signal line value
    macd_hist: float    # Histogram
    bb_upper: float     # Bollinger upper band
    bb_mid: float       # Bollinger middle (SMA)
    bb_lower: float     # Bollinger lower band
    price: float        # current price

    # Momentum / volume indicators — None when candle data unavailable
    vwap: float | None = None             # Volume-weighted average price
    volume_surge: float | None = None     # current_vol / avg_vol(N)
    volume_avg: float | None = None       # N-period average volume (reference)
    current_volume: float | None = None   # most recent candle's volume
    roc: float | None = None              # Rate of Change (%)
    adx: float | None = None             # Average Directional Index
    adx_plus_di: float | None = None     # +DI (bullish directional)
    adx_minus_di: float | None = None    # -DI (bearish directional)

    # Strategy-specific indicators
    ema9: float | None = None            # EMA 9-period  (prices)
    ema21: float | None = None           # EMA 21-period (prices)
    ema55: float | None = None           # EMA 55-period (None until 55 bars)
    stoch_k: float | None = None         # Stochastic %K  (requires candle OHLC)
    stoch_d: float | None = None         # Stochastic %D signal line
    donchian_high: float | None = None   # 20-period high excl. current bar
    donchian_low: float | None = None    # 20-period low  excl. current bar
    atr: float | None = None             # ATR 14-period  (requires candle OHLC)

    # ---------------------------------------------------------------------------
    # Classic indicator properties (unchanged)
    # ---------------------------------------------------------------------------

    @property
    def macd_bullish(self) -> bool:
        return self.macd > self.macd_signal

    @property
    def macd_hist_rising(self) -> bool:
        """Histogram is positive (momentum building bullish)."""
        return self.macd_hist > 0

    @property
    def rsi_oversold(self) -> bool:
        """Strong oversold — high conviction buy."""
        return self.rsi < 35

    @property
    def rsi_weak_oversold(self) -> bool:
        """Mild oversold — lower conviction buy signal."""
        return self.rsi < 45

    @property
    def rsi_overbought(self) -> bool:
        """Strong overbought — high conviction sell."""
        return self.rsi > 65

    @property
    def rsi_weak_overbought(self) -> bool:
        """Mild overbought — lower conviction sell signal."""
        return self.rsi > 55

    @property
    def price_near_lower_bb(self) -> bool:
        return self.price <= self.bb_lower * 1.015

    @property
    def price_near_upper_bb(self) -> bool:
        return self.price >= self.bb_upper * 0.985

    @property
    def price_below_mid_bb(self) -> bool:
        """Price below the BB midline (SMA) — mild bearish bias."""
        return self.price < self.bb_mid

    @property
    def price_above_mid_bb(self) -> bool:
        """Price above the BB midline (SMA) — mild bullish bias."""
        return self.price > self.bb_mid

    # ---------------------------------------------------------------------------
    # Momentum / volume properties
    # ---------------------------------------------------------------------------

    @property
    def price_above_vwap(self) -> bool:
        """Price above VWAP — bullish intraday bias."""
        return self.vwap is not None and self.price > self.vwap

    @property
    def price_below_vwap(self) -> bool:
        """Price below VWAP — bearish intraday bias."""
        return self.vwap is not None and self.price < self.vwap

    @property
    def trend_bullish(self) -> bool:
        """ADX confirms a strong bullish trend (+DI dominant)."""
        return (
            self.adx is not None
            and self.adx_plus_di is not None
            and self.adx_minus_di is not None
            and self.adx_plus_di > self.adx_minus_di
        )

    @property
    def trend_bearish(self) -> bool:
        """ADX confirms a strong bearish trend (-DI dominant)."""
        return (
            self.adx is not None
            and self.adx_plus_di is not None
            and self.adx_minus_di is not None
            and self.adx_minus_di > self.adx_plus_di
        )


@dataclass
class Signal:
    direction: str              # "buy", "sell", "hold"
    confidence: float           # 0.0 to 1.0
    reasoning: str              # explanation
    indicators: Indicators | None = None


class Predictor:
    def __init__(self) -> None:
        logger.debug("Predictor initialized")

    # ---------------------------------------------------------------------------
    # Indicator computation helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _compute_vwap(candles: list[CandleTick]) -> float | None:
        """Rolling VWAP over all provided candles: sum(typical_price*vol)/sum(vol)."""
        if not candles:
            return None
        total_vol = sum(c.volume for c in candles)
        if total_vol <= 0:
            return None
        return sum(c.typical_price * c.volume for c in candles) / total_vol

    @staticmethod
    def _compute_volume_surge(
        candles: list[CandleTick],
        period: int = 20,
    ) -> tuple[float, float, float] | None:
        """Return (surge_ratio, current_volume, avg_volume).

        surge_ratio = current_candle_volume / mean(last *period* prior candles).
        Returns None when there is insufficient history.
        """
        if len(candles) < period + 1:
            return None
        prior_vols = [c.volume for c in candles[-(period + 1):-1]]
        avg = float(np.mean(prior_vols))
        if avg <= 0:
            return None
        current = candles[-1].volume
        return current / avg, current, avg

    @staticmethod
    def _compute_roc(prices: list[float], period: int) -> float | None:
        """Rate of Change: (price[-1] - price[-period-1]) / price[-period-1] * 100."""
        if len(prices) < period + 1:
            return None
        past = prices[-(period + 1)]
        if past <= 0:
            return None
        return (prices[-1] - past) / past * 100.0

    @staticmethod
    def _compute_adx(
        candles: list[CandleTick],
        window: int = 14,
    ) -> tuple[float, float, float] | None:
        """Return (adx, plus_di, minus_di) using the ta library.

        Requires at least window*2 candles for reliable ADX.
        """
        if len(candles) < window * 2:
            return None
        try:
            high = pd.Series([c.high for c in candles], dtype=float)
            low = pd.Series([c.low for c in candles], dtype=float)
            close = pd.Series([c.close for c in candles], dtype=float)

            adx_obj = ta.trend.ADXIndicator(high=high, low=low, close=close, window=window)
            adx_val = adx_obj.adx().iloc[-1]
            plus_di = adx_obj.adx_pos().iloc[-1]
            minus_di = adx_obj.adx_neg().iloc[-1]

            if any(np.isnan(v) for v in (adx_val, plus_di, minus_di)):
                return None
            return float(adx_val), float(plus_di), float(minus_di)
        except Exception as exc:
            logger.debug("ADX computation failed: %s", exc)
            return None

    @staticmethod
    def _safe_ema(series: pd.Series, window: int) -> float | None:
        """Compute EMA for the given window; returns None if result is NaN."""
        try:
            val = ta.trend.EMAIndicator(series, window=window).ema_indicator().iloc[-1]
            if np.isnan(val):
                return None
            return float(val)
        except Exception as exc:
            logger.debug("EMA(%d) computation failed: %s", window, exc)
            return None

    @staticmethod
    def _compute_stochastic(
        candles: list["CandleTick"],
        window: int = 14,
        smooth_window: int = 3,
    ) -> tuple[float, float] | None:
        """Return (stoch_k, stoch_d) or None on failure."""
        try:
            high = pd.Series([c.high for c in candles], dtype=float)
            low = pd.Series([c.low for c in candles], dtype=float)
            close = pd.Series([c.close for c in candles], dtype=float)
            stoch = ta.momentum.StochasticOscillator(
                high=high, low=low, close=close,
                window=window, smooth_window=smooth_window,
            )
            k_val = stoch.stoch().iloc[-1]
            d_val = stoch.stoch_signal().iloc[-1]
            if np.isnan(k_val) or np.isnan(d_val):
                return None
            return float(k_val), float(d_val)
        except Exception as exc:
            logger.debug("Stochastic computation failed: %s", exc)
            return None

    @staticmethod
    def _compute_atr(
        candles: list["CandleTick"],
        window: int = 14,
    ) -> float | None:
        """Return ATR value or None on failure."""
        try:
            high = pd.Series([c.high for c in candles], dtype=float)
            low = pd.Series([c.low for c in candles], dtype=float)
            close = pd.Series([c.close for c in candles], dtype=float)
            atr = ta.volatility.AverageTrueRange(
                high=high, low=low, close=close, window=window
            )
            val = atr.average_true_range().iloc[-1]
            if np.isnan(val):
                return None
            return float(val)
        except Exception as exc:
            logger.debug("ATR computation failed: %s", exc)
            return None

    # ---------------------------------------------------------------------------
    # Main indicator computation
    # ---------------------------------------------------------------------------

    def compute_indicators(
        self,
        prices: list[float],
        candles: list[CandleTick] | None = None,
    ) -> Indicators | None:
        """Compute RSI, MACD, Bollinger Bands, and optional momentum indicators.

        Requires at least 26 prices for MACD computation. Returns None if the
        list is too short or if any core indicator value is NaN.

        When *candles* is provided, also computes VWAP, volume surge ratio,
        Rate of Change, and ADX — stored as optional fields on the returned
        Indicators. Callers that pass only *prices* receive None for those fields.
        """
        if len(prices) < 26:
            logger.debug(
                "Not enough prices to compute indicators: need 26, got %d",
                len(prices),
            )
            return None

        series = pd.Series(prices, dtype=float)

        try:
            # RSI
            rsi_val = ta.momentum.RSIIndicator(series, window=14).rsi().iloc[-1]

            # MACD
            macd_obj = ta.trend.MACD(
                series, window_slow=26, window_fast=12, window_sign=9
            )
            macd_val = macd_obj.macd().iloc[-1]
            macd_signal_val = macd_obj.macd_signal().iloc[-1]
            macd_hist_val = macd_obj.macd_diff().iloc[-1]

            # Bollinger Bands
            bb_obj = ta.volatility.BollingerBands(series, window=20, window_dev=2)
            bb_upper_val = bb_obj.bollinger_hband().iloc[-1]
            bb_mid_val = bb_obj.bollinger_mavg().iloc[-1]
            bb_lower_val = bb_obj.bollinger_lband().iloc[-1]

        except Exception as exc:
            logger.warning("Failed to compute indicators: %s", exc)
            return None

        core_values = {
            "rsi": rsi_val,
            "macd": macd_val,
            "macd_signal": macd_signal_val,
            "macd_hist": macd_hist_val,
            "bb_upper": bb_upper_val,
            "bb_mid": bb_mid_val,
            "bb_lower": bb_lower_val,
        }

        for name, val in core_values.items():
            if val is None or (isinstance(val, float) and np.isnan(val)):
                logger.debug("Indicator '%s' is NaN — skipping signal", name)
                return None

        ind = Indicators(
            rsi=float(rsi_val),
            macd=float(macd_val),
            macd_signal=float(macd_signal_val),
            macd_hist=float(macd_hist_val),
            bb_upper=float(bb_upper_val),
            bb_mid=float(bb_mid_val),
            bb_lower=float(bb_lower_val),
            price=float(prices[-1]),
        )

        # Optional momentum indicators (require candle OHLCV data)
        if candles and len(candles) >= 26:
            ind.vwap = self._compute_vwap(candles)

            surge_result = self._compute_volume_surge(candles)
            if surge_result is not None:
                ind.volume_surge, ind.current_volume, ind.volume_avg = surge_result

            ind.roc = self._compute_roc(prices, period=10)  # default; overridden by risk

            adx_result = self._compute_adx(candles)
            if adx_result is not None:
                ind.adx, ind.adx_plus_di, ind.adx_minus_di = adx_result

        # EMA crossover indicators (price series only; always attempt)
        ind.ema9 = self._safe_ema(series, 9)
        ind.ema21 = self._safe_ema(series, 21)
        ind.ema55 = self._safe_ema(series, 55)  # None when < 55 bars

        # Donchian channel (20-period, exclude current bar)
        if len(prices) >= 21:
            ind.donchian_high = float(max(prices[-21:-1]))
            ind.donchian_low = float(min(prices[-21:-1]))

        # Stochastic K/D (candles required, 17+ bars for warm-up)
        if candles and len(candles) >= 17:
            stoch_result = self._compute_stochastic(candles)
            if stoch_result is not None:
                ind.stoch_k, ind.stoch_d = stoch_result

        # ATR (candles required, 28+ for reliable values)
        if candles and len(candles) >= 28:
            atr_val = self._compute_atr(candles)
            if atr_val is not None:
                ind.atr = atr_val

        return ind

    # ---------------------------------------------------------------------------
    # Rule-based signal
    # ---------------------------------------------------------------------------

    def rule_based_signal(
        self,
        ind: Indicators,
        risk: "RiskProfile | None" = None,
        strategy: "TradingStrategy | None" = None,
        sentiment: float = 0.0,
    ) -> Signal:
        """Derive a directional signal by delegating to the active strategy.

        The strategy's ``score()`` method returns (buy_score, sell_score,
        buy_triggers, sell_triggers).  This method applies the decision logic:
        whichever score is higher (and > 0) determines direction; confidence
        is clamped to [0.0, 1.0] using the strategy's ``max_score``.

        Args:
            ind:       Computed technical indicators.
            risk:      Risk profile supplying threshold values.
            strategy:  Strategy to use for scoring.  Defaults to ClassicStrategy.
            sentiment: News sentiment score passed through to strategies that
                       use it (e.g. SentimentStrategy).
        """
        if risk is None:
            from src.trading.risk import MEDIUM
            risk = MEDIUM

        if strategy is None:
            from src.trading.strategy import DEFAULT_STRATEGY
            strategy = DEFAULT_STRATEGY

        buy_score, sell_score, buy_triggers, sell_triggers = strategy.score(
            ind, risk, sentiment
        )
        max_score = strategy.max_score

        # ---------------------------------------------------------------
        # Decision
        # ---------------------------------------------------------------
        if buy_score == 0.0 and sell_score == 0.0:
            return Signal(
                direction="hold",
                confidence=0.0,
                reasoning="No buy or sell conditions triggered; holding.",
                indicators=ind,
            )

        if buy_score >= sell_score:
            confidence = min(buy_score / max_score, 1.0)
            reasoning = "BUY — " + "; ".join(buy_triggers) + "."
            return Signal(
                direction="buy",
                confidence=confidence,
                reasoning=reasoning,
                indicators=ind,
            )
        else:
            confidence = min(sell_score / max_score, 1.0)
            reasoning = "SELL — " + "; ".join(sell_triggers) + "."
            return Signal(
                direction="sell",
                confidence=confidence,
                reasoning=reasoning,
                indicators=ind,
            )

    # ---------------------------------------------------------------------------
    # Public entry point
    # ---------------------------------------------------------------------------

    async def get_signal(
        self,
        symbol: str,
        prices: list[float],
        news_sentiment: float = 0.0,
        risk: "RiskProfile | None" = None,
        candles: list[CandleTick] | None = None,
        strategy: "TradingStrategy | None" = None,
    ) -> Signal:
        """Produce a final trading signal blending technical analysis and news sentiment.

        Steps:
          1. Compute technical indicators from the price series (and optionally candles).
          2. Derive a rule-based directional signal (using risk profile thresholds).
          3. If the rule-based signal is "hold", apply news sentiment nudge:
               weighted = sentiment * risk.sentiment_weight
               weighted >  threshold  -> nudge to "buy"
               weighted < -threshold  -> nudge to "sell"
          4. Return the final Signal.

        If indicators cannot be computed (insufficient data), a hold signal with
        zero confidence is returned immediately.
        """
        if risk is None:
            from src.trading.risk import MEDIUM
            risk = MEDIUM

        logger.debug("Computing signal for %s (prices=%d)", symbol, len(prices))

        ind = self.compute_indicators(prices, candles=candles)
        if ind is None:
            return Signal(
                direction="hold",
                confidence=0.0,
                reasoning=(
                    f"Insufficient price data for {symbol} "
                    f"(need 26 prices, got {len(prices)})."
                ),
                indicators=None,
            )

        signal = self.rule_based_signal(ind, risk, strategy=strategy, sentiment=news_sentiment)

        # Apply news sentiment nudge only when the technical signal is neutral
        # AND the active strategy does not handle sentiment itself.
        skip_nudge = strategy is not None and strategy.skip_sentiment_nudge
        if signal.direction == "hold" and not skip_nudge:
            threshold = risk.sentiment_nudge_threshold
            weighted = news_sentiment * risk.sentiment_weight
            if weighted > threshold:
                nudge_confidence = (
                    min((weighted - threshold) / max(1.0 - threshold, 1e-9), 1.0) * 0.5
                )
                signal = Signal(
                    direction="buy",
                    confidence=nudge_confidence,
                    reasoning=(
                        f"Neutral technical signal nudged to BUY by positive news "
                        f"sentiment (weighted={weighted:.2f} > {threshold})."
                    ),
                    indicators=ind,
                )
                logger.debug(
                    "%s: hold -> buy (sentiment nudge, weighted=%.2f)",
                    symbol, weighted,
                )
            elif weighted < -threshold:
                nudge_confidence = (
                    min((abs(weighted) - threshold) / max(1.0 - threshold, 1e-9), 1.0) * 0.5
                )
                signal = Signal(
                    direction="sell",
                    confidence=nudge_confidence,
                    reasoning=(
                        f"Neutral technical signal nudged to SELL by negative news "
                        f"sentiment (weighted={weighted:.2f} < -{threshold})."
                    ),
                    indicators=ind,
                )
                logger.debug(
                    "%s: hold -> sell (sentiment nudge, weighted=%.2f)",
                    symbol, weighted,
                )

        logger.info(
            "%s signal: %s (confidence=%.2f) sentiment=%.2f",
            symbol,
            signal.direction,
            signal.confidence,
            news_sentiment,
        )
        return signal
