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

        return ind

    # ---------------------------------------------------------------------------
    # Rule-based signal
    # ---------------------------------------------------------------------------

    def rule_based_signal(
        self,
        ind: Indicators,
        risk: "RiskProfile | None" = None,
    ) -> Signal:
        """Derive a directional signal from technical indicator rules.

        Classic conditions (weight 1.0 each, max 4 in each direction):
          BUY:  1. RSI strongly/mildly oversold
                2. MACD bullish + histogram rising
                3. Price near lower Bollinger Band
                4. Price above SMA + MACD bullish (trend confirm)

          SELL: 1. RSI strongly/mildly overbought
                2. MACD bearish + histogram falling
                3. Price near upper Bollinger Band
                4. Price below SMA + MACD bearish (trend confirm)

        Momentum conditions (volume/ADX/VWAP — only scored when data available):
          BUY:  5. Volume surge + RSI oversold         (weight 1.5)
                6. Momentum ROC + MACD bullish          (weight 1.0)
                7. ADX trend confirmed + VWAP above     (weight 0.75)
                8. VWAP reclaim + improving MACD hist   (weight 0.5)

          SELL: 5. Volume surge + RSI overbought        (weight 1.5)
                6. Momentum ROC negative + MACD bearish (weight 1.0)
                7. ADX trend confirmed + VWAP below     (weight 0.75)
                8. VWAP breakdown + worsening MACD hist (weight 0.5)

        Confidence = score / max_score, clamped to [0.0, 1.0].
        max_score = 4.0 (the 4 classic conditions).  Momentum signals are
        bonus score that can push confidence above 1.0 before clamping,
        so they increase conviction without inflating the denominator.
        Typical thresholds by profile: LOW≥0.60 (3 cond.), MEDIUM≥0.45 (2 cond.),
        HIGH≥0.30 (2 cond.), EXTREME≥0.20 (1 cond.).
        """
        if risk is None:
            from src.trading.risk import MEDIUM
            risk = MEDIUM

        # Resolve risk-profile-based ROC period (default to stored if unavailable)
        roc_val = ind.roc  # computed with default period; good enough for signal

        # Denominator = the 4 classic conditions only (each worth 1.0).
        # Momentum signals add bonus score ON TOP; the denominator stays at 4.0
        # so that 3 classic conditions fires at 0.75 confidence regardless of
        # whether candle data (and therefore momentum indicators) is available.
        # Using the full theoretical max (7.75) would require 3.5 points to
        # reach the 0.45 MEDIUM threshold, which never happens in normal markets.
        max_score = 4.0

        # ---------------------------------------------------------------
        # BUY scoring
        # ---------------------------------------------------------------
        buy_score: float = 0.0
        buy_triggers: list[str] = []

        # 1. RSI oversold (strong or mild — mutually exclusive)
        if ind.rsi < risk.rsi_oversold:
            buy_score += 1.0
            buy_triggers.append(f"RSI strongly oversold ({ind.rsi:.1f}<{risk.rsi_oversold})")
        elif ind.rsi < risk.rsi_weak_oversold:
            buy_score += 1.0
            buy_triggers.append(f"RSI mildly oversold ({ind.rsi:.1f}<{risk.rsi_weak_oversold})")

        # 2. MACD bullish + histogram rising
        if ind.macd_bullish and ind.macd_hist_rising:
            buy_score += 1.0
            buy_triggers.append(
                f"MACD bullish+rising hist ({ind.macd:.4f}>{ind.macd_signal:.4f})"
            )

        # 3. Price near lower Bollinger Band
        if ind.price <= ind.bb_lower * risk.bb_lower_tolerance:
            buy_score += 1.0
            buy_triggers.append(
                f"Price near lower BB ({ind.price:.4f}≤{ind.bb_lower * risk.bb_lower_tolerance:.4f})"
            )

        # 4. Price above SMA + MACD bullish (trend confirmation)
        if ind.price_above_mid_bb and ind.macd_bullish:
            buy_score += 1.0
            buy_triggers.append("Price above SMA+MACD bullish (trend confirm)")

        # 5. Volume surge + RSI oversold (weight 1.5)
        if (
            ind.volume_surge is not None
            and ind.volume_surge >= risk.volume_surge_threshold
            and ind.rsi < risk.rsi_oversold
        ):
            buy_score += 1.5
            buy_triggers.append(
                f"Volume surge {ind.volume_surge:.1f}x + RSI oversold (accumulation)"
            )

        # 6. Positive ROC momentum + MACD bullish (weight 1.0)
        if (
            roc_val is not None
            and roc_val > risk.roc_momentum_threshold
            and ind.macd_bullish
            and ind.macd_hist_rising
        ):
            buy_score += 1.0
            buy_triggers.append(
                f"ROC momentum +{roc_val:.2f}% + MACD bullish"
            )

        # 7. ADX strong trend + +DI dominant + price above VWAP (weight 0.75)
        if (
            ind.adx is not None
            and ind.adx > risk.adx_trend_threshold
            and ind.trend_bullish
            and ind.price_above_vwap
        ):
            buy_score += 0.75
            buy_triggers.append(
                f"ADX {ind.adx:.1f} trend + bullish DI + above VWAP"
            )

        # 8. Price reclaims VWAP + RSI mild oversold + hist improving (weight 0.5)
        if (
            ind.price_above_vwap
            and ind.rsi < risk.rsi_weak_oversold
            and ind.macd_hist_rising
        ):
            buy_score += 0.5
            buy_triggers.append("VWAP reclaim + improving momentum")

        # ---------------------------------------------------------------
        # SELL scoring
        # ---------------------------------------------------------------
        sell_score: float = 0.0
        sell_triggers: list[str] = []

        # 1. RSI overbought (strong or mild — mutually exclusive)
        if ind.rsi > risk.rsi_overbought:
            sell_score += 1.0
            sell_triggers.append(
                f"RSI strongly overbought ({ind.rsi:.1f}>{risk.rsi_overbought})"
            )
        elif ind.rsi > risk.rsi_weak_overbought:
            sell_score += 1.0
            sell_triggers.append(
                f"RSI mildly overbought ({ind.rsi:.1f}>{risk.rsi_weak_overbought})"
            )

        # 2. MACD bearish + histogram falling
        if not ind.macd_bullish and not ind.macd_hist_rising:
            sell_score += 1.0
            sell_triggers.append(
                f"MACD bearish+falling hist ({ind.macd:.4f}<{ind.macd_signal:.4f})"
            )

        # 3. Price near upper Bollinger Band
        if ind.price >= ind.bb_upper * risk.bb_upper_tolerance:
            sell_score += 1.0
            sell_triggers.append(
                f"Price near upper BB ({ind.price:.4f}≥{ind.bb_upper * risk.bb_upper_tolerance:.4f})"
            )

        # 4. Price below SMA + MACD bearish (trend confirmation)
        if ind.price_below_mid_bb and not ind.macd_bullish:
            sell_score += 1.0
            sell_triggers.append("Price below SMA+MACD bearish (trend confirm)")

        # 5. Volume surge + RSI overbought (distribution, weight 1.5)
        if (
            ind.volume_surge is not None
            and ind.volume_surge >= risk.volume_surge_threshold
            and ind.rsi > risk.rsi_overbought
        ):
            sell_score += 1.5
            sell_triggers.append(
                f"Volume surge {ind.volume_surge:.1f}x + RSI overbought (distribution)"
            )

        # 6. Negative ROC momentum + MACD bearish (weight 1.0)
        if (
            roc_val is not None
            and roc_val < -risk.roc_momentum_threshold
            and not ind.macd_bullish
            and not ind.macd_hist_rising
        ):
            sell_score += 1.0
            sell_triggers.append(
                f"ROC momentum {roc_val:.2f}% + MACD bearish"
            )

        # 7. ADX strong trend + -DI dominant + price below VWAP (weight 0.75)
        if (
            ind.adx is not None
            and ind.adx > risk.adx_trend_threshold
            and ind.trend_bearish
            and ind.price_below_vwap
        ):
            sell_score += 0.75
            sell_triggers.append(
                f"ADX {ind.adx:.1f} trend + bearish DI + below VWAP"
            )

        # 8. Price breaks below VWAP + RSI mild overbought + hist deteriorating (weight 0.5)
        if (
            ind.price_below_vwap
            and ind.rsi > risk.rsi_weak_overbought
            and not ind.macd_hist_rising
        ):
            sell_score += 0.5
            sell_triggers.append("VWAP breakdown + deteriorating momentum")

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

        signal = self.rule_based_signal(ind, risk)

        # Apply news sentiment nudge only when the technical signal is neutral.
        if signal.direction == "hold":
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
