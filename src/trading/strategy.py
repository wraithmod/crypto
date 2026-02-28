"""Trading strategy implementations.

Each strategy encapsulates signal-scoring logic for a specific market regime.
Strategies are decoupled from risk profiles: ``--strategy`` controls which
indicators fire; ``--risk`` controls position sizing, thresholds, and
stop-loss levels.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.prediction.predictor import Indicators
    from src.trading.risk import RiskProfile


class TradingStrategy(ABC):
    """Abstract base for all signal strategies."""

    name: str
    description: str
    max_score: float = 4.0
    skip_sentiment_nudge: bool = False

    @abstractmethod
    def score(
        self,
        ind: "Indicators",
        risk: "RiskProfile",
        sentiment: float = 0.0,
    ) -> tuple[float, float, list[str], list[str]]:
        """Compute buy/sell scores and trigger descriptions.

        Returns:
            (buy_score, sell_score, buy_triggers, sell_triggers)
        """
        ...


class ClassicStrategy(TradingStrategy):
    """RSI + MACD + Bollinger Bands — original default strategy."""

    name = "classic"
    description = "RSI + MACD + BB + Sentiment — best for ranging/sideways markets"
    max_score = 4.0
    skip_sentiment_nudge = False

    def score(
        self,
        ind: "Indicators",
        risk: "RiskProfile",
        sentiment: float = 0.0,
    ) -> tuple[float, float, list[str], list[str]]:
        roc_val = ind.roc

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
            buy_triggers.append(f"ROC momentum +{roc_val:.2f}% + MACD bullish")

        # 7. ADX strong trend + +DI dominant + price above VWAP (weight 0.75)
        if (
            ind.adx is not None
            and ind.adx > risk.adx_trend_threshold
            and ind.trend_bullish
            and ind.price_above_vwap
        ):
            buy_score += 0.75
            buy_triggers.append(f"ADX {ind.adx:.1f} trend + bullish DI + above VWAP")

        # 8. Price reclaims VWAP + RSI mild oversold + hist improving (weight 0.5)
        if ind.price_above_vwap and ind.rsi < risk.rsi_weak_oversold and ind.macd_hist_rising:
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

        # 5. Volume surge + RSI overbought (weight 1.5)
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
            sell_triggers.append(f"ROC momentum {roc_val:.2f}% + MACD bearish")

        # 7. ADX strong trend + -DI dominant + price below VWAP (weight 0.75)
        if (
            ind.adx is not None
            and ind.adx > risk.adx_trend_threshold
            and ind.trend_bearish
            and ind.price_below_vwap
        ):
            sell_score += 0.75
            sell_triggers.append(f"ADX {ind.adx:.1f} trend + bearish DI + below VWAP")

        # 8. Price breaks below VWAP + RSI mild overbought + hist deteriorating (weight 0.5)
        if (
            ind.price_below_vwap
            and ind.rsi > risk.rsi_weak_overbought
            and not ind.macd_hist_rising
        ):
            sell_score += 0.5
            sell_triggers.append("VWAP breakdown + deteriorating momentum")

        return buy_score, sell_score, buy_triggers, sell_triggers


class TrendStrategy(TradingStrategy):
    """EMA crossover + ADX confirmation — best for sustained bull/bear runs."""

    name = "trend"
    description = "EMA 9/21/55 crossover + ADX confirmation — best for trending markets"
    max_score = 4.0
    skip_sentiment_nudge = False

    def score(
        self,
        ind: "Indicators",
        risk: "RiskProfile",
        sentiment: float = 0.0,
    ) -> tuple[float, float, list[str], list[str]]:
        buy_score: float = 0.0
        buy_triggers: list[str] = []

        # 1. EMA alignment (exclusive: full w2.0, partial w1.0)
        if (
            ind.ema9 is not None and ind.ema21 is not None
            and ind.ema55 is not None
            and ind.ema9 > ind.ema21 > ind.ema55
        ):
            buy_score += 2.0
            buy_triggers.append(
                f"Full EMA bullish alignment (EMA9={ind.ema9:.2f}>EMA21={ind.ema21:.2f}>EMA55={ind.ema55:.2f})"
            )
        elif (
            ind.ema9 is not None and ind.ema21 is not None
            and ind.ema9 > ind.ema21
        ):
            buy_score += 1.0
            buy_triggers.append(
                f"Partial EMA bullish alignment (EMA9={ind.ema9:.2f}>EMA21={ind.ema21:.2f})"
            )

        # 2. ADX confirmed bullish trend
        if (
            ind.adx is not None
            and ind.adx > risk.adx_trend_threshold
            and ind.trend_bullish
        ):
            buy_score += 1.0
            buy_triggers.append(
                f"ADX {ind.adx:.1f}>{risk.adx_trend_threshold} confirmed bullish trend"
            )

        # 3. ROC positive + MACD bullish
        if (
            ind.roc is not None
            and ind.roc > risk.roc_momentum_threshold
            and ind.macd_bullish
        ):
            buy_score += 0.5
            buy_triggers.append(f"ROC +{ind.roc:.2f}% + MACD bullish momentum")

        # 4. Price above EMA21 + above VWAP
        if (
            ind.ema21 is not None
            and ind.price > ind.ema21
            and ind.price_above_vwap
        ):
            buy_score += 0.5
            buy_triggers.append("Price above EMA21 and VWAP")

        # ---------------------------------------------------------------
        # SELL scoring
        # ---------------------------------------------------------------
        sell_score: float = 0.0
        sell_triggers: list[str] = []

        # 1. EMA alignment (exclusive)
        if (
            ind.ema9 is not None and ind.ema21 is not None
            and ind.ema55 is not None
            and ind.ema9 < ind.ema21 < ind.ema55
        ):
            sell_score += 2.0
            sell_triggers.append(
                f"Full EMA bearish alignment (EMA9={ind.ema9:.2f}<EMA21={ind.ema21:.2f}<EMA55={ind.ema55:.2f})"
            )
        elif (
            ind.ema9 is not None and ind.ema21 is not None
            and ind.ema9 < ind.ema21
        ):
            sell_score += 1.0
            sell_triggers.append(
                f"Partial EMA bearish alignment (EMA9={ind.ema9:.2f}<EMA21={ind.ema21:.2f})"
            )

        # 2. ADX confirmed bearish trend
        if (
            ind.adx is not None
            and ind.adx > risk.adx_trend_threshold
            and ind.trend_bearish
        ):
            sell_score += 1.0
            sell_triggers.append(
                f"ADX {ind.adx:.1f}>{risk.adx_trend_threshold} confirmed bearish trend"
            )

        # 3. ROC negative + MACD bearish
        if (
            ind.roc is not None
            and ind.roc < -risk.roc_momentum_threshold
            and not ind.macd_bullish
        ):
            sell_score += 0.5
            sell_triggers.append(f"ROC {ind.roc:.2f}% + MACD bearish momentum")

        # 4. Price below EMA21 + below VWAP
        if (
            ind.ema21 is not None
            and ind.price < ind.ema21
            and ind.price_below_vwap
        ):
            sell_score += 0.5
            sell_triggers.append("Price below EMA21 and VWAP")

        return buy_score, sell_score, buy_triggers, sell_triggers


class BreakoutStrategy(TradingStrategy):
    """Donchian channel breakout + volume confirmation — best for volatile altcoins."""

    name = "breakout"
    description = "Donchian channel N-period high/low breaks + volume — best for volatile altcoins"
    max_score = 4.0
    skip_sentiment_nudge = False

    def score(
        self,
        ind: "Indicators",
        risk: "RiskProfile",
        sentiment: float = 0.0,
    ) -> tuple[float, float, list[str], list[str]]:
        buy_score: float = 0.0
        buy_triggers: list[str] = []

        # 1. Price breaks above Donchian high (20-period)
        if ind.donchian_high is not None and ind.price > ind.donchian_high:
            buy_score += 2.0
            buy_triggers.append(
                f"Price {ind.price:.4f} breaks above Donchian high {ind.donchian_high:.4f}"
            )

        # 2. Volume surge confirms breakout
        if (
            ind.volume_surge is not None
            and ind.volume_surge >= risk.volume_surge_threshold
        ):
            buy_score += 1.0
            buy_triggers.append(f"Volume surge {ind.volume_surge:.1f}x confirms breakout")

        # 3. MACD bullish + hist rising
        if ind.macd_bullish and ind.macd_hist_rising:
            buy_score += 0.5
            buy_triggers.append("MACD bullish momentum")

        # 4. RSI below overbought — room to run
        if ind.rsi < risk.rsi_overbought:
            buy_score += 0.5
            buy_triggers.append(f"RSI {ind.rsi:.1f} not overbought — room to run")

        # ---------------------------------------------------------------
        # SELL scoring
        # ---------------------------------------------------------------
        sell_score: float = 0.0
        sell_triggers: list[str] = []

        # 1. Price breaks below Donchian low
        if ind.donchian_low is not None and ind.price < ind.donchian_low:
            sell_score += 2.0
            sell_triggers.append(
                f"Price {ind.price:.4f} breaks below Donchian low {ind.donchian_low:.4f}"
            )

        # 2. Volume surge confirms breakdown
        if (
            ind.volume_surge is not None
            and ind.volume_surge >= risk.volume_surge_threshold
        ):
            sell_score += 1.0
            sell_triggers.append(f"Volume surge {ind.volume_surge:.1f}x confirms breakdown")

        # 3. MACD bearish + hist falling
        if not ind.macd_bullish and not ind.macd_hist_rising:
            sell_score += 0.5
            sell_triggers.append("MACD bearish momentum")

        # 4. RSI above oversold — still has downside
        if ind.rsi > risk.rsi_oversold:
            sell_score += 0.5
            sell_triggers.append(f"RSI {ind.rsi:.1f} not oversold — downside potential")

        return buy_score, sell_score, buy_triggers, sell_triggers


class ScalpStrategy(TradingStrategy):
    """Stochastic + MACD histogram micro-moves — best for high-frequency short positions."""

    name = "scalp"
    description = "Stochastic %K/%D + MACD histogram — best for high-frequency short positions"
    max_score = 4.0
    skip_sentiment_nudge = False

    def score(
        self,
        ind: "Indicators",
        risk: "RiskProfile",
        sentiment: float = 0.0,
    ) -> tuple[float, float, list[str], list[str]]:
        buy_score: float = 0.0
        buy_triggers: list[str] = []
        sell_score: float = 0.0
        sell_triggers: list[str] = []

        if ind.stoch_k is not None:
            # 1. Stochastic oversold (exclusive: <20 w1.5, <30 w1.0)
            if ind.stoch_k < 20:
                buy_score += 1.5
                buy_triggers.append(f"Stochastic strongly oversold K={ind.stoch_k:.1f}<20")
            elif ind.stoch_k < 30:
                buy_score += 1.0
                buy_triggers.append(f"Stochastic oversold K={ind.stoch_k:.1f}<30")

            # 2. K crosses D bullish
            if ind.stoch_d is not None and ind.stoch_k > ind.stoch_d:
                buy_score += 1.0
                buy_triggers.append(
                    f"Stochastic K({ind.stoch_k:.1f}) crosses above D({ind.stoch_d:.1f})"
                )

            # Stochastic overbought SELL (exclusive: >80 w1.5, >70 w1.0)
            if ind.stoch_k > 80:
                sell_score += 1.5
                sell_triggers.append(f"Stochastic strongly overbought K={ind.stoch_k:.1f}>80")
            elif ind.stoch_k > 70:
                sell_score += 1.0
                sell_triggers.append(f"Stochastic overbought K={ind.stoch_k:.1f}>70")

            # K crosses D bearish
            if ind.stoch_d is not None and ind.stoch_k < ind.stoch_d:
                sell_score += 1.0
                sell_triggers.append(
                    f"Stochastic K({ind.stoch_k:.1f}) crosses below D({ind.stoch_d:.1f})"
                )

        else:
            # Fallback when no stochastic data (e.g. ASX): use MACD only
            if ind.macd_bullish and ind.macd_hist_rising:
                buy_score += 1.5
                buy_triggers.append("MACD bullish (stochastic fallback)")
            if not ind.macd_bullish and not ind.macd_hist_rising:
                sell_score += 1.5
                sell_triggers.append("MACD bearish (stochastic fallback)")

        # 3. MACD histogram positive and rising (BUY) / negative and falling (SELL)
        if ind.macd_bullish and ind.macd_hist_rising:
            buy_score += 1.0
            buy_triggers.append("MACD histogram rising and positive")
        if not ind.macd_bullish and not ind.macd_hist_rising:
            sell_score += 1.0
            sell_triggers.append("MACD histogram negative and falling")

        # 4. VWAP position
        if ind.price_above_vwap:
            buy_score += 0.5
            buy_triggers.append("Price above VWAP")
        if ind.price_below_vwap:
            sell_score += 0.5
            sell_triggers.append("Price below VWAP")

        return buy_score, sell_score, buy_triggers, sell_triggers


class SentimentStrategy(TradingStrategy):
    """News sentiment primary, RSI+MACD as technical filter — best for news-driven events."""

    name = "sentiment"
    description = "News sentiment primary + RSI/MACD filter — best for news-driven coins"
    max_score = 4.0
    skip_sentiment_nudge = True  # sentiment is the primary signal here

    def score(
        self,
        ind: "Indicators",
        risk: "RiskProfile",
        sentiment: float = 0.0,
    ) -> tuple[float, float, list[str], list[str]]:
        buy_score: float = 0.0
        buy_triggers: list[str] = []

        # 1. Sentiment primary (exclusive: >0.3 w2.0, >0.1 w1.0)
        if sentiment > 0.3:
            buy_score += 2.0
            buy_triggers.append(f"Strong positive news sentiment ({sentiment:.2f}>0.3)")
        elif sentiment > 0.1:
            buy_score += 1.0
            buy_triggers.append(f"Positive news sentiment ({sentiment:.2f}>0.1)")

        # 2. RSI not overbought (technical confirmation)
        if ind.rsi < risk.rsi_overbought:
            buy_score += 1.0
            buy_triggers.append(f"RSI {ind.rsi:.1f} not overbought (technical confirm)")

        # 3. MACD bullish
        if ind.macd_bullish:
            buy_score += 0.5
            buy_triggers.append("MACD bullish technical confirmation")

        # 4. Price above VWAP or SMA
        if ind.price_above_vwap or ind.price_above_mid_bb:
            buy_score += 0.5
            buy_triggers.append("Price above VWAP or SMA")

        # ---------------------------------------------------------------
        # SELL scoring
        # ---------------------------------------------------------------
        sell_score: float = 0.0
        sell_triggers: list[str] = []

        # 1. Sentiment primary (exclusive: <-0.3 w2.0, <-0.1 w1.0)
        if sentiment < -0.3:
            sell_score += 2.0
            sell_triggers.append(f"Strong negative news sentiment ({sentiment:.2f}<-0.3)")
        elif sentiment < -0.1:
            sell_score += 1.0
            sell_triggers.append(f"Negative news sentiment ({sentiment:.2f}<-0.1)")

        # 2. RSI not oversold (room for more downside)
        if ind.rsi > risk.rsi_oversold:
            sell_score += 1.0
            sell_triggers.append(f"RSI {ind.rsi:.1f} not oversold (technical confirm)")

        # 3. MACD bearish
        if not ind.macd_bullish:
            sell_score += 0.5
            sell_triggers.append("MACD bearish technical confirmation")

        # 4. Price below VWAP or SMA
        if ind.price_below_vwap or ind.price_below_mid_bb:
            sell_score += 0.5
            sell_triggers.append("Price below VWAP or SMA")

        return buy_score, sell_score, buy_triggers, sell_triggers


STRATEGIES: dict[str, TradingStrategy] = {
    "classic":   ClassicStrategy(),
    "trend":     TrendStrategy(),
    "breakout":  BreakoutStrategy(),
    "scalp":     ScalpStrategy(),
    "sentiment": SentimentStrategy(),
}

DEFAULT_STRATEGY: TradingStrategy = STRATEGIES["classic"]
