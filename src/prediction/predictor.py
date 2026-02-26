import asyncio
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import ta

logger = logging.getLogger(__name__)


@dataclass
class Indicators:
    rsi: float          # 0-100
    macd: float         # MACD line value
    macd_signal: float  # Signal line value
    macd_hist: float    # Histogram
    bb_upper: float     # Bollinger upper band
    bb_mid: float       # Bollinger middle (SMA)
    bb_lower: float     # Bollinger lower band
    price: float        # current price

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


@dataclass
class Signal:
    direction: str              # "buy", "sell", "hold"
    confidence: float           # 0.0 to 1.0
    reasoning: str              # explanation
    indicators: Indicators | None = None


class Predictor:
    def __init__(self) -> None:
        logger.debug("Predictor initialized")

    def compute_indicators(self, prices: list[float]) -> Indicators | None:
        """Compute RSI, MACD, and Bollinger Bands from a price series.

        Requires at least 26 prices for MACD computation. Returns None if the
        list is too short or if any computed indicator value is NaN.
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

        values = {
            "rsi": rsi_val,
            "macd": macd_val,
            "macd_signal": macd_signal_val,
            "macd_hist": macd_hist_val,
            "bb_upper": bb_upper_val,
            "bb_mid": bb_mid_val,
            "bb_lower": bb_lower_val,
        }

        for name, val in values.items():
            if val is None or (isinstance(val, float) and np.isnan(val)):
                logger.debug("Indicator '%s' is NaN — skipping signal", name)
                return None

        return Indicators(
            rsi=float(rsi_val),
            macd=float(macd_val),
            macd_signal=float(macd_signal_val),
            macd_hist=float(macd_hist_val),
            bb_upper=float(bb_upper_val),
            bb_mid=float(bb_mid_val),
            bb_lower=float(bb_lower_val),
            price=float(prices[-1]),
        )

    def rule_based_signal(self, ind: Indicators) -> Signal:
        """Derive a directional signal from technical indicator rules.

        Four BUY conditions (confidence = triggers / 4):
          1. RSI strongly oversold (< 35)
          2. RSI mildly oversold (< 45)  — only if not already counted by #1
          3. MACD bullish AND histogram positive (momentum)
          4. Price near or below lower Bollinger Band

        Four SELL conditions (confidence = triggers / 4):
          1. RSI strongly overbought (> 65)
          2. RSI mildly overbought (> 55)  — only if not already counted by #1
          3. MACD bearish AND histogram negative
          4. Price near or above upper Bollinger Band

        Confidence grades: 0.25 (1/4), 0.50 (2/4), 0.75 (3/4), 1.0 (4/4).
        """
        total_conditions = 4

        # --- BUY scoring ---
        buy_triggers: list[str] = []
        if ind.rsi_oversold:
            buy_triggers.append(f"RSI strongly oversold ({ind.rsi:.1f}<35)")
        elif ind.rsi_weak_oversold:
            buy_triggers.append(f"RSI mildly oversold ({ind.rsi:.1f}<45)")
        if ind.macd_bullish and ind.macd_hist_rising:
            buy_triggers.append(f"MACD bullish+rising hist ({ind.macd:.4f}>{ind.macd_signal:.4f})")
        if ind.price_near_lower_bb:
            buy_triggers.append(f"Price near lower BB ({ind.price:.4f}≤{ind.bb_lower*1.015:.4f})")
        if ind.price_above_mid_bb and ind.macd_bullish:
            buy_triggers.append(f"Price above SMA+MACD bullish (trend confirm)")

        # --- SELL scoring ---
        sell_triggers: list[str] = []
        if ind.rsi_overbought:
            sell_triggers.append(f"RSI strongly overbought ({ind.rsi:.1f}>65)")
        elif ind.rsi_weak_overbought:
            sell_triggers.append(f"RSI mildly overbought ({ind.rsi:.1f}>55)")
        if not ind.macd_bullish and not ind.macd_hist_rising:
            sell_triggers.append(f"MACD bearish+falling hist ({ind.macd:.4f}<{ind.macd_signal:.4f})")
        if ind.price_near_upper_bb:
            sell_triggers.append(f"Price near upper BB ({ind.price:.4f}≥{ind.bb_upper*0.985:.4f})")
        if ind.price_below_mid_bb and not ind.macd_bullish:
            sell_triggers.append(f"Price below SMA+MACD bearish (trend confirm)")

        buy_count = len(buy_triggers)
        sell_count = len(sell_triggers)

        if buy_count == 0 and sell_count == 0:
            return Signal(
                direction="hold",
                confidence=0.0,
                reasoning="No buy or sell conditions triggered; holding.",
                indicators=ind,
            )

        if buy_count >= sell_count:
            confidence = buy_count / total_conditions
            reasoning = "BUY — " + "; ".join(buy_triggers) + "."
            return Signal(
                direction="buy",
                confidence=min(confidence, 1.0),
                reasoning=reasoning,
                indicators=ind,
            )
        else:
            confidence = sell_count / total_conditions
            reasoning = "SELL — " + "; ".join(sell_triggers) + "."
            return Signal(
                direction="sell",
                confidence=min(confidence, 1.0),
                reasoning=reasoning,
                indicators=ind,
            )

    async def get_signal(
        self,
        symbol: str,
        prices: list[float],
        news_sentiment: float = 0.0,
    ) -> Signal:
        """Produce a final trading signal blending technical analysis and news sentiment.

        Steps:
          1. Compute technical indicators from the price series.
          2. Derive a rule-based directional signal.
          3. If the rule-based signal is "hold", apply news sentiment nudge:
               sentiment >  0.3  -> nudge to "buy"
               sentiment < -0.3  -> nudge to "sell"
          4. Return the final Signal.

        If indicators cannot be computed (insufficient data), a hold signal with
        zero confidence is returned immediately.
        """
        logger.debug("Computing signal for %s (prices=%d)", symbol, len(prices))

        ind = self.compute_indicators(prices)
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

        signal = self.rule_based_signal(ind)

        # Apply news sentiment nudge only when the technical signal is neutral.
        if signal.direction == "hold":
            if news_sentiment > 0.3:
                nudge_confidence = min((news_sentiment - 0.3) / 0.7, 1.0) * 0.5
                signal = Signal(
                    direction="buy",
                    confidence=nudge_confidence,
                    reasoning=(
                        f"Neutral technical signal nudged to BUY by positive news "
                        f"sentiment ({news_sentiment:.2f} > 0.3)."
                    ),
                    indicators=ind,
                )
                logger.debug(
                    "%s: hold -> buy (sentiment nudge, sentiment=%.2f)",
                    symbol,
                    news_sentiment,
                )
            elif news_sentiment < -0.3:
                nudge_confidence = min((abs(news_sentiment) - 0.3) / 0.7, 1.0) * 0.5
                signal = Signal(
                    direction="sell",
                    confidence=nudge_confidence,
                    reasoning=(
                        f"Neutral technical signal nudged to SELL by negative news "
                        f"sentiment ({news_sentiment:.2f} < -0.3)."
                    ),
                    indicators=ind,
                )
                logger.debug(
                    "%s: hold -> sell (sentiment nudge, sentiment=%.2f)",
                    symbol,
                    news_sentiment,
                )

        logger.info(
            "%s signal: %s (confidence=%.2f) sentiment=%.2f",
            symbol,
            signal.direction,
            signal.confidence,
            news_sentiment,
        )
        return signal
