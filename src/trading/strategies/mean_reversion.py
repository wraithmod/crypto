"""Mean-reversion trading strategies for crypto HFT.

This module provides four complementary mean-reversion strategies, each
targeting a different statistical or technical measure of price extremity:

- CciReversionStrategy   — CCI 20-period extremes (<-100 / >100)
- KeltnerReversionStrategy — Keltner channel band touches with squeeze filter
- WilliamsRStrategy      — Williams %R momentum oscillator extremes
- ZScoreReversionStrategy — Rolling 20-period z-score statistical extremes

All four implement the TradingStrategy ABC defined in src.trading.strategy.
Each strategy is stateless; a single instance is safe to share across
concurrent evaluation loops.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.trading.strategy import TradingStrategy

if TYPE_CHECKING:
    from src.prediction.predictor import Indicators
    from src.trading.risk import RiskProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. CCI Reversion Strategy
# ---------------------------------------------------------------------------

class CciReversionStrategy(TradingStrategy):
    """Mean-reversion via CCI 20-period extremes.

    Uses the Commodity Channel Index to identify statistically extreme
    deviations from the 20-period mean price.  Extreme readings
    (abs(CCI) > 150) are scored more aggressively than moderate readings
    (abs(CCI) > 100).  RSI and MACD act as confirmation overlays.

    Scoring summary (buy / sell are mirrored):
        1. CCI < -150              →  +3.0  (exclusive with rule 2)
        2. CCI < -100 (not < -150) →  +2.0
        3. RSI < rsi_oversold      →  +0.75
        4. MACD bullish + hist ↑   →  +0.5
    """

    name: str = "cci_reversion"
    description: str = (
        "CCI 20-period extreme oscillations + RSI/MACD confirmation"
        " — best for ranging crypto markets"
    )
    max_score: float = 4.0
    skip_sentiment_nudge: bool = False

    def score(
        self,
        ind: "Indicators",
        risk: "RiskProfile",
        sentiment: float = 0.0,
    ) -> tuple[float, float, list[str], list[str]]:
        """Return (buy_score, sell_score, buy_triggers, sell_triggers)."""
        buy_score: float = 0.0
        buy_triggers: list[str] = []

        # 1 & 2 — CCI buy signal (mutually exclusive tiers)
        if ind.cci is not None:
            if ind.cci < -150:
                buy_score += 3.0
                buy_triggers.append(f"CCI extreme oversold <-150 ({ind.cci:.1f})")
            elif ind.cci < -100:
                buy_score += 2.0
                buy_triggers.append(f"CCI oversold <-100 ({ind.cci:.1f})")

        # 3 — RSI oversold confirmation
        if ind.rsi < risk.rsi_oversold:
            buy_score += 0.75
            buy_triggers.append(
                f"RSI confirms oversold ({ind.rsi:.1f}<{risk.rsi_oversold})"
            )

        # 4 — MACD bullish momentum
        if ind.macd_bullish and ind.macd_hist_rising:
            buy_score += 0.5
            buy_triggers.append(
                f"MACD bullish momentum ({ind.macd:.4f}>{ind.macd_signal:.4f})"
            )

        # ---------------------------------------------------------------
        # SELL scoring (mirror of buy)
        # ---------------------------------------------------------------
        sell_score: float = 0.0
        sell_triggers: list[str] = []

        # 1 & 2 — CCI sell signal (mutually exclusive tiers)
        if ind.cci is not None:
            if ind.cci > 150:
                sell_score += 3.0
                sell_triggers.append(f"CCI extreme overbought >150 ({ind.cci:.1f})")
            elif ind.cci > 100:
                sell_score += 2.0
                sell_triggers.append(f"CCI overbought >100 ({ind.cci:.1f})")

        # 3 — RSI overbought confirmation
        if ind.rsi > risk.rsi_overbought:
            sell_score += 0.75
            sell_triggers.append(
                f"RSI confirms overbought ({ind.rsi:.1f}>{risk.rsi_overbought})"
            )

        # 4 — MACD bearish (not bullish)
        if not ind.macd_bullish:
            sell_score += 0.5
            sell_triggers.append(
                f"MACD bearish ({ind.macd:.4f}<{ind.macd_signal:.4f})"
            )

        return buy_score, sell_score, buy_triggers, sell_triggers


# ---------------------------------------------------------------------------
# 2. Keltner Channel Reversion Strategy
# ---------------------------------------------------------------------------

class KeltnerReversionStrategy(TradingStrategy):
    """Mean-reversion via Keltner channel band touches.

    Keltner channels use ATR-based width rather than standard deviation,
    making them more adaptive to recent volatility than Bollinger Bands.
    When price touches the lower channel and Bollinger Bands are *inside*
    the Keltner channels, it indicates a compression (squeeze) that often
    precedes a sharp mean-reversion move.

    Scoring summary (buy / sell are mirrored):
        1. price <= keltner_lower           →  +2.0
        2. RSI < rsi_oversold               →  +1.0
        3. bb_lower < keltner_lower         →  +1.0  (squeeze release signal)
        4. price_above_vwap + price > keltner_mid → +0.5  (midline reclaim)
    """

    name: str = "keltner_reversion"
    description: str = (
        "Keltner channel band touches + BB squeeze detection"
        " — best for volatility-compression breakout reversal"
    )
    max_score: float = 4.0
    skip_sentiment_nudge: bool = False

    def score(
        self,
        ind: "Indicators",
        risk: "RiskProfile",
        sentiment: float = 0.0,
    ) -> tuple[float, float, list[str], list[str]]:
        """Return (buy_score, sell_score, buy_triggers, sell_triggers)."""
        buy_score: float = 0.0
        buy_triggers: list[str] = []

        # 1 — Price at or below Keltner lower band
        if ind.keltner_lower is not None and ind.price <= ind.keltner_lower:
            buy_score += 2.0
            buy_triggers.append(
                f"Price at/below Keltner lower"
                f" ({ind.price:.4f}<={ind.keltner_lower:.4f})"
            )

        # 2 — RSI confirms oversold at Keltner
        if ind.rsi < risk.rsi_oversold:
            buy_score += 1.0
            buy_triggers.append(
                f"RSI confirms oversold at Keltner ({ind.rsi:.1f}<{risk.rsi_oversold})"
            )

        # 3 — BB inside Keltner: squeeze release
        if ind.keltner_lower is not None and ind.bb_lower < ind.keltner_lower:
            buy_score += 1.0
            buy_triggers.append(
                f"BB inside Keltner — squeeze release"
                f" (BB lower {ind.bb_lower:.4f} < Keltner lower {ind.keltner_lower:.4f})"
            )

        # 4 — Keltner midline reclaim with VWAP confirmation
        if (
            ind.price_above_vwap
            and ind.keltner_mid is not None
            and ind.price > ind.keltner_mid
        ):
            buy_score += 0.5
            buy_triggers.append(
                f"Keltner midline reclaim ({ind.price:.4f}>{ind.keltner_mid:.4f})"
                " + above VWAP"
            )

        # ---------------------------------------------------------------
        # SELL scoring (mirror of buy)
        # ---------------------------------------------------------------
        sell_score: float = 0.0
        sell_triggers: list[str] = []

        # 1 — Price at or above Keltner upper band
        if ind.keltner_upper is not None and ind.price >= ind.keltner_upper:
            sell_score += 2.0
            sell_triggers.append(
                f"Price at/above Keltner upper"
                f" ({ind.price:.4f}>={ind.keltner_upper:.4f})"
            )

        # 2 — RSI confirms overbought at Keltner
        if ind.rsi > risk.rsi_overbought:
            sell_score += 1.0
            sell_triggers.append(
                f"RSI confirms overbought at Keltner ({ind.rsi:.1f}>{risk.rsi_overbought})"
            )

        # 3 — BB outside Keltner upper: expansion / exhaustion
        if ind.keltner_upper is not None and ind.bb_upper > ind.keltner_upper:
            sell_score += 1.0
            sell_triggers.append(
                f"BB above Keltner upper — expansion exhaustion"
                f" (BB upper {ind.bb_upper:.4f} > Keltner upper {ind.keltner_upper:.4f})"
            )

        # 4 — Price below VWAP and below Keltner midline
        if (
            ind.price_below_vwap
            and ind.keltner_mid is not None
            and ind.price < ind.keltner_mid
        ):
            sell_score += 0.5
            sell_triggers.append(
                f"Below VWAP + below Keltner midline ({ind.price:.4f}<{ind.keltner_mid:.4f})"
            )

        return buy_score, sell_score, buy_triggers, sell_triggers


# ---------------------------------------------------------------------------
# 3. Williams %R Strategy
# ---------------------------------------------------------------------------

class WilliamsRStrategy(TradingStrategy):
    """Mean-reversion via Williams %R momentum oscillator.

    Williams %R oscillates between -100 (most oversold) and 0 (most
    overbought).  Extreme readings near -100 or 0 signal potential
    reversals.  This strategy pairs Williams %R with MACD directional
    confirmation and Stochastic double-confirmation for the highest
    conviction entries.

    Scoring summary (buy / sell are mirrored):
        1. williams_r < -90                  →  +2.5  (exclusive with rule 2)
        2. williams_r < -80 (not < -90)      →  +1.5
        3. MACD bullish                      →  +0.75
        4. stoch_k < 30                      →  +1.0  (double oscillator confirm)
    """

    name: str = "williams_r"
    description: str = (
        "Williams %R 14-period extremes + Stochastic double confirmation"
        " — best for short-duration mean-reversion scalps"
    )
    max_score: float = 4.0
    skip_sentiment_nudge: bool = False

    def score(
        self,
        ind: "Indicators",
        risk: "RiskProfile",
        sentiment: float = 0.0,
    ) -> tuple[float, float, list[str], list[str]]:
        """Return (buy_score, sell_score, buy_triggers, sell_triggers)."""
        buy_score: float = 0.0
        buy_triggers: list[str] = []

        # 1 & 2 — Williams %R oversold (mutually exclusive tiers)
        if ind.williams_r is not None:
            if ind.williams_r < -90:
                buy_score += 2.5
                buy_triggers.append(
                    f"Williams %R extreme oversold <-90 ({ind.williams_r:.1f})"
                )
            elif ind.williams_r < -80:
                buy_score += 1.5
                buy_triggers.append(
                    f"Williams %R oversold <-80 ({ind.williams_r:.1f})"
                )

        # 3 — MACD confirms bullish direction
        if ind.macd_bullish:
            buy_score += 0.75
            buy_triggers.append(
                f"MACD confirms bullish ({ind.macd:.4f}>{ind.macd_signal:.4f})"
            )

        # 4 — Stochastic + Williams %R double oversold confirmation
        if ind.stoch_k is not None and ind.stoch_k < 30:
            buy_score += 1.0
            buy_triggers.append(
                f"Stoch+Williams double oversold (K={ind.stoch_k:.1f}<30)"
            )

        # ---------------------------------------------------------------
        # SELL scoring (mirror of buy)
        # ---------------------------------------------------------------
        sell_score: float = 0.0
        sell_triggers: list[str] = []

        # 1 & 2 — Williams %R overbought (mutually exclusive tiers)
        if ind.williams_r is not None:
            if ind.williams_r > -10:
                sell_score += 2.5
                sell_triggers.append(
                    f"Williams %R extreme overbought >-10 ({ind.williams_r:.1f})"
                )
            elif ind.williams_r > -20:
                sell_score += 1.5
                sell_triggers.append(
                    f"Williams %R overbought >-20 ({ind.williams_r:.1f})"
                )

        # 3 — MACD confirms bearish direction
        if not ind.macd_bullish:
            sell_score += 0.75
            sell_triggers.append(
                f"MACD confirms bearish ({ind.macd:.4f}<{ind.macd_signal:.4f})"
            )

        # 4 — Stochastic + Williams %R double overbought confirmation
        if ind.stoch_k is not None and ind.stoch_k > 70:
            sell_score += 1.0
            sell_triggers.append(
                f"Stoch+Williams double overbought (K={ind.stoch_k:.1f}>70)"
            )

        return buy_score, sell_score, buy_triggers, sell_triggers


# ---------------------------------------------------------------------------
# 4. Z-Score Reversion Strategy
# ---------------------------------------------------------------------------

class ZScoreReversionStrategy(TradingStrategy):
    """Mean-reversion via rolling 20-period price z-score.

    The z-score measures how many standard deviations the current price
    lies above or below the 20-period rolling mean.  Readings beyond
    ±2σ are statistically unlikely under a normal distribution and
    suggest reversion candidates; readings beyond ±3σ are extreme.
    MACD histogram and RSI are used as momentum recovery filters.

    Scoring summary (buy / sell are mirrored):
        1. price_zscore < -3.0              →  +3.0  (exclusive with rule 2)
        2. price_zscore < -2.0 (not < -3)  →  +2.0
        3. macd_hist_rising                 →  +0.5  (momentum recovery)
        4. RSI < rsi_oversold               →  +0.75
    """

    name: str = "zscore_reversion"
    description: str = (
        "Rolling 20-period price z-score statistical extremes + MACD/RSI confirmation"
        " — best for statistically driven mean-reversion"
    )
    max_score: float = 4.0
    skip_sentiment_nudge: bool = False

    def score(
        self,
        ind: "Indicators",
        risk: "RiskProfile",
        sentiment: float = 0.0,
    ) -> tuple[float, float, list[str], list[str]]:
        """Return (buy_score, sell_score, buy_triggers, sell_triggers)."""
        buy_score: float = 0.0
        buy_triggers: list[str] = []

        # 1 & 2 — Z-score oversold (mutually exclusive tiers)
        if ind.price_zscore is not None:
            if ind.price_zscore < -3.0:
                buy_score += 3.0
                buy_triggers.append(
                    f"Z-score extreme <-3\u03c3 ({ind.price_zscore:.2f})"
                )
            elif ind.price_zscore < -2.0:
                buy_score += 2.0
                buy_triggers.append(
                    f"Z-score oversold <-2\u03c3 ({ind.price_zscore:.2f})"
                )

        # 3 — MACD histogram rising (momentum recovering)
        if ind.macd_hist_rising:
            buy_score += 0.5
            buy_triggers.append("MACD hist rising — momentum recovering")

        # 4 — RSI confirms oversold
        if ind.rsi < risk.rsi_oversold:
            buy_score += 0.75
            buy_triggers.append(
                f"RSI confirms oversold ({ind.rsi:.1f}<{risk.rsi_oversold})"
            )

        # ---------------------------------------------------------------
        # SELL scoring (mirror of buy)
        # ---------------------------------------------------------------
        sell_score: float = 0.0
        sell_triggers: list[str] = []

        # 1 & 2 — Z-score overbought (mutually exclusive tiers)
        if ind.price_zscore is not None:
            if ind.price_zscore > 3.0:
                sell_score += 3.0
                sell_triggers.append(
                    f"Z-score extreme >+3\u03c3 ({ind.price_zscore:.2f})"
                )
            elif ind.price_zscore > 2.0:
                sell_score += 2.0
                sell_triggers.append(
                    f"Z-score overbought >+2\u03c3 ({ind.price_zscore:.2f})"
                )

        # 3 — MACD histogram falling (momentum deteriorating)
        if not ind.macd_hist_rising:
            sell_score += 0.5
            sell_triggers.append("MACD hist falling — momentum deteriorating")

        # 4 — RSI confirms overbought
        if ind.rsi > risk.rsi_overbought:
            sell_score += 0.75
            sell_triggers.append(
                f"RSI confirms overbought ({ind.rsi:.1f}>{risk.rsi_overbought})"
            )

        return buy_score, sell_score, buy_triggers, sell_triggers


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

MEAN_REVERSION_STRATEGIES: dict[str, TradingStrategy] = {
    "cci_reversion":     CciReversionStrategy(),
    "keltner_reversion": KeltnerReversionStrategy(),
    "williams_r":        WilliamsRStrategy(),
    "zscore_reversion":  ZScoreReversionStrategy(),
}
