"""Volume-flow trading strategies for the crypto HFT platform.

Three complementary strategies that use volume-weighted indicators to confirm
price signals and distinguish genuine moves from low-volume fake-outs:

- MFIFlowStrategy   — Money Flow Index (volume-weighted RSI)
- VWAPBandsStrategy — VWAP ± 1σ/2σ standard deviation bands (mean reversion)
- OBVTrendStrategy  — On-Balance Volume trend confirmation (accumulation/distribution)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from src.trading.strategy import TradingStrategy

if TYPE_CHECKING:
    from src.prediction.predictor import Indicators
    from src.trading.risk import RiskProfile


class MFIFlowStrategy(TradingStrategy):
    """Money Flow Index — volume-weighted RSI for HFT signal confirmation.

    Unlike plain RSI which only uses price, MFI requires volume to confirm a
    directional move.  Low-volume fake-outs score zero on the primary signal,
    preventing false entries during illiquid periods.
    """

    name = "mfi_flow"
    description = (
        "Money Flow Index (volume-weighted RSI) — low-volume fake-outs score zero"
    )
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

        # 1. MFI extreme oversold <20 (exclusive, strongest signal)
        if ind.mfi is not None and ind.mfi < 20:
            buy_score += 2.5
            buy_triggers.append(f"MFI extreme oversold <20 ({ind.mfi:.1f})")
        # 1b. MFI mildly oversold <35 (only when not already <20)
        elif ind.mfi is not None and ind.mfi < 35:
            buy_score += 1.5
            buy_triggers.append(f"MFI oversold <35 ({ind.mfi:.1f})")

        # 2. MACD bullish + histogram rising
        if ind.macd_bullish and ind.macd_hist_rising:
            buy_score += 1.0
            buy_triggers.append("MACD bullish momentum")

        # 3. RSI confirms oversold
        if ind.rsi < risk.rsi_oversold:
            buy_score += 0.75
            buy_triggers.append(
                f"RSI confirms oversold ({ind.rsi:.1f}<{risk.rsi_oversold})"
            )

        # 4. Price above VWAP
        if ind.price_above_vwap:
            buy_score += 0.5
            buy_triggers.append("Price above VWAP")

        # ---------------------------------------------------------------
        # SELL scoring
        # ---------------------------------------------------------------
        sell_score: float = 0.0
        sell_triggers: list[str] = []

        # 1. MFI extreme overbought >80 (exclusive, strongest signal)
        if ind.mfi is not None and ind.mfi > 80:
            sell_score += 2.5
            sell_triggers.append(f"MFI extreme overbought >80 ({ind.mfi:.1f})")
        # 1b. MFI mildly overbought >65 (only when not already >80)
        elif ind.mfi is not None and ind.mfi > 65:
            sell_score += 1.5
            sell_triggers.append(f"MFI overbought >65 ({ind.mfi:.1f})")

        # 2. MACD not bullish — momentum fading
        if not ind.macd_bullish:
            sell_score += 1.0
            sell_triggers.append(
                f"MACD bearish ({ind.macd:.4f}<{ind.macd_signal:.4f})"
            )

        # 3. RSI overbought confirmation
        if ind.rsi > risk.rsi_overbought:
            sell_score += 0.75
            sell_triggers.append(
                f"RSI confirms overbought ({ind.rsi:.1f}>{risk.rsi_overbought})"
            )

        # 4. Price below VWAP
        if not ind.price_above_vwap:
            sell_score += 0.5
            sell_triggers.append("Price below VWAP")

        return buy_score, sell_score, buy_triggers, sell_triggers


class VWAPBandsStrategy(TradingStrategy):
    """VWAP ± 1σ/2σ standard deviation bands — intraday mean reversion.

    Price reaching ±2σ from VWAP is statistically extreme intraday and tends
    to revert toward the session mean.  Scores scale with distance from VWAP:
    2σ touch is the strongest entry signal; crossing the centerline is weakest.
    """

    name = "vwap_bands"
    description = (
        "VWAP ±1σ/2σ bands — intraday mean reversion; ±2σ is statistically extreme"
    )
    max_score = 5.0
    skip_sentiment_nudge = False

    def score(
        self,
        ind: "Indicators",
        risk: "RiskProfile",
        sentiment: float = 0.0,
    ) -> tuple[float, float, list[str], list[str]]:
        buy_score: float = 0.0
        buy_triggers: list[str] = []

        # 1. Price at/below VWAP -2σ (exclusive, most extreme mean-reversion entry)
        if ind.vwap_lower2 is not None and ind.price <= ind.vwap_lower2:
            buy_score += 3.0
            buy_triggers.append(
                f"Price at/below VWAP -2σ band ({ind.price:.4f}<={ind.vwap_lower2:.4f})"
            )
        # 1b. Price below VWAP -1σ (only when not already at -2σ)
        elif ind.vwap_lower1 is not None and ind.price <= ind.vwap_lower1:
            buy_score += 2.0
            buy_triggers.append(
                f"Price below VWAP -1σ band ({ind.price:.4f}<={ind.vwap_lower1:.4f})"
            )
        # 1c. Below VWAP centerline only — weak signal; only when band data unavailable
        elif ind.vwap is not None and ind.vwap_lower1 is None and ind.price < ind.vwap:
            buy_score += 0.5
            buy_triggers.append(
                f"Below VWAP centerline — slight oversold bias ({ind.price:.4f}<{ind.vwap:.4f})"
            )

        # 2. MACD histogram recovering
        if ind.macd_hist_rising:
            buy_score += 1.0
            buy_triggers.append("MACD histogram recovering")

        # 3. RSI oversold confirmation
        if ind.rsi < risk.rsi_oversold:
            buy_score += 1.0
            buy_triggers.append(
                f"RSI oversold confirmation ({ind.rsi:.1f}<{risk.rsi_oversold})"
            )

        # ---------------------------------------------------------------
        # SELL scoring
        # ---------------------------------------------------------------
        sell_score: float = 0.0
        sell_triggers: list[str] = []

        # 1. Price at/above VWAP +2σ (exclusive, most extreme mean-reversion exit)
        if ind.vwap_upper2 is not None and ind.price >= ind.vwap_upper2:
            sell_score += 3.0
            sell_triggers.append(
                f"Price at/above VWAP +2σ band ({ind.price:.4f}>={ind.vwap_upper2:.4f})"
            )
        # 1b. Price above VWAP +1σ (only when not already at +2σ)
        elif ind.vwap_upper1 is not None and ind.price >= ind.vwap_upper1:
            sell_score += 2.0
            sell_triggers.append(
                f"Price above VWAP +1σ band ({ind.price:.4f}>={ind.vwap_upper1:.4f})"
            )
        # 1c. Above VWAP centerline only — weak signal; only when band data unavailable
        elif ind.vwap is not None and ind.vwap_upper1 is None and ind.price > ind.vwap:
            sell_score += 0.5
            sell_triggers.append(
                f"Above VWAP centerline — slight overbought bias ({ind.price:.4f}>{ind.vwap:.4f})"
            )

        # 2. MACD histogram not rising (momentum fading)
        if not ind.macd_hist_rising:
            sell_score += 1.0
            sell_triggers.append("MACD histogram not rising")

        # 3. RSI overbought confirmation
        if ind.rsi > risk.rsi_overbought:
            sell_score += 1.0
            sell_triggers.append(
                f"RSI overbought confirmation ({ind.rsi:.1f}>{risk.rsi_overbought})"
            )

        return buy_score, sell_score, buy_triggers, sell_triggers


class OBVTrendStrategy(TradingStrategy):
    """On-Balance Volume trend confirmation — accumulation vs. distribution.

    OBV rising above its EMA signals institutional accumulation; OBV falling
    below its EMA signals distribution even when price looks flat.  OBV
    divergence (price up, OBV down) is an early warning of a reversal.
    """

    name = "obv_trend"
    description = (
        "On-Balance Volume trend — OBV above EMA signals accumulation; "
        "divergence warns of distribution"
    )
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

        # 1. OBV rising above EMA — institutional accumulation
        if ind.obv_rising is not None and ind.obv_rising:
            buy_score += 2.0
            buy_triggers.append("OBV rising above EMA — accumulation")

        # 2. MACD price momentum confirms OBV direction
        if ind.macd_bullish:
            buy_score += 1.0
            buy_triggers.append("MACD price momentum confirms OBV")

        # 3. Volume surge confirms breakout
        if (
            ind.volume_surge is not None
            and ind.volume_surge >= risk.volume_surge_threshold
        ):
            buy_score += 1.0
            buy_triggers.append(
                f"Volume surge {ind.volume_surge:.1f}x confirms breakout"
            )

        # 4. RSI not overbought — upside room remains
        if ind.rsi < risk.rsi_overbought:
            buy_score += 0.5
            buy_triggers.append(
                f"RSI {ind.rsi:.1f} not overbought — upside room"
            )

        # ---------------------------------------------------------------
        # SELL scoring
        # ---------------------------------------------------------------
        sell_score: float = 0.0
        sell_triggers: list[str] = []

        # 1. OBV falling below EMA — distribution signal
        if ind.obv_rising is not None and not ind.obv_rising:
            sell_score += 2.0
            sell_triggers.append("OBV falling below EMA — distribution")

        # 2. MACD not bullish — price momentum confirms OBV distribution
        if not ind.macd_bullish:
            sell_score += 1.0
            sell_triggers.append("MACD bearish confirms OBV distribution")

        # 3. Volume surge on declining OBV — institutional selling
        if (
            ind.volume_surge is not None
            and ind.volume_surge >= risk.volume_surge_threshold
        ):
            sell_score += 0.5
            sell_triggers.append(
                f"Volume surge {ind.volume_surge:.1f}x on declining OBV — selling pressure"
            )

        # 4. RSI overbought — downside room exists
        if ind.rsi > risk.rsi_overbought:
            sell_score += 0.75
            sell_triggers.append(
                f"RSI {ind.rsi:.1f} overbought — downside room"
            )

        return buy_score, sell_score, buy_triggers, sell_triggers


VOLUME_FLOW_STRATEGIES: dict[str, TradingStrategy] = {
    "mfi_flow":   MFIFlowStrategy(),
    "vwap_bands": VWAPBandsStrategy(),
    "obv_trend":  OBVTrendStrategy(),
}
