"""Trading risk profiles — four preset tiers from conservative to extreme.

Each profile bundles all strategy parameters that scale with risk tolerance.
Pass a RiskProfile to TradeEngine and Predictor; MEDIUM replicates the
original hardcoded defaults so existing behaviour is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskProfile:
    """Immutable bundle of strategy parameters for one risk tier."""

    name: str

    # RSI thresholds — lower oversold / higher overbought = more selective
    rsi_oversold: float         # strong buy signal when rsi < this
    rsi_weak_oversold: float    # mild buy signal when rsi < this
    rsi_overbought: float       # strong sell signal when rsi > this
    rsi_weak_overbought: float  # mild sell signal when rsi > this

    # Bollinger Band proximity — how close to the band edge triggers a signal
    # e.g. 1.015 means price <= bb_lower * 1.015 triggers a buy
    bb_lower_tolerance: float
    bb_upper_tolerance: float

    # Execution gate — minimum signal confidence required to place a trade
    confidence_threshold: float

    # Sentiment nudge — how strongly news biases a neutral technical signal
    sentiment_nudge_threshold: float  # weighted sentiment must exceed this
    sentiment_weight: float           # multiplier applied to raw sentiment score

    # Position sizing
    trade_fraction: float        # fraction of available cash deployed per trade
    max_position_fraction: float # max fraction of total portfolio in one asset

    # Risk controls
    stop_loss_pct: float      # auto-sell if position falls this far from avg cost
    min_profit_to_sell: float # minimum gain (above fees) before a sell fires

    # Loop timing
    hft_interval: float  # seconds between HFT evaluation cycles

    # Momentum / volume parameters
    volume_surge_threshold: float   # current_vol / avg_vol > this = surge event
    roc_period: int                 # bars back for Rate-of-Change calculation
    roc_momentum_threshold: float   # ROC% above this = strong momentum signal
    adx_trend_threshold: float      # ADX above this = trend confirmed

    # Momentum-hold gate: suppress normal sell signals on profitable positions
    # when upward momentum is still strong.  Value = fraction of available
    # momentum indicators (MACD, ROC, ADX, VWAP) that must be bullish to hold.
    # 1.0 = all must be bullish (conservative); 0.4 = any two-ish (aggressive).
    momentum_hold_fraction: float


# ---------------------------------------------------------------------------
# Preset profiles
# ---------------------------------------------------------------------------

LOW = RiskProfile(
    name="low",
    rsi_oversold=28.0,
    rsi_weak_oversold=38.0,
    rsi_overbought=72.0,
    rsi_weak_overbought=62.0,
    bb_lower_tolerance=1.008,
    bb_upper_tolerance=0.992,
    confidence_threshold=0.60,
    sentiment_nudge_threshold=0.50,
    sentiment_weight=0.6,
    trade_fraction=0.05,
    max_position_fraction=0.15,
    stop_loss_pct=0.015,
    min_profit_to_sell=0.008,
    hft_interval=5.0,
    volume_surge_threshold=3.0,
    roc_period=14,
    roc_momentum_threshold=3.0,
    adx_trend_threshold=30.0,
    momentum_hold_fraction=1.0,   # all indicators must agree to hold
)

MEDIUM = RiskProfile(
    name="medium",
    rsi_oversold=35.0,
    rsi_weak_oversold=45.0,
    rsi_overbought=65.0,
    rsi_weak_overbought=55.0,
    bb_lower_tolerance=1.015,
    bb_upper_tolerance=0.985,
    confidence_threshold=0.45,
    sentiment_nudge_threshold=0.30,
    sentiment_weight=1.0,
    trade_fraction=0.10,
    max_position_fraction=0.30,
    stop_loss_pct=0.025,
    min_profit_to_sell=0.0035,
    hft_interval=1.0,
    volume_surge_threshold=2.0,
    roc_period=10,
    roc_momentum_threshold=2.0,
    adx_trend_threshold=25.0,
    momentum_hold_fraction=0.75,  # 3 of 4 indicators bullish
)

HIGH = RiskProfile(
    name="high",
    rsi_oversold=40.0,
    rsi_weak_oversold=50.0,
    rsi_overbought=60.0,
    rsi_weak_overbought=50.0,
    bb_lower_tolerance=1.025,
    bb_upper_tolerance=0.975,
    confidence_threshold=0.30,
    sentiment_nudge_threshold=0.20,
    sentiment_weight=1.4,
    trade_fraction=0.20,
    max_position_fraction=0.50,
    stop_loss_pct=0.050,
    min_profit_to_sell=0.002,
    hft_interval=0.5,
    volume_surge_threshold=1.5,
    roc_period=7,
    roc_momentum_threshold=1.5,
    adx_trend_threshold=20.0,
    momentum_hold_fraction=0.60,  # any 3 of 4 / slight majority
)

EXTREME = RiskProfile(
    name="extreme",
    rsi_oversold=48.0,
    rsi_weak_oversold=54.0,
    rsi_overbought=52.0,
    rsi_weak_overbought=46.0,
    bb_lower_tolerance=1.040,
    bb_upper_tolerance=0.960,
    confidence_threshold=0.20,
    sentiment_nudge_threshold=0.10,
    sentiment_weight=2.0,
    trade_fraction=0.35,
    max_position_fraction=0.75,
    stop_loss_pct=0.080,
    min_profit_to_sell=0.001,
    hft_interval=0.2,
    volume_surge_threshold=1.2,
    roc_period=5,
    roc_momentum_threshold=1.0,
    adx_trend_threshold=15.0,
    momentum_hold_fraction=0.40,  # any 2 of 4 sufficient to hold
)

PROFILES: dict[str, RiskProfile] = {
    "low":     LOW,
    "medium":  MEDIUM,
    "high":    HIGH,
    "extreme": EXTREME,
}
