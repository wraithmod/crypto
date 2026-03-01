"""Tests for the 11 new trading strategies (mean_reversion, volume_flow, advanced)
and the new Indicators fields added to predictor.py.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prediction.predictor import Indicators
from src.trading.risk import MEDIUM, HIGH
from src.trading.strategy import STRATEGIES
from src.trading.strategies.mean_reversion import (
    CciReversionStrategy,
    KeltnerReversionStrategy,
    WilliamsRStrategy,
    ZScoreReversionStrategy,
    MEAN_REVERSION_STRATEGIES,
)
from src.trading.strategies.volume_flow import (
    MFIFlowStrategy,
    VWAPBandsStrategy,
    OBVTrendStrategy,
    VOLUME_FLOW_STRATEGIES,
)
from src.trading.strategies.advanced import (
    SuperTrendStrategy,
    DualRSIStrategy,
    BBSqueezeStrategy,
    TripleConfluenceStrategy,
    ADVANCED_STRATEGIES,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _ind(
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
    adx: float | None = None,
    adx_plus_di: float | None = None,
    adx_minus_di: float | None = None,
    roc: float | None = None,
    ema9: float | None = None,
    ema21: float | None = None,
    ema55: float | None = None,
    stoch_k: float | None = None,
    stoch_d: float | None = None,
    donchian_high: float | None = None,
    donchian_low: float | None = None,
    atr: float | None = None,
    # Mean-reversion fields
    cci: float | None = None,
    keltner_upper: float | None = None,
    keltner_mid: float | None = None,
    keltner_lower: float | None = None,
    williams_r: float | None = None,
    price_zscore: float | None = None,
    # Volume-flow fields
    mfi: float | None = None,
    obv: float | None = None,
    obv_ema: float | None = None,
    obv_rising: bool | None = None,
    cmf: float | None = None,
    vwap_std: float | None = None,
    vwap_upper1: float | None = None,
    vwap_upper2: float | None = None,
    vwap_lower1: float | None = None,
    vwap_lower2: float | None = None,
    # Advanced trend/regime fields
    supertrend: float | None = None,
    supertrend_bullish: bool | None = None,
    rsi_fast: float | None = None,
    rsi_slow: float | None = None,
    bb_width: float | None = None,
    bb_width_percentile: float | None = None,
    squeeze_active: bool | None = None,
) -> Indicators:
    return Indicators(
        rsi=rsi, macd=macd, macd_signal=macd_signal, macd_hist=macd_hist,
        bb_upper=bb_upper, bb_mid=bb_mid, bb_lower=bb_lower, price=price,
        vwap=vwap, volume_surge=volume_surge,
        adx=adx, adx_plus_di=adx_plus_di, adx_minus_di=adx_minus_di,
        roc=roc,
        ema9=ema9, ema21=ema21, ema55=ema55,
        stoch_k=stoch_k, stoch_d=stoch_d,
        donchian_high=donchian_high, donchian_low=donchian_low,
        atr=atr,
        cci=cci,
        keltner_upper=keltner_upper, keltner_mid=keltner_mid, keltner_lower=keltner_lower,
        williams_r=williams_r, price_zscore=price_zscore,
        mfi=mfi, obv=obv, obv_ema=obv_ema, obv_rising=obv_rising, cmf=cmf,
        vwap_std=vwap_std,
        vwap_upper1=vwap_upper1, vwap_upper2=vwap_upper2,
        vwap_lower1=vwap_lower1, vwap_lower2=vwap_lower2,
        supertrend=supertrend, supertrend_bullish=supertrend_bullish,
        rsi_fast=rsi_fast, rsi_slow=rsi_slow,
        bb_width=bb_width, bb_width_percentile=bb_width_percentile,
        squeeze_active=squeeze_active,
    )


# ---------------------------------------------------------------------------
# STRATEGIES registry
# ---------------------------------------------------------------------------

class TestStrategiesRegistry:
    """All 16 strategies must be present in the global STRATEGIES dict."""

    expected_keys = {
        # Original 5
        "classic", "trend", "breakout", "scalp", "sentiment",
        # Mean-reversion 4
        "cci_reversion", "keltner_reversion", "williams_r", "zscore_reversion",
        # Volume-flow 3
        "mfi_flow", "vwap_bands", "obv_trend",
        # Advanced 4
        "supertrend", "dual_rsi", "bb_squeeze", "triple_confluence",
    }

    def test_all_strategies_registered(self):
        assert self.expected_keys == set(STRATEGIES.keys())

    def test_mean_reversion_sub_registry(self):
        assert set(MEAN_REVERSION_STRATEGIES.keys()) == {
            "cci_reversion", "keltner_reversion", "williams_r", "zscore_reversion"
        }

    def test_volume_flow_sub_registry(self):
        assert set(VOLUME_FLOW_STRATEGIES.keys()) == {"mfi_flow", "vwap_bands", "obv_trend"}

    def test_advanced_sub_registry(self):
        assert set(ADVANCED_STRATEGIES.keys()) == {
            "supertrend", "dual_rsi", "bb_squeeze", "triple_confluence"
        }


# ---------------------------------------------------------------------------
# CciReversionStrategy
# ---------------------------------------------------------------------------

class TestCciReversionStrategy:
    strat = CciReversionStrategy()

    def test_extreme_oversold_scores_buy(self):
        ind = _ind(cci=-160.0, rsi=25.0, macd=0.01, macd_signal=0.0, macd_hist=0.01)
        buy, sell, buy_t, sell_t = self.strat.score(ind, MEDIUM)
        assert buy >= 3.0
        assert any("extreme" in t.lower() for t in buy_t)

    def test_oversold_scores_buy(self):
        ind = _ind(cci=-110.0)
        buy, sell, buy_t, sell_t = self.strat.score(ind, MEDIUM)
        assert buy >= 2.0

    def test_extreme_overbought_scores_sell(self):
        ind = _ind(cci=160.0, rsi=75.0)
        buy, sell, buy_t, sell_t = self.strat.score(ind, MEDIUM)
        assert sell >= 3.0

    def test_neutral_cci_no_signal(self):
        # CCI is neutral (0.0) and RSI is neutral (50.0).
        # Neither CCI tier fires; only MACD confirmation may contribute up to 0.5.
        # Score from CCI tiers should be zero for both buy and sell.
        ind = _ind(cci=0.0, rsi=50.0)
        buy, sell, buy_t, sell_t = self.strat.score(ind, MEDIUM)
        # No CCI-tier trigger should appear
        assert not any("cci" in t.lower() for t in buy_t)
        assert not any("cci" in t.lower() for t in sell_t)
        # Total scores are low — at most the 0.5 MACD confirmation rule
        assert buy <= 0.5
        assert sell <= 0.5

    def test_none_cci_does_not_crash(self):
        ind = _ind(cci=None)
        buy, sell, _, _ = self.strat.score(ind, MEDIUM)
        assert buy >= 0.0 and sell >= 0.0

    def test_tiers_are_exclusive(self):
        """CCI at -150 should not score both tiers."""
        ind_extreme = _ind(cci=-160.0)
        ind_mild    = _ind(cci=-110.0)
        buy_extreme, _, _, _ = self.strat.score(ind_extreme, MEDIUM)
        buy_mild, _, _, _    = self.strat.score(ind_mild, MEDIUM)
        assert buy_extreme > buy_mild


# ---------------------------------------------------------------------------
# KeltnerReversionStrategy
# ---------------------------------------------------------------------------

class TestKeltnerReversionStrategy:
    strat = KeltnerReversionStrategy()

    def test_price_at_lower_keltner_buy(self):
        ind = _ind(
            price=44_000.0,
            keltner_lower=44_100.0,  # price <= keltner_lower
            keltner_mid=45_000.0,
            keltner_upper=46_000.0,
        )
        buy, sell, buy_t, _ = self.strat.score(ind, MEDIUM)
        assert buy >= 2.0
        assert any("keltner lower" in t.lower() for t in buy_t)

    def test_price_at_upper_keltner_sell(self):
        ind = _ind(
            price=47_000.0,
            keltner_upper=46_500.0,
            keltner_mid=45_000.0,
            keltner_lower=44_000.0,
        )
        buy, sell, _, sell_t = self.strat.score(ind, MEDIUM)
        assert sell >= 2.0

    def test_bb_inside_keltner_squeeze_release_buy(self):
        ind = _ind(
            price=44_000.0,
            bb_lower=43_500.0,
            keltner_lower=43_800.0,  # bb_lower < keltner_lower
            keltner_mid=45_000.0,
            keltner_upper=46_200.0,
        )
        buy, sell, buy_t, _ = self.strat.score(ind, MEDIUM)
        assert any("squeeze" in t.lower() for t in buy_t)

    def test_no_keltner_no_crash(self):
        ind = _ind(keltner_lower=None, keltner_mid=None, keltner_upper=None)
        buy, sell, _, _ = self.strat.score(ind, MEDIUM)
        assert buy >= 0.0 and sell >= 0.0


# ---------------------------------------------------------------------------
# WilliamsRStrategy
# ---------------------------------------------------------------------------

class TestWilliamsRStrategy:
    strat = WilliamsRStrategy()

    def test_extreme_oversold_buy(self):
        ind = _ind(williams_r=-95.0, macd=0.01, macd_signal=0.0)
        buy, sell, buy_t, _ = self.strat.score(ind, MEDIUM)
        assert buy >= 2.5
        assert any("extreme" in t.lower() for t in buy_t)

    def test_mild_oversold_buy(self):
        ind = _ind(williams_r=-85.0)
        buy, sell, _, _ = self.strat.score(ind, MEDIUM)
        assert buy >= 1.5

    def test_extreme_overbought_sell(self):
        ind = _ind(williams_r=-5.0)
        buy, sell, _, sell_t = self.strat.score(ind, MEDIUM)
        assert sell >= 2.5

    def test_stoch_double_confirmation(self):
        ind = _ind(williams_r=-92.0, stoch_k=25.0)
        buy, sell, buy_t, _ = self.strat.score(ind, MEDIUM)
        assert any("double" in t.lower() or "stoch" in t.lower() for t in buy_t)
        assert buy >= 3.5

    def test_none_williams_r_no_crash(self):
        ind = _ind(williams_r=None)
        buy, sell, _, _ = self.strat.score(ind, MEDIUM)
        assert buy >= 0.0 and sell >= 0.0


# ---------------------------------------------------------------------------
# ZScoreReversionStrategy
# ---------------------------------------------------------------------------

class TestZScoreReversionStrategy:
    strat = ZScoreReversionStrategy()

    def test_extreme_negative_zscore_buy(self):
        ind = _ind(price_zscore=-3.5)
        buy, sell, buy_t, _ = self.strat.score(ind, MEDIUM)
        assert buy >= 3.0
        assert any("3" in t for t in buy_t)

    def test_mild_negative_zscore_buy(self):
        ind = _ind(price_zscore=-2.5)
        buy, sell, _, _ = self.strat.score(ind, MEDIUM)
        assert buy >= 2.0

    def test_extreme_positive_zscore_sell(self):
        ind = _ind(price_zscore=3.5)
        buy, sell, _, sell_t = self.strat.score(ind, MEDIUM)
        assert sell >= 3.0

    def test_none_zscore_no_crash(self):
        ind = _ind(price_zscore=None)
        buy, sell, _, _ = self.strat.score(ind, MEDIUM)
        assert buy >= 0.0 and sell >= 0.0

    def test_macd_hist_rising_adds_buy(self):
        ind_rising  = _ind(price_zscore=-2.2, macd=0.01, macd_signal=0.0, macd_hist=0.01)
        ind_falling = _ind(price_zscore=-2.2, macd=-0.01, macd_signal=0.0, macd_hist=-0.01)
        buy_r, _, _, _ = self.strat.score(ind_rising, MEDIUM)
        buy_f, _, _, _ = self.strat.score(ind_falling, MEDIUM)
        assert buy_r > buy_f


# ---------------------------------------------------------------------------
# MFIFlowStrategy
# ---------------------------------------------------------------------------

class TestMFIFlowStrategy:
    strat = MFIFlowStrategy()

    def test_extreme_mfi_oversold_buy(self):
        ind = _ind(mfi=15.0)
        buy, sell, buy_t, _ = self.strat.score(ind, MEDIUM)
        assert buy >= 2.5
        assert any("extreme" in t.lower() for t in buy_t)

    def test_mild_mfi_oversold_buy(self):
        ind = _ind(mfi=30.0)
        buy, sell, _, _ = self.strat.score(ind, MEDIUM)
        assert buy >= 1.5

    def test_extreme_mfi_overbought_sell(self):
        ind = _ind(mfi=85.0)
        buy, sell, _, sell_t = self.strat.score(ind, MEDIUM)
        assert sell >= 2.5

    def test_none_mfi_no_crash(self):
        ind = _ind(mfi=None)
        buy, sell, _, _ = self.strat.score(ind, MEDIUM)
        assert buy >= 0.0 and sell >= 0.0

    def test_mfi_tiers_exclusive(self):
        ind_extreme = _ind(mfi=15.0)
        ind_mild    = _ind(mfi=30.0)
        buy_e, _, _, _ = self.strat.score(ind_extreme, MEDIUM)
        buy_m, _, _, _ = self.strat.score(ind_mild, MEDIUM)
        assert buy_e > buy_m


# ---------------------------------------------------------------------------
# VWAPBandsStrategy
# ---------------------------------------------------------------------------

class TestVWAPBandsStrategy:
    strat = VWAPBandsStrategy()

    def test_at_lower2_band_strong_buy(self):
        ind = _ind(
            price=43_000.0,
            vwap=45_000.0,
            vwap_lower1=44_000.0,
            vwap_lower2=43_100.0,  # price <= lower2
        )
        buy, sell, buy_t, _ = self.strat.score(ind, MEDIUM)
        assert buy >= 3.0
        assert any("2σ" in t or "2" in t for t in buy_t)

    def test_at_lower1_band_buy(self):
        ind = _ind(
            price=43_800.0,
            vwap=45_000.0,
            vwap_lower1=44_000.0,   # price <= lower1 but not lower2
            vwap_lower2=43_000.0,
        )
        buy, sell, buy_t, _ = self.strat.score(ind, MEDIUM)
        assert buy >= 2.0

    def test_at_upper2_band_strong_sell(self):
        ind = _ind(
            price=47_200.0,
            vwap=45_000.0,
            vwap_upper1=46_000.0,
            vwap_upper2=47_000.0,   # price >= upper2
        )
        buy, sell, _, sell_t = self.strat.score(ind, MEDIUM)
        assert sell >= 3.0

    def test_no_bands_falls_back_to_centerline(self):
        ind = _ind(price=44_000.0, vwap=45_000.0)
        buy, sell, buy_t, _ = self.strat.score(ind, MEDIUM)
        assert any("centerline" in t.lower() for t in buy_t)
        assert buy >= 0.5

    def test_none_vwap_no_crash(self):
        ind = _ind(vwap=None, vwap_lower1=None, vwap_lower2=None)
        buy, sell, _, _ = self.strat.score(ind, MEDIUM)
        assert buy >= 0.0 and sell >= 0.0


# ---------------------------------------------------------------------------
# OBVTrendStrategy
# ---------------------------------------------------------------------------

class TestOBVTrendStrategy:
    strat = OBVTrendStrategy()

    def test_obv_rising_accumulation_buy(self):
        ind = _ind(obv_rising=True, macd=0.01, macd_signal=0.0)
        buy, sell, buy_t, _ = self.strat.score(ind, MEDIUM)
        assert buy >= 2.0
        assert any("accumulation" in t.lower() for t in buy_t)

    def test_obv_falling_distribution_sell(self):
        ind = _ind(obv_rising=False)
        buy, sell, _, sell_t = self.strat.score(ind, MEDIUM)
        assert sell >= 2.0
        assert any("distribution" in t.lower() for t in sell_t)

    def test_none_obv_rising_no_crash(self):
        ind = _ind(obv_rising=None)
        buy, sell, _, _ = self.strat.score(ind, MEDIUM)
        assert buy >= 0.0 and sell >= 0.0

    def test_volume_surge_stacks_with_obv(self):
        ind = _ind(obv_rising=True, volume_surge=3.0)
        buy_surge, _, _, _ = self.strat.score(ind, MEDIUM)
        ind_no = _ind(obv_rising=True, volume_surge=None)
        buy_no, _, _, _    = self.strat.score(ind_no, MEDIUM)
        assert buy_surge > buy_no


# ---------------------------------------------------------------------------
# SuperTrendStrategy
# ---------------------------------------------------------------------------

class TestSuperTrendStrategy:
    strat = SuperTrendStrategy()

    def test_supertrend_bullish_buy(self):
        ind = _ind(supertrend_bullish=True)
        buy, sell, buy_t, _ = self.strat.score(ind, MEDIUM)
        assert buy >= 2.5
        assert any("supertrend" in t.lower() for t in buy_t)

    def test_supertrend_bearish_sell(self):
        ind = _ind(supertrend_bullish=False)
        buy, sell, _, sell_t = self.strat.score(ind, MEDIUM)
        assert sell >= 2.5

    def test_ema9_above_ema21_confirmation(self):
        ind = _ind(supertrend_bullish=True, ema9=45_100.0, ema21=45_000.0)
        buy_with, _, _, _ = self.strat.score(ind, MEDIUM)
        ind_no = _ind(supertrend_bullish=True)
        buy_no, _, _, _   = self.strat.score(ind_no, MEDIUM)
        assert buy_with > buy_no

    def test_none_supertrend_no_crash(self):
        ind = _ind(supertrend_bullish=None)
        buy, sell, _, _ = self.strat.score(ind, MEDIUM)
        assert buy >= 0.0 and sell >= 0.0


# ---------------------------------------------------------------------------
# DualRSIStrategy
# ---------------------------------------------------------------------------

class TestDualRSIStrategy:
    strat = DualRSIStrategy()

    def test_rsi_fast_above_slow_from_oversold_buy(self):
        ind = _ind(rsi_fast=42.0, rsi_slow=38.0)  # fast > slow, both < 50
        buy, sell, buy_t, _ = self.strat.score(ind, MEDIUM)
        assert buy >= 2.0
        assert any("cross" in t.lower() or "rsi-7" in t.lower() for t in buy_t)

    def test_rsi_fast_oversold_stacks(self):
        ind = _ind(rsi_fast=25.0, rsi_slow=22.0)  # fast > slow AND fast < rsi_oversold
        buy, sell, _, _ = self.strat.score(ind, MEDIUM)
        assert buy >= 3.5

    def test_rsi_fast_below_slow_from_overbought_sell(self):
        ind = _ind(rsi_fast=58.0, rsi_slow=62.0)  # fast < slow, fast > 50
        buy, sell, _, sell_t = self.strat.score(ind, MEDIUM)
        assert sell >= 2.0

    def test_none_rsi_fast_no_crash(self):
        ind = _ind(rsi_fast=None, rsi_slow=None)
        buy, sell, _, _ = self.strat.score(ind, MEDIUM)
        assert buy >= 0.0 and sell >= 0.0


# ---------------------------------------------------------------------------
# BBSqueezeStrategy
# ---------------------------------------------------------------------------

class TestBBSqueezeStrategy:
    strat = BBSqueezeStrategy()

    def test_full_squeeze_macd_bullish_buy(self):
        ind = _ind(
            squeeze_active=True,
            bb_width_percentile=0.10,
            macd=0.01, macd_signal=0.0, macd_hist=0.01,
        )
        buy, sell, buy_t, _ = self.strat.score(ind, MEDIUM)
        assert buy >= 3.0
        assert any("squeeze" in t.lower() for t in buy_t)

    def test_full_squeeze_macd_bearish_sell(self):
        ind = _ind(
            squeeze_active=True,
            bb_width_percentile=0.10,
            macd=-0.01, macd_signal=0.0, macd_hist=-0.01,
        )
        buy, sell, _, sell_t = self.strat.score(ind, MEDIUM)
        assert sell >= 3.0

    def test_narrowing_macd_bullish_partial_buy(self):
        ind = _ind(
            squeeze_active=False,
            bb_width_percentile=0.30,
            macd=0.01, macd_signal=0.0, macd_hist=0.01,
        )
        buy, sell, buy_t, _ = self.strat.score(ind, MEDIUM)
        assert buy >= 1.5
        assert any("narrow" in t.lower() for t in buy_t)

    def test_no_squeeze_data_no_crash(self):
        ind = _ind(squeeze_active=None, bb_width_percentile=None)
        buy, sell, _, _ = self.strat.score(ind, MEDIUM)
        assert buy >= 0.0 and sell >= 0.0


# ---------------------------------------------------------------------------
# TripleConfluenceStrategy
# ---------------------------------------------------------------------------

class TestTripleConfluenceStrategy:
    strat = TripleConfluenceStrategy()

    def test_full_confluence_buy(self):
        """All four conditions fire → capped at max_score=4.0."""
        ind = _ind(
            rsi=28.0,
            macd=0.01, macd_signal=0.0, macd_hist=0.01,
            stoch_k=20.0,
            price=45_000.0, vwap=44_900.0,  # price_above_vwap
        )
        buy, sell, buy_t, _ = self.strat.score(ind, MEDIUM)
        assert buy == 4.0
        assert len(buy_t) == 4

    def test_full_confluence_sell(self):
        """All four sell conditions → capped at max_score=4.0."""
        ind = _ind(
            rsi=75.0,
            macd=-0.01, macd_signal=0.0, macd_hist=-0.01,
            stoch_k=80.0,
            price=44_000.0, vwap=44_500.0,  # price_below_vwap
        )
        buy, sell, _, sell_t = self.strat.score(ind, MEDIUM)
        assert sell == 4.0

    def test_partial_confluence_buy(self):
        """Only MACD fires → lower score."""
        ind = _ind(macd=0.01, macd_signal=0.0, macd_hist=0.01, rsi=50.0, stoch_k=50.0)
        buy, sell, buy_t, _ = self.strat.score(ind, MEDIUM)
        assert 0.0 < buy < 4.0

    def test_score_capped_at_max(self):
        """Score must not exceed max_score even when all conditions fire."""
        ind = _ind(
            rsi=25.0,
            macd=0.01, macd_signal=0.0, macd_hist=0.01,
            stoch_k=20.0,
            price=45_000.0, vwap=44_900.0,
        )
        buy, sell, _, _ = self.strat.score(ind, MEDIUM)
        assert buy <= self.strat.max_score


# ---------------------------------------------------------------------------
# New Indicators field smoke tests
# ---------------------------------------------------------------------------

class TestNewIndicatorFields:
    """Ensure new Indicators fields initialise correctly and have sane defaults."""

    def test_new_fields_default_to_none(self):
        ind = Indicators(
            rsi=50.0, macd=0.0, macd_signal=0.0, macd_hist=0.0,
            bb_upper=46000.0, bb_mid=45000.0, bb_lower=44000.0, price=45000.0,
        )
        for field in (
            "cci", "keltner_upper", "keltner_mid", "keltner_lower",
            "williams_r", "price_zscore",
            "mfi", "obv", "obv_ema", "obv_rising", "cmf",
            "vwap_std", "vwap_upper1", "vwap_upper2", "vwap_lower1", "vwap_lower2",
            "supertrend", "supertrend_bullish",
            "rsi_fast", "rsi_slow", "bb_width", "bb_width_percentile", "squeeze_active",
        ):
            assert getattr(ind, field) is None, f"Expected {field} to default to None"

    def test_new_fields_accept_values(self):
        ind = _ind(
            cci=-120.0, keltner_upper=47000.0, keltner_mid=45000.0, keltner_lower=43000.0,
            williams_r=-85.0, price_zscore=-2.1,
            mfi=22.0, obv=1_000_000.0, obv_ema=950_000.0, obv_rising=True, cmf=0.15,
            vwap_std=200.0, vwap_upper1=45200.0, vwap_upper2=45400.0,
            vwap_lower1=44800.0, vwap_lower2=44600.0,
            supertrend=44500.0, supertrend_bullish=True,
            rsi_fast=35.0, rsi_slow=40.0,
            bb_width=0.04, bb_width_percentile=0.15, squeeze_active=True,
        )
        assert ind.cci == -120.0
        assert ind.keltner_upper == 47000.0
        assert ind.williams_r == -85.0
        assert ind.price_zscore == -2.1
        assert ind.mfi == 22.0
        assert ind.obv_rising is True
        assert ind.supertrend_bullish is True
        assert ind.squeeze_active is True
        assert ind.bb_width_percentile == pytest.approx(0.15)

    def test_safe_float_helper(self):
        """Predictor._safe_float should handle None, NaN, and valid floats."""
        import math
        from src.prediction.predictor import Predictor
        assert Predictor._safe_float(None) is None
        assert Predictor._safe_float(float("nan")) is None
        assert Predictor._safe_float(42.0) == pytest.approx(42.0)
        assert Predictor._safe_float("3.14") == pytest.approx(3.14)
        assert Predictor._safe_float("not_a_number") is None
