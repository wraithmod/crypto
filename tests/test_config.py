"""Tests for config.py — AppConfig defaults relevant to US stocks and feature flags."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import AppConfig


# ---------------------------------------------------------------------------
# US stocks config
# ---------------------------------------------------------------------------

class TestUSStocksConfig:
    """Validate AppConfig fields related to the us_stocks feature."""

    def setup_method(self):
        # Use a fresh instance so tests are independent of the live singleton.
        self.cfg = AppConfig()

    def test_us_stocks_enabled_is_true_by_default(self):
        assert self.cfg.us_stocks_enabled is True

    def test_us_stocks_symbols_has_exactly_20_items(self):
        assert len(self.cfg.us_stocks_symbols) == 20

    def test_nvda_in_us_stocks_symbols(self):
        assert "NVDA" in self.cfg.us_stocks_symbols

    def test_aapl_in_us_stocks_symbols(self):
        assert "AAPL" in self.cfg.us_stocks_symbols

    def test_msft_in_us_stocks_symbols(self):
        assert "MSFT" in self.cfg.us_stocks_symbols

    def test_amzn_in_us_stocks_symbols(self):
        assert "AMZN" in self.cfg.us_stocks_symbols

    def test_googl_in_us_stocks_symbols(self):
        assert "GOOGL" in self.cfg.us_stocks_symbols

    def test_meta_in_us_stocks_symbols(self):
        assert "META" in self.cfg.us_stocks_symbols

    def test_avgo_in_us_stocks_symbols(self):
        assert "AVGO" in self.cfg.us_stocks_symbols

    def test_tsla_in_us_stocks_symbols(self):
        assert "TSLA" in self.cfg.us_stocks_symbols

    def test_brk_b_in_us_stocks_symbols(self):
        assert "BRK-B" in self.cfg.us_stocks_symbols

    def test_wmt_in_us_stocks_symbols(self):
        assert "WMT" in self.cfg.us_stocks_symbols

    def test_lly_in_us_stocks_symbols(self):
        assert "LLY" in self.cfg.us_stocks_symbols

    def test_jpm_in_us_stocks_symbols(self):
        assert "JPM" in self.cfg.us_stocks_symbols

    def test_xom_in_us_stocks_symbols(self):
        assert "XOM" in self.cfg.us_stocks_symbols

    def test_v_in_us_stocks_symbols(self):
        assert "V" in self.cfg.us_stocks_symbols

    def test_jnj_in_us_stocks_symbols(self):
        assert "JNJ" in self.cfg.us_stocks_symbols

    def test_mu_in_us_stocks_symbols(self):
        assert "MU" in self.cfg.us_stocks_symbols

    def test_ma_in_us_stocks_symbols(self):
        assert "MA" in self.cfg.us_stocks_symbols

    def test_cost_in_us_stocks_symbols(self):
        assert "COST" in self.cfg.us_stocks_symbols

    def test_orcl_in_us_stocks_symbols(self):
        assert "ORCL" in self.cfg.us_stocks_symbols

    def test_abbv_in_us_stocks_symbols(self):
        assert "ABBV" in self.cfg.us_stocks_symbols

    def test_unh_not_in_us_stocks_symbols(self):
        """UNH is not in the current 20-stock list."""
        assert "UNH" not in self.cfg.us_stocks_symbols

    def test_us_stocks_symbols_has_no_duplicates(self):
        syms = self.cfg.us_stocks_symbols
        assert len(syms) == len(set(syms))

    def test_all_us_stocks_are_non_empty_strings(self):
        for sym in self.cfg.us_stocks_symbols:
            assert isinstance(sym, str) and len(sym) > 0

    def test_no_ax_suffix_in_us_stocks(self):
        """US stock symbols must not accidentally contain ASX tickers."""
        for sym in self.cfg.us_stocks_symbols:
            assert not sym.endswith(".AX"), f"{sym} should not be an ASX ticker"

    def test_no_caret_prefix_in_us_stocks(self):
        """US stock symbols must not accidentally contain index symbols."""
        for sym in self.cfg.us_stocks_symbols:
            assert not sym.startswith("^"), f"{sym} should not be an index symbol"
