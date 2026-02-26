"""Tests for src/portfolio/portfolio.py."""
import pytest
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.portfolio.portfolio import Portfolio, Trade, Holding, make_trade


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trade(symbol: str, side: str, qty: float, price: float) -> Trade:
    """Create a Trade with a stable UUID and a fixed timestamp for tests."""
    return Trade(
        symbol=symbol,
        side=side,
        quantity=qty,
        price=price,
        timestamp=time.time(),
        trade_id=str(uuid.uuid4()),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInitialState:
    """Portfolio is empty right after construction."""

    def test_initial_cash(self):
        p = Portfolio(initial_cash=10_000.0)
        assert p.get_cash() == pytest.approx(10_000.0)

    def test_initial_holdings_empty(self):
        p = Portfolio(initial_cash=10_000.0)
        assert p.get_holdings() == {}

    def test_initial_realized_pnl_zero(self):
        p = Portfolio(initial_cash=10_000.0)
        assert p.get_realized_pnl() == pytest.approx(0.0)

    def test_initial_trade_history_empty(self):
        p = Portfolio(initial_cash=10_000.0)
        assert p.get_trade_history() == []


class TestBuyTrade:
    """Buying an asset updates holdings and deducts cash."""

    def test_buy_creates_holding(self):
        p = Portfolio(initial_cash=50_000.0)
        p.add_trade(_trade("BTCUSDT", "buy", 1.0, 45_000.0))

        holdings = p.get_holdings()
        assert "BTCUSDT" in holdings

    def test_buy_quantity_correct(self):
        p = Portfolio(initial_cash=100_000.0)
        p.add_trade(_trade("BTCUSDT", "buy", 1.0, 45_000.0))

        assert p.get_holdings()["BTCUSDT"].quantity == pytest.approx(1.0)

    def test_buy_avg_cost_equals_price(self):
        p = Portfolio(initial_cash=100_000.0)
        p.add_trade(_trade("BTCUSDT", "buy", 1.0, 45_000.0))

        assert p.get_holdings()["BTCUSDT"].avg_cost == pytest.approx(45_000.0)

    def test_buy_deducts_cash(self):
        p = Portfolio(initial_cash=100_000.0)
        p.add_trade(_trade("BTCUSDT", "buy", 1.0, 45_000.0))

        assert p.get_cash() == pytest.approx(100_000.0 - 45_000.0)

    def test_buy_appends_trade_history(self):
        p = Portfolio(initial_cash=100_000.0)
        trade = _trade("BTCUSDT", "buy", 1.0, 45_000.0)
        p.add_trade(trade)

        history = p.get_trade_history()
        assert len(history) == 1
        assert history[0].trade_id == trade.trade_id


class TestSellTrade:
    """Selling calculates realized P&L and restores cash."""

    def test_sell_realized_pnl_profit(self):
        p = Portfolio(initial_cash=100_000.0)
        buy_price = 45_000.0
        sell_price = 50_000.0
        qty = 1.0

        p.add_trade(_trade("BTCUSDT", "buy", qty, buy_price))
        p.add_trade(_trade("BTCUSDT", "sell", qty, sell_price))

        expected_pnl = (sell_price - buy_price) * qty
        assert p.get_realized_pnl() == pytest.approx(expected_pnl)

    def test_sell_realized_pnl_loss(self):
        p = Portfolio(initial_cash=100_000.0)
        buy_price = 50_000.0
        sell_price = 45_000.0
        qty = 2.0

        p.add_trade(_trade("BTCUSDT", "buy", qty, buy_price))
        p.add_trade(_trade("BTCUSDT", "sell", qty, sell_price))

        expected_pnl = (sell_price - buy_price) * qty  # negative
        assert p.get_realized_pnl() == pytest.approx(expected_pnl)

    def test_sell_removes_holding_when_fully_liquidated(self):
        p = Portfolio(initial_cash=100_000.0)
        p.add_trade(_trade("BTCUSDT", "buy", 1.0, 45_000.0))
        p.add_trade(_trade("BTCUSDT", "sell", 1.0, 50_000.0))

        assert "BTCUSDT" not in p.get_holdings()

    def test_sell_increases_cash(self):
        p = Portfolio(initial_cash=100_000.0)
        p.add_trade(_trade("BTCUSDT", "buy", 1.0, 45_000.0))
        cash_after_buy = p.get_cash()

        p.add_trade(_trade("BTCUSDT", "sell", 1.0, 50_000.0))
        assert p.get_cash() == pytest.approx(cash_after_buy + 50_000.0)

    def test_sell_without_holding_raises(self):
        p = Portfolio(initial_cash=100_000.0)
        with pytest.raises(ValueError, match="no position held"):
            p.add_trade(_trade("BTCUSDT", "sell", 1.0, 45_000.0))

    def test_sell_more_than_held_raises(self):
        p = Portfolio(initial_cash=100_000.0)
        p.add_trade(_trade("BTCUSDT", "buy", 0.5, 45_000.0))
        with pytest.raises(ValueError, match="only"):
            p.add_trade(_trade("BTCUSDT", "sell", 1.0, 45_000.0))


class TestWeightedAvgCost:
    """Two buys at different prices yield a correct weighted average cost."""

    def test_weighted_avg_cost_two_buys(self):
        p = Portfolio(initial_cash=200_000.0)

        qty1, price1 = 1.0, 40_000.0
        qty2, price2 = 2.0, 50_000.0

        p.add_trade(_trade("BTCUSDT", "buy", qty1, price1))
        p.add_trade(_trade("BTCUSDT", "buy", qty2, price2))

        expected_avg = (qty1 * price1 + qty2 * price2) / (qty1 + qty2)
        assert p.get_holdings()["BTCUSDT"].avg_cost == pytest.approx(expected_avg)

    def test_weighted_avg_cost_total_quantity(self):
        p = Portfolio(initial_cash=200_000.0)
        p.add_trade(_trade("BTCUSDT", "buy", 1.0, 40_000.0))
        p.add_trade(_trade("BTCUSDT", "buy", 2.0, 50_000.0))

        assert p.get_holdings()["BTCUSDT"].quantity == pytest.approx(3.0)


class TestUnrealizedPnl:
    """Unrealized P&L reflects mark-to-market movement since purchase."""

    def test_unrealized_pnl_positive(self):
        p = Portfolio(initial_cash=100_000.0)
        p.add_trade(_trade("BTCUSDT", "buy", 1.0, 45_000.0))

        prices = {"BTCUSDT": 50_000.0}
        assert p.get_unrealized_pnl(prices) == pytest.approx(5_000.0)

    def test_unrealized_pnl_negative(self):
        p = Portfolio(initial_cash=100_000.0)
        p.add_trade(_trade("BTCUSDT", "buy", 1.0, 45_000.0))

        prices = {"BTCUSDT": 40_000.0}
        assert p.get_unrealized_pnl(prices) == pytest.approx(-5_000.0)

    def test_unrealized_pnl_missing_price(self):
        """Missing price for a held symbol is skipped gracefully (no crash)."""
        p = Portfolio(initial_cash=100_000.0)
        p.add_trade(_trade("BTCUSDT", "buy", 1.0, 45_000.0))

        # Pass an empty prices dict — no price for BTCUSDT
        assert p.get_unrealized_pnl({}) == pytest.approx(0.0)

    def test_unrealized_pnl_zero_when_no_holdings(self):
        p = Portfolio(initial_cash=10_000.0)
        assert p.get_unrealized_pnl({"BTCUSDT": 45_000.0}) == pytest.approx(0.0)


class TestTotalValue:
    """Total portfolio value equals cash plus marked-to-market holdings."""

    def test_total_value_no_holdings(self):
        p = Portfolio(initial_cash=10_000.0)
        assert p.get_total_value({"BTCUSDT": 45_000.0}) == pytest.approx(10_000.0)

    def test_total_value_with_holdings(self):
        p = Portfolio(initial_cash=100_000.0)
        p.add_trade(_trade("BTCUSDT", "buy", 1.0, 45_000.0))

        prices = {"BTCUSDT": 50_000.0}
        expected = (100_000.0 - 45_000.0) + 1.0 * 50_000.0  # 105_000.0
        assert p.get_total_value(prices) == pytest.approx(expected)

    def test_total_value_multiple_assets(self):
        p = Portfolio(initial_cash=200_000.0)
        p.add_trade(_trade("BTCUSDT", "buy", 1.0, 45_000.0))
        p.add_trade(_trade("ETHUSDT", "buy", 10.0, 3_000.0))

        prices = {"BTCUSDT": 50_000.0, "ETHUSDT": 3_500.0}
        cash = 200_000.0 - 45_000.0 - 30_000.0
        expected = cash + 1.0 * 50_000.0 + 10.0 * 3_500.0
        assert p.get_total_value(prices) == pytest.approx(expected)


class TestInsufficientCash:
    """Buying with insufficient cash raises ValueError (engine handles sizing)."""

    def test_insufficient_cash_raises_value_error(self):
        p = Portfolio(initial_cash=1_000.0)  # only $1000
        with pytest.raises(ValueError, match="Insufficient cash"):
            p.add_trade(_trade("BTCUSDT", "buy", 1.0, 45_000.0))

    def test_exact_cash_buy_succeeds(self):
        """Spending every dollar of cash should not raise."""
        p = Portfolio(initial_cash=45_000.0)
        # Should not raise
        p.add_trade(_trade("BTCUSDT", "buy", 1.0, 45_000.0))
        assert p.get_cash() == pytest.approx(0.0)


class TestMakeTradeHelper:
    """make_trade factory generates valid Trade objects."""

    def test_make_trade_fields(self):
        trade = make_trade("BTCUSDT", "buy", 0.5, 45_000.0)
        assert trade.symbol == "BTCUSDT"
        assert trade.side == "buy"
        assert trade.quantity == pytest.approx(0.5)
        assert trade.price == pytest.approx(45_000.0)
        assert trade.trade_id != ""

    def test_make_trade_custom_timestamp(self):
        ts = 1_700_000_000.0
        trade = make_trade("ETHUSDT", "sell", 2.0, 3_000.0, timestamp=ts)
        assert trade.timestamp == pytest.approx(ts)

    def test_make_trade_auto_timestamp(self):
        before = time.time()
        trade = make_trade("ETHUSDT", "buy", 1.0, 3_000.0)
        after = time.time()
        assert before <= trade.timestamp <= after
