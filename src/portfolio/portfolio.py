import uuid
import logging
import time
from threading import Lock
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    symbol: str
    side: str           # "buy" or "sell"
    quantity: float
    price: float
    timestamp: float
    trade_id: str       # uuid4


@dataclass
class Holding:
    symbol: str
    quantity: float
    avg_cost: float     # weighted average cost basis

    @property
    def total_cost(self) -> float:
        return self.quantity * self.avg_cost


class Portfolio:
    def __init__(self, initial_cash: float = 10_000.0) -> None:
        self._cash: float = initial_cash
        self._holdings: dict[str, Holding] = {}
        self._trade_history: list[Trade] = []
        self._realized_pnl: float = 0.0
        self._lock: Lock = Lock()
        logger.info("Portfolio initialized with cash=%.2f", initial_cash)

    def add_trade(self, trade: Trade) -> None:
        with self._lock:
            side = trade.side.lower()
            if side not in ("buy", "sell"):
                raise ValueError(f"Trade side must be 'buy' or 'sell', got '{trade.side}'")

            if side == "buy":
                cost = trade.quantity * trade.price
                if cost > self._cash:
                    raise ValueError(
                        f"Insufficient cash for buy: need {cost:.2f}, have {self._cash:.2f}"
                    )

                self._cash -= cost

                if trade.symbol in self._holdings:
                    existing = self._holdings[trade.symbol]
                    new_total_qty = existing.quantity + trade.quantity
                    new_avg_cost = (
                        (existing.quantity * existing.avg_cost)
                        + (trade.quantity * trade.price)
                    ) / new_total_qty
                    existing.quantity = new_total_qty
                    existing.avg_cost = new_avg_cost
                    logger.debug(
                        "BUY %s qty=%.6f @ %.6f | new_qty=%.6f new_avg_cost=%.6f",
                        trade.symbol, trade.quantity, trade.price,
                        existing.quantity, existing.avg_cost,
                    )
                else:
                    self._holdings[trade.symbol] = Holding(
                        symbol=trade.symbol,
                        quantity=trade.quantity,
                        avg_cost=trade.price,
                    )
                    logger.debug(
                        "BUY %s qty=%.6f @ %.6f | new position opened",
                        trade.symbol, trade.quantity, trade.price,
                    )

            else:  # sell
                if trade.symbol not in self._holdings:
                    raise ValueError(
                        f"Cannot sell {trade.symbol}: no position held"
                    )

                existing = self._holdings[trade.symbol]
                if trade.quantity > existing.quantity:
                    raise ValueError(
                        f"Cannot sell {trade.quantity:.6f} {trade.symbol}: "
                        f"only {existing.quantity:.6f} held"
                    )

                realized = (trade.price - existing.avg_cost) * trade.quantity
                self._realized_pnl += realized
                proceeds = trade.quantity * trade.price
                self._cash += proceeds

                existing.quantity -= trade.quantity

                logger.debug(
                    "SELL %s qty=%.6f @ %.6f | realized_pnl=%.6f proceeds=%.6f",
                    trade.symbol, trade.quantity, trade.price, realized, proceeds,
                )

                if existing.quantity == 0.0:
                    del self._holdings[trade.symbol]
                    logger.debug("Position closed for %s", trade.symbol)

            self._trade_history.append(trade)
            logger.info(
                "Trade recorded: %s %s %.6f @ %.6f (id=%s)",
                trade.side.upper(), trade.symbol, trade.quantity,
                trade.price, trade.trade_id,
            )

    def get_holdings(self) -> dict[str, Holding]:
        with self._lock:
            return dict(self._holdings)

    def get_cash(self) -> float:
        with self._lock:
            return self._cash

    def get_unrealized_pnl(self, prices: dict[str, float]) -> float:
        with self._lock:
            total = 0.0
            for symbol, holding in self._holdings.items():
                if symbol in prices:
                    total += (prices[symbol] - holding.avg_cost) * holding.quantity
                else:
                    logger.warning(
                        "No price available for %s when calculating unrealized P&L", symbol
                    )
            return total

    def get_realized_pnl(self) -> float:
        with self._lock:
            return self._realized_pnl

    def get_total_value(self, prices: dict[str, float]) -> float:
        with self._lock:
            holdings_value = sum(
                holding.quantity * prices[symbol]
                for symbol, holding in self._holdings.items()
                if symbol in prices
            )
            return self._cash + holdings_value

    def get_trade_history(self) -> list[Trade]:
        with self._lock:
            return list(self._trade_history)

    def get_position_value(self, symbol: str, prices: dict[str, float]) -> float:
        with self._lock:
            if symbol not in self._holdings:
                return 0.0
            if symbol not in prices:
                logger.warning(
                    "No price available for %s when calculating position value", symbol
                )
                return 0.0
            return self._holdings[symbol].quantity * prices[symbol]


def make_trade(
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    timestamp: Optional[float] = None,
) -> Trade:
    """Factory helper that auto-generates trade_id and defaults timestamp to now."""
    return Trade(
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        timestamp=timestamp if timestamp is not None else time.time(),
        trade_id=str(uuid.uuid4()),
    )
