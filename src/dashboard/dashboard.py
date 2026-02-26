"""
Live console dashboard for the crypto trading platform.

Renders a 5-panel rich Layout that refreshes in-place without scrolling:
  - Header  : platform title and current timestamp
  - Prices  : live bid/ask/price per tracked symbol
  - Portfolio: holdings with current value
  - P&L     : realized, unrealized, total
  - Strategy: active strategy and last trade action
  - News    : recent headlines with sentiment labels

Expected dependency interfaces (injected at runtime):

    MarketFeed
        feed.symbols: list[str]
        feed.get_latest(symbol) -> PriceTick | None
            PriceTick.price, .volume, .bid, .ask

    Portfolio
        portfolio.get_holdings() -> dict[str, Holding]
            Holding.symbol, .quantity, .avg_cost
        portfolio.get_cash() -> float
        portfolio.get_unrealized_pnl(prices: dict[str, float]) -> float
        portfolio.get_realized_pnl() -> float
        portfolio.get_total_value(prices: dict[str, float]) -> float

    NewsFeed
        news_feed.get_latest() -> list[NewsItem]
            NewsItem.title, .source
            NewsItem.sentiment: SentimentResult | None
                SentimentResult.score: float   (-1 … +1 or 0 … 1)
                SentimentResult.label: str     e.g. "bullish"/"neutral"/"bearish"

    TradeEngine
        engine.get_last_action() -> str
        engine.get_active_strategy() -> str
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    # Avoid hard import cycles; these are structural/duck-typed at runtime.
    from typing import Any

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SENTIMENT_COLORS: dict[str, str] = {
    "bullish": "green",
    "positive": "green",
    "neutral": "yellow",
    "mixed": "yellow",
    "bearish": "red",
    "negative": "red",
}


def _sentiment_color(label: str) -> str:
    """Return a rich color name for a sentiment label string."""
    return _SENTIMENT_COLORS.get(label.lower(), "white")


def _pnl_color(value: float) -> str:
    """Return green for positive, red for negative, white for zero."""
    if value > 0:
        return "green"
    if value < 0:
        return "red"
    return "white"


def _fmt_price(price: float) -> str:
    """Format a price with commas and two decimal places."""
    return f"${price:,.2f}"


def _fmt_signed(value: float) -> str:
    """Format a signed dollar value with explicit + or - prefix."""
    sign = "+" if value >= 0 else ""
    return f"{sign}${value:,.2f}"


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


class Dashboard:
    """
    Async live dashboard for the crypto trading platform.

    Usage::

        dashboard = Dashboard()
        await dashboard.run(market_feed, portfolio, news_feed, engine)
    """

    def __init__(self) -> None:
        self.console = Console()
        # Tracks the previous price of each symbol so we can show ▲/▼.
        self._prev_prices: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Panel / table builders
    # ------------------------------------------------------------------

    def _make_prices_table(self, market_feed: Any) -> Table:
        """
        Build the LIVE PRICES table.

        Columns: Symbol | Price | Bid | Ask | Volume | Change
        A ▲ (green) or ▼ (red) indicator shows whether the price moved up
        or down since the previous refresh.  On the first render the
        direction indicator is omitted.
        """
        table = Table(
            title="[bold cyan]LIVE PRICES[/bold cyan]",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold magenta",
            expand=True,
        )
        table.add_column("Symbol", style="bold", no_wrap=True)
        table.add_column("Price", justify="right")
        table.add_column("Bid", justify="right", style="dim")
        table.add_column("Ask", justify="right", style="dim")
        table.add_column("Volume", justify="right", style="dim")
        table.add_column("Dir", justify="center", no_wrap=True)

        symbols: list[str] = getattr(market_feed, "symbols", []) or []

        for symbol in symbols:
            tick = market_feed.get_latest(symbol)
            if tick is None:
                table.add_row(
                    symbol,
                    "[dim]—[/dim]",
                    "[dim]—[/dim]",
                    "[dim]—[/dim]",
                    "[dim]—[/dim]",
                    "[dim]?[/dim]",
                )
                continue

            price: float = float(tick.price)
            bid: float = float(tick.bid) if tick.bid is not None else price
            ask: float = float(tick.ask) if tick.ask is not None else price
            volume: float = float(tick.volume) if tick.volume is not None else 0.0

            # Direction indicator vs previous refresh
            prev = self._prev_prices.get(symbol)
            if prev is None:
                direction = Text("•", style="dim")
            elif price > prev:
                direction = Text("▲", style="bold green")
            elif price < prev:
                direction = Text("▼", style="bold red")
            else:
                direction = Text("=", style="dim")

            # Price colour based on direction
            if prev is not None and price != prev:
                price_color = "green" if price > prev else "red"
            else:
                price_color = "white"

            # Volume formatting: show K / M suffixes for readability
            if volume >= 1_000_000:
                vol_str = f"{volume / 1_000_000:.2f}M"
            elif volume >= 1_000:
                vol_str = f"{volume / 1_000:.1f}K"
            else:
                vol_str = f"{volume:.4f}"

            table.add_row(
                f"[bold]{symbol}[/bold]",
                f"[{price_color}]{_fmt_price(price)}[/{price_color}]",
                f"[dim]{_fmt_price(bid)}[/dim]",
                f"[dim]{_fmt_price(ask)}[/dim]",
                f"[dim]{vol_str}[/dim]",
                direction,
            )

        return table

    def _make_portfolio_table(self, portfolio: Any, prices: dict[str, float]) -> Table:
        """
        Build the PORTFOLIO table.

        Shows each holding's symbol, quantity, average cost, current price,
        current value, and unrealized P&L for that position.  A cash row and
        total value row are appended at the bottom.
        """
        table = Table(
            title="[bold cyan]PORTFOLIO[/bold cyan]",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold magenta",
            expand=True,
        )
        table.add_column("Asset", style="bold", no_wrap=True)
        table.add_column("Qty", justify="right")
        table.add_column("Avg Cost", justify="right", style="dim")
        table.add_column("Price", justify="right")
        table.add_column("Value", justify="right")
        table.add_column("Unreal. P&L", justify="right")

        holdings: dict[str, Any] = portfolio.get_holdings()

        for symbol, holding in holdings.items():
            qty: float = float(holding.quantity)
            avg_cost: float = float(holding.avg_cost)
            current_price: float = prices.get(symbol, avg_cost)
            current_value: float = qty * current_price
            pos_pnl: float = (current_price - avg_cost) * qty
            pnl_color = _pnl_color(pos_pnl)

            table.add_row(
                symbol,
                f"{qty:.6g}",
                f"[dim]{_fmt_price(avg_cost)}[/dim]",
                _fmt_price(current_price),
                _fmt_price(current_value),
                f"[{pnl_color}]{_fmt_signed(pos_pnl)}[/{pnl_color}]",
            )

        # Cash row
        cash: float = portfolio.get_cash()
        table.add_row(
            "[italic]Cash[/italic]",
            "—",
            "—",
            "—",
            f"[green]{_fmt_price(cash)}[/green]",
            "—",
        )

        # Total value row
        total_value: float = portfolio.get_total_value(prices)
        table.add_row(
            "[bold]TOTAL[/bold]",
            "",
            "",
            "",
            f"[bold]{_fmt_price(total_value)}[/bold]",
            "",
        )

        return table

    def _make_pnl_panel(self, portfolio: Any, prices: dict[str, float]) -> Panel:
        """
        Build the P&L panel.

        Shows realized P&L, unrealized P&L, and the combined total.
        Each figure is coloured green/red depending on its sign.
        """
        realized: float = portfolio.get_realized_pnl()
        unrealized: float = portfolio.get_unrealized_pnl(prices)
        total: float = realized + unrealized

        r_color = _pnl_color(realized)
        u_color = _pnl_color(unrealized)
        t_color = _pnl_color(total)

        content = Text()
        content.append("Realized:    ", style="dim")
        content.append(f"{_fmt_signed(realized)}\n", style=f"bold {r_color}")
        content.append("Unrealized:  ", style="dim")
        content.append(f"{_fmt_signed(unrealized)}\n", style=f"bold {u_color}")
        content.append("─" * 22 + "\n", style="dim")
        content.append("Total P&L:   ", style="bold")
        content.append(f"{_fmt_signed(total)}", style=f"bold {t_color}")

        return Panel(
            content,
            title="[bold cyan]P&L[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
            expand=True,
        )

    def _make_strategy_panel(self, engine: Any) -> Panel:
        """
        Build the STRATEGY panel.

        Displays the active strategy name and the most recent trade action
        taken by the engine.
        """
        strategy: str = engine.get_active_strategy() or "[dim]None[/dim]"
        last_action: str = engine.get_last_action() or "[dim]None[/dim]"

        content = Text()
        content.append("Strategy:    ", style="dim")
        content.append(f"{strategy}\n", style="bold yellow")
        content.append("Last Action: ", style="dim")
        content.append(last_action, style="bold white")

        return Panel(
            content,
            title="[bold cyan]STRATEGY[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
            expand=True,
        )

    def _make_markets_table(self, indices_feed: Any) -> Table:
        """
        Build the GLOBAL MARKETS table.

        Columns: Index | Value | Change | % Change
        VIX is highlighted separately as a fear gauge.
        Change columns are green for positive, red for negative.
        """
        table = Table(
            title="[bold cyan]GLOBAL MARKETS[/bold cyan]",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold magenta",
            expand=True,
        )
        table.add_column("Index", style="bold", no_wrap=True, min_width=14)
        table.add_column("Value", justify="right")
        table.add_column("Change", justify="right")
        table.add_column("% Change", justify="right")

        ticks: dict[str, Any] = {}
        if indices_feed is not None:
            ticks = getattr(indices_feed, "get_all", lambda: {})()

        if not ticks:
            table.add_row("[dim]Fetching…[/dim]", "", "", "")
            return table

        for sym, tick in ticks.items():
            price: float = float(tick.price)
            change: float = float(tick.change)
            change_pct: float = float(tick.change_pct)
            name: str = str(tick.name)

            # VIX is a fear gauge — invert color logic (high VIX = red)
            if sym == "^VIX":
                price_style = "bold red" if price > 20 else "bold green"
                chg_color = "red" if change > 0 else "green"
            else:
                price_style = "white"
                chg_color = "green" if change >= 0 else "red"

            sign = "+" if change >= 0 else ""
            table.add_row(
                f"[bold]{name}[/bold]",
                f"[{price_style}]{price:,.2f}[/{price_style}]",
                f"[{chg_color}]{sign}{change:,.2f}[/{chg_color}]",
                f"[{chg_color}]{sign}{change_pct:.2f}%[/{chg_color}]",
            )

        return table

    def _make_news_panel(self, news_feed: Any) -> Panel:
        """
        Build the NEWS & SENTIMENT panel.

        Each item shows:
            [LABEL score] Headline title — Source
        Label colour: green=bullish, yellow=neutral, red=bearish.
        Up to 8 items are displayed; older items are truncated.
        """
        items: list[Any] = news_feed.get_latest() or []

        content = Text()

        if not items:
            content.append("No recent news available.", style="dim italic")
        else:
            # Show at most 8 headlines to avoid overflowing the panel
            for item in items[:8]:
                sentiment = item.sentiment

                if sentiment is not None:
                    label: str = str(sentiment.label).upper()
                    score: float = float(sentiment.score)
                    color = _sentiment_color(str(sentiment.label))
                    badge = f"[{label} {score:+.2f}]"
                    content.append(badge, style=f"bold {color}")
                else:
                    content.append("[UNSCORED]", style="dim")

                # Truncate long titles so they fit on one line
                title: str = str(item.title)
                source: str = str(item.source) if item.source else "Unknown"
                if len(title) > 55:
                    title = title[:52] + "..."

                content.append(f" {title}", style="white")
                content.append(f" — {source}\n", style="dim")

        return Panel(
            content,
            title="[bold cyan]NEWS & SENTIMENT[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
            expand=True,
        )

    # ------------------------------------------------------------------
    # Layout assembly
    # ------------------------------------------------------------------

    def _make_header(self) -> Panel:
        """Render the top header bar with platform name and current time."""
        now: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header_text = Text(justify="center")
        header_text.append("CRYPTO TRADING PLATFORM", style="bold white on dark_blue")
        header_text.append(f"    {now}", style="dim cyan")

        return Panel(
            header_text,
            box=box.HEAVY,
            border_style="blue",
            padding=(0, 1),
        )

    def _build_layout(
        self,
        market_feed: Any,
        portfolio: Any,
        news_feed: Any,
        engine: Any,
        indices_feed: Any = None,
    ) -> Layout:
        """
        Assemble all panels into the full 6-region Layout:

            ┌────────────────────── header ──────────────────────┐
            │ crypto prices │    portfolio      │      p&l        │
            ├────────────────── global markets ──────────────────┤
            ├────────────────────── strategy ────────────────────┤
            └────────────────────── news ─────────────────────────┘
        """
        # Collect current prices once; reused by portfolio and P&L panels.
        symbols: list[str] = getattr(market_feed, "symbols", []) or []
        prices: dict[str, float] = {}
        for sym in symbols:
            tick = market_feed.get_latest(sym)
            if tick is not None:
                prices[sym] = float(tick.price)

        # Build individual panels/tables
        prices_table = self._make_prices_table(market_feed)
        portfolio_table = self._make_portfolio_table(portfolio, prices)
        pnl_panel = self._make_pnl_panel(portfolio, prices)
        strategy_panel = self._make_strategy_panel(engine)
        news_panel = self._make_news_panel(news_feed)
        header_panel = self._make_header()
        markets_table = self._make_markets_table(indices_feed)

        # Update internal price history after rendering prices table
        for sym, price in prices.items():
            self._prev_prices[sym] = price

        # ------------------------------------------------------------------
        # Layout tree
        # ------------------------------------------------------------------
        layout = Layout()

        # Split vertically: header / main / markets / strategy / news
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=5),
            Layout(name="markets", size=12),
            Layout(name="strategy", size=4),
            Layout(name="news", ratio=3),
        )

        layout["header"].update(header_panel)

        # Main row: crypto prices | portfolio | pnl  (ratios 3 : 5 : 2)
        layout["main"].split_row(
            Layout(name="prices", ratio=3),
            Layout(name="portfolio", ratio=5),
            Layout(name="pnl", ratio=2),
        )
        layout["main"]["prices"].update(
            Panel(
                prices_table,
                border_style="blue",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )
        layout["main"]["portfolio"].update(
            Panel(
                portfolio_table,
                border_style="blue",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )
        layout["main"]["pnl"].update(pnl_panel)

        layout["markets"].update(
            Panel(
                markets_table,
                border_style="magenta",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )
        layout["strategy"].update(strategy_panel)
        layout["news"].update(news_panel)

        return layout

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        market_feed: Any,
        portfolio: Any,
        news_feed: Any,
        engine: Any,
        refresh: float = 0.5,
        indices_feed: Any = None,
    ) -> None:
        """
        Start the live dashboard loop.

        Parameters
        ----------
        market_feed:
            Object satisfying the MarketFeed interface.
        portfolio:
            Object satisfying the Portfolio interface.
        news_feed:
            Object satisfying the NewsFeed interface.
        engine:
            Object satisfying the TradeEngine interface.
        refresh:
            Seconds between each screen redraw.  Defaults to 0.5 s (2 fps).
            rich's refresh_per_second is set to the reciprocal of this value.
        indices_feed:
            Optional :class:`~src.market.indices.IndicesFeed` instance.
            When provided, renders the GLOBAL MARKETS panel with real-time
            equity index data (VIX, S&P 500, etc.).
        """
        refresh_per_second: int = max(1, int(1 / refresh))

        with Live(
            console=self.console,
            refresh_per_second=refresh_per_second,
            screen=True,
        ) as live:
            while True:
                layout = self._build_layout(
                    market_feed, portfolio, news_feed, engine,
                    indices_feed=indices_feed,
                )
                live.update(layout)
                await asyncio.sleep(refresh)
