"""Central configuration loaded from key files and environment."""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent

def _read_key(filename: str) -> str:
    path = ROOT / filename
    if path.exists():
        return path.read_text().strip()
    return ""

@dataclass
class AppConfig:
    # Tracked symbols
    symbols: list[str] = field(default_factory=lambda: [
        # Core / large-cap (liquid, established)
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
        "XRPUSDT", "ADAUSDT", "AVAXUSDT", "DOGEUSDT",
        # Mid-cap momentum
        "LINKUSDT", "NEARUSDT", "SUIUSDT", "INJUSDT",
        # Speculative / high-volatility (wide swings, HFT-friendly)
        "PEPEUSDT", "WIFUSDT", "SHIBUSDT",
    ])

    # Paper trading seed capital
    initial_cash: float = 10_000.0

    # LLM provider selection: "gemini" | "openai" | "claude"
    llm_provider: str = "gemini"

    # Binance WebSocket
    binance_ws_url: str = "wss://stream.binance.com:9443/stream"

    # Dashboard refresh rate (seconds)
    dashboard_refresh: float = 0.5

    # Price history length for indicators
    price_history_len: int = 200

    # HFT loop interval (seconds)
    hft_interval: float = 1.0

    # News fetch interval (seconds)
    news_interval: float = 60.0

    # Max position size as fraction of portfolio
    max_position_fraction: float = 0.30

    # Trade quantity (fraction of available cash per trade)
    trade_fraction: float = 0.10

    # Binance taker fee per trade (0.1% standard; 0.075% with BNB discount)
    binance_fee_rate: float = 0.001

    # Minimum profit % above avg_cost before a SELL fires (must exceed round-trip fees)
    # 0.2% fees + 0.15% profit margin = 0.35%
    min_profit_to_sell: float = 0.0035

    # Stop-loss: sell immediately if position is down this fraction from avg cost
    stop_loss_pct: float = 0.025

    # API Keys (loaded from files)
    claude_api_key: str = field(default_factory=lambda: _read_key("claude.key"))
    gemini_api_key: str = field(default_factory=lambda: _read_key("gemini.key"))
    openai_api_key: str = field(default_factory=lambda: _read_key("openai.key"))

    # Global market indices (Yahoo Finance symbols)
    tracked_indices: list[str] = field(default_factory=lambda: [
        "^VIX",    # CBOE Volatility Index
        "^GSPC",   # S&P 500
        "^IXIC",   # NASDAQ Composite
        "^DJI",    # Dow Jones Industrial Average
        "^N225",   # Nikkei 225 (Japan)
        "^GDAXI",  # DAX (Germany)
        "^FTSE",   # FTSE 100 (UK)
        "^HSI",    # Hang Seng (Hong Kong)
    ])

    # How often to poll Yahoo Finance for index data (seconds)
    indices_poll_interval: float = 30.0

    # News RSS feeds — grouped by tier (used for credibility weighting)
    news_feeds: list[str] = field(default_factory=lambda: [
        # --- Tier 1: High-credibility crypto ---
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
        "https://decrypt.co/feed",
        "https://www.theblock.co/rss.xml",
        "https://bitcoinmagazine.com/.rss/full/",
        # --- Tier 1: Macro / traditional finance ---
        "https://feeds.reuters.com/reuters/businessNews",
        "https://feeds.reuters.com/reuters/technologyNews",
        "https://www.cnbc.com/id/10001147/device/rss/rss.html",   # CNBC Finance
        "https://www.cnbc.com/id/100727362/device/rss/rss.html",  # CNBC Crypto
        # --- Tier 2: Crypto mid-tier ---
        "https://cryptoslate.com/feed/",
        "https://beincrypto.com/feed/",
        "https://www.newsbtc.com/feed/",
        "https://ambcrypto.com/feed/",
        "https://cryptonews.com/news/feed/",
        # --- Tier 2: DeFi / Web3 ---
        "https://blockworks.co/feed",
        "https://thedefiant.io/feed",
        # --- Tier 2: Yahoo Finance crypto pairs ---
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=BTC-USD&region=US&lang=en-US",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=ETH-USD&region=US&lang=en-US",
        # --- Tier 3: Community / speculative ---
        "https://cointelegraph.com/rss/tag/altcoin",
        "https://cointelegraph.com/rss/tag/defi",
    ])

# Singleton
config = AppConfig()
