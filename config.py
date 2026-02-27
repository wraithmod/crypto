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
    # Tracked crypto symbols (Binance trading pairs)
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

    # How often to poll Yahoo Finance for global index data (seconds)
    indices_poll_interval: float = 30.0

    # -----------------------------------------------------------------------
    # ASX (Australian Stock Exchange) — experimental feature flag
    # Set asx_enabled=False to disable the ASX panel entirely.
    # -----------------------------------------------------------------------
    asx_enabled: bool = True

    # How often to poll ASX prices (seconds) — slower than crypto/global indices
    asx_poll_interval: float = 60.0

    # Top 50 ASX stocks by market cap (Yahoo Finance format: TICKER.AX)
    # NOTE: NCM.AX (Newcrest) acquired by Newmont 2023 — excluded.
    #       SYD.AX (Sydney Airport) privatised 2022 — excluded.
    asx_symbols: list[str] = field(default_factory=lambda: [
        # --- Banking & Financial Services ---
        "CBA.AX",   # Commonwealth Bank         ~$230B
        "NAB.AX",   # National Australia Bank    ~$115B
        "WBC.AX",   # Westpac Banking            ~$105B
        "ANZ.AX",   # ANZ Group                  ~$85B
        "MQG.AX",   # Macquarie Group            ~$75B
        "QBE.AX",   # QBE Insurance              ~$25B
        "SUN.AX",   # Suncorp Group              ~$18B
        "IAG.AX",   # Insurance Australia Group  ~$15B
        "CPU.AX",   # Computershare              ~$18B
        "ASX.AX",   # ASX Limited (the exchange) ~$15B
        # --- Mining & Resources ---
        "BHP.AX",   # BHP Group                  ~$220B
        "RIO.AX",   # Rio Tinto                  ~$180B
        "FMG.AX",   # Fortescue Metals           ~$65B
        "S32.AX",   # South32                    ~$20B
        "MIN.AX",   # Mineral Resources          ~$12B
        "PLS.AX",   # Pilbara Minerals (lithium) ~$8B
        "IGO.AX",   # IGO Limited (lithium/nickel) ~$5B
        "LYC.AX",   # Lynas Rare Earths          ~$7B
        # --- Energy ---
        "WDS.AX",   # Woodside Energy            ~$60B
        "STO.AX",   # Santos                     ~$25B
        "ORG.AX",   # Origin Energy              ~$20B
        "AGL.AX",   # AGL Energy                 ~$8B
        # --- Healthcare & Biotech ---
        "CSL.AX",   # CSL Limited (biotech)      ~$145B
        "COH.AX",   # Cochlear (hearing implants) ~$20B
        "RMD.AX",   # ResMed (sleep apnea)       ~$35B
        "RHC.AX",   # Ramsay Health Care         ~$15B
        "PME.AX",   # Pro Medicus (med imaging)  ~$15B
        "SHL.AX",   # Sonic Healthcare           ~$10B
        # --- Consumer & Retail ---
        "WES.AX",   # Wesfarmers (Bunnings)      ~$80B
        "WOW.AX",   # Woolworths Group           ~$40B
        "COL.AX",   # Coles Group                ~$25B
        "JBH.AX",   # JB Hi-Fi                   ~$8B
        "HVN.AX",   # Harvey Norman              ~$7B
        "TWE.AX",   # Treasury Wine Estates      ~$8B
        # --- Technology ---
        "WTC.AX",   # WiseTech Global (logistics) ~$30B
        "XRO.AX",   # Xero (accounting SaaS)     ~$20B
        "REA.AX",   # REA Group (property)       ~$25B
        "CAR.AX",   # Carsales.com               ~$10B
        "SEK.AX",   # Seek (jobs)                ~$7B
        "NXT.AX",   # NextDC (data centres)      ~$5B
        # --- Telco & Infrastructure ---
        "TLS.AX",   # Telstra                    ~$45B
        "TCL.AX",   # Transurban (toll roads)    ~$45B
        "QAN.AX",   # Qantas Airways             ~$12B
        "TPG.AX",   # TPG Telecom                ~$6B
        # --- REITs & Property ---
        "GMG.AX",   # Goodman Group (industrial REIT) ~$60B
        "SCG.AX",   # Scentre Group (Westfield)  ~$15B
        "DXS.AX",   # Dexus                      ~$7B
        "MGR.AX",   # Mirvac Group               ~$8B
        # --- Gold & Precious Metals ---
        "NST.AX",   # Northern Star Resources    ~$15B
        "EVN.AX",   # Evolution Mining           ~$10B
        # --- Diversified / Other ---
        "ALL.AX",   # Aristocrat Leisure (gaming) ~$25B
    ])

    # ASX-specific news RSS feeds (Australian finance)
    asx_news_feeds: list[str] = field(default_factory=lambda: [
        # Tier 1 — major Australian business news
        "https://www.abc.net.au/news/feed/52278/rss.xml",         # ABC News Business
        "https://www.afr.com/rss",                                 # Australian Financial Review
        # Tier 2 — ASX-focused
        "https://stockhead.com.au/feed/",                          # Stockhead (small/mid ASX)
        "https://www.businessinsider.com.au/feed",                 # Business Insider AU
        "https://www.investsmart.com.au/rss",                      # InvestSMART
        # Yahoo Finance ASX pairs
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^AXJO&region=AU&lang=en-AU",  # ASX 200
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=CBA.AX&region=AU&lang=en-AU",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=BHP.AX&region=AU&lang=en-AU",
    ])

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
