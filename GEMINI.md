# GEMINI Sub-Agent

## Role
Market research, news source discovery, financial data analysis, and sentiment system design for the crypto trading platform at `/home/wraith/making`.

## Capabilities
- Web research: finding live RSS feeds, verifying URLs, assessing source credibility
- Financial analysis: evaluating sentiment signals, indicator quality, market data sources
- News pipeline design: feed weighting, relevance scoring, per-coin filtering

## Verified RSS Feeds (Feb 2026)

### Tier-1 Crypto (weight=1.0) — VERIFIED LIVE
- CoinDesk: `https://www.coindesk.com/arc/outboundfeeds/rss/`
- CoinTelegraph: `https://cointelegraph.com/rss`
- Decrypt: `https://decrypt.co/feed`

### Tier-1 Crypto (weight=1.0) — Unverified but standard
- Bitcoin Magazine: `https://bitcoinmagazine.com/news/feed`

### Tier-1 Macro (weight=1.0)
- Reuters Business: `https://feeds.reuters.com/reuters/businessNews`
- Reuters Tech: `https://feeds.reuters.com/reuters/technologyNews`
- Yahoo Finance BTC: `https://feeds.finance.yahoo.com/rss/2.0/headline?s=BTC-USD`
- Yahoo Finance ETH: `https://feeds.finance.yahoo.com/rss/2.0/headline?s=ETH-USD`

### Tier-2 Crypto (weight=0.65)
- CryptoSlate: `https://cryptoslate.com/feed/`
- BeInCrypto: `https://beincrypto.com/feed/`
- NewsBTC: `https://www.newsbtc.com/feed/`
- AMBCrypto: `https://ambcrypto.com/feed/` — best for meme/altcoins
- CoinGape: `https://coingape.com/feed/` — breaks meme trends early

### Tier-2 DeFi/Web3 (weight=0.85)
- The Defiant: `https://thedefiant.io/feed`
- Blockworks: `https://blockworks.co/feed`

### Tier-2 Finance (weight=0.6)
- CNBC Finance: `https://www.cnbc.com/id/10001147/device/rss/rss.html`
- CNBC Crypto: `https://www.cnbc.com/id/100727362/device/rss/rss.html`

### Topic Subfeeds (Tier-1 source, weight=1.0)
- CoinTelegraph Altcoins: `https://cointelegraph.com/rss/tag/altcoin`
- CoinTelegraph DeFi: `https://cointelegraph.com/rss/tag/defi`
- CoinTelegraph Bitcoin: `https://cointelegraph.com/rss/tag/bitcoin`

### Blocked/Unavailable
- The Block: 403 Forbidden
- MarketWatch: CDN blocking
- Cryptonews.com: 403 Forbidden

## Meme Coin Coverage (PEPE, WIF, SHIB)
Best sources: AMBCrypto, CoinGape, CoinTelegraph Altcoins subfeed, CryptoSlate

## Headline Scoring Recommendations
- Free tier LLM: 5-10 headlines per 60s cycle
- Paid tier: 15-20 headlines per cycle
- Run sentiment concurrently (asyncio.gather)
- Deduplicate by title hash before scoring

## Credibility Weights
```
Tier-1 crypto/macro: 1.0
DeFi-specific:       0.85
Tier-2 crypto:       0.65
Macro (CNBC etc):    0.6
Tier-3 aggregators:  0.4
```
