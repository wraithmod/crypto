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

---

## Trading Strategy Research (Feb 2026)

*Research conducted: 2026-02-26. Sources: QuantifiedStrategies, Gate.io Wiki, FXOpen, BingX, CoinDesk,
FinanceFeeds, MindMathMoney, Changelly, Phemex Academy, StoicAI, and multiple backtested strategy databases.*

*Context note: Bitcoin's 14-day RSI dropped below 30 for only the third time in history in February 2026,
following a >50% correction from October 2025 highs. This provides real-world validation for the mean
reversion analysis below.*

---

### 1. Scalping Strategies

Scalping targets small, rapid price movements — typically 0.1% to 0.5% per trade — with very high
trade frequency. The goal is compounding many small wins rather than waiting for large directional moves.

#### RSI/MACD/BB Thresholds for High-Frequency Entries

The standard RSI(14) with 70/30 levels is **too slow** for scalping. Effective scalping uses faster
lookback periods with more extreme thresholds to reduce noise while still catching genuine extremes.

| Parameter | Conservative Scalp | Moderate Scalp | Aggressive Scalp |
|---|---|---|---|
| RSI period | 9 | 7 | 5 |
| RSI oversold entry | < 30 | < 25 | < 20 |
| RSI overbought exit | > 70 | > 75 | > 80 |
| MACD settings | 12-26-9 (default) | 8-17-9 (faster) | 3-10-16 (Raschke) |
| BB period / std dev | 20 / 2.0 | 15 / 1.8 | 10 / 1.5 |
| BB touch tolerance | 1.5% from band | 1.0% from band | 0.5% from band |
| Hold time target | 5-15 min | 1-5 min | 30s-2 min |
| Target profit per trade | 0.3-0.5% | 0.15-0.3% | 0.05-0.15% |

**Key principle for scalping:** Use multi-timeframe RSI confirmation. If both the 1-minute and 5-minute
RSI indicate deeply overbought (>80), the probability of a meaningful pullback increases substantially.
Single-timeframe scalping has significantly more false signals.

#### Win Rates and Risk/Reward for Scalping

| Strategy Type | Typical Win Rate | Risk/Reward | Notes |
|---|---|---|---|
| RSI+BB mean reversion scalp | 60-70% | 1:1 to 1:1.5 | Works best in range-bound markets |
| MACD+BB breakout scalp | 50-60% | 1:2 to 1:3 | Higher variance, better in trending markets |
| Triple-indicator (RSI+MACD+BB) | 65-75% | 1:1.5 to 1:2 | Fewer signals but higher quality |
| BB band-touch only | 55-65% | 1:1 | Simple, noisy, requires volume confirm |

**Win rate threshold for profitability:** With a 1:1 R/R ratio you need >55% win rate to profit after
fees. With 1:2 R/R, you only need >35%. For this system's ~0.2% round-trip fee cost, each trade must
exceed that threshold before counting as a winner.

**The existing system's RSI thresholds (35/65) are appropriate for medium-frequency trading (HFT loop
at 1s interval) but are too conservative for true sub-minute scalping. They serve well for the current
architecture where signals are evaluated on WebSocket tick data sampled into a 200-bar history.**

---

### 2. Momentum / Breakout Strategies

Momentum strategies chase directional moves after confirmation, rather than fading extremes. They have
lower win rates than mean reversion but larger average gains per winning trade.

#### BB Squeeze + MACD Crossover Combination

The Bollinger Band squeeze is one of the most reliable crypto momentum precursors. When bands contract
to their tightest in 20+ bars, volatility compression is at a maximum and a directional expansion is
imminent. The direction is confirmed by the MACD crossover.

**Entry conditions (bullish breakout):**
1. BB width falls to 20-bar minimum (squeeze detected)
2. Price closes above the upper BB (breakout candle)
3. MACD line crosses above signal line (momentum confirmation)
4. MACD histogram turns positive (momentum accelerating)
5. Optional: RSI above 50 (avoids buying into a still-bearish regime)

**Entry conditions (bearish breakdown):**
1. BB width at 20-bar minimum
2. Price closes below lower BB
3. MACD line crosses below signal line
4. MACD histogram turns negative
5. Optional: RSI below 50

**Performance metrics from backtested MACD+BB strategies:**
- Win rate: 65-78% (QuantifiedStrategies.com, backtested 2001-present on liquid instruments)
- Average gain per trade: 1.4% net of fees and slippage
- CAGR: ~12% with only 11% time in market
- Maximum drawdown: ~15%
- MACD divergences near BB extremes: 65% success rate, 1.8:1 reward/risk

**Important caveat for crypto:** Crypto has much higher volatility than equities. False breakouts
(price breaches band then snaps back) are common. A "robust candle close" beyond the band — not just
an intrabar wick — is essential for signal validity.

#### MACD-Only Crossover Performance

| Indicator Combo | Win Rate | Notes |
|---|---|---|
| MACD crossover alone | 50-55% | Near coin-flip, needs filters |
| MACD + RSI filter | 65-73% | RSI confirms direction, reduces whipsaws |
| MACD + BB breakout | 65-78% | Best documented multi-indicator combination |
| MACD golden cross + RSI overbought | 73-77% | Market reversal identification specifically |

**For the current system:** The existing `macd_bullish and macd_hist_rising` condition is correct
in principle. Enhancement: also check that the MACD histogram is *expanding* (current hist > prior hist)
not just positive, to catch only accelerating momentum, not fading moves.

---

### 3. Mean Reversion Strategies

Mean reversion exploits the tendency of assets to return to an equilibrium after extreme moves. Crypto
markets are particularly prone to overshoots due to leverage, retail panic, and thin liquidity — making
mean reversion viable but also dangerous in sustained downtrends.

#### RSI Oversold Reversal Thresholds in Crypto

| RSI Level | Market Regime | Typical Outcome | Recovery % (median) | Reliability |
|---|---|---|---|---|
| RSI < 45 (mild) | Mild pullback | Bounce within 2-4 bars | 1-3% | Low (65% false signals) |
| RSI < 35 (moderate) | Clear oversold | Bounce within 5-10 bars | 2-5% | Moderate (55% reliable) |
| RSI < 30 (strong) | Strong oversold | Multi-bar bounce or consolidation | 4-10% | Moderate-high (62% reliable) |
| RSI < 20 (extreme) | Capitulation event | Extended consolidation then rally | 10-30% | High but slow (75%+ rate, 3-8 week recovery) |

**Real-world February 2026 data point:** Bitcoin's 14-day RSI dropped below 30 for only the third time
in its entire history in Feb 2026, following a >50% correction from October 2025 highs. Historical
precedent from the prior two occurrences:
- **Jan 2015 (RSI ~28, price ~$200):** ~8 months of consolidation before sustained rally
- **Dec 2018 (RSI <30, price ~$3,500):** ~3 months of accumulation before upward move

**Critical lesson:** RSI < 30 in crypto does NOT necessarily mean an imminent sharp bounce. It often
signals extended consolidation. The strategy must account for this with patience in position building
and wide stop-losses, or avoid holding through the consolidation at all.

#### Mean Reversion Win Rates by RSI Threshold

| RSI Threshold | Win Rate (range-bound market) | Win Rate (downtrend) | Win Rate (uptrend) |
|---|---|---|---|
| < 45 | 45-55% | 30-40% | 60-70% |
| < 35 | 55-65% | 35-45% | 65-75% |
| < 30 | 60-68% | 40-50% | 70-80% |
| < 20 | 70-80% | 50-60% | 80-90% |

**Key observation:** RSI mean reversion is fundamentally a market-regime-dependent strategy. In a
confirmed downtrend (price below 200-day SMA, declining MACD), even extreme RSI oversold readings
produce only weak bounces that often fail. The VIX index feeds already in this system can help
detect regime: VIX > 25 signals fear/downtrend and reduces mean-reversion reliability.

---

### 4. Risk-Tiered Parameters

All parameters are calibrated for the existing system architecture: RSI(14), MACD(12-26-9),
BB(20, 2.0 stddev), Binance WebSocket with 200-bar price history, 1s HFT loop, $10,000 paper capital.

#### Complete Parameter Table by Risk Level

| Parameter | LOW | MEDIUM | HIGH | EXTREME |
|---|---|---|---|---|
| **RSI oversold entry threshold** | < 28 | < 32 | < 38 | < 45 |
| **RSI overbought exit threshold** | > 72 | > 68 | > 62 | > 55 |
| **MACD histogram min value (buy)** | > +0.0010 | > +0.0005 | > +0.0001 | any positive |
| **MACD histogram min value (sell)** | < -0.0010 | < -0.0005 | < -0.0001 | any negative |
| **BB lower band tolerance % (buy)** | within 0.5% | within 1.0% | within 1.5% | within 2.5% |
| **BB upper band tolerance % (sell)** | within 0.5% | within 1.0% | within 1.5% | within 2.5% |
| **Min confidence threshold** | 0.65 | 0.55 | 0.45 | 0.30 |
| **Position size (% of cash per trade)** | 5% | 10% | 20% | 35% |
| **Max position concentration (% of portfolio)** | 15% | 25% | 40% | 60% |
| **Stop-loss % from avg cost** | 1.5% | 2.5% | 4.0% | 7.0% |
| **Min profit % before selling (after fees)** | 0.5% | 0.35% | 0.20% | 0.10% |
| **HFT loop interval (seconds)** | 5.0 | 1.0 | 0.5 | 0.25 |
| **News sentiment nudge threshold** | ±0.5 | ±0.3 | ±0.2 | ±0.1 |
| **Sentiment override buy dip** | No | No | Partial | Yes |
| **Max open positions simultaneously** | 2 | 4 | 6 | No limit |
| **Sell on partial profit target** | Yes | Yes | No | No |

#### Current System Config vs. Recommended Levels

The current `config.py` maps to approximately **MEDIUM** risk level:

| Config Field | Current Value | LOW equiv | MEDIUM equiv | HIGH equiv | EXTREME equiv |
|---|---|---|---|---|---|
| `trade_fraction` | 0.10 (10%) | 5% | 10% | 20% | 35% |
| `max_position_fraction` | 0.30 (30%) | 15% | 25% | 40% | 60% |
| `stop_loss_pct` | 0.025 (2.5%) | 1.5% | 2.5% | 4.0% | 7.0% |
| `min_profit_to_sell` | 0.0035 (0.35%) | 0.5% | 0.35% | 0.20% | 0.10% |
| `hft_interval` | 1.0s | 5.0s | 1.0s | 0.5s | 0.25s |
| `_CONFIDENCE_THRESHOLD` (engine.py) | 0.45 | 0.65 | 0.55 | 0.45 | 0.30 |

**Conclusion:** The current system is miscalibrated — the confidence threshold (0.45) matches HIGH risk
while position sizing (10%) matches MEDIUM. These should be aligned. Recommend either:
- Full MEDIUM: raise confidence threshold to 0.55
- Full HIGH: raise trade_fraction to 0.20 and lower stop_loss tolerance to 4%

---

### 5. Sentiment Weighting by Risk Level

News sentiment adds an orthogonal signal to price-based indicators. However, its reliability varies
dramatically with risk appetite and market conditions.

#### Sentiment Influence by Risk Level

| Risk Level | Nudge Threshold | Max Confidence from Sentiment | Override Technical? | Buy Dip Despite Negative Sentiment? |
|---|---|---|---|---|
| LOW | ±0.5 | 20% (weak nudge only) | Never | Never |
| MEDIUM | ±0.3 | 35% (moderate nudge) | Only when technical is neutral hold | No |
| HIGH | ±0.2 | 50% (strong nudge) | Can flip weak holds to trades | Rarely, only if RSI < 30 |
| EXTREME | ±0.1 | 70% (near-primary signal) | Yes, can override weak technicals | Yes — treats panic as buying opportunity |

#### Sentiment Override Logic for EXTREME Risk

At extreme risk, the rationale is contrarian: negative sentiment during technically oversold conditions
is treated as a *buy signal amplifier*, not a suppressor. The logic:

1. If RSI < 30 AND sentiment < -0.4 (panic selling): **increase buy confidence** rather than decrease it.
   Rationale: retail panic selling into technical oversold = forced liquidation bottom.
2. If RSI > 65 AND sentiment > 0.5 (euphoria): **reduce sell confidence** — wait for MACD turn.
   Rationale: euphoric news often front-runs further price gains; don't sell prematurely.
3. If sentiment < -0.6 AND no technical signal: **initiate small speculative buy** (position_size * 0.5).
   Rationale: extreme fear at market bottoms is the highest-conviction dip-buy entry.

**Warning:** This logic should only activate for assets with sufficient liquidity (BTC, ETH, SOL) where
the news event is systemic (macro fear) rather than asset-specific (exploit, hack, regulatory action).
Asset-specific negative news should always suppress buys regardless of risk level.

#### Sentiment Decay and Staleness

Crypto sentiment has a very short half-life. A headline from 4+ hours ago has diminishing relevance.
Recommended decay schedule:
- 0-30 minutes: 100% weight
- 30-60 minutes: 75% weight
- 1-4 hours: 40% weight
- 4+ hours: 10% weight (essentially noise)

The current system's 60-second news poll interval is appropriate given the LLM cost constraint.

---

### 6. Known Pitfalls and Essential Guard Rails

Even at extreme risk, certain failure modes are catastrophic and must be guarded against.

#### Top Failure Modes in Aggressive Crypto Strategies

| Failure Mode | Description | Frequency | Typical Loss | Guard Rail |
|---|---|---|---|---|
| **Whipsaw / signal churn** | Price oscillates rapidly; system buys and sells repeatedly without directional move; fees accumulate | Very common in low-volatility periods | 0.2% per churn cycle, can stack to 5-10% daily | Enforce minimum hold time (30s-5min); require 2+ confirming bars before signal |
| **Liquidity trap (meme coins)** | Position built in PEPE/WIF/SHIB during thin hours; large sell order causes 2-5% slippage | Common for positions >0.5% of daily volume | 2-15% instant loss | Cap position size in low-volume assets; detect order-book depth |
| **Stop-loss hunting** | Makers temporarily push price below common stop levels to trigger liquidations, then reverse | Common on 5-30min timeframes | Triggers stop, then misses rally | Use percentage stops from avg_cost (current impl), NOT fixed price levels |
| **Trend-riding mean reversion** | Buying "oversold" in a genuine sustained downtrend; RSI can stay < 30 for weeks | Moderate | 15-40% loss before recovery | Only buy mean reversion if MACD histogram is showing deceleration (histogram rising even if negative) |
| **Sentiment echo chamber** | News sentiment is already priced in; acting on 1-hour-old "positive" news buys the top | Common | 1-5% per instance | Apply time decay to sentiment scores (see section 5) |
| **Concentration risk** | All capital deployed in one asset when a black swan hits | Rare but catastrophic | 30-80% of portfolio | Hard cap: no single asset > 60% even at EXTREME risk |
| **Fee erosion** | High frequency + small positions = fees exceed gains | Very common at 0.25s intervals | 0.2% per round-trip × 1000s of trades | Enforce min_profit_to_sell > 2× fee rate; use BNB for fee discount |
| **Cascade liquidation** | Leveraged positions trigger margin calls; price drops further; loop repeats | Rare on paper trading, real on live | Total position wipe | For paper trading: never model leverage; for live: max 2× leverage |
| **False BB squeeze breakout** | Price breaches upper BB but on low volume; snaps back | Common (40-50% of band touches are false) | 1-3% per false breakout | Require volume above 20-bar average to validate breakout signals |
| **Overfit parameter drift** | Parameters optimized for one market regime fail in another | Inevitable over 3-6 month cycles | Strategy loses edge | Backtest quarterly; monitor win rate rolling 30-day; reset if <45% |

#### Essential Guard Rails (Non-Negotiable at Any Risk Level)

These rules must hold regardless of risk tier configuration:

1. **Never trade on a position that already hit its stop-loss.** The existing pre-signal stop-loss check
   in `engine.py` (lines 176-190) is correct and essential.

2. **Minimum 2 confirming bars for any signal.** A single bar breaching a band or RSI threshold is
   insufficient. The signal should persist for at least 2 evaluation cycles before execution.
   *Currently not implemented — recommended addition.*

3. **Daily loss limit: 5% of portfolio.** If total unrealized + realized PnL for the day exceeds -5%,
   halt all new buys until next UTC midnight. Sells (stop-losses) still execute.

4. **Asset-specific negative news hard stop.** If an asset receives sentiment score < -0.7 AND the news
   is asset-specific (hack, SEC action, de-peg), suppress all buys for that asset for 60 minutes
   regardless of technical signals.

5. **No new buys when VIX > 35.** Extreme market fear index readings indicate systemic risk. Mean
   reversion during systemic events has historically very poor outcomes. The VIX feed already exists
   in this system via `IndicesFeed`.

6. **Slippage model for paper trading.** Paper trades should model 0.05-0.15% slippage on fills
   (depending on asset liquidity: BTC=0.05%, mid-caps=0.10%, meme coins=0.15-0.25%). Without this,
   paper trading results are systematically overoptimistic.

7. **Fee compounding awareness.** At a 1-second HFT loop with 15 symbols, the system can generate
   hundreds of signals per hour. Each round-trip costs ~0.2%. Even at 65% win rate with 0.3% avg
   target profit, excessive churn destroys returns. The `min_profit_to_sell` gate is critical.

---

### 7. Symbol-Specific Strategy Recommendations

Different assets on the tracked symbols list benefit from different parameter tuning:

| Symbol Group | Examples | Best Strategy | RSI Period | BB StdDev | Notes |
|---|---|---|---|---|---|
| Large-cap anchors | BTC, ETH | Mean reversion + momentum | 14 | 2.0 | Highest liquidity; tightest spreads; most indicator-reliable |
| Mid-cap momentum | SOL, BNB, XRP, ADA, AVAX | Momentum/breakout | 14 | 2.0 | Good volume; reliable breakouts; trend-follow works well |
| Volatile mid-cap | LINK, NEAR, SUI, INJ | Mixed; tighter stops | 10-14 | 2.0-2.5 | Higher beta; wider swings; reduce position size 25% vs BTC |
| Meme/speculative | PEPE, WIF, SHIB | Extreme caution; sentiment-primary | 7-10 | 2.5 | News-driven; technicals unreliable; trade only on strong sentiment + RSI confluence |

**Meme coin specific notes:** PEPE, WIF, and SHIB can move 20-50% in hours on a single viral tweet.
Technical indicators lag badly on these. The sentiment feed (AMBCrypto, CoinGape) is more predictive
than RSI/MACD for meme coins. Recommended approach: use technical indicators as a veto (don't buy
memes when RSI > 70) but use sentiment as the primary entry trigger.

---

### 8. Implementation Recommendations for This System

Based on the research and the existing codebase analysis, these are prioritized improvements:

#### High Priority

1. **Raise confidence threshold to 0.55** (from current 0.45 in `engine.py`). The threshold
   currently mismatches position sizing (MEDIUM risk but LOW confidence bar). This causes too many
   low-conviction trades.

2. **Add minimum hold time guard.** Track last trade timestamp per symbol. Do not re-enter a
   position within 60 seconds of exiting it. Prevents churn on noisy signals.

3. **Add VIX-based trade suppression.** When `IndicesFeed` returns VIX > 35, suppress all new
   BUY orders. This uses the already-integrated VIX data feed.

4. **Model slippage in paper trades.** Add a slippage factor to `place_paper_trade()` based on
   symbol tier: BTC/ETH=0.05%, others=0.10%, meme coins=0.20%. Prevents overoptimistic paper results.

5. **Sentiment time decay.** Apply exponential decay to sentiment scores based on article age.
   A 3-hour-old headline should contribute 30% of its original weight.

#### Medium Priority

6. **MACD histogram expansion filter.** In `rule_based_signal()`, check that `macd_hist > prior_macd_hist`
   (histogram is *expanding*, not just positive). This eliminates stale momentum signals.

7. **Symbol-tier position sizing.** Halve `trade_fraction` for PEPE/WIF/SHIB vs BTC/ETH.
   Meme coin volatility requires smaller position sizing even at the same risk level.

8. **Daily loss circuit breaker.** Track cumulative daily realized PnL. Halt new buys if daily
   losses exceed 5% of portfolio start-of-day value.

#### Low Priority / Future

9. **Multi-timeframe signal confirmation.** Aggregate 1-minute, 5-minute, and 15-minute RSI.
   Only trade when at least 2 of 3 timeframes agree on direction.

10. **BB width squeeze detector.** Compute BB width as (upper-lower)/mid. Flag squeeze when
    width falls to 20-bar minimum. Boost confidence score by +0.15 when squeeze precedes signal.

---

*Research compiled by GEMINI sub-agent, 2026-02-26.*
*Sources consulted:*
- *[MACD and Bollinger Bands Strategy (78% Win Rate) — QuantifiedStrategies](https://www.quantifiedstrategies.com/macd-and-bollinger-bands-strategy/)*
- *[RSI Trading Strategy (91% Win Rate) — QuantifiedStrategies](https://www.quantifiedstrategies.com/rsi-trading-strategy/)*
- *[How to Use Technical Indicators for Crypto Trading 2026 — Gate.io Wiki](https://web3.gate.com/crypto-wiki/article/how-to-use-technical-indicators-macd-rsi-and-bollinger-bands-for-crypto-trading-in-2026-20260204)*
- *[5 Crypto Scalping Strategies — FXOpen](https://fxopen.com/blog/en/5-crypto-scalping-strategies/)*
- *[Bollinger Band Squeeze Breakout Strategy — MindMathMoney](https://www.mindmathmoney.com/articles/the-bollinger-band-squeeze-trading-strategy-a-comprehensive-guide)*
- *[Bollinger Bands and MACD Entry Rules — LuxAlgo](https://www.luxalgo.com/blog/bollinger-bands-and-macd-entry-rules-explained/)*
- *[Top Crypto Scalping Strategies — BingX](https://bingx.com/en/learn/article/top-crypto-scalping-strategies-for-short-term-price-movements-trading)*
- *[Bitcoin RSI Below 30 for Third Time Ever — CoinDesk, Feb 2026](https://www.coindesk.com/markets/2026/02/19/bitcoin-s-14-day-rsi-falls-below-30-for-third-time-ever-months-of-consolidation-likely)*
- *[Crypto Risk Management 2025 — Changelly](https://changelly.com/blog/risk-management-in-crypto-trading/)*
- *[MACD Indicator in Crypto Trading — Zignaly](https://zignaly.com/crypto-trading/indicators/macd-crypto-indicator)*
- *[Linda Raschke MACD Settings 3-10-16 — MindMathMoney](https://www.mindmathmoney.com/articles/linda-raschke-trading-strategy-macd-indicator-settings-for-trading-stocks-forex-and-crypto)*
- *[Crypto Liquidity Crisis Oct 2025 — FTI Consulting](https://www.fticonsulting.com/insights/articles/crypto-crash-october-2025-leverage-met-liquidity)*
- *[Crypto Slippage Guide 2025 — FinanceFeeds](https://financefeeds.com/crypto-slippage-guide-2025-causes-and-effects/)*
