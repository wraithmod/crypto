# CODEX Sub-Agent

## Role
Programming, code organisation, auditing, and system design for the crypto trading platform at `/home/wraith/making`.

## Capabilities
- Python code generation, refactoring, and review
- Async patterns, dataclass design, module architecture
- Test writing (pytest, mocks)
- Performance analysis and optimisation

## Completed Design: News Sentiment Overhaul (Feb 2026)

### SentimentResult dataclass
Added: `affected_coins: list[str]`, `impact_magnitude: str`, `source_weight: float`
All new fields have defaults — no breaking changes to existing callsites.

### LLM Prompt
Enhanced to request `affected_coins` and `impact_magnitude` in JSON response.
System prompt reframed as "crypto market impact analyst".
max_tokens reduced to 300 (sufficient for new schema).

### Source Credibility
Module-level `_SOURCE_WEIGHTS` dict keyed by registered domain.
`_source_weight_from_url(url)` extracts domain via `urllib.parse.urlparse`.
Assigned to `item.sentiment.source_weight` after `analyze_sentiment` returns.

### Per-Symbol Sentiment
`get_symbol_sentiment(symbol)` — filters by `affected_coins` OR title text.
Weights: source_weight × magnitude_weight × time_decay.
Returns `None` if no relevant items (caller falls back to `get_avg_sentiment()`).
`_extract_coin_ticker("BTCUSDT") -> "BTC"` strips known quote currencies.

### Concurrent Scoring
`fetch_and_analyze` uses `asyncio.gather(*[analyze_sentiment(item) for ...], return_exceptions=True)`.
Scores top 8 headlines per cycle (was 3, sequential).

### Time Decay
`_time_decay_factor(published)`: age<1h=1.0, 1h-6h=0.5, >6h=0.1.
Uses `email.utils.parsedate_to_datetime` (stdlib) with `dateutil` fallback.
Applied in both `get_avg_sentiment()` and `get_symbol_sentiment()`.

### get_avg_sentiment update
Now weighted by `source_weight × time_decay` instead of simple average.

### engine.py change
`evaluate_symbol` calls `news_feed.get_symbol_sentiment(symbol)` first,
falls back to `news_feed.get_avg_sentiment()` if None returned.

## Coding Conventions
- Python 3.13, full async (asyncio), no sync I/O on event loop
- dataclasses for data objects, type annotations throughout
- All log output to `trading.log` (never stdout)
- Keep changes minimal and focused — no unnecessary refactors

---

## Risk Profile Architecture (Feb 2026)

### Overview

A 4-tier risk profile system controlled by a CLI flag `--risk low|medium|high|extreme`.
Each tier encodes a coherent set of signal-sensitivity, position-sizing, and risk-management
parameters. The profile is instantiated once at startup and threaded through `Predictor` and
`TradeEngine` so every decision adapts to the chosen risk appetite without touching
`AppConfig` or the exchange connectors.

---

### 1. New file: `src/trading/risk.py`

```python
"""Risk profile presets for the trading platform.

A RiskProfile encodes all tunable thresholds for one risk tier.  The four
presets (LOW, MEDIUM, HIGH, EXTREME) are module-level constants; callers
should import and use them directly rather than constructing ad-hoc instances.

Usage
-----
    from src.trading.risk import RiskProfile, PROFILES
    profile = PROFILES["medium"]   # or LOW, MEDIUM, HIGH, EXTREME
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskProfile:
    # ------------------------------------------------------------------
    # RSI thresholds — controls buy/sell trigger sensitivity
    # ------------------------------------------------------------------
    rsi_oversold: float          # Strong oversold BUY threshold (default 35)
    rsi_weak_oversold: float     # Mild oversold BUY threshold   (default 45)
    rsi_overbought: float        # Strong overbought SELL threshold (default 65)
    rsi_weak_overbought: float   # Mild overbought SELL threshold   (default 55)

    # ------------------------------------------------------------------
    # Bollinger Band proximity multipliers
    # ------------------------------------------------------------------
    # price_near_lower_bb  = price <= bb_lower * bb_lower_tolerance
    # price_near_upper_bb  = price >= bb_upper * bb_upper_tolerance
    bb_lower_tolerance: float    # > 1.0 widens the "near lower band" zone
    bb_upper_tolerance: float    # < 1.0 widens the "near upper band" zone

    # ------------------------------------------------------------------
    # Trade execution gates
    # ------------------------------------------------------------------
    confidence_threshold: float  # Minimum signal confidence before a trade fires
    sentiment_nudge_threshold: float  # |sentiment| must exceed this to nudge a hold
    sentiment_weight: float      # Multiplier applied to sentiment-derived confidence

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------
    trade_fraction: float        # Fraction of available cash deployed per BUY
    max_position_fraction: float # Maximum single-symbol exposure as % of portfolio

    # ------------------------------------------------------------------
    # Risk management
    # ------------------------------------------------------------------
    stop_loss_pct: float         # Liquidate immediately if position is down this far
    min_profit_to_sell: float    # Minimum gain above avg_cost to permit a signal-SELL

    # ------------------------------------------------------------------
    # HFT loop timing
    # ------------------------------------------------------------------
    hft_interval: float          # Seconds between evaluation cycles


# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------

LOW = RiskProfile(
    # RSI — tight bands; only act on strong conviction signals
    rsi_oversold=28.0,
    rsi_weak_oversold=38.0,
    rsi_overbought=72.0,
    rsi_weak_overbought=62.0,
    # BB — must be very close to the band edge to trigger
    bb_lower_tolerance=1.008,
    bb_upper_tolerance=0.992,
    # Execution — high bar; needs at least 2 of 4 conditions
    confidence_threshold=0.60,
    sentiment_nudge_threshold=0.50,
    sentiment_weight=0.6,
    # Position sizing — small bites, limited exposure
    trade_fraction=0.05,
    max_position_fraction=0.15,
    # Risk management — tight stop, meaningful profit before exit
    stop_loss_pct=0.015,
    min_profit_to_sell=0.008,
    # Timing — conservative cycle; fewer trades, lower fees
    hft_interval=5.0,
)

MEDIUM = RiskProfile(
    # RSI — current production defaults
    rsi_oversold=35.0,
    rsi_weak_oversold=45.0,
    rsi_overbought=65.0,
    rsi_weak_overbought=55.0,
    # BB — current production defaults
    bb_lower_tolerance=1.015,
    bb_upper_tolerance=0.985,
    # Execution — single condition sufficient (0.25 confidence)
    confidence_threshold=0.45,
    sentiment_nudge_threshold=0.30,
    sentiment_weight=1.0,
    # Position sizing — current production defaults
    trade_fraction=0.10,
    max_position_fraction=0.30,
    # Risk management — current production defaults
    stop_loss_pct=0.025,
    min_profit_to_sell=0.0035,
    # Timing — 1-second cycle
    hft_interval=1.0,
)

HIGH = RiskProfile(
    # RSI — relaxed bands; fires earlier in a move
    rsi_oversold=40.0,
    rsi_weak_oversold=50.0,
    rsi_overbought=60.0,
    rsi_weak_overbought=50.0,
    # BB — trigger further from the band edge
    bb_lower_tolerance=1.025,
    bb_upper_tolerance=0.975,
    # Execution — lower bar; acts on weak signals
    confidence_threshold=0.30,
    sentiment_nudge_threshold=0.20,
    sentiment_weight=1.4,
    # Position sizing — larger bites, higher concentration
    trade_fraction=0.20,
    max_position_fraction=0.50,
    # Risk management — wider stop, quicker profit take
    stop_loss_pct=0.050,
    min_profit_to_sell=0.002,
    # Timing — 0.5-second cycle
    hft_interval=0.5,
)

EXTREME = RiskProfile(
    # RSI — very wide; captures almost any momentum reading
    rsi_oversold=48.0,
    rsi_weak_oversold=54.0,
    rsi_overbought=52.0,
    rsi_weak_overbought=46.0,
    # BB — wide proximity zone; nearly always near a band
    bb_lower_tolerance=1.040,
    bb_upper_tolerance=0.960,
    # Execution — fires on minimal signal; momentum trading mode
    confidence_threshold=0.20,
    sentiment_nudge_threshold=0.10,
    sentiment_weight=2.0,
    # Position sizing — aggressive; near-full concentration allowed
    trade_fraction=0.35,
    max_position_fraction=0.75,
    # Risk management — very wide stop, takes profits quickly
    stop_loss_pct=0.080,
    min_profit_to_sell=0.001,
    # Timing — 0.2-second cycle (true HFT pace)
    hft_interval=0.2,
)

PROFILES: dict[str, RiskProfile] = {
    "low": LOW,
    "medium": MEDIUM,
    "high": HIGH,
    "extreme": EXTREME,
}
```

---

### 2. Parameter rationale table

| Field | LOW | MEDIUM | HIGH | EXTREME | Notes |
|-------|-----|--------|------|---------|-------|
| `rsi_oversold` | 28 | 35 | 40 | 48 | Lower = harder to trigger BUY |
| `rsi_weak_oversold` | 38 | 45 | 50 | 54 | |
| `rsi_overbought` | 72 | 65 | 60 | 52 | Higher = harder to trigger SELL |
| `rsi_weak_overbought` | 62 | 55 | 50 | 46 | |
| `bb_lower_tolerance` | 1.008 | 1.015 | 1.025 | 1.040 | Wider = catches price earlier |
| `bb_upper_tolerance` | 0.992 | 0.985 | 0.975 | 0.960 | |
| `confidence_threshold` | 0.60 | 0.45 | 0.30 | 0.20 | Minimum to execute a trade |
| `sentiment_nudge_threshold` | 0.50 | 0.30 | 0.20 | 0.10 | Smaller = sentiment nudges more |
| `sentiment_weight` | 0.6x | 1.0x | 1.4x | 2.0x | Scales nudge confidence |
| `trade_fraction` | 5% | 10% | 20% | 35% | Cash deployed per trade |
| `max_position_fraction` | 15% | 30% | 50% | 75% | Max single-symbol exposure |
| `stop_loss_pct` | 1.5% | 2.5% | 5.0% | 8.0% | Hard floor before liquidation |
| `min_profit_to_sell` | 0.8% | 0.35% | 0.2% | 0.1% | Min gain to permit signal-SELL |
| `hft_interval` | 5.0s | 1.0s | 0.5s | 0.2s | Seconds between eval cycles |

**RSI note for EXTREME:** `rsi_overbought` (52) < `rsi_weak_oversold` (54) — intentional.
In extreme mode the engine is essentially always in a momentum-following state: any RSI
above 52 looks overbought, below 48 looks oversold. This maximises signal frequency at
the expense of accuracy.

---

### 3. Changes to `src/prediction/predictor.py`

#### 3a. Signature change for `rule_based_signal`

Add an optional `risk: RiskProfile | None = None` parameter. When `None`, fall back to
the MEDIUM preset so all existing call-sites remain valid without modification.

```python
from src.trading.risk import RiskProfile, MEDIUM as _DEFAULT_RISK

def rule_based_signal(self, ind: Indicators, risk: RiskProfile | None = None) -> Signal:
    risk = risk or _DEFAULT_RISK
    total_conditions = 4

    # BUY scoring — use risk thresholds instead of Indicators properties
    buy_triggers: list[str] = []
    if ind.rsi < risk.rsi_oversold:
        buy_triggers.append(f"RSI strongly oversold ({ind.rsi:.1f}<{risk.rsi_oversold})")
    elif ind.rsi < risk.rsi_weak_oversold:
        buy_triggers.append(f"RSI mildly oversold ({ind.rsi:.1f}<{risk.rsi_weak_oversold})")
    if ind.macd > ind.macd_signal and ind.macd_hist > 0:
        buy_triggers.append(f"MACD bullish+rising hist ({ind.macd:.4f}>{ind.macd_signal:.4f})")
    if ind.price <= ind.bb_lower * risk.bb_lower_tolerance:
        buy_triggers.append(f"Price near lower BB ({ind.price:.4f}≤{ind.bb_lower * risk.bb_lower_tolerance:.4f})")
    if ind.price > ind.bb_mid and ind.macd > ind.macd_signal:
        buy_triggers.append("Price above SMA+MACD bullish (trend confirm)")

    # SELL scoring
    sell_triggers: list[str] = []
    if ind.rsi > risk.rsi_overbought:
        sell_triggers.append(f"RSI strongly overbought ({ind.rsi:.1f}>{risk.rsi_overbought})")
    elif ind.rsi > risk.rsi_weak_overbought:
        sell_triggers.append(f"RSI mildly overbought ({ind.rsi:.1f}>{risk.rsi_weak_overbought})")
    if ind.macd <= ind.macd_signal and ind.macd_hist <= 0:
        sell_triggers.append(f"MACD bearish+falling hist ({ind.macd:.4f}<{ind.macd_signal:.4f})")
    if ind.price >= ind.bb_upper * risk.bb_upper_tolerance:
        sell_triggers.append(f"Price near upper BB ({ind.price:.4f}≥{ind.bb_upper * risk.bb_upper_tolerance:.4f})")
    if ind.price < ind.bb_mid and ind.macd <= ind.macd_signal:
        sell_triggers.append("Price below SMA+MACD bearish (trend confirm)")

    # ... rest of scoring logic unchanged ...
```

#### 3b. Signature change for `get_signal`

Pass `risk` through to `rule_based_signal`, and parameterise the sentiment nudge
threshold and weight using the profile:

```python
async def get_signal(
    self,
    symbol: str,
    prices: list[float],
    news_sentiment: float = 0.0,
    risk: RiskProfile | None = None,
) -> Signal:
    risk = risk or _DEFAULT_RISK
    ...
    signal = self.rule_based_signal(ind, risk)

    # Sentiment nudge uses risk thresholds
    if signal.direction == "hold":
        if news_sentiment > risk.sentiment_nudge_threshold:
            raw_confidence = (news_sentiment - risk.sentiment_nudge_threshold) / \
                             (1.0 - risk.sentiment_nudge_threshold)
            nudge_confidence = min(raw_confidence, 1.0) * 0.5 * risk.sentiment_weight
            signal = Signal(direction="buy", confidence=nudge_confidence, ...)
        elif news_sentiment < -risk.sentiment_nudge_threshold:
            raw_confidence = (abs(news_sentiment) - risk.sentiment_nudge_threshold) / \
                             (1.0 - risk.sentiment_nudge_threshold)
            nudge_confidence = min(raw_confidence, 1.0) * 0.5 * risk.sentiment_weight
            signal = Signal(direction="sell", confidence=nudge_confidence, ...)
```

#### 3c. Indicators boolean properties

The four RSI boolean properties on `Indicators` (`rsi_oversold`, `rsi_weak_oversold`,
`rsi_overbought`, `rsi_weak_overbought`) and the two BB proximity properties
(`price_near_lower_bb`, `price_near_upper_bb`) encode hardcoded thresholds that are now
superseded by `RiskProfile`. They should be kept as-is on the dataclass for backwards
compatibility (dashboard display, tests), but `rule_based_signal` must stop using them
and instead compare `ind.rsi` and `ind.price` directly against profile values.

---

### 4. Changes to `src/trading/engine.py`

#### 4a. Constructor signature

```python
from src.trading.risk import RiskProfile, MEDIUM as _DEFAULT_RISK

class TradeEngine:
    def __init__(
        self,
        portfolio: Portfolio,
        predictor: Predictor,
        config: AppConfig = default_config,
        risk: RiskProfile = _DEFAULT_RISK,
    ) -> None:
        self._portfolio = portfolio
        self._predictor = predictor
        self._config = config
        self._risk = risk
```

#### 4b. Replace module-level `_CONFIDENCE_THRESHOLD` constant

The module constant `_CONFIDENCE_THRESHOLD = 0.45` is replaced by `self._risk.confidence_threshold`
throughout `evaluate_symbol`. The constant can be removed or kept as a fallback comment.

#### 4c. `evaluate_symbol` — confidence gate

```python
# Gate on confidence  (was: _CONFIDENCE_THRESHOLD)
if signal.confidence < self._risk.confidence_threshold:
    ...
    return
```

#### 4d. `evaluate_symbol` — stop-loss check

```python
# was: self._config.stop_loss_pct
if pct_change <= -self._risk.stop_loss_pct:
```

Both the pre-signal stop-loss block and the in-signal stop-loss block use
`self._risk.stop_loss_pct` instead of `self._config.stop_loss_pct`.

#### 4e. `evaluate_symbol` — position sizing

```python
# was: self._config.max_position_fraction, self._config.trade_fraction
max_position_value = self._risk.max_position_fraction * total_value
trade_cash = self._risk.trade_fraction * cash
```

#### 4f. `evaluate_symbol` — min profit gate

```python
# was: self._config.min_profit_to_sell
min_required = round_trip_fee + self._risk.min_profit_to_sell
```

#### 4g. `run_hft_loop` — interval

```python
# was: self._config.hft_interval
await asyncio.sleep(self._risk.hft_interval)
```

The log message should also emit the risk-derived interval:

```python
logger.info(
    "HFT loop starting for symbols=%s interval=%.2fs (risk=%s)",
    symbols,
    self._risk.hft_interval,
    type(self._risk).__name__,
)
```

#### 4h. `get_active_strategy` — include risk label

```python
def get_active_strategy(self) -> str:
    return f"RSI+MACD+BB with News Sentiment Blend [{self._risk_label}]"
```

Where `self._risk_label` is stored in `__init__` from the profile name (passed as an
extra string arg, or derived from a `PROFILES` reverse-lookup). Simplest approach: the
caller passes the label string alongside the profile.

Alternative — add `name: str` field to `RiskProfile` frozen dataclass so the profile is
self-describing. This is the preferred approach as it keeps the engine's constructor clean:

```python
@dataclass(frozen=True)
class RiskProfile:
    name: str   # "low" | "medium" | "high" | "extreme"
    ...
```

Then `get_active_strategy` returns:
```python
f"RSI+MACD+BB + News Sentiment [{self._risk.name.upper()}]"
```

#### 4i. `get_signal` call — forward risk profile

```python
signal = await self._predictor.get_signal(
    symbol, price_history, news_sentiment, risk=self._risk
)
```

---

### 5. Changes to `src/main.py`

#### 5a. Argument parsing — add `--risk`

```python
from src.trading.risk import PROFILES, MEDIUM as _DEFAULT_RISK

parser.add_argument(
    "--risk",
    choices=["low", "medium", "high", "extreme"],
    default="medium",
    metavar="LEVEL",
    help="Risk profile: low | medium | high | extreme (default: medium)",
)
```

#### 5b. Load profile and pass to TradeEngine

```python
args = parser.parse_args()
config.initial_cash = args.cash
risk_profile = PROFILES[args.risk]
```

Inside `main()`, the profile must be available. The cleanest pattern is to store it on
`config` (add `risk_profile: RiskProfile = MEDIUM` to `AppConfig`) OR pass it as a
function argument. Given that `AppConfig` already carries runtime overrides (`initial_cash`),
storing it on config is consistent:

**Option A — store on config (preferred for simplicity):**

```python
# In AppConfig (config.py):
from src.trading.risk import RiskProfile, MEDIUM
risk_profile: RiskProfile = field(default_factory=lambda: MEDIUM)

# In __main__ block:
config.risk_profile = PROFILES[args.risk]

# In main():
engine = TradeEngine(
    portfolio=portfolio,
    predictor=predictor,
    config=config,
    risk=config.risk_profile,
)
```

**Option B — pass directly (avoids coupling config to risk module):**

```python
# Pass risk_profile as a parameter into main()
async def main(risk_profile: RiskProfile) -> None:
    ...
    engine = TradeEngine(..., risk=risk_profile)

# In __main__ block:
asyncio.run(main(risk_profile=PROFILES[args.risk]))
```

Option B is preferred — it avoids a circular import risk (config.py importing from
`src/trading/risk.py`), keeps `AppConfig` free of trading-logic types, and makes the
data flow explicit. `AppConfig` retains `stop_loss_pct`, `trade_fraction`, etc. as
documentation / fallback defaults; the `TradeEngine` uses `RiskProfile` values at
runtime.

#### 5c. Log the chosen profile at startup

```python
logger.info(
    "Risk profile: %s | hft_interval=%.2fs | confidence_threshold=%.2f | "
    "stop_loss=%.1f%% | trade_fraction=%.0f%%",
    args.risk.upper(),
    risk_profile.hft_interval,
    risk_profile.confidence_threshold,
    risk_profile.stop_loss_pct * 100,
    risk_profile.trade_fraction * 100,
)
```

---

### 6. Complete parameter values summary

```
RiskProfile fields        LOW        MEDIUM     HIGH       EXTREME
─────────────────────────────────────────────────────────────────
name                      "low"      "medium"   "high"     "extreme"
rsi_oversold              28.0       35.0       40.0       48.0
rsi_weak_oversold         38.0       45.0       50.0       54.0
rsi_overbought            72.0       65.0       60.0       52.0
rsi_weak_overbought       62.0       55.0       50.0       46.0
bb_lower_tolerance        1.008      1.015      1.025      1.040
bb_upper_tolerance        0.992      0.985      0.975      0.960
confidence_threshold      0.60       0.45       0.30       0.20
sentiment_nudge_threshold 0.50       0.30       0.20       0.10
sentiment_weight          0.6        1.0        1.4        2.0
trade_fraction            0.05       0.10       0.20       0.35
max_position_fraction     0.15       0.30       0.50       0.75
stop_loss_pct             0.015      0.025      0.050      0.080
min_profit_to_sell        0.008      0.0035     0.002      0.001
hft_interval (sec)        5.0        1.0        0.5        0.2
```

---

### 7. File change summary

| File | Change |
|------|--------|
| `src/trading/risk.py` | **NEW** — `RiskProfile` dataclass + 4 presets + `PROFILES` dict |
| `src/prediction/predictor.py` | `rule_based_signal(ind, risk=None)` and `get_signal(..., risk=None)` — thresholds from profile |
| `src/trading/engine.py` | Constructor gains `risk: RiskProfile`; `_CONFIDENCE_THRESHOLD` constant removed; 6 call-sites use `self._risk.*` |
| `src/main.py` | `--risk` argparse arg; `main(risk_profile)` signature; `TradeEngine(..., risk=risk_profile)` |
| `config.py` | No change required (Option B above) |
| `tests/` | New unit tests for each preset: assert confidence gate, position sizing, stop-loss at boundary values |

---

### 8. Backwards compatibility

- `rule_based_signal(ind)` (no `risk` arg) continues to work — defaults to MEDIUM.
- `get_signal(symbol, prices, sentiment)` continues to work — defaults to MEDIUM.
- `TradeEngine(portfolio, predictor, config)` continues to work — defaults to MEDIUM.
- Running `python src/main.py` without `--risk` defaults to MEDIUM, identical behaviour to today.
- The four boolean properties on `Indicators` are not removed; they remain for display
  and testing. Only `rule_based_signal` switches away from them.

---

### 9. Testing strategy

Each risk preset should have a dedicated test fixture:

```python
# tests/trading/test_risk_profiles.py

@pytest.mark.parametrize("profile,expected_direction", [
    (LOW,     "hold"),   # RSI=40 does not trigger LOW's oversold threshold of 28
    (MEDIUM,  "buy"),    # RSI=40 triggers MEDIUM's weak_oversold threshold of 45
    (HIGH,    "buy"),    # RSI=40 triggers HIGH's oversold threshold of 40
    (EXTREME, "buy"),    # RSI=40 triggers EXTREME's oversold threshold of 48
])
def test_rsi_40_direction_by_profile(profile, expected_direction):
    ind = make_indicators(rsi=40.0, ...)
    predictor = Predictor()
    signal = predictor.rule_based_signal(ind, risk=profile)
    assert signal.direction == expected_direction
```

Boundary tests for confidence gate, stop-loss trigger, and position sizing should mock
`Portfolio` and `MarketFeed` as existing tests do.
