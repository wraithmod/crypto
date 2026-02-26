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
