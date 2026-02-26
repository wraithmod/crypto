# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Mission

This is a crypto financial trading research and automation platform. The goal is a console app that:
- Monitors live crypto market prices with a real-time display
- Ingests recent financial news with economic and political impact analysis
- Makes directional market predictions
- Simulates pseudo-trades to model buy/sell outcomes before execution
- Executes real trades starting from seed capital to grow a portfolio
- Supports high-frequency trading strategies
- Displays a live dashboard: current prices, portfolio holdings, and P&L

## Agent Coordination Model

Claude acts as **coordinator/orchestrator**. Two sub-agents assist:

| Agent | File | Role |
|-------|------|------|
| GEMINI | `GEMINI.md` | Research, market analysis, news ingestion, financial reasoning |
| CODEX | `CODE.md` | Programming, code organization, auditing, system design |

- Delegate token-intensive tasks (large data analysis, bulk code generation) to GEMINI or CODEX
- Parallel agent instances may be spawned for concurrent processing
- Each agent maintains its own `.md` file; update them when their capabilities expand

## API Keys Available

Key files are stored in the project root (do NOT commit these):
- `claude.key` — Anthropic/Claude API
- `gemini.key` — Google Gemini API
- `openai.key` — OpenAI API (CODEX)
- `huggingface.key` — HuggingFace API

Load keys from these files at runtime; never hardcode them.

## Planned App Architecture

```
making/
├── src/
│   ├── main.py              # Entry point, console UI loop
│   ├── market/              # Price feeds, exchange connectors
│   ├── news/                # News ingestion, sentiment analysis
│   ├── prediction/          # ML/LLM-based directional models
│   ├── trading/             # Order logic, pseudo-trade simulator, HFT engine
│   ├── portfolio/           # Holdings tracker, P&L calculation
│   └── agents/              # Sub-agent communication wrappers
├── tests/                   # Mirrors src/ structure
├── CLAUDE.md
├── GEMINI.md
└── CODE.md
```

## Console UI Requirements

The live dashboard must show (refreshing in-place, not scrolling):
1. Current prices for tracked crypto pairs (BTC, ETH, etc.)
2. Current portfolio: asset, quantity, value, avg cost
3. Overall P&L (realized + unrealized)
4. Active strategy / last trade action
5. Recent news headlines with sentiment indicator

## Development Conventions

- Language: Python (primary). Use `asyncio` for concurrent price feeds and HFT loops.
- Use `curses` or `rich` for the console dashboard.
- Exchange connectivity: start with Binance or Kraken REST/WebSocket APIs (or paper trading endpoints).
- Predictions: combine technical indicators (RSI, MACD, Bollinger) with LLM news sentiment via GEMINI.
- Tests: use `pytest`. Mock all external API calls in tests.

## Current State

No source files exist yet. Begin by scaffolding `src/` and `tests/`, then implement market data feeds before adding trading logic.
