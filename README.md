# Crypto Trading Platform (AI-Assisted)

Async paper-trading platform for crypto with:
- Binance market data streaming
- News ingestion + LLM sentiment analysis (`gemini`, `openai`, or `claude`)
- Portfolio/trading engine
- Console dashboard
- Optional global indices and ASX feeds

## Requirements

- Python 3.11+ (project currently runs on newer versions too)
- Internet access (Binance, RSS feeds, LLM APIs)
- One LLM API key:
  - Google Gemini
  - OpenAI
  - Anthropic Claude

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Create local key files (recommended for this project):

```bash
printf '%s' 'YOUR_GEMINI_KEY' > gemini.key
printf '%s' 'YOUR_OPENAI_KEY' > openai.key
printf '%s' 'YOUR_CLAUDE_KEY' > claude.key
chmod 600 *.key
```

Run:

```bash
python src/main.py
```

## LLM Provider Selection

Set the provider in `config.py` (`AppConfig.llm_provider`):
- `gemini`
- `openai`
- `claude`

The code reads API keys from files in the project root:
- `gemini.key`
- `openai.key`
- `claude.key`

## Safer Local Secrets Workflow

This repo tracks `.env.example` as a template only.

1. Copy `.env.example` to `.env`
2. Fill in your real keys locally
3. Write the key files used by the app from `.env`

Example:

```bash
set -a
. ./.env
set +a
printf '%s' "$GEMINI_API_KEY" > gemini.key
printf '%s' "$OPENAI_API_KEY" > openai.key
printf '%s' "$CLAUDE_API_KEY" > claude.key
chmod 600 gemini.key openai.key claude.key
```

## Tests

```bash
pytest
```

## Notes

- `trading.log` is local runtime output and is gitignored.
- API key files (`*.key`) are gitignored.
- The repo may contain local experiment docs (`CLAUDE.md`, `GEMINI.md`, `CODE.md`) that describe workflows.
