# API Configuration

## Which API
This project uses the **Anthropic API** (Claude) via the official `anthropic` Python SDK.

## Authentication
The API key is read from an environment variable — **never hard-coded in source**. This is standard practice: hard-coded keys get committed to git by accident and leak.

### Setup
1. Get a key from the [Anthropic Console](https://console.anthropic.com/settings/keys).
2. Copy `.env.example` to `.env` and fill in the real key:
   ```
   cp .env.example .env
   ```
3. Either export it in your shell before running scripts:
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-your-real-key"
   ```
   or load `.env` automatically with `python-dotenv` (not included by default, to keep dependencies minimal — add it and a `load_dotenv()` call at the top of a script if you prefer this over `export`).

`.env` is listed in `.gitignore` so it is never committed.

## Model selection
The model is also configurable via environment variable, defaulting to `claude-sonnet-5`:

```bash
export ANTHROPIC_MODEL="claude-sonnet-5"   # default if unset
```

Override it to use a different model (e.g. a faster/cheaper one for high-volume tasks, or a different tier for harder reasoning) without touching code.

## No-key fallback (demo mode)
If `ANTHROPIC_API_KEY` is not set, `src/llm_client.py` automatically switches to **demo mode**: it returns clearly-labeled placeholder text instead of making network calls. This means:
- The application never crashes just because a key is missing.
- All four Week 4 scripts can be run and reviewed end-to-end with zero setup.
- Every demo-mode response is explicitly prefixed `[DEMO MODE RESPONSE]` so it's never mistaken for a real model output.

Once a real key is set, every script automatically switches to live calls — no code changes needed.

## Rate limits & error handling
`llm_client.py` wraps every live call with:
- Retry with exponential backoff on `RateLimitError` and `APIConnectionError` (up to 3 attempts).
- Immediate, non-retried failure on 4xx errors like authentication failures (retrying a bad key won't fix it).
- A deliberate live demo of this (`src/02_api_application.py`, `demo_error_handling()`) that calls the real API with an invalid key and shows the resulting `401 authentication_error` being caught cleanly.

## Cost awareness
`UsageTracker` in `llm_client.py` records input/output tokens for every call made in a run and estimates cost using illustrative per-1K-token pricing constants at the top of the file (`PRICE_PER_1K_INPUT`, `PRICE_PER_1K_OUTPUT`). Check Anthropic's current pricing page before relying on these figures for real budgeting — they're placeholders for demonstrating the tracking pattern, not live pricing.

## Hugging Face configuration
`src/03_huggingface_transformers.py` needs no API key — Hugging Face `pipeline()` calls download model weights directly from the Hub on first use and cache them locally (default cache: `~/.cache/huggingface`). No `.env` entry is required for this part of the project.
