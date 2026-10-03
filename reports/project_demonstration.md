# Project Demonstration

**Project:** Week 4 — Generative AI, LLMs & Hugging Face
**Author:** Sikander

This document walks through what the project does end to end, for a live demo or submission review.

## 0. One-time setup
```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in a real ANTHROPIC_API_KEY to see live output
```
The project runs without a key too — it automatically drops into demo mode (see `reports/api_configuration.md`). The walkthrough below describes behavior in both modes.

## 1. Day 1 — Prompting Fundamentals
Run:
```bash
cd src && python3 01_prompting_fundamentals.py
```
- Prints a short primer on every Day 1 term (tokens, context window, temperature, etc.) as executable documentation.
- Runs the **same task** (sentiment + reasoning on one review) through zero-shot, few-shot, and role-based prompts, so the three techniques are genuinely comparable side by side.
- Runs the same prompt at three temperatures (0.0, 0.7, 1.3) to show the determinism-to-creativity spectrum.
- Saves everything, plus token/cost totals, to `outputs/day1_prompting_results.json`.

**Demo talking point:** with a live key, run it twice at temperature 1.3 and show the tagline output differs between runs, then run twice at temperature 0.0 and show it's identical — the clearest possible illustration of what temperature controls.

## 2. Day 2 — LLM API Application
Run:
```bash
python3 02_api_application.py
```
- **Conversation history:** a 3-turn exchange where turn 3 ("summarize what we've discussed") only works correctly if history was actually passed back to the model.
- **Streaming:** prints the response token-by-token as it arrives, not after a single blocking wait.
- **Structured output:** extracts event/day/time/topic from a free-text note into parsed JSON, with a shown error path if parsing fails.
- **Error handling:** deliberately calls the live Anthropic API with an invalid key and prints the real `401` it gets back — proof the retry/error-handling logic actually fires against the real API, not just in theory.
- Saves full transcript + usage to `outputs/day2_api_application_results.json`.

**Demo talking point:** the error-handling demo is run against the live API every time (it doesn't need a valid key — it's testing the invalid-key path on purpose), so it's a true, reproducible demonstration regardless of whether you have a key configured.

## 3. Day 3 — Hugging Face Transformers
Run:
```bash
python3 03_huggingface_transformers.py
```
- Probes Hub connectivity first; if reachable, runs a real `distilbert` sentiment-analysis pipeline and a real `distilgpt2` text-generation pipeline locally.
- If the Hub isn't reachable (as in this development sandbox — see the note at the top of the script), falls back to a clearly-labeled offline heuristic so the script still completes and the pipeline *code* can still be reviewed.
- Covers embeddings conceptually, with a small real TF-IDF similarity demo that needs no model download.
- Saves results, including which mode ran, to `outputs/day3_huggingface_results.json`.

**Demo talking point:** show `hub_available_in_this_run` in the saved JSON — it's an honest, logged record of which mode actually executed, not a hidden assumption.

## 4. Day 4 — AI Assistant Patterns
Run:
```bash
python3 04_ai_assistant_patterns.py
```
- **Classification:** routes three support tickets into one of four fixed categories using a reusable `CLASSIFICATION_TEMPLATE`.
- **Extraction:** pulls recipient/address/deadline/order-reference from a free-text message into structured JSON, explicitly instructed to use `null` rather than inventing missing fields (a concrete hallucination-reduction technique).
- **Summarization:** grounded 1-sentence summary, combining a system-level and a per-prompt grounding instruction.
- **Tool-calling pattern:** simulates the model deciding to call a `get_weather` tool and the application executing it — illustrates the pattern without needing the full tool-use API surface.
- Saves everything to `outputs/day4_ai_assistant_results.json`.

**Demo talking point:** `reports/prompt_documentation.md` has every template used here written out with the reasoning behind its design — useful to walk through alongside the code.

## 5. Summary for Reviewers
- Runs end-to-end with zero configuration (demo mode), and automatically upgrades to live Claude API calls the moment a key is set — no code changes needed either way.
- Error handling isn't just described, it's demonstrated against the real live API on every run.
- Hugging Face section is honest about the sandbox network restriction rather than silently faking success — the fallback is clearly labeled, and the real pipeline code is unmodified from what would run with full internet access.
- Every script writes its results to `outputs/*.json` for inspection without re-running anything.
