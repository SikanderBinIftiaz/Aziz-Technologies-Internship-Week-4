# Week 4 — Generative AI, LLMs & Hugging Face

**Al Aziz Technologies — Python Programming & AI Development Track**
**Author:** Sikander (BS(AI), SZABIST)

## Overview
This project covers Week 4 of the internship: generative AI and LLM fundamentals, building applications against the Anthropic (Claude) API, local inference with Hugging Face Transformers, and prompt-engineering/application patterns (structured outputs, grounding, tool-calling).

## Project Structure
```
week4_project/
├── README.md
├── requirements.txt
├── .env.example                  # API key configuration template
├── .gitignore
├── src/
│   ├── llm_client.py                   # Shared client: auth, retries, streaming, cost tracking
│   ├── 01_prompting_fundamentals.py    # Day 1: concepts, zero/few-shot/role-based prompting
│   ├── 02_api_application.py           # Day 2: history, streaming, structured output, error handling
│   ├── 03_huggingface_transformers.py  # Day 3: local Transformer pipelines
│   └── 04_ai_assistant_patterns.py     # Day 4: classification/extraction/summarization, tool-calling
├── outputs/                      # JSON results from each script run
└── reports/
    ├── prompt_documentation.md
    ├── api_configuration.md
    └── project_demonstration.md
```

## Daily Breakdown

**Day 1 — Generative AI & LLM Fundamentals**
Core concepts (tokens, context window, temperature, parameters, inference, message roles) plus a direct comparison of zero-shot, few-shot, and role-based prompting on the same task.

**Day 2 — LLM APIs & AI Application Development**
A working Python application using the Anthropic API: environment-variable authentication, conversation history, streaming responses, structured JSON output, retry/error handling (demonstrated live against the real API), and token/cost tracking.

**Day 3 — Hugging Face Transformers**
Local inference pipelines for text classification and text generation, plus an embeddings overview and notes on local-vs-API inference and GPU/CPU trade-offs. (Note: this sandbox's network doesn't reach huggingface.co, so the script runs with a clearly-labeled offline fallback here — the pipeline code itself is unmodified and will run live on a normal machine. See `reports/project_demonstration.md`.)

**Day 4 — Prompt Engineering & LLM Application Patterns**
Reusable prompt templates for classification, extraction, and summarization, with explicit grounding instructions to reduce hallucination, plus a function/tool-calling pattern demo.

**Day 5 — Weekly Project & Revision**
All four days integrate through the shared `llm_client.py` module — one authentication/error-handling/cost-tracking layer used everywhere, rather than four one-off scripts.

## How to Run
```bash
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY for live output (optional — demo mode works without it)

cd src
python3 01_prompting_fundamentals.py
python3 02_api_application.py
python3 03_huggingface_transformers.py
python3 04_ai_assistant_patterns.py
```

Full configuration details: [`reports/api_configuration.md`](reports/api_configuration.md).
Prompt design rationale: [`reports/prompt_documentation.md`](reports/prompt_documentation.md).
Walkthrough for a live demo: [`reports/project_demonstration.md`](reports/project_demonstration.md).

## Friday Deliverables Checklist
- [x] Working AI application
- [x] GitHub repository
- [x] README.md
- [x] Prompt documentation
- [x] API configuration
- [x] Project demonstration
