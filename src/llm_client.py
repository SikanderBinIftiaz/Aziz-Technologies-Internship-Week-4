"""
Week 4 - Shared LLM client module
Day 1 & 2 concepts: API authentication, environment variables, model selection,
text generation, streaming responses, conversation history, error handling,
rate limits, token usage, cost awareness.

This module wraps the Anthropic Python SDK and is imported by every other
script in this project so the authentication/retry/cost-tracking logic
lives in exactly one place.

Author: Sikander
"""

import os
import json
import time
from dataclasses import dataclass, field
from typing import Optional

from anthropic import Anthropic, APIStatusError, APIConnectionError, RateLimitError

# ---------------------------------------------------------------------------
# API authentication & model selection (Day 2)
# ---------------------------------------------------------------------------
# The API key is NEVER hard-coded. It is read from an environment variable,
# which is the standard, secure way to handle credentials. See
# reports/api_configuration.md and .env.example for setup instructions.

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 2

# Pricing used only to demonstrate cost-awareness tracking (illustrative
# figures per 1K tokens; check Anthropic's pricing page for current rates).
PRICE_PER_1K_INPUT = 0.003
PRICE_PER_1K_OUTPUT = 0.015


@dataclass
class UsageTracker:
    """DAY 2 - Token usage & cost awareness."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    calls: list = field(default_factory=list)

    def record(self, input_tokens: int, output_tokens: int, label: str = ""):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.calls.append({
            "label": label,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        })

    @property
    def estimated_cost_usd(self) -> float:
        return (
            (self.total_input_tokens / 1000) * PRICE_PER_1K_INPUT
            + (self.total_output_tokens / 1000) * PRICE_PER_1K_OUTPUT
        )

    def summary(self) -> dict:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "calls": self.calls,
        }


class LLMClient:
    """
    Thin wrapper around the Anthropic API that adds:
      - environment-variable based authentication (Day 2)
      - retry with exponential backoff on rate limits / transient errors (Day 2)
      - conversation history management (Day 2)
      - token usage & cost tracking (Day 2)
      - a DEMO fallback mode so the app is runnable and demonstrable even
        without a live API key (useful for grading/offline review; error
        handling + user messaging is itself part of the Day 2 curriculum).
    """

    def __init__(self, model: str = MODEL, system: Optional[str] = None):
        self.model = model
        self.system = system
        self.history = []  # DAY 2 - conversation history
        self.usage = UsageTracker()
        self.demo_mode = API_KEY is None
        self.client = Anthropic(api_key=API_KEY) if API_KEY else None

        if self.demo_mode:
            print("[llm_client] No ANTHROPIC_API_KEY found in the environment — "
                  "running in DEMO MODE (canned responses, no network calls). "
                  "Set the key to make real calls. See reports/api_configuration.md.")

    # -- conversation history -------------------------------------------------
    def reset_history(self):
        self.history = []

    def add_user_message(self, content: str):
        self.history.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        self.history.append({"role": "assistant", "content": content})

    # -- core text generation (Day 1 & 2) --------------------------------------
    def generate(self, prompt: str, temperature: float = 1.0,
                 max_tokens: int = 512, use_history: bool = False,
                 label: str = "") -> dict:
        """
        Single text-generation call. Returns a dict with text, token usage,
        and whether the call succeeded, errored, or fell back to demo mode.
        """
        messages = (self.history if use_history else []) + [
            {"role": "user", "content": prompt}
        ]

        if self.demo_mode:
            text = self._demo_response(prompt)
            result = {"text": text, "input_tokens": len(prompt.split()),
                      "output_tokens": len(text.split()), "mode": "demo"}
        else:
            result = self._call_with_retry(messages, temperature, max_tokens)

        self.usage.record(result["input_tokens"], result["output_tokens"], label or prompt[:40])

        if use_history:
            self.add_user_message(prompt)
            self.add_assistant_message(result["text"])

        return result

    def _call_with_retry(self, messages, temperature, max_tokens) -> dict:
        """DAY 2 - Error handling & rate limits, with exponential backoff."""
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                kwargs = {
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": messages,
                }
                if self.system:
                    kwargs["system"] = self.system

                response = self.client.messages.create(**kwargs)
                text = "".join(
                    block.text for block in response.content if block.type == "text"
                )
                return {
                    "text": text,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "mode": "live",
                }

            except RateLimitError as e:
                last_error = e
                wait = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
                print(f"[llm_client] Rate limited (attempt {attempt}/{MAX_RETRIES}). "
                      f"Backing off {wait}s...")
                time.sleep(wait)

            except APIConnectionError as e:
                last_error = e
                print(f"[llm_client] Connection error (attempt {attempt}/{MAX_RETRIES}): {e}")
                time.sleep(BASE_BACKOFF_SECONDS)

            except APIStatusError as e:
                # 4xx (e.g. auth errors) are not worth retrying
                last_error = e
                print(f"[llm_client] API error {e.status_code}: {e.message}")
                break

        return {
            "text": f"[ERROR] Request failed after {MAX_RETRIES} attempts: {last_error}",
            "input_tokens": 0,
            "output_tokens": 0,
            "mode": "error",
        }

    # -- streaming (Day 2) -----------------------------------------------------
    def generate_stream(self, prompt: str, temperature: float = 1.0, max_tokens: int = 512):
        """
        DAY 2 - Streaming responses. Yields text chunks as they arrive.
        Falls back to a simulated chunk-by-chunk yield in demo mode so the
        calling code (and a reader of this file) sees identical streaming
        behavior either way.
        """
        if self.demo_mode:
            text = self._demo_response(prompt)
            for word in text.split():
                yield word + " "
                time.sleep(0.02)
            return

        stream_kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.system:
            stream_kwargs["system"] = self.system

        try:
            with self.client.messages.stream(**stream_kwargs) as stream:
                for text_chunk in stream.text_stream:
                    yield text_chunk
        except (APIStatusError, APIConnectionError, RateLimitError) as e:
            yield f"[ERROR during streaming] {e}"

    # -- structured / JSON output (Day 2 & 4) ----------------------------------
    def generate_json(self, prompt: str, schema_hint: str, temperature: float = 0.0,
                       max_tokens: int = 512, label: str = "") -> dict:
        """
        DAY 2/4 - Structured outputs. Instructs the model to return only JSON
        matching schema_hint, then parses it. Includes error handling for
        malformed JSON (a real, common failure mode).
        """
        structured_prompt = (
            f"{prompt}\n\nRespond with ONLY valid JSON, no prose, no markdown "
            f"code fences, matching this shape:\n{schema_hint}"
        )
        result = self.generate(structured_prompt, temperature=temperature,
                                max_tokens=max_tokens, label=label)
        raw_text = result["text"].strip()
        # Defensive cleanup in case the model wraps the JSON in ```json fences anyway
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            if raw_text.lower().startswith("json"):
                raw_text = raw_text[4:].strip()

        try:
            parsed = json.loads(raw_text)
            result["parsed"] = parsed
            result["parse_ok"] = True
        except json.JSONDecodeError as e:
            result["parsed"] = None
            result["parse_ok"] = False
            result["parse_error"] = str(e)

        return result

    # -- demo-mode canned responses --------------------------------------------
    @staticmethod
    def _demo_response(prompt: str) -> str:
        """Deterministic, offline stand-in response used only when no API key
        is configured, so the application remains runnable end-to-end."""
        return (
            "[DEMO MODE RESPONSE] This is a placeholder answer generated "
            "locally because no ANTHROPIC_API_KEY was found. In live mode, "
            f"this would be Claude's real answer to: \"{prompt[:80]}...\""
            if len(prompt) > 80 else
            "[DEMO MODE RESPONSE] This is a placeholder answer generated "
            f"locally because no ANTHROPIC_API_KEY was found. In live mode, "
            f"this would be Claude's real answer to: \"{prompt}\""
        )
