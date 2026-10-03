# Prompt Documentation

This project uses four prompt patterns across Days 1 and 4. Each is documented here with its purpose, structure, and when to use it.

## 1. Zero-shot prompting
**Used in:** `src/01_prompting_fundamentals.py`

Give the model a task with no examples, relying entirely on its pretrained knowledge.

```
Classify the sentiment of this review and explain why in one sentence.

Review: "{review}"
```

**When to use:** simple, well-understood tasks where the model doesn't need to be shown the expected format.

## 2. Few-shot prompting
**Used in:** `src/01_prompting_fundamentals.py`

Give the model 2-3 worked examples before the real input, so it infers the expected output format/style from the pattern rather than a written instruction alone.

```
Review: "Fast shipping, but the box arrived crushed."
Sentiment: Mixed — positive on logistics, negative on packaging.

Review: "Absolutely love it, works perfectly every day."
Sentiment: Positive — consistent satisfaction, no complaints.

Review: "{review}"
Sentiment:
```

**When to use:** when output format/style matters and zero-shot gives inconsistent results, or the task is unusual enough that examples clarify it faster than a longer instruction would.

## 3. Role-based prompting
**Used in:** `src/01_prompting_fundamentals.py`

Assign the model a persona/role in the system or opening instruction, which shapes tone, priorities, and the kind of reasoning it leans on.

```
You are a senior product analyst who writes concise, balanced verdicts
for an internal weekly report. {task}
```

**When to use:** when you want consistent tone/voice across many calls, or want the model to prioritize certain considerations (e.g. "senior analyst" implies brevity and balance) without spelling every rule out.

## 4. Structured-output prompting
**Used in:** `src/02_api_application.py`, `src/04_ai_assistant_patterns.py`

Explicitly instruct the model to return only JSON matching a given shape, then parse and validate the result.

```
{task_instruction}

Respond with ONLY valid JSON, no prose, no markdown code fences,
matching this shape:
{"field_a": string, "field_b": string|null}
```

**When to use:** any time the output feeds into code rather than being read directly by a person — extraction, classification with a fixed label set, API responses.

**Design notes that reduce failure rate:**
- Explicitly forbid markdown code fences (models often wrap JSON in ` ```json ` blocks anyway, so the parser also strips fences defensively — see `llm_client.generate_json()`).
- Use `null` rather than inventing a value for missing fields — this is stated directly in the extraction prompt (`EXTRACTION_TEMPLATE`), which is a concrete hallucination-reduction technique (Day 4).
- Keep temperature low (0.0) for structured/classification tasks — randomness is a liability, not a feature, when the output must be parsed or must match a fixed label set.

## Grounding / hallucination reduction
**Used in:** `src/04_ai_assistant_patterns.py` (`task_summarization`, system prompt)

Two techniques are combined:
1. **System-level instruction:** `"You are a precise, grounded assistant. Never state information that isn't given to you."` — sets a standing constraint for every call in that session.
2. **Per-prompt instruction:** `"Stay strictly grounded in the text — do not add information that isn't present."` — repeats the constraint at the point of use, since per-prompt instructions are generally followed more reliably than system-level ones alone.

## Prompt template reference
All reusable templates live in `src/04_ai_assistant_patterns.py` as the `CLASSIFICATION_TEMPLATE`, `EXTRACTION_TEMPLATE`, and `SUMMARIZATION_TEMPLATE` constants — parameterized with Python `.format()` placeholders so they can be reused across inputs without rewriting the instruction each time.
