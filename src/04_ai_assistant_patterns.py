"""
Week 4 - Day 4
Prompt Engineering & LLM Application Patterns
Al Aziz Technologies - Python Programming & AI Development Track

Builds a small AI assistant using reusable prompt templates and structured
(JSON) outputs for three tasks: classification, extraction, and
summarization. Also demonstrates grounding/hallucination-reduction
instructions and a simple function/tool-calling pattern.

Author: Sikander
"""

import json
import sys
import os

sys.path.append(os.path.dirname(__file__))
from llm_client import LLMClient

OUTPUTS_DIR = "../outputs"

# ---------------------------------------------------------------------------
# DAY 4 - Prompt templates (reusable, parameterized prompt structure)
# ---------------------------------------------------------------------------

CLASSIFICATION_TEMPLATE = """You are classifying support tickets into exactly one category.
Categories: {categories}

Ticket: "{text}"

Respond with ONLY the category name, nothing else."""

EXTRACTION_TEMPLATE = """Extract the requested fields from the text below.
If a field is not present, use null — do not guess or invent a value.

Fields to extract: {fields}

Text: "{text}"

Respond with ONLY valid JSON matching this shape: {schema}"""

SUMMARIZATION_TEMPLATE = """Summarize the text below in {max_sentences} sentence(s) or fewer.
Stay strictly grounded in the text — do not add information that isn't present.

Text: "{text}"

Summary:"""


# ---------------------------------------------------------------------------
# DAY 4 - Task 1: Classification
# ---------------------------------------------------------------------------

def task_classification(client: LLMClient):
    print("=" * 70)
    print("DAY 4 - Task: Classification")
    print("=" * 70)

    categories = ["Billing", "Technical Issue", "Feature Request", "General Question"]
    tickets = [
        "I was charged twice for my subscription this month.",
        "The app crashes every time I try to export a report.",
        "Could you add dark mode to the dashboard?",
    ]

    results = []
    for ticket in tickets:
        prompt = CLASSIFICATION_TEMPLATE.format(
            categories=", ".join(categories), text=ticket
        )
        result = client.generate(prompt, temperature=0.0, max_tokens=20,
                                  label="classification")
        print(f"\nTicket: {ticket}\nPredicted category ({result['mode']}): {result['text'].strip()}")
        results.append({"ticket": ticket, "predicted_category": result["text"].strip()})

    return results


# ---------------------------------------------------------------------------
# DAY 4 - Task 2: Extraction (structured output)
# ---------------------------------------------------------------------------

def task_extraction(client: LLMClient):
    print("\n" + "=" * 70)
    print("DAY 4 - Task: Extraction (structured output)")
    print("=" * 70)

    text = ("Please ship the replacement part to Ayesha Khan at 42 Mall Road, "
            "Lahore by next Thursday. Order reference ORD-7731.")
    fields = ["recipient_name", "address", "deadline", "order_reference"]
    schema = ('{"recipient_name": string|null, "address": string|null, '
              '"deadline": string|null, "order_reference": string|null}')

    prompt = EXTRACTION_TEMPLATE.format(
        fields=", ".join(fields), text=text, schema=schema
    )
    result = client.generate_json(prompt, schema_hint=schema, label="extraction")

    print(f"\nText: {text}")
    print(f"Parsed successfully: {result['parse_ok']}")
    if result["parse_ok"]:
        print(f"Extracted: {json.dumps(result['parsed'], indent=2)}")

    return {"text": text, "parse_ok": result["parse_ok"], "extracted": result.get("parsed")}


# ---------------------------------------------------------------------------
# DAY 4 - Task 3: Summarization (with grounding instruction)
# ---------------------------------------------------------------------------

def task_summarization(client: LLMClient):
    print("\n" + "=" * 70)
    print("DAY 4 - Task: Summarization (grounded)")
    print("=" * 70)

    text = (
        "AeroSentinel is a proposed drone system for disaster response in "
        "Pakistan. It would use thermal cameras to locate survivors after "
        "floods and deliver small medical supply packages to areas cut off "
        "by road damage. The team is currently at the concept and proposal "
        "stage, with no hardware built yet."
    )

    prompt = SUMMARIZATION_TEMPLATE.format(text=text, max_sentences=1)
    result = client.generate(prompt, temperature=0.2, max_tokens=80, label="summarization")

    print(f"\nText: {text}")
    print(f"Summary ({result['mode']}): {result['text'].strip()}")

    return {"text": text, "summary": result["text"].strip()}


# ---------------------------------------------------------------------------
# DAY 4 - Function/tool-calling concept demo
# ---------------------------------------------------------------------------

def tool_calling_pattern_demo():
    """
    Illustrates the function/tool-calling pattern conceptually: the model is
    given a description of available tools/functions, decides whether one
    is needed, and (in a real tool-calling API) returns a structured call
    the application code then executes. This demo simulates that decision
    step with the same generate_json() structured-output helper used above,
    since it's the same underlying pattern without needing the full
    tool-use API surface.
    """
    print("\n" + "=" * 70)
    print("DAY 4 - Function/tool-calling pattern (conceptual demo)")
    print("=" * 70)

    available_tools = {
        "get_weather": "Get current weather for a city. Args: city (string).",
        "convert_currency": "Convert an amount between currencies. "
                             "Args: amount (number), from_currency, to_currency.",
    }

    def get_weather(city: str) -> str:
        return f"[MOCK] It's 29°C and clear in {city}."

    def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
        return f"[MOCK] {amount} {from_currency} ≈ {amount * 0.0036:.2f} {to_currency}"

    tool_functions = {"get_weather": get_weather, "convert_currency": convert_currency}

    user_request = "What's the weather like in Islamabad right now?"

    print(f"\nAvailable tools: {list(available_tools.keys())}")
    print(f"User request: {user_request}")
    print(
        "\nIn a real tool-calling API, the model would return a structured "
        "call like {\"tool\": \"get_weather\", \"args\": {\"city\": \"Islamabad\"}} "
        "which the application then executes and feeds back to the model. "
        "Simulating that decision here without a live call:"
    )

    # Simple simulated routing (stands in for the model's tool-choice decision)
    simulated_call = {"tool": "get_weather", "args": {"city": "Islamabad"}}
    print(f"\nSimulated model tool call: {simulated_call}")

    tool_result = tool_functions[simulated_call["tool"]](**simulated_call["args"])
    print(f"Tool execution result: {tool_result}")

    return {
        "available_tools": list(available_tools.keys()),
        "user_request": user_request,
        "simulated_tool_call": simulated_call,
        "tool_result": tool_result,
    }


def main():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    client = LLMClient(system="You are a precise, grounded assistant. "
                               "Never state information that isn't given to you.")

    classification_results = task_classification(client)
    extraction_result = task_extraction(client)
    summarization_result = task_summarization(client)
    tool_calling_result = tool_calling_pattern_demo()

    output = {
        "classification": classification_results,
        "extraction": extraction_result,
        "summarization": summarization_result,
        "tool_calling_demo": tool_calling_result,
        "usage_summary": client.usage.summary(),
    }

    with open(f"{OUTPUTS_DIR}/day4_ai_assistant_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Token usage this run: {client.usage.summary()['total_tokens']} "
          f"(est. cost: ${client.usage.estimated_cost_usd:.6f})")
    print(f"Results saved to {OUTPUTS_DIR}/day4_ai_assistant_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
