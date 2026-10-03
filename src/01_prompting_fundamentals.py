"""
Week 4 - Day 1
Generative AI & LLM Fundamentals
Al Aziz Technologies - Python Programming & AI Development Track

Covers: what an LLM/tokens/context window/parameters/inference/temperature are,
system/user/assistant messages, and hands-on prompt engineering — zero-shot,
few-shot, role-based prompting — with a side-by-side output comparison.

Author: Sikander
"""

import json
import sys
import os

sys.path.append(os.path.dirname(__file__))
from llm_client import LLMClient

OUTPUTS_DIR = "../outputs"


def explain_concepts():
    """Short printed primer on the Day 1 vocabulary — kept as executable
    documentation rather than a separate static file, so running this script
    doubles as a review."""
    concepts = {
        "Generative AI": "AI that creates new content (text, images, code, audio) "
                          "rather than just classifying or predicting a label, "
                          "as traditional ML typically does.",
        "Generative AI vs traditional ML": "Traditional ML (e.g. a classifier) maps "
                          "input -> a fixed label/number. Generative AI maps "
                          "input -> new, open-ended output drawn from a learned "
                          "probability distribution over possible content.",
        "LLM (Large Language Model)": "A neural network, usually Transformer-based, "
                          "trained on huge text corpora to predict the next token, "
                          "which lets it generate, summarize, translate and reason "
                          "over text.",
        "Tokens / Tokenization": "Text is split into tokens (often sub-word pieces) "
                          "before being fed to the model. 'Tokenization' is this "
                          "splitting step; a token is roughly 3/4 of an English word "
                          "on average.",
        "Context window": "The maximum number of tokens (input + output combined) "
                          "a model can attend to in a single request.",
        "Parameters": "The learned weights inside the neural network — the larger "
                          "the parameter count, generally the more capacity the "
                          "model has to represent patterns (with diminishing returns "
                          "and cost trade-offs).",
        "Inference": "The act of running a trained model forward to produce an "
                          "output for a given input (as opposed to training it).",
        "Temperature": "A sampling parameter controlling randomness: near 0 is "
                          "deterministic/focused, higher values (e.g. 1.0+) produce "
                          "more varied, creative output.",
        "System/user/assistant messages": "The three roles in a chat-style LLM "
                          "conversation: 'system' sets behavior/instructions, "
                          "'user' is the human's input, 'assistant' is the model's "
                          "prior or current reply.",
    }

    print("=" * 70)
    print("DAY 1 - Core concepts")
    print("=" * 70)
    for term, definition in concepts.items():
        print(f"\n{term}:\n  {definition}")

    return concepts


def run_prompting_techniques(client: LLMClient):
    """Hands-on: zero-shot, few-shot, and role-based prompting on the same
    underlying task, so the outputs are genuinely comparable."""
    print("\n" + "=" * 70)
    print("DAY 1 - Prompt engineering techniques (same task, three approaches)")
    print("=" * 70)

    task_description = "Classify the sentiment of this review and explain why in one sentence."
    review = "The battery dies by noon even with light use, but the camera is excellent."

    prompts = {
        "zero_shot": (
            f"{task_description}\n\nReview: \"{review}\""
        ),
        "few_shot": (
            f"{task_description}\n\n"
            "Review: \"Fast shipping, but the box arrived crushed.\"\n"
            "Sentiment: Mixed — positive on logistics, negative on packaging.\n\n"
            "Review: \"Absolutely love it, works perfectly every day.\"\n"
            "Sentiment: Positive — consistent satisfaction, no complaints.\n\n"
            f"Review: \"{review}\"\nSentiment:"
        ),
        "role_based": (
            "You are a senior product analyst who writes concise, balanced "
            "verdicts for an internal weekly report. "
            f"{task_description}\n\nReview: \"{review}\""
        ),
    }

    results = {}
    for technique, prompt in prompts.items():
        print(f"\n--- {technique.replace('_', ' ').title()} ---")
        print(f"Prompt:\n{prompt}\n")
        result = client.generate(prompt, temperature=0.3, max_tokens=150, label=technique)
        print(f"Output ({result['mode']} mode): {result['text']}")
        results[technique] = {
            "prompt": prompt,
            "output": result["text"],
            "mode": result["mode"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
        }

    return results


def compare_temperatures(client: LLMClient):
    """Hands-on: show how temperature changes output variety on an identical prompt."""
    print("\n" + "=" * 70)
    print("DAY 1 - Temperature comparison (same prompt, different temperature)")
    print("=" * 70)

    prompt = "Suggest a one-line tagline for a drone-based disaster response startup."
    comparisons = {}

    for temp in [0.0, 0.7, 1.3]:
        result = client.generate(prompt, temperature=temp, max_tokens=40,
                                  label=f"temperature_{temp}")
        print(f"\nTemperature {temp}: {result['text']}")
        comparisons[str(temp)] = result["text"]

    return comparisons


def main():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    concepts = explain_concepts()
    client = LLMClient(system="You are a concise, helpful assistant for a weekly engineering report.")

    technique_results = run_prompting_techniques(client)
    temperature_results = compare_temperatures(client)

    output = {
        "concepts_covered": list(concepts.keys()),
        "prompting_techniques": technique_results,
        "temperature_comparison": temperature_results,
        "usage_summary": client.usage.summary(),
    }

    with open(f"{OUTPUTS_DIR}/day1_prompting_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Token usage this run: {client.usage.summary()['total_tokens']} "
          f"(est. cost: ${client.usage.estimated_cost_usd:.6f})")
    print(f"Results saved to {OUTPUTS_DIR}/day1_prompting_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
