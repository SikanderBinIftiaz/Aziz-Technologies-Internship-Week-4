"""
Week 4 - Day 2
LLM APIs & AI Application Development
Al Aziz Technologies - Python Programming & AI Development Track

A small Python LLM application demonstrating: API authentication via env vars,
model selection, text generation, structured (JSON) responses, streaming
responses, conversation history, error handling, rate limits, and token/cost
tracking.

Author: Sikander
"""

import json
import sys
import os

sys.path.append(os.path.dirname(__file__))
from llm_client import LLMClient

OUTPUTS_DIR = "../outputs"


def demo_conversation_history(client: LLMClient):
    """DAY 2 - Conversation history: a 3-turn conversation where each turn
    can reference what was said before."""
    print("=" * 70)
    print("DAY 2 - Multi-turn conversation with history")
    print("=" * 70)

    client.reset_history()
    turns = [
        "My project is a drone-based disaster response system called AeroSentinel.",
        "What's one technical risk I should plan for?",
        "Summarize what we've discussed so far in one sentence.",
    ]

    transcript = []
    for turn in turns:
        print(f"\nUser: {turn}")
        result = client.generate(turn, use_history=True, temperature=0.4,
                                  max_tokens=150, label="conversation_turn")
        print(f"Assistant ({result['mode']}): {result['text']}")
        transcript.append({"user": turn, "assistant": result["text"]})

    return transcript


def demo_streaming(client: LLMClient):
    """DAY 2 - Streaming responses: print tokens as they arrive instead of
    waiting for the full response."""
    print("\n" + "=" * 70)
    print("DAY 2 - Streaming response")
    print("=" * 70)

    prompt = "List three benefits of using drones for disaster response, briefly."
    print(f"\nPrompt: {prompt}\n\nStreaming output: ", end="", flush=True)

    full_text = ""
    for chunk in client.generate_stream(prompt, temperature=0.5, max_tokens=150):
        print(chunk, end="", flush=True)
        full_text += chunk
    print()  # newline after stream ends

    return full_text


def demo_structured_output(client: LLMClient):
    """DAY 2 - Structured/JSON responses with error handling for malformed JSON."""
    print("\n" + "=" * 70)
    print("DAY 2 - Structured JSON output")
    print("=" * 70)

    prompt = ("Extract structured info from this note: "
              "\"Meeting with the SZABIST robotics club on Friday at 3pm "
              "about the AeroSentinel drone prototype demo.\"")
    schema = (
        '{"event": string, "day": string, "time": string, "topic": string}'
    )

    result = client.generate_json(prompt, schema_hint=schema, label="structured_extract")
    print(f"\nRaw model output: {result['text']}")
    print(f"Parsed successfully: {result['parse_ok']}")
    if result["parse_ok"]:
        print(f"Parsed JSON: {json.dumps(result['parsed'], indent=2)}")
    else:
        print(f"Parse error (handled gracefully): {result.get('parse_error')}")

    return result


def demo_error_handling():
    """DAY 2 - Error handling & rate limits: deliberately trigger a real
    authentication error against the live API (no key / a bad key) to show
    the error-handling path actually firing, not just being theoretical."""
    print("\n" + "=" * 70)
    print("DAY 2 - Error handling demo (deliberately invalid credentials)")
    print("=" * 70)

    from anthropic import Anthropic, APIStatusError

    bad_client = Anthropic(api_key="invalid-key-for-demo-purposes")
    try:
        bad_client.messages.create(
            model="claude-sonnet-5",
            max_tokens=10,
            messages=[{"role": "user", "content": "hi"}],
        )
    except APIStatusError as e:
        print(f"Caught expected error — status {e.status_code}: {e.message}")
        print("This confirms the error-handling path works: a bad/missing "
              "key fails loudly and predictably instead of crashing silently.")
        return {"status_code": e.status_code, "message": e.message}

    return {"status_code": None, "message": "No error raised (unexpected)"}


def main():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    client = LLMClient(system="You are a helpful assistant for a student engineering project.")

    transcript = demo_conversation_history(client)
    streamed_text = demo_streaming(client)
    structured_result = demo_structured_output(client)
    error_demo = demo_error_handling()

    output = {
        "conversation_history_demo": transcript,
        "streaming_demo_output": streamed_text,
        "structured_output_demo": {
            "parse_ok": structured_result["parse_ok"],
            "parsed": structured_result.get("parsed"),
        },
        "error_handling_demo": error_demo,
        "usage_summary": client.usage.summary(),
    }

    with open(f"{OUTPUTS_DIR}/day2_api_application_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Token usage this run: {client.usage.summary()['total_tokens']} "
          f"(est. cost: ${client.usage.estimated_cost_usd:.6f})")
    print(f"Results saved to {OUTPUTS_DIR}/day2_api_application_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
