"""
Week 4 - Day 3
Hugging Face Transformers
Al Aziz Technologies - Python Programming & AI Development Track

Loads pretrained Transformer models from the Hugging Face Hub and runs local
inference for text classification and text generation, with a brief
embeddings overview and local-vs-API / GPU-vs-CPU discussion.

NOTE ON THIS SANDBOX: this development container's network allowlist does
not include huggingface.co, so model downloads cannot complete here. The
code below is the real, correct implementation — unmodified from what you'd
run locally — and will work as-is on any machine with normal internet
access (or with `huggingface-cli login` / HF_HOME cache pre-populated).
A clearly-labeled OFFLINE FALLBACK kicks in only so this script still runs
to completion and produces output for review in this sandbox.

Author: Sikander
"""

import json
import os

OUTPUTS_DIR = "../outputs"


def check_hf_availability() -> bool:
    """Try a tiny real network probe to the Hub before committing to a full
    model download, so the failure (if any) is fast and the reason is clear."""
    import urllib.request
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=5)
        return True
    except Exception as e:
        print(f"[huggingface] Hub unreachable from this environment ({e}).")
        print("[huggingface] Falling back to offline demo mode for this run.")
        return False


# ---------------------------------------------------------------------------
# DAY 3 - Text classification pipeline
# ---------------------------------------------------------------------------

def run_text_classification(hub_available: bool):
    print("=" * 70)
    print("DAY 3 - Text classification pipeline")
    print("=" * 70)

    samples = [
        "The drone's battery life exceeded every test we ran this week.",
        "The deployment kept crashing and support never responded.",
        "It's an average product, does the job, nothing special.",
    ]

    if hub_available:
        from transformers import pipeline
        classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )
        results = classifier(samples)
        source = "live Hugging Face pipeline (distilbert-base-uncased-finetuned-sst-2-english)"
    else:
        # OFFLINE FALLBACK — simple keyword heuristic, clearly not a real model.
        results = []
        for text in samples:
            lower = text.lower()
            positive_hits = sum(w in lower for w in ["exceeded", "great", "love", "excellent"])
            negative_hits = sum(w in lower for w in ["crash", "never", "bad", "poor"])
            if positive_hits > negative_hits:
                label, score = "POSITIVE", 0.75
            elif negative_hits > positive_hits:
                label, score = "NEGATIVE", 0.75
            else:
                label, score = "NEUTRAL", 0.50
            results.append({"label": label, "score": score})
        source = "OFFLINE FALLBACK (keyword heuristic — not a real Transformer model)"

    print(f"\nSource: {source}")
    output = []
    for text, result in zip(samples, results):
        print(f"\nText: {text}\nPrediction: {result}")
        output.append({"text": text, "prediction": result})

    return {"source": source, "results": output}


# ---------------------------------------------------------------------------
# DAY 3 - Text generation pipeline
# ---------------------------------------------------------------------------

def run_text_generation(hub_available: bool):
    print("\n" + "=" * 70)
    print("DAY 3 - Text generation pipeline")
    print("=" * 70)

    prompt = "Disaster response drones can"

    if hub_available:
        from transformers import pipeline
        generator = pipeline("text-generation", model="distilgpt2")
        result = generator(prompt, max_new_tokens=30, num_return_sequences=1)
        generated_text = result[0]["generated_text"]
        source = "live Hugging Face pipeline (distilgpt2)"
    else:
        generated_text = (
            prompt + " [OFFLINE FALLBACK — no model download available in this "
            "sandbox; on a machine with internet access this pipeline call "
            "would return real distilgpt2-generated continuation text here.]"
        )
        source = "OFFLINE FALLBACK (no real generation performed)"

    print(f"\nSource: {source}")
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")

    return {"source": source, "prompt": prompt, "generated_text": generated_text}


# ---------------------------------------------------------------------------
# DAY 3 - Embeddings overview (conceptual + a tiny real demo using sklearn
# as a stand-in vectorizer, since that runs anywhere with no model download)
# ---------------------------------------------------------------------------

def embeddings_overview():
    print("\n" + "=" * 70)
    print("DAY 3 - Embeddings overview")
    print("=" * 70)
    print(
        "\nAn embedding maps text (a word, sentence, or document) to a dense "
        "vector of floating-point numbers such that semantically similar "
        "texts end up close together in that vector space (by cosine "
        "similarity, typically). Hugging Face hosts dedicated sentence-"
        "embedding models (e.g. the sentence-transformers family) for this. "
        "Below is a lightweight illustration using TF-IDF vectors (not a "
        "neural embedding, but it demonstrates the 'similar text -> close "
        "vectors' idea without requiring a model download)."
    )

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    docs = [
        "The drone surveyed the flooded area.",
        "The UAV scanned the flooded zone.",
        "The chef prepared a three-course meal.",
    ]
    vectors = TfidfVectorizer().fit_transform(docs)
    similarity = cosine_similarity(vectors)

    print("\nPairwise cosine similarity:")
    for i, doc_i in enumerate(docs):
        for j, doc_j in enumerate(docs):
            if i < j:
                print(f"  \"{doc_i[:30]}...\" vs \"{doc_j[:30]}...\" -> {similarity[i][j]:.3f}")

    print(
        "\nNote: with real neural embeddings (e.g. sentence-transformers), "
        "sentences 1 & 2 above (same meaning, different words) would score "
        "far higher than with TF-IDF, because neural embeddings capture "
        "meaning, not just shared vocabulary."
    )

    return {"docs": docs, "similarity_matrix": similarity.tolist()}


def discuss_deployment_considerations():
    """DAY 3 - Local vs API-based inference, model selection, GPU/CPU."""
    notes = {
        "local_vs_api_inference": (
            "Local inference (loading weights with `transformers` and running "
            "on your own machine) avoids per-request API costs and network "
            "latency, and keeps data on-device — but you pay for compute, "
            "storage, and maintenance yourself. API-based inference (calling "
            "a hosted model) trades that for simplicity, elastic scaling, and "
            "access to much larger models than most local hardware can run."
        ),
        "model_selection": (
            "Pick the smallest model that meets the accuracy bar for the task. "
            "distilbert/distilgpt2-class models are fast and cheap for "
            "prototyping; move to larger models only if evaluation shows the "
            "smaller one underperforms on your actual data."
        ),
        "gpu_cpu_considerations": (
            "CPU inference works fine for small models and low request volume "
            "(as used in this script). GPU inference becomes worthwhile once "
            "you're running larger models, batching many requests, or need "
            "low per-request latency at scale."
        ),
    }

    print("\n" + "=" * 70)
    print("DAY 3 - Deployment considerations")
    print("=" * 70)
    for key, text in notes.items():
        print(f"\n{key}:\n  {text}")

    return notes


def main():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    hub_available = check_hf_availability()

    classification_result = run_text_classification(hub_available)
    generation_result = run_text_generation(hub_available)
    embeddings_result = embeddings_overview()
    deployment_notes = discuss_deployment_considerations()

    output = {
        "hub_available_in_this_run": hub_available,
        "classification": classification_result,
        "generation": generation_result,
        "embeddings_demo": embeddings_result,
        "deployment_notes": deployment_notes,
    }

    with open(f"{OUTPUTS_DIR}/day3_huggingface_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {OUTPUTS_DIR}/day3_huggingface_results.json")


if __name__ == "__main__":
    main()
