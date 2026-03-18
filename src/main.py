"""
Phase 1 Proof of Concept: Run Faithfulness metric on Variant A (N=5) across all judges.
"""

import os
import ast
import time
import pandas as pd
from ragas.metrics.collections import Faithfulness

from judges import get_all_judges

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "variants")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
POC_N = 5  # Proof of concept sample size


def load_variant(variant_name, n=None):
    """Load a variant CSV and return as list of dicts for batch_score."""
    path = os.path.join(DATA_DIR, f"{variant_name}.csv")
    df = pd.read_csv(path)
    if n:
        df = df.head(n)

    inputs = []
    for _, row in df.iterrows():
        # Parse contexts back from string representation
        contexts = row["contexts"]
        if isinstance(contexts, str):
            try:
                contexts = ast.literal_eval(contexts)
            except (ValueError, SyntaxError):
                contexts = [contexts]

        inputs.append({
            "user_input": row["question"],
            "retrieved_contexts": contexts,
            "response": str(row["answer"]),
        })

    return inputs


RATE_LIMIT_DELAY = 15  # seconds between calls for free-tier rate limits


def run_evaluation(judge_name, judge_llm, inputs):
    """Run Faithfulness evaluation with a single judge, one sample at a time.

    Sequential scoring avoids hitting free-tier rate limits (e.g. Gemini 5 req/min)
    and gives per-sample error isolation.
    """
    print(f"\n  Running Faithfulness with {judge_name}...")
    metric = Faithfulness(llm=judge_llm)
    scores = []

    for i, inp in enumerate(inputs):
        try:
            result = metric.score(
                user_input=inp["user_input"],
                response=inp["response"],
                retrieved_contexts=inp["retrieved_contexts"],
            )
            scores.append(result.value)
            print(f"    [{i+1}/{len(inputs)}] score={result.value:.2f}")
        except Exception as e:
            scores.append(float("nan"))
            print(f"    [{i+1}/{len(inputs)}] ERROR: {e}")

        # Delay between calls to stay under free-tier rate limits
        if i < len(inputs) - 1:
            time.sleep(RATE_LIMIT_DELAY)

    return pd.DataFrame({
        "judge": [judge_name] * len(inputs),
        "user_input": [inp["user_input"] for inp in inputs],
        "faithfulness": scores,
    })


def main():
    print("=" * 60)
    print("RAGAS Phase 1 — Proof of Concept")
    print("Metric: Faithfulness | Variant: A (Gold) | N=5")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1] Loading Variant A...")
    inputs = load_variant("A", n=POC_N)
    print(f"    Loaded {len(inputs)} samples")

    # Step 2: Initialize judges
    print("\n[2] Initializing judges...")
    judges = get_all_judges()

    if not judges:
        print("\nNo judges available. Check your .env file (GOOGLE_API_KEY, GROQ_API_KEY).")
        return

    # Step 3: Run evaluations
    print("\n[3] Running evaluations...")
    all_results = []

    for judge_name, judge_llm in judges.items():
        start = time.time()
        result_df = run_evaluation(judge_name, judge_llm, inputs)
        elapsed = time.time() - start
        print(f"      {judge_name} completed in {elapsed:.1f}s")
        all_results.append(result_df)

    # Step 4: Combine and save
    combined = pd.concat(all_results, ignore_index=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "test_run.csv")
    combined.to_csv(output_path, index=False)
    print(f"\n[4] Results saved to {output_path}")

    # Step 5: Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    summary = combined.groupby("judge")["faithfulness"].agg(["mean", "std", "count"])
    print(summary.to_string())

    print(f"\nTotal rows: {len(combined)}")


if __name__ == "__main__":
    main()
