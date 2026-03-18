"""
Phase 1 Proof of Concept: Run Faithfulness metric on Variant A (N=5) across all judges.
"""

import os
import ast
import time
import pandas as pd
from ragas import evaluate
from ragas.metrics.collections import Faithfulness
from ragas import EvaluationDataset, SingleTurnSample

from judges import get_all_judges

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "variants")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
POC_N = 5  # Proof of concept sample size


def load_variant(variant_name, n=None):
    """Load a variant CSV and return as list of SingleTurnSample."""
    path = os.path.join(DATA_DIR, f"{variant_name}.csv")
    df = pd.read_csv(path)
    if n:
        df = df.head(n)

    samples = []
    for _, row in df.iterrows():
        # Parse contexts back from string representation
        contexts = row["contexts"]
        if isinstance(contexts, str):
            try:
                contexts = ast.literal_eval(contexts)
            except (ValueError, SyntaxError):
                contexts = [contexts]

        sample = SingleTurnSample(
            user_input=row["question"],
            retrieved_contexts=contexts,
            response=str(row["answer"]),
            reference=str(row.get("ground_truth", row["answer"])),
        )
        samples.append(sample)

    return EvaluationDataset(samples=samples)


def run_evaluation(judge_name, judge_llm, dataset):
    """Run Faithfulness evaluation with a single judge. Returns results DataFrame."""
    print(f"\n  Running Faithfulness with {judge_name}...")
    metric = Faithfulness(llm=judge_llm)

    try:
        result = evaluate(
            dataset=dataset,
            metrics=[metric],
        )
        result_df = result.to_pandas()
        result_df["judge"] = judge_name
        return result_df
    except Exception as e:
        print(f"  [ERROR] {judge_name}: {e}")
        # Return DataFrame with NaN scores
        return pd.DataFrame({
            "judge": [judge_name] * len(dataset),
            "user_input": [s.user_input for s in dataset.samples],
            "faithfulness": [float("nan")] * len(dataset),
        })


def main():
    print("=" * 60)
    print("RAGAS Phase 1 — Proof of Concept")
    print("Metric: Faithfulness | Variant: A (Gold) | N=5")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1] Loading Variant A...")
    dataset = load_variant("A", n=POC_N)
    print(f"    Loaded {len(dataset)} samples")

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
        result_df = run_evaluation(judge_name, judge_llm, dataset)
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
    if "faithfulness" in combined.columns:
        summary = combined.groupby("judge")["faithfulness"].agg(["mean", "std", "count"])
        print(summary.to_string())
    else:
        print(combined.to_string())

    print(f"\nTotal rows: {len(combined)}")


if __name__ == "__main__":
    main()
