"""
Dataset loading and variant generation for RAGAS evaluation.
Loads vibrantlabsai/WikiEval and creates 4 dataset variants.
"""

import os
import ast
import pandas as pd
from datasets import load_dataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "variants")
N_SAMPLES = 50


def load_wikieval(n=N_SAMPLES):
    """Load WikiEval dataset and return first n rows as DataFrame."""
    ds = load_dataset("vibrantlabsai/WikiEval", split="train")
    df = ds.to_pandas().head(n)
    print(f"Loaded {len(df)} samples from WikiEval")
    return df


def ensure_list(value):
    """Ensure context value is a Python list (RAGAS requirement)."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
        return [value]
    return [str(value)]


def normalize_answer_lengths(df):
    """
    Bias mitigation: if average token lengths of answer columns differ by >20%
    from the mean, truncate the longer ones to normalize.
    """
    answer_cols = ["answer", "ungrounded_answer", "poor_answer"]
    present_cols = [c for c in answer_cols if c in df.columns]

    avg_lengths = {}
    for col in present_cols:
        avg_lengths[col] = df[col].dropna().apply(lambda x: len(str(x).split())).mean()

    if not avg_lengths:
        return df

    mean_length = sum(avg_lengths.values()) / len(avg_lengths)
    print(f"Average token lengths — {avg_lengths}")
    print(f"Overall mean: {mean_length:.1f}")

    for col, avg_len in avg_lengths.items():
        if avg_len > mean_length * 1.2:
            target_tokens = int(mean_length)
            print(f"  Truncating '{col}' from avg {avg_len:.1f} to ~{target_tokens} tokens")
            df[col] = df[col].apply(
                lambda x: " ".join(str(x).split()[:target_tokens]) if pd.notna(x) else x
            )

    return df


def create_variants(df):
    """Create the 4 dataset variants from WikiEval data."""
    df = normalize_answer_lengths(df)

    variants = {}

    # Variant A (Gold): question, context_v1, answer
    variants["A"] = pd.DataFrame({
        "question": df["question"],
        "contexts": df["context_v1"].apply(ensure_list),
        "answer": df["answer"],
        "ground_truth": df["answer"],
    })

    # Variant B (Hallucination): question, context_v1, ungrounded_answer
    variants["B"] = pd.DataFrame({
        "question": df["question"],
        "contexts": df["context_v1"].apply(ensure_list),
        "answer": df["ungrounded_answer"],
        "ground_truth": df["answer"],
    })

    # Variant C (Poor Quality): question, context_v1, poor_answer
    variants["C"] = pd.DataFrame({
        "question": df["question"],
        "contexts": df["context_v1"].apply(ensure_list),
        "answer": df["poor_answer"],
        "ground_truth": df["answer"],
    })

    # Variant D (Noisy Context): question, context_v2, answer
    variants["D"] = pd.DataFrame({
        "question": df["question"],
        "contexts": df["context_v2"].apply(ensure_list),
        "answer": df["answer"],
        "ground_truth": df["answer"],
    })

    return variants


def save_variants(variants):
    """Save variant DataFrames to CSV files."""
    os.makedirs(DATA_DIR, exist_ok=True)

    for name, df in variants.items():
        path = os.path.join(DATA_DIR, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"Saved Variant {name}: {len(df)} rows -> {path}")


def main():
    df = load_wikieval()
    variants = create_variants(df)
    save_variants(variants)

    # Validation
    print("\n--- Validation ---")
    for name, vdf in variants.items():
        print(f"Variant {name}: {len(vdf)} rows, columns={list(vdf.columns)}")
        assert len(vdf) == N_SAMPLES, f"Expected {N_SAMPLES} rows, got {len(vdf)}"
        assert "question" in vdf.columns
        assert "contexts" in vdf.columns
        assert "answer" in vdf.columns
    print("All variants validated successfully.")


if __name__ == "__main__":
    main()
