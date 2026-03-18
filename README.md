# RAGAS LLM Judge Reliability Study

A research project investigating the reliability and consistency of RAGAS (Retrieval-Augmented Generation Assessment) metrics when evaluated by different LLM judges.

Conducted at the **Auckland Institute of Studies**.

## Research Objective

Determine whether RAGAS metric scores are significantly influenced by the choice of underlying LLM judge. The study evaluates scoring consistency, sensitivity to answer quality degradation, and inter-judge agreement across controlled dataset variants.

## Experimental Design

**Dataset:** `vibrantlabsai/WikiEval` (N=50), structured into four controlled variants:

| Variant | Description |
|---------|-------------|
| A — Gold | Original question, context, and correct answer |
| B — Hallucination | Ungrounded answer not supported by context |
| C — Poor Quality | Low-quality answer with partial information |
| D — Noisy | Original answer evaluated against a noisier context |

**LLM Judges:**

| Judge | Model | Provider |
|-------|-------|----------|
| Judge 1 | gemini-2.5-flash | Google AI |
| Judge 2 | llama-3.1-8b-instant | Groq |

**Metrics:** Faithfulness, Answer Relevance, Answer Correctness

**Statistical Methods:** Spearman's Rank Correlation (ranking agreement), Mean Absolute Error (absolute score variance)

## Project Structure

```
.
├── src/
│   ├── data_prep.py     # Loads WikiEval, creates 4 dataset variants with bias mitigation
│   ├── judges.py        # Initializes LLM judge clients with error handling
│   └── main.py          # Evaluation pipeline (PoC: Faithfulness on Variant A, N=5)
├── data/
│   └── variants/        # Generated CSV files: A.csv, B.csv, C.csv, D.csv
├── results/             # Evaluation output (e.g. test_run.csv)
├── RESEARCH_PLAN.md     # Architecture and methodology documentation
├── requirements.txt
├── .env.template        # Environment variable template
└── .gitignore
```

## Setup

**1. Clone and create a virtual environment:**
```bash
git clone https://github.com/NPZlatu/RAGAS-LLM-Judge-Reliability-Study.git
cd RAGAS-LLM-Judge-Reliability-Study
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Configure API keys:**
```bash
cp .env.template .env
# Add your GOOGLE_API_KEY and GROQ_API_KEY to .env
```

## Usage

Run all commands from the project root with the venv activated:

```bash
python src/data_prep.py   # Generate dataset variants
python src/judges.py      # Verify judge connectivity
python src/main.py        # Run evaluation pipeline
```

## Phase 1 Results (PoC)

Faithfulness on Variant A (Gold), N=5:

| Judge | Mean Faithfulness |
|-------|-------------------|
| Gemini 2.5 Flash | 1.0 |
| Llama 3.1 8B (Groq) | 1.0 |

Both judges correctly scored gold-standard answers at maximum faithfulness, validating the pipeline before full-scale evaluation.

## Bias Mitigation

- **Verbosity bias:** Answer length normalized across variants
- **Position bias:** Consistent context-answer ordering
- **Non-determinism:** Temperature set to 0 across all judges

## Roadmap

- [x] Phase 1 — Pipeline setup, bias mitigation, PoC evaluation (Faithfulness, Variant A, N=5)
- [ ] Phase 2 — Full evaluation (all 4 variants, all 3 metrics, N=50)
- [ ] Phase 3 — Statistical analysis and reporting (Spearman correlation, MAE)
