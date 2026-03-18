# Research Plan: RAGAS LLM-Judge Reliability Study

## 1. Directory Structure

```
RAGAS-Evaluation/
├── .env                          # API keys (not committed)
├── .env.template                 # Template with placeholder keys
├── .gitignore
├── requirements.txt
├── RESEARCH_PLAN.md              # This file
├── PHASE1_IMPLEMENTATION_PLAN.md
├── data/
│   └── variants/
│       ├── A.csv                 # Gold standard triples
│       ├── B.csv                 # Hallucinated answers
│       ├── C.csv                 # Poor quality answers
│       └── D.csv                 # Noisy/irrelevant contexts
├── results/
│   └── test_run.csv              # Phase 1 PoC output
└── src/
    ├── data_prep.py              # Dataset loading & variant generation
    ├── judges.py                 # LLM judge initialization
    └── main.py                   # Evaluation execution
```

## 2. Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `data_prep.py` | Load WikiEval, apply bias mitigation (verbosity normalization), generate 4 variant CSVs |
| `judges.py` | Initialize 3 LLM judges via LangchainLLMWrapper, pre-flight checks, error handling |
| `main.py` | Load variant data, build RAGAS EvaluationDataset, run metrics across judges, save results |

## 3. Data Flow

```
WikiEval (HuggingFace)
    │
    ▼
data_prep.py
    ├── Normalize answer lengths (bias mitigation)
    ├── Format contexts as lists
    └── Output: data/variants/{A,B,C,D}.csv
          │
          ▼
main.py
    ├── Load variant CSV → EvaluationDataset
    ├── For each judge in judges.py:
    │     └── evaluate(dataset, metrics=[Faithfulness])
    └── Output: results/test_run.csv
```

## 4. Error & Rate-Limit Strategy

| Scenario | Handling |
|----------|----------|
| Groq rate limit (429) | Exponential backoff, 3 retries, 60s max wait |
| Gemini quota exhausted | Catch `ResourceExhausted`, log error, skip row |
| Ollama not running | Pre-flight `ollama list` check; clear error message |
| RAGAS parse failure | Catch per-sample exception, mark score as NaN, continue |
| Missing API key | Fail fast with specific error message identifying which key is missing |

## 5. Judges

| Judge | Provider | Model | Notes |
|-------|----------|-------|-------|
| 1 | Google AI (free) | gemini-2.5-flash | Requires GOOGLE_API_KEY |
| 2 | Groq (free) | llama-3.1-8b-instant | Requires GROQ_API_KEY |
| 3 | Local Ollama | llama3.2:3b | Requires `ollama pull llama3.2:3b` |

## 6. Metrics (Phase 1: Faithfulness only)

- **Faithfulness**: Measures whether the answer is grounded in the provided context
- Phase 2 will add: Answer Relevance, Answer Correctness
