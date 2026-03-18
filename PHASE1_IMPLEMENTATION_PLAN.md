# Phase 1 Implementation Plan: RAGAS LLM-Judge Reliability Study

## Context

This is Phase 1 of a research project studying the reliability of RAGAS metrics (Faithfulness, Answer Relevance, Answer Correctness) across different LLM judges. The goal of Phase 1 is to set up the environment, engineer the dataset variants, configure three LLM judges (all free), and run a proof-of-concept evaluation. No paid APIs are used.

---

## Judge Configuration (All Free)

| Judge | Provider | Model ID | Integration |
|-------|----------|----------|-------------|
| Judge 1 | Google AI (free tier) | `gemini-2.5-flash` | `langchain-google-genai` |
| Judge 2 | Groq (free tier) | `llama-3.1-8b-instant` | `langchain-groq` |
| Judge 3 | Local Ollama | `llama3.2:3b` (pull needed) | `langchain-ollama` |

---

## Directory Structure

```
RAGAS-Evaluation/
├── RESEARCH_PLAN.md           # Step 0: architecture doc (generated first)
├── PHASE1_IMPLEMENTATION_PLAN.md  # This file
├── .env                       # API keys (Gemini, Groq only) — DO NOT COMMIT
├── .env.template              # Template committed to repo
├── requirements.txt
├── data/
│   └── variants/
│       ├── A.csv              # Gold
│       ├── B.csv              # Hallucination
│       ├── C.csv              # Poor Quality
│       └── D.csv              # Noisy Context
├── results/
│   └── test_run.csv           # PoC output (N=5, Variant A, Faithfulness)
└── src/
    ├── data_prep.py           # Dataset loading + variant generation
    ├── judges.py              # LLM judge initialization
    └── main.py                # PoC execution script
```

---

## Step-by-Step Implementation

### Step 0: Generate `RESEARCH_PLAN.md`
- Create a markdown file documenting directory structure, module responsibilities, data flow, and error/rate-limit strategy
- This is the first deliverable — no code written until confirmed

### Step 1: Environment Setup
**Files to create:**
- `requirements.txt` with: `ragas`, `datasets`, `langchain-google-genai`, `langchain-groq`, `langchain-ollama`, `python-dotenv`, `pandas`, `scipy`
- `.env.template` with placeholders: `GOOGLE_API_KEY=`, `GROQ_API_KEY=`
- `.env` (user fills in their actual keys)

**Actions:**
- Pull Ollama model: `ollama pull llama3.2:3b`
- Create and activate venv: `python3 -m venv .venv && source .venv/bin/activate`
- Install: `pip install -r requirements.txt`

### Step 2: `src/data_prep.py`
- Load `vibrantlabsai/WikiEval` (split: `train`), take N=50
- Map to 4 variants:
  - **A (Gold):** `question`, `context_v1`, `answer`
  - **B (Hallucination):** `question`, `context_v1`, `ungrounded_answer`
  - **C (Poor Quality):** `question`, `context_v1`, `poor_answer`
  - **D (Noisy):** `question`, `context_v2`, `answer`
- Convert context columns to Python lists (RAGAS requirement)
- **Bias mitigation:** compute average token length of `answer`, `ungrounded_answer`, `poor_answer`; if any differs by >20% from the mean, truncate to normalize
- Save to `data/variants/{A,B,C,D}.csv`

### Step 3: `src/judges.py`
Initialize all three judges using `ragas.llms.LangchainLLMWrapper` with `temperature=0`:

```python
# Judge 1: Gemini 2.5 Flash (Google free tier)
from langchain_google_genai import ChatGoogleGenerativeAI
gemini_judge = LangchainLLMWrapper(ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0))

# Judge 2: Llama 3.1 8B (Groq free tier)
from langchain_groq import ChatGroq
groq_judge = LangchainLLMWrapper(ChatGroq(model="llama-3.1-8b-instant", temperature=0))

# Judge 3: Llama 3.2 3B (Local Ollama)
from langchain_ollama import ChatOllama
ollama_judge = LangchainLLMWrapper(ChatOllama(model="llama3.2:3b", temperature=0))
```

Error handling: wrap each initialization in try/except; fail fast with clear error messages if API key or Ollama connection is missing.

### Step 4: `src/main.py` (Proof of Concept)
- Load Variant A (N=5 rows only)
- Build RAGAS `EvaluationDataset` from the 5 rows
- Run `Faithfulness` metric against all 3 judges sequentially
- Collect results into a DataFrame with columns: `judge`, `question`, `faithfulness_score`
- Save to `results/test_run.csv`
- Print a summary table to stdout

---

## Rate Limit & Error Strategy

| Issue | Strategy |
|-------|----------|
| Groq rate limit hit | Exponential backoff with 3 retries (60s max wait) |
| Gemini quota exhausted | Catch `ResourceExhausted`, log and skip row |
| Ollama not running | Pre-flight check: `ollama list` before evaluation; clear error message |
| RAGAS parse failure | Catch per-sample exceptions; mark score as `NaN`, continue |

---

## Verification / Testing

1. **Data validation:** After running `data_prep.py`, check all 4 CSVs exist with 50 rows and correct columns
2. **Judge connectivity:** Run `judges.py` standalone to confirm all 3 judges initialize without error
3. **PoC run:** Execute `main.py` and verify `results/test_run.csv` is created with 15 rows (5 samples × 3 judges) and no all-NaN columns
4. **Ollama pre-check:** Confirm `ollama list` shows `llama3.2:3b` before running

---

## Keys You'll Need

- `GOOGLE_API_KEY` — from https://aistudio.google.com/app/apikey (free)
- `GROQ_API_KEY` — from https://console.groq.com/keys (free)
- No OpenAI key needed
