# RAGAS LLM Judge Reliability Study

Can you trust an AI to mark another AI's answers?

RAGAS scores a RAG system by asking an LLM (the "judge") to mark its answers. This study tests the test: if you change only the judge model, do the scores change? Master of Information Technology applied research, Auckland Institute of Studies, 2026. Supervisor: Dr. Shabana Ali.

## Result

**RAGAS is reliable for comparing systems, but not for absolute scores.**

- **The judges agree on the order.** Rank agreement is strong for Faithfulness (Spearman 0.82) and Answer Correctness (0.89). That is enough to compare two versions of a system or to catch a regression.
- **The judges do not agree on the number.** Changing the judge moves a score by about 0.1 on a 0 to 1 scale (mean absolute error 0.097). A pass mark set with one judge does not carry to another.
- **Answer Relevancy cannot detect hallucination.** Its score barely moved when answers were made up, on all three judges. Faithfulness is the metric that catches it.
- **The judges catch bad answers.** In 15 of 18 comparisons, a deliberately degraded answer set scored significantly lower than the gold set. The 3 misses are all Answer Relevancy on hallucinated answers.

## Design

**Dataset:** [WikiEval](https://huggingface.co/datasets/vibrantlabsai/WikiEval), 50 questions, rebuilt into four variants.

| Variant | What it is |
|---|---|
| A, Gold | Correct answer with relevant context |
| B, Hallucination | Answer not supported by the context |
| C, Poor quality | Incomplete answer |
| D, Noisy | Correct answer with irrelevant context added |

**Judges:** GPT-4o mini (OpenAI), Gemini 2.5 Flash-Lite (Google), Llama 3.1 8B (Cerebras). Temperature 0, and one shared embedding model (`text-embedding-3-small`) so the judge is the only variable.

**Metrics:** Faithfulness, Answer Relevancy, Answer Correctness (RAGAS 0.4.3).

**Size:** 4 variants × 3 judges × 3 metrics × 50 rows = 1,800 scores.

**Analysis:** mean absolute error for score agreement, Mann-Whitney U and Kruskal-Wallis for sensitivity, Spearman rank correlation for ranking stability.

## Numbers

Mean score per judge, metric and variant:

| Judge | Metric | A Gold | B Halluc. | C Poor | D Noisy |
|---|---|---|---|---|---|
| Gemini | Faithfulness | 0.974 | 0.449 | 0.444 | 1.000 |
| Gemini | Answer Relevancy | 0.876 | 0.847 | 0.648 | 0.876 |
| Gemini | Answer Correctness | 1.000 | 0.431 | 0.293 | 1.000 |
| GPT-4o mini | Faithfulness | 0.980 | 0.431 | 0.444 | 0.997 |
| GPT-4o mini | Answer Relevancy | 0.862 | 0.839 | 0.586 | 0.861 |
| GPT-4o mini | Answer Correctness | 1.000 | 0.498 | 0.393 | 1.000 |
| Llama 3.1 | Faithfulness | 0.872 | 0.306 | 0.348 | 0.866 |
| Llama 3.1 | Answer Relevancy | 0.825 | 0.826 | 0.571 | 0.825 |
| Llama 3.1 | Answer Correctness | 0.990 | 0.565 | 0.443 | 0.990 |

Look at the Answer Relevancy rows: variant B (hallucinated) scores almost the same as variant A (gold) for every judge. That is the blind spot.

Mean absolute error between judges (lower means closer agreement):

| Metric | Gemini vs GPT | Gemini vs Llama | GPT vs Llama |
|---|---|---|---|
| Faithfulness | 0.070 | 0.169 | 0.170 |
| Answer Relevancy | 0.080 | 0.096 | 0.066 |
| Answer Correctness | 0.051 | 0.093 | 0.080 |

Average rank agreement (Spearman): Faithfulness 0.816, Answer Correctness 0.893, Answer Relevancy 0.594.

## What to do with this

- Treat a RAGAS score as relative. Allow about ±0.1 before reading anything into a difference.
- Use Faithfulness, not Answer Relevancy, to check that an answer is grounded.
- Pin the judge model and version. A silent model update changes your scores.
- For a decision that matters, use more than one judge. Where judges disagree widely on a sample, send that sample to a person.

## Limits

One dataset from one domain (Wikipedia questions), 50 rows per variant, three lightweight judges, no human-marked baseline, and 4.3% of scores came back empty and were excluded.

## Code

```
src/
  data_prep.py   Builds the four dataset variants from WikiEval
  judges.py      Sets up the judge clients
  main.py        Runs the evaluation
RESEARCH_PLAN.md Method and architecture notes
```

This repository currently holds the pipeline from the first phase of the project. The full tournament run, the analysis script and the raw scores are being added.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.template .env        # add your API keys
python src/data_prep.py
python src/judges.py
python src/main.py
```
