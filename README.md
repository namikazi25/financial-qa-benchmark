# Financial QA Benchmark

**Evaluating LLMs for grounded financial question answering on 10-K filings.**

![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Models](https://img.shields.io/badge/Models-GPT--4o--mini%20%7C%20Gemini%202.5%20Flash--Lite-orange)
![Judge](https://img.shields.io/badge/Judge-Claude%20Sonnet%204.5-purple)

A production-grade evaluation pipeline comparing two cost-optimized LLMs on financial question answering. Features a curated 50-sample benchmark (40 standard + 10 adversarial edge cases), multi-metric evaluation via LLM-as-judge, numeric consistency scoring, and BERTScore.

Built with: **Python** · **OpenRouter** · **structured logging** · **YAML config** · **Makefile automation**

---

## Results

| Metric | GPT-4o-mini | Gemini 2.5 Flash-Lite |
|--------|:-----------:|:---------------------:|
| Avg Judge Score (/9) | 8.10 | **8.55** |
| Correct Answers (>=7/9) | 36 / 50 | **41 / 50** |
| Numeric Consistency | 0.443 | **0.677** |
| Avg Latency | 1.71s | **0.87s** |
| Cost per Correct Answer | $0.00012 | **$0.00008** |
| Edge-Case Abstention | **60%** | 40% |

**Recommendation:** Gemini 2.5 Flash-Lite for production financial QA. Higher accuracy, better numeric precision, 2x faster, 25% cheaper. GPT-4o-mini is more conservative on broken context (higher abstention), which suits deployments where avoiding unsupported answers is critical.

---

## Pipeline Architecture

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐     ┌──────────────┐
│ Dataset Curation │────>│  Answer Generation   │────>│   LLM-as-Judge      │────>│   Analysis   │
│                  │     │                      │     │                     │     │              │
│ 40 standard      │     │ GPT-4o-mini          │     │ Claude Sonnet 4.5   │     │ Judge scores │
│ 10 edge cases    │     │ Gemini Flash-Lite    │     │ Correctness (1-3)   │     │ Numeric cons │
│ Stratified by    │     │ Identical prompts    │     │ Completeness (1-3)  │     │ BERTScore    │
│ question type    │     │ JSON enforced        │     │ Groundedness (1-3)  │     │ Cost/latency │
└─────────────────┘     └──────────────────────┘     └─────────────────────┘     └──────────────┘
```

---

## Key Design Decisions

### Why LLM-as-judge over BLEU/ROUGE?
Financial answers can be semantically equivalent with different phrasing. "$1.35 billion" and "$1.35B" mean the same thing but BLEU scores them as different. An LLM judge evaluates meaning, not string overlap.

### Why a neutral third-party judge?
Using GPT to judge GPT introduces self-serving bias. Claude Sonnet 4.5 is independent of both answer models.

### Why 40 standard + 10 edge cases?
Standard rows test core QA quality. Edge cases (truncated context, sparse tables, empty fields) test whether models abstain rather than hallucinate. The 80/20 split deliberately oversamples adversarial inputs.

### Why numeric consistency as a separate metric?
In 10-K filings, the exact dollar amount matters. The judge might score a paraphrase highly, but if "$14.16 per diluted share" became "approximately $14", the number is wrong. Regex extraction catches this independently.

---

## Notable Failure Cases

**Sample 17 -- Table-format context (Americas net revenue %)**
- Gemini extracted both years and direction of change -> 9/9
- GPT-4o-mini returned only "79.3%", missed the 2022 comparison -> 6/9
- *Insight: Gemini handles sparse tabular context better*

**Sample 44 -- Unnecessary abstention**
- GPT-4o-mini abstained despite sufficient context -> lost points
- Gemini answered correctly -> 7/9
- *Insight: GPT-4o-mini is over-cautious on ambiguous but answerable rows*

**Sample 16 -- Correct abstention (context: "Gothia")**
- Both models correctly abstained on garbled context
- *Insight: Abstention logic works when context is genuinely insufficient*

---

## Engineering Principles

- **Structured logging** over print statements -- configurable verbosity via `config.yaml`, file + console output
- **Externalized configuration** -- all thresholds, paths, model IDs, and delays in `config.yaml`, not hardcoded
- **Reproducible by default** -- pre-generated outputs committed, `uv.lock` pinned, `Makefile` for one-command runs
- **Separation of concerns** -- prompts, models, evaluation, and analysis in separate modules
- **Type-safe config** -- dataclass-based config loader with validation on startup

---

## Quick Start

```bash
make setup
```

Create `.env`:
```
OPENROUTER_API_KEY=your_key_here
```

Download dataset from [Kaggle](https://www.kaggle.com/datasets/yousefsaeedian/financial-q-and-a-10k) -> `data/Financial-QA-10k.csv`

```bash
make run        # Full pipeline
make evaluate   # Judge only
make bertscore  # BERTScore only
make lint       # Ruff check
```

Pre-generated outputs are in `outputs/` if you want to inspect results without API calls.

---

## Configuration

All tunable parameters live in [`config.yaml`](config.yaml):

- **logging:** level, directory, filename
- **dataset:** paths, sample counts (40 standard / 10 edge)
- **models:** identifiers, temperature, max tokens, retries, cost table
- **pipeline:** output paths, API call delays
- **evaluation:** judge threshold (>=7/9), summary path
- **bertscore:** model name, output path

Secrets stay in `.env` and are never committed.

---

## Project Structure

```
├── config.yaml              # All tunable parameters
├── config.py                # Dataclass config loader
├── logger.py                # Shared logging setup
├── Makefile                 # Build/run/lint targets
├── dataset.py               # Benchmark curation (40+10 split)
├── prompts.py               # Answer + judge prompt templates
├── models.py                # OpenRouter API interface
├── main.py                  # Answer generation pipeline
├── evaluate.py              # LLM-as-judge + numeric consistency
├── bertscore.py             # BERTScore computation
├── data/                    # Raw + curated benchmark
└── outputs/                 # All prediction and evaluation artifacts
```

---

## Cost

Generation: ~$0.004 per model (50 rows). Judge: ~$0.16 per model (Claude Sonnet 4.5). BERTScore: free (local). Total benchmark cost under $0.40.

---

## Limitations

- 50 samples is directional, not statistically conclusive
- Benchmark selection is heuristic-based, not human-annotated
- LLM-as-judge scores are indicative, not ground truth
- Edge cases intentionally overrepresented vs production distribution
