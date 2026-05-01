# GLiNER large-v2.1 vs openai/privacy-filter — PII Detection Evaluation

> **Built and evaluated with [Neo](https://heyneo.com)** — an autonomous AI engineering agent.

A rigorous, reproducible evaluation comparing two open-weight PII detection models on the `ai4privacy/pii-masking-400k` dataset. Uses a three-tier MUC/SemEval scoring framework to eliminate tokenizer-induced measurement bias.

Read the full write-up: [blog.md](blog.md)

---

## What This Is

Most model comparisons use a single metric. When comparing a span-extraction model (GLiNER) against a token classifier (openai/privacy-filter), that single metric — strict exact character-span match — quietly favours the span extractor because BPE tokenizers introduce systematic ±1 character offsets. This pipeline fixes that with three scoring tiers:

- **Strict**: exact `(start, end, label)` match
- **Boundary**: span overlap + label match (the fair metric)
- **Partial (MUC)**: any overlap with partial credit

**Key finding**: Under strict scoring, GLiNER wins (0.37 vs 0.15 macro F1). Under boundary scoring — the fair metric — openai/privacy-filter wins (0.50 vs 0.42 on English, 0.56 vs 0.45 on multilingual).

---

## Models Evaluated

| Model | HuggingFace ID | Type | Parameters |
|---|---|---|---|
| GLiNER large-v2.1 | `urchade/gliner_large-v2.1` | Zero-shot NER | ~300M |
| OpenAI Privacy Filter | `openai/privacy-filter` | Fine-tuned token classifier | 1.5B total / ~50M active (MoE) |

---

## Dataset

`ai4privacy/pii-masking-400k` — 406K synthetic multilingual PII entries. We use the validation split only.

- 50 English samples: GLiNER threshold tuning (dev set)
- 400 English samples: main English evaluation
- 200 multilingual samples: 40 each from French, German, Spanish, Italian, Dutch

---

## Results Summary

### English (400 samples)

| Model | Strict F1 | Boundary F1 | Partial F1 |
|---|---|---|---|
| GLiNER large-v2.1 | **0.3667** | 0.4162 | **0.3915** |
| openai/privacy-filter | 0.1547 | **0.4976** | 0.3261 |

### Multilingual (200 samples: FR/DE/ES/IT/NL)

| Model | Strict F1 | Boundary F1 | Partial F1 |
|---|---|---|---|
| GLiNER large-v2.1 | **0.3695** | 0.4508 | **0.4101** |
| openai/privacy-filter | 0.1813 | **0.5559** | 0.3686 |

![F1 Comparison Chart](results/f1_comparison_v2.png)

*Macro F1 across all three scoring tiers for both models on English (left group) and multilingual (right group) evaluation sets.*

### GLiNER Threshold Tuning

Best threshold: **0.7** (macro F1 = 0.5022 on dev set, vs 0.4225 at default 0.5)

---

## Project Structure

```
gliner_vs_openai/
├── src/
│   ├── load_dataset_v2.py        # Dataset loading and gold span extraction
│   ├── tune_gliner_threshold.py  # GLiNER threshold sweep on dev set
│   ├── run_gliner_v2.py          # GLiNER inference on eval sets
│   ├── run_openai_filter_v2.py   # openai/privacy-filter inference on eval sets
│   ├── evaluate_v2.py            # Three-tier MUC/SemEval scoring
│   └── generate_report_v2.py     # Report and chart generation
├── data/
│   ├── dev_samples.jsonl         # 50 English dev samples
│   ├── eval_en_samples.jsonl     # 400 English eval samples
│   └── eval_ml_samples.jsonl     # 200 multilingual eval samples
├── results/
│   ├── gliner_threshold_sweep.csv
│   ├── gliner_en_predictions.jsonl
│   ├── gliner_ml_predictions.jsonl
│   ├── openai_en_predictions.jsonl
│   ├── openai_ml_predictions.jsonl
│   ├── metrics_en.csv
│   ├── metrics_ml.csv
│   ├── f1_comparison_v2.png
│   └── evaluation_report_v2.md
├── plans/
│   └── plan.md
├── blog.md
└── README.md
```

---

## Setup and Reproduction

### Requirements

- Python 3.10+
- ~8GB RAM (for loading both models simultaneously)
- No GPU required (CPU inference)

### Install

```bash
# Clone or download this repo
cd gliner_vs_openai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install gliner datasets pandas matplotlib tqdm seqeval

# Install transformers dev branch (required for openai/privacy-filter)
# The model type 'openai_privacy_filter' is not in any stable transformers release yet
pip install git+https://github.com/huggingface/transformers.git
```

> **Note**: `gliner` pins `transformers<5.2.0` but works fine with the dev branch despite the version conflict warning. Both models load and run correctly.

### Run the Full Pipeline

```bash
source venv/bin/activate

# Step 1: Load and sample the dataset
python src/load_dataset_v2.py

# Step 2: Tune GLiNER threshold on dev set
python src/tune_gliner_threshold.py

# Step 3: Run GLiNER inference on eval sets
python src/run_gliner_v2.py

# Step 4: Run openai/privacy-filter inference on eval sets
python src/run_openai_filter_v2.py

# Step 5: Score with three-tier evaluation
python src/evaluate_v2.py

# Step 6: Generate report and chart
python src/generate_report_v2.py
```

Results are written to `results/`. The full report is `results/evaluation_report_v2.md`. The comparison chart is `results/f1_comparison_v2.png`.

### Expected Runtime (CPU only)

| Step | Approximate Time |
|---|---|
| Dataset loading | 2-5 min (first run downloads ~500MB) |
| GLiNER threshold tuning | 5-10 min |
| GLiNER inference (600 samples) | 15-20 min |
| openai/privacy-filter inference (600 samples) | 10-15 min |
| Evaluation + report | < 1 min |

---

## Extending This Evaluation

### Add a new model

1. Write a new inference script in `src/` following the pattern of `run_gliner_v2.py` or `run_openai_filter_v2.py`
2. Output predictions to `results/<model_name>_en_predictions.jsonl` and `results/<model_name>_ml_predictions.jsonl`
3. Each line: `{"id": <int>, "text": "...", "predictions": [{"start": <int>, "end": <int>, "label": "<CATEGORY>"}]}`
4. Add the new model to `evaluate_v2.py` and `generate_report_v2.py`

### Change the evaluation categories

Edit the `AI4PRIVACY_TO_COMMON` mapping in `src/load_dataset_v2.py` and the label mappings in the inference scripts. The evaluator in `evaluate_v2.py` uses whatever categories appear in the gold spans.

### Use a different dataset

Replace `src/load_dataset_v2.py` with a script that produces the same output format: JSONL files where each line has `{"id": <int>, "text": "...", "gold_spans": [{"start": <int>, "end": <int>, "label": "<CATEGORY>"}]}`.

### Run on more samples

Change the `N_EVAL_EN` and `N_EVAL_ML` constants in `src/load_dataset_v2.py`. The ai4privacy validation split has ~81K samples, so there is plenty of headroom.

---

## Built with Neo

This evaluation pipeline was built entirely by [Neo](https://neo.helixml.tech) — an autonomous AI engineering agent — from a single high-level prompt. Neo researched both models, identified the tokenizer bias problem, designed the three-tier scoring framework, wrote all the code, debugged failures (including a dataset ID issue that would have silently zeroed out all scores), and produced the final report and chart.

Neo works inside VS Code and Cursor as an extension. You give it a goal; it plans, codes, runs, and iterates until the work is done.

To extend this evaluation or build something new on top of it, describe what you want to Neo and it will take it from there.

---

## Citation / Attribution

Models:
- GLiNER: Zaratiana et al., "GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer" (2023)
- openai/privacy-filter: OpenAI (2025), Apache 2.0 license

Dataset:
- ai4privacy/pii-masking-400k: AI4Privacy (HuggingFace), CC BY 4.0

Evaluation framework:
- MUC/SemEval three-tier NER scoring as implemented in the `nervaluate` methodology (SemEval 2013 Task 9.1)
