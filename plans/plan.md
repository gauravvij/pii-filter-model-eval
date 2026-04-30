# GLiNER large-v2.1 vs openai/privacy-filter — Upgraded Unbiased Evaluation v2

## Goal
Produce a rigorous, unbiased, standardized comparison of `urchade/gliner_large-v2.1` vs `openai/privacy-filter` on PII/privacy filtering across English AND multilingual samples, using three industry-standard scoring tiers, GLiNER threshold tuning, and a scaled-up 600-sample eval set.

## Research Summary
- **Bias root cause identified**: The previous run used strict exact character-span match only. `openai/privacy-filter` uses a GPT-style BPE tokenizer that adds leading whitespace to tokens (e.g., `" Alice"` vs `"Alice"`), causing systematic offset drift of 1 character. This is a measurement artefact — not a model failure — and unfairly penalises the OpenAI model.
- **Standard fix**: Use the **MUC/SemEval three-tier scoring framework** — the industry standard for NER evaluation:
  - **Strict**: exact (start, end, label) match — what we had before
  - **Boundary**: (start, end) overlap match regardless of exact offsets — fixes tokenizer drift
  - **Partial**: any character overlap between predicted and gold span (same label) — most lenient
- **MUC partial scoring**: COR + 0.5×PAR in numerator, giving credit for partial overlaps
- **GLiNER threshold tuning**: sweep threshold ∈ {0.3, 0.4, 0.5, 0.6, 0.7} on a held-out 50-sample dev set, pick best macro F1 threshold for the main eval
- **Scale-up**: 600 samples total — 400 English + 200 multilingual (French, German, Spanish, Italian, Dutch — all supported by ai4privacy dataset)
- **Multilingual note**: openai/privacy-filter docs explicitly state "performance may drop on non-English text" — this is a documented characteristic, not a bias. We test both models equally on the same multilingual samples.
- **GLiNER multilingual**: urchade/gliner_large-v2.1 supports multilingual NER natively.
- **Dataset**: `ai4privacy/pii-masking-400k` — 81,379 validation samples across 6 languages. Enough for 600 samples easily.
- **Fairness principles applied**:
  1. Same dataset, same samples, same gold labels for both models
  2. Three scoring tiers so neither tokenizer architecture is disadvantaged
  3. GLiNER threshold tuned on dev set (not test set) to avoid overfitting
  4. Multilingual samples evaluated separately from English — no mixing of language strata in final tables
  5. URL category (0 gold spans in prior run) — will be reported as "insufficient data" if still <5 gold spans

## Approach
Complete rewrite of all pipeline scripts to implement:
1. Scaled dataset: 400 English + 200 multilingual (balanced across available non-English languages)
2. GLiNER threshold sweep on 50-sample dev set → best threshold selected
3. Three-tier MUC/SemEval scoring: Strict / Boundary / Partial
4. Separate result tables: English-only and Multilingual
5. Full report with per-tier, per-language, per-category breakdown

## Subtasks

### 1. Environment verification
Verify the existing venv at `/root/gliner_vs_openai/venv` has all required packages (gliner, transformers dev, datasets, pandas, matplotlib, tqdm, seqeval). Install any missing ones. Confirm both models load cleanly. Expected output: clean import test, no errors.

### 2. Dataset loading v2 — scaled + multilingual
Load `ai4privacy/pii-masking-400k` validation split. Sample:
- 50 English samples → `data/dev_samples.jsonl` (GLiNER threshold tuning only, NOT used in final eval)
- 400 English samples → `data/eval_en_samples.jsonl`
- 200 non-English samples → `data/eval_ml_samples.jsonl` (balanced across French, German, Spanish, Italian, Dutch — ~40 per language)
All three sets must be disjoint (no overlap). Parse `privacy_mask` JSON → character-level gold spans, apply 6-category label mapping. Save all three files.
Verify: correct sample counts, no ID overlap between sets, spot-check spans.

### 3. GLiNER threshold tuning
Load `urchade/gliner_large-v2.1`. Run `predict_entities` on the 50 dev samples with each threshold in `[0.3, 0.4, 0.5, 0.6, 0.7]`. For each threshold, compute macro F1 (strict match) on the dev set. Select the threshold with the highest macro F1. Save threshold sweep results to `results/gliner_threshold_sweep.csv`.
Verify: CSV has 5 rows with valid F1 values, best threshold identified and printed.

### 4. GLiNER inference on eval sets
Using the best threshold from subtask 3, run GLiNER large-v2.1 on:
- `data/eval_en_samples.jsonl` → `results/gliner_en_predictions.jsonl`
- `data/eval_ml_samples.jsonl` → `results/gliner_ml_predictions.jsonl`
Labels: `["person name", "email address", "phone number", "street address", "url", "date"]`
Map GLiNER output labels to common schema. Save predictions.
Verify: both files exist, non-empty predictions, entity types vary.

### 5. openai/privacy-filter inference on eval sets
Load `openai/privacy-filter` with `AutoModelForTokenClassification` + `AutoTokenizer` (trust_remote_code=True, dtype=torch.float32). Run on:
- `data/eval_en_samples.jsonl` → `results/openai_en_predictions.jsonl`
- `data/eval_ml_samples.jsonl` → `results/openai_ml_predictions.jsonl`
Decode BIES tag sequences (B/I/E/S) → character-level spans using offset_mapping from tokenizer. Map to common schema. Save predictions.
Verify: both files exist, non-empty predictions, entity types vary.

### 6. Three-tier evaluation & scoring
Implement MUC/SemEval three-tier span scoring for both models on both eval sets:

**Tier 1 — Strict**: (start, end, label) must all match exactly. TP only if perfect match.

**Tier 2 — Boundary** (fixes tokenizer offset drift): predicted span overlaps gold span AND labels match. Overlap = predicted interval and gold interval share at least 1 character. This is the fair metric for comparing tokenizer-based vs span-extraction models.

**Tier 3 — Partial (MUC)**: any character overlap between predicted and gold span (same label). Score = (COR + 0.5×PAR) / total, where COR=exact match, PAR=partial overlap. This is the most lenient and gives partial credit.

Compute P/R/F1 for each: model × tier × language_group × category.
Save to:
- `results/metrics_en.csv` (English results, all tiers)
- `results/metrics_ml.csv` (Multilingual results, all tiers)
Verify: no all-zero rows for both models under Boundary scoring (if still zero, investigate).

### 7. Report generation
Produce `results/evaluation_report_v2.md` containing:
- Methodology section explaining the three-tier approach and why it eliminates tokenizer bias
- GLiNER threshold sweep table
- **English results**: per-category P/R/F1 table for all three tiers, both models
- **Multilingual results**: per-category P/R/F1 table for all three tiers, both models
- **Summary table**: macro F1 by model × tier × language group
- Key findings narrative: which model wins under which scoring tier and why
- Limitations section

Produce `results/f1_comparison_v2.png`: grouped bar chart showing macro F1 for both models across all three tiers (English + Multilingual side by side).

Verify: report file ≥ 5KB, chart file exists, all tables populated with non-trivial values.

## Deliverables
| File Path | Description |
|-----------|-------------|
| `/root/gliner_vs_openai/data/dev_samples.jsonl` | 50 English dev samples for threshold tuning |
| `/root/gliner_vs_openai/data/eval_en_samples.jsonl` | 400 English eval samples |
| `/root/gliner_vs_openai/data/eval_ml_samples.jsonl` | 200 multilingual eval samples |
| `/root/gliner_vs_openai/results/gliner_threshold_sweep.csv` | GLiNER threshold sweep results |
| `/root/gliner_vs_openai/results/gliner_en_predictions.jsonl` | GLiNER predictions on English set |
| `/root/gliner_vs_openai/results/gliner_ml_predictions.jsonl` | GLiNER predictions on multilingual set |
| `/root/gliner_vs_openai/results/openai_en_predictions.jsonl` | openai/privacy-filter predictions on English set |
| `/root/gliner_vs_openai/results/openai_ml_predictions.jsonl` | openai/privacy-filter predictions on multilingual set |
| `/root/gliner_vs_openai/results/metrics_en.csv` | English P/R/F1 — all tiers, both models |
| `/root/gliner_vs_openai/results/metrics_ml.csv` | Multilingual P/R/F1 — all tiers, both models |
| `/root/gliner_vs_openai/results/evaluation_report_v2.md` | Full unbiased evaluation report |
| `/root/gliner_vs_openai/results/f1_comparison_v2.png` | Comparison bar chart |
| `/root/gliner_vs_openai/src/load_dataset_v2.py` | Updated dataset loading script |
| `/root/gliner_vs_openai/src/tune_gliner_threshold.py` | GLiNER threshold tuning script |
| `/root/gliner_vs_openai/src/run_gliner_v2.py` | Updated GLiNER inference script |
| `/root/gliner_vs_openai/src/run_openai_filter_v2.py` | Updated openai inference script |
| `/root/gliner_vs_openai/src/evaluate_v2.py` | Three-tier evaluation script |
| `/root/gliner_vs_openai/src/generate_report_v2.py` | Report generation script |

## Evaluation Criteria
- GLiNER threshold sweep produces a best threshold ≠ default 0.5 OR confirms 0.5 is optimal
- Both models produce non-trivially varied predictions on both language sets
- Under Boundary scoring, openai/privacy-filter PERSON F1 > 0.05 (if still 0, investigate tokenizer offset decoding)
- Multilingual results exist for both models with non-zero predictions
- Report clearly states which model wins under each scoring tier
- No silent failures — any model scoring 0% on all categories under any tier triggers investigation

## Notes
- Use existing venv: `source /root/gliner_vs_openai/venv/bin/activate`
- Dev set (50 samples) must NOT be included in the English eval set (400 samples) — use disjoint indices
- openai/privacy-filter requires `trust_remote_code=True` and transformers dev (already installed)
- BIES decoding: B=begin multi-token, I=inside, E=end multi-token, S=single-token entity
- Offset mapping from HuggingFace tokenizer gives exact (char_start, char_end) per token — use this for character span recovery, do NOT reconstruct from token strings
- For multilingual GLiNER inference: use the same English label prompts (GLiNER handles cross-lingual matching natively)
- CPU only — batch size 1 for openai model, GLiNER processes one text at a time
