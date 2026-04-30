"""
Subtask 5: Evaluation & scoring
- Exact character span match scoring
- Compute TP/FP/FN per category per model
- Derive Precision/Recall/F1
- Save to /root/gliner_vs_openai/results/metrics.csv
"""

import json
import os
import csv
from collections import defaultdict

CATEGORIES = ["PERSON", "EMAIL", "PHONE", "ADDRESS", "URL", "DATE"]


def load_jsonl(path):
    records = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            records[rec["id"]] = rec
    return records


def spans_to_set(spans, label=None):
    """Convert list of span dicts to a set of (start, end, label) tuples."""
    result = set()
    for s in spans:
        if label is None or s["label"] == label:
            result.add((s["start"], s["end"], s["label"]))
    return result


def compute_metrics(gold_records, pred_records):
    """
    Compute TP/FP/FN per category using exact character span match.
    Returns dict: {category: {tp, fp, fn, precision, recall, f1}}
    """
    # Per-category accumulators
    tp_counts = defaultdict(int)
    fp_counts = defaultdict(int)
    fn_counts = defaultdict(int)

    for sample_id, gold_rec in gold_records.items():
        pred_rec = pred_records.get(sample_id, {"predictions": []})

        gold_spans = gold_rec.get("gold_spans", [])
        pred_spans = pred_rec.get("predictions", [])

        for cat in CATEGORIES:
            gold_set = spans_to_set(gold_spans, label=cat)
            pred_set = spans_to_set(pred_spans, label=cat)

            tp = len(gold_set & pred_set)
            fp = len(pred_set - gold_set)
            fn = len(gold_set - pred_set)

            tp_counts[cat] += tp
            fp_counts[cat] += fp
            fn_counts[cat] += fn

    # Compute P/R/F1 per category
    results = {}
    for cat in CATEGORIES:
        tp = tp_counts[cat]
        fp = fp_counts[cat]
        fn = fn_counts[cat]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[cat] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    # Overall (macro average)
    macro_p = sum(results[c]["precision"] for c in CATEGORIES) / len(CATEGORIES)
    macro_r = sum(results[c]["recall"] for c in CATEGORIES) / len(CATEGORIES)
    macro_f1 = sum(results[c]["f1"] for c in CATEGORIES) / len(CATEGORIES)

    # Overall micro (aggregate TP/FP/FN)
    total_tp = sum(tp_counts[c] for c in CATEGORIES)
    total_fp = sum(fp_counts[c] for c in CATEGORIES)
    total_fn = sum(fn_counts[c] for c in CATEGORIES)
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    results["MACRO_AVG"] = {
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "precision": round(macro_p, 4),
        "recall": round(macro_r, 4),
        "f1": round(macro_f1, 4),
    }
    results["MICRO_AVG"] = {
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "precision": round(micro_p, 4),
        "recall": round(micro_r, 4),
        "f1": round(micro_f1, 4),
    }

    return results


def main():
    os.makedirs("/root/gliner_vs_openai/results", exist_ok=True)

    # Load gold labels
    print("Loading gold labels...")
    gold_records = load_jsonl("/root/gliner_vs_openai/data/eval_samples.jsonl")
    print(f"  Loaded {len(gold_records)} gold records")

    # Load predictions
    print("Loading GLiNER predictions...")
    gliner_preds = load_jsonl("/root/gliner_vs_openai/results/gliner_predictions.jsonl")
    print(f"  Loaded {len(gliner_preds)} GLiNER prediction records")

    print("Loading openai/privacy-filter predictions...")
    openai_preds = load_jsonl("/root/gliner_vs_openai/results/openai_predictions.jsonl")
    print(f"  Loaded {len(openai_preds)} openai prediction records")

    # Compute metrics
    print("\nComputing GLiNER metrics...")
    gliner_metrics = compute_metrics(gold_records, gliner_preds)

    print("Computing openai/privacy-filter metrics...")
    openai_metrics = compute_metrics(gold_records, openai_preds)

    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS — Exact Character Span Match")
    print("="*80)

    header = f"{'Category':<15} {'GLiNER P':>10} {'GLiNER R':>10} {'GLiNER F1':>10} | {'OpenAI P':>10} {'OpenAI R':>10} {'OpenAI F1':>10}"
    print(header)
    print("-" * len(header))

    all_cats = CATEGORIES + ["MACRO_AVG", "MICRO_AVG"]
    for cat in all_cats:
        gm = gliner_metrics[cat]
        om = openai_metrics[cat]
        print(f"{cat:<15} {gm['precision']:>10.4f} {gm['recall']:>10.4f} {gm['f1']:>10.4f} | {om['precision']:>10.4f} {om['recall']:>10.4f} {om['f1']:>10.4f}")

    # Save to CSV
    output_path = "/root/gliner_vs_openai/results/metrics.csv"
    fieldnames = ["model", "category", "tp", "fp", "fn", "precision", "recall", "f1"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for cat in all_cats:
            gm = gliner_metrics[cat]
            writer.writerow({
                "model": "gliner_large_v2.1",
                "category": cat,
                "tp": gm["tp"],
                "fp": gm["fp"],
                "fn": gm["fn"],
                "precision": gm["precision"],
                "recall": gm["recall"],
                "f1": gm["f1"],
            })

        for cat in all_cats:
            om = openai_metrics[cat]
            writer.writerow({
                "model": "openai_privacy_filter",
                "category": cat,
                "tp": om["tp"],
                "fp": om["fp"],
                "fn": om["fn"],
                "precision": om["precision"],
                "recall": om["recall"],
                "f1": om["f1"],
            })

    print(f"\n✅ Metrics saved to {output_path}")

    # Sanity check: verify no all-zero rows for both models
    print("\n=== SANITY CHECK ===")
    gliner_all_zero = all(gliner_metrics[c]["f1"] == 0.0 for c in CATEGORIES)
    openai_all_zero = all(openai_metrics[c]["f1"] == 0.0 for c in CATEGORIES)
    print(f"  GLiNER all-zero F1: {gliner_all_zero} {'⚠️ PROBLEM' if gliner_all_zero else '✅ OK'}")
    print(f"  OpenAI all-zero F1: {openai_all_zero} {'⚠️ PROBLEM' if openai_all_zero else '✅ OK'}")

    # Check gold span counts
    total_gold = sum(len(r["gold_spans"]) for r in gold_records.values())
    total_gliner = sum(len(r["predictions"]) for r in gliner_preds.values())
    total_openai = sum(len(r["predictions"]) for r in openai_preds.values())
    print(f"\n  Total gold spans: {total_gold}")
    print(f"  Total GLiNER predictions: {total_gliner}")
    print(f"  Total OpenAI predictions: {total_openai}")

    # Per-category gold counts
    print("\n  Gold span distribution:")
    gold_cat_counts = defaultdict(int)
    for rec in gold_records.values():
        for span in rec["gold_spans"]:
            gold_cat_counts[span["label"]] += 1
    for cat in CATEGORIES:
        print(f"    {cat}: {gold_cat_counts[cat]}")

    return gliner_metrics, openai_metrics


if __name__ == "__main__":
    main()
