"""
run_gliner_v2.py — GLiNER inference on eval sets using best threshold from tuning.
"""
import json, os, csv
from gliner import GLiNER

PROJECT_DIR = "/root/gliner_vs_openai"
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
os.makedirs(RESULTS_DIR, exist_ok=True)

LABELS = ["person name", "email address", "phone number", "street address", "url", "date"]

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records)} records → {path}")

def get_best_threshold():
    sweep_path = os.path.join(RESULTS_DIR, "gliner_threshold_sweep.csv")
    best_thresh, best_f1 = 0.5, 0.0
    with open(sweep_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            f1 = float(row["macro_f1"])
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(row["threshold"])
    print(f"Best threshold from sweep: {best_thresh} (macro_f1={best_f1:.4f})")
    return best_thresh

def run_inference(model, samples, threshold):
    records = []
    for i, sample in enumerate(samples):
        text = sample["text"]
        preds = model.predict_entities(text, LABELS, threshold=threshold)
        predictions = [{"start": p["start"], "end": p["end"],
                        "label": p["label"], "score": round(p["score"], 4)} for p in preds]
        records.append({"id": sample["id"], "language": sample.get("language",""),
                        "text": text, "predictions": predictions})
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(samples)} processed")
    return records

def main():
    print("Loading GLiNER large-v2.1...")
    model = GLiNER.from_pretrained("urchade/gliner_large-v2.1")
    threshold = get_best_threshold()

    # English eval
    print(f"\nRunning GLiNER on English eval (threshold={threshold})...")
    en_samples = load_jsonl(os.path.join(DATA_DIR, "eval_en_samples.jsonl"))
    en_records = run_inference(model, en_samples, threshold)
    save_jsonl(en_records, os.path.join(RESULTS_DIR, "gliner_en_predictions.jsonl"))

    # Multilingual eval
    print(f"\nRunning GLiNER on Multilingual eval (threshold={threshold})...")
    ml_samples = load_jsonl(os.path.join(DATA_DIR, "eval_ml_samples.jsonl"))
    ml_records = run_inference(model, ml_samples, threshold)
    save_jsonl(ml_records, os.path.join(RESULTS_DIR, "gliner_ml_predictions.jsonl"))

    # Verify
    from collections import Counter
    en_labels = Counter(p["label"] for r in en_records for p in r["predictions"])
    ml_labels = Counter(p["label"] for r in ml_records for p in r["predictions"])
    en_nonempty = sum(1 for r in en_records if r["predictions"])
    ml_nonempty = sum(1 for r in ml_records if r["predictions"])
    print(f"\nEN: {en_nonempty}/{len(en_records)} samples with predictions, labels: {dict(en_labels)}")
    print(f"ML: {ml_nonempty}/{len(ml_records)} samples with predictions, labels: {dict(ml_labels)}")
    assert en_nonempty > len(en_records)*0.3, "Too few EN predictions"
    assert ml_nonempty > len(ml_records)*0.3, "Too few ML predictions"
    assert len(en_labels) >= 2, "Too few distinct EN labels"
    print("\n✓ GLiNER inference v2 complete!")

if __name__=="__main__":
    main()
