"""
tune_gliner_threshold.py — Sweep GLiNER threshold on dev set, pick best macro F1.
"""
import json, os, csv
from gliner import GLiNER

PROJECT_DIR = "/root/gliner_vs_openai"
DEV_FILE = os.path.join(PROJECT_DIR, "data", "dev_samples.jsonl")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

LABELS = ["person name", "email address", "phone number", "street address", "url", "date"]
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def strict_f1(gold_spans, pred_spans):
    """Strict exact match F1 per category, returns macro F1."""
    cats = set(LABELS)
    tp = {c:0 for c in cats}; fp = {c:0 for c in cats}; fn = {c:0 for c in cats}
    gold_set = {(s["start"],s["end"],s["label"]) for s in gold_spans}
    pred_set = {(s["start"],s["end"],s["label"]) for s in pred_spans}
    for t in pred_set:
        if t in gold_set: tp[t[2]] = tp.get(t[2],0)+1
        else: fp[t[2]] = fp.get(t[2],0)+1
    for t in gold_set:
        if t not in pred_set: fn[t[2]] = fn.get(t[2],0)+1
    return tp, fp, fn

def compute_macro_f1(all_tp, all_fp, all_fn):
    f1s = []
    for c in LABELS:
        tp = all_tp.get(c,0); fp = all_fp.get(c,0); fn = all_fn.get(c,0)
        p = tp/(tp+fp) if (tp+fp)>0 else 0.0
        r = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1 = 2*p*r/(p+r) if (p+r)>0 else 0.0
        f1s.append(f1)
    return sum(f1s)/len(f1s)

def main():
    print("Loading GLiNER large-v2.1...")
    model = GLiNER.from_pretrained("urchade/gliner_large-v2.1")
    print("Model loaded.")

    dev_samples = load_jsonl(DEV_FILE)
    print(f"Dev samples: {len(dev_samples)}")

    results = []
    for thresh in THRESHOLDS:
        print(f"\nThreshold={thresh}...")
        all_tp = {c:0 for c in LABELS}
        all_fp = {c:0 for c in LABELS}
        all_fn = {c:0 for c in LABELS}
        for i, sample in enumerate(dev_samples):
            text = sample["text"]
            gold_spans = sample["gold_spans"]
            preds = model.predict_entities(text, LABELS, threshold=thresh)
            pred_spans = [{"start":p["start"],"end":p["end"],"label":p["label"]} for p in preds]
            tp, fp, fn = strict_f1(gold_spans, pred_spans)
            for c in LABELS:
                all_tp[c] += tp.get(c,0)
                all_fp[c] += fp.get(c,0)
                all_fn[c] += fn.get(c,0)
            if (i+1) % 10 == 0:
                print(f"  {i+1}/{len(dev_samples)}")

        macro_f1 = compute_macro_f1(all_tp, all_fp, all_fn)
        per_cat = {}
        for c in LABELS:
            tp=all_tp[c]; fp=all_fp[c]; fn=all_fn[c]
            p=tp/(tp+fp) if (tp+fp)>0 else 0.0
            r=tp/(tp+fn) if (tp+fn)>0 else 0.0
            f1=2*p*r/(p+r) if (p+r)>0 else 0.0
            per_cat[c] = round(f1,4)
        print(f"  macro_f1={macro_f1:.4f}, per_cat={per_cat}")
        results.append({"threshold": thresh, "macro_f1": round(macro_f1,4), **per_cat})

    # Save CSV
    out_path = os.path.join(RESULTS_DIR, "gliner_threshold_sweep.csv")
    fieldnames = ["threshold","macro_f1"] + LABELS
    with open(out_path,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved → {out_path}")

    best = max(results, key=lambda x: x["macro_f1"])
    print(f"\n★ Best threshold: {best['threshold']} (macro_f1={best['macro_f1']})")

if __name__=="__main__":
    main()
