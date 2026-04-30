"""
Subtask 3: GLiNER large-v2.1 inference
- Load urchade/gliner_large-v2.1
- Run predict_entities on all 300 texts
- Labels: ['person name','email address','phone number','street address','url','date']
- Map to common schema
- Save to /root/gliner_vs_openai/results/gliner_predictions.jsonl
"""

import json
import os
from tqdm import tqdm

# GLiNER label → common schema mapping
GLINER_LABEL_TO_COMMON = {
    "person name": "PERSON",
    "email address": "EMAIL",
    "phone number": "PHONE",
    "street address": "ADDRESS",
    "url": "URL",
    "date": "DATE",
}

GLINER_LABELS = list(GLINER_LABEL_TO_COMMON.keys())

def main():
    os.makedirs("/root/gliner_vs_openai/results", exist_ok=True)

    # Load GLiNER model
    print("Loading urchade/gliner_large-v2.1...")
    from gliner import GLiNER
    model = GLiNER.from_pretrained("urchade/gliner_large-v2.1")
    model.eval()
    print("Model loaded successfully.")

    # Load eval samples
    input_path = "/root/gliner_vs_openai/data/eval_samples.jsonl"
    samples = []
    with open(input_path) as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"Loaded {len(samples)} eval samples.")

    # Run inference
    output_path = "/root/gliner_vs_openai/results/gliner_predictions.jsonl"
    category_counts = {}
    total_entities = 0
    samples_with_preds = 0

    with open(output_path, "w") as f_out:
        for sample in tqdm(samples, desc="GLiNER inference"):
            text = sample["text"]

            # Truncate very long texts to avoid memory issues (GLiNER has token limits)
            if len(text) > 2000:
                text = text[:2000]

            try:
                entities = model.predict_entities(text, GLINER_LABELS, threshold=0.5)
            except Exception as e:
                print(f"  ERROR on sample {sample['id']}: {e}")
                entities = []

            # Map to common schema
            predictions = []
            for ent in entities:
                common_label = GLINER_LABEL_TO_COMMON.get(ent["label"], None)
                if common_label is None:
                    continue
                predictions.append({
                    "start": ent["start"],
                    "end": ent["end"],
                    "label": common_label,
                    "gliner_label": ent["label"],
                    "text": ent["text"],
                    "score": round(ent["score"], 4),
                })
                category_counts[common_label] = category_counts.get(common_label, 0) + 1

            if predictions:
                samples_with_preds += 1
            total_entities += len(predictions)

            record = {
                "id": sample["id"],
                "text": text,
                "predictions": predictions,
            }
            f_out.write(json.dumps(record) + "\n")

    print(f"\n✅ GLiNER inference complete!")
    print(f"  Samples with predictions: {samples_with_preds}/{len(samples)}")
    print(f"  Total entities predicted: {total_entities}")
    print(f"  Category distribution:")
    for cat, count in sorted(category_counts.items()):
        print(f"    {cat}: {count}")

    # Spot-check
    print("\n=== SPOT CHECK: First 5 predictions ===")
    with open(output_path) as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            rec = json.loads(line)
            if rec["predictions"]:
                print(f"\nSample {rec['id']} ({len(rec['predictions'])} entities):")
                for p in rec["predictions"][:3]:
                    print(f"  [{p['start']}:{p['end']}] {p['label']} = {p['text']!r} (score={p['score']})")


if __name__ == "__main__":
    main()
