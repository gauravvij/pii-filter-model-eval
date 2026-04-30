"""
run_openai_filter_v2.py — openai/privacy-filter inference using BIES decoding + offset_mapping.
Uses trust_remote_code=True (required for custom model type openai_privacy_filter).
"""
import json, os, torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

PROJECT_DIR = "/root/gliner_vs_openai"
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
os.makedirs(RESULTS_DIR, exist_ok=True)

# openai/privacy-filter label → common schema mapping
OPENAI_LABEL_MAP = {
    "private_person": "person name",
    "private_email": "email address",
    "private_phone": "phone number",
    "private_address": "street address",
    "private_url": "url",
    "private_date": "date",
    # account_number and secret have no mapping → skip
}

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records)} records → {path}")

def decode_bies(token_labels, offsets, text):
    """
    Decode BIES tag sequence to character-level spans.
    token_labels: list of tag strings like 'B-private_person', 'O', etc.
    offsets: list of (char_start, char_end) tuples from tokenizer offset_mapping
    Returns list of {start, end, label} dicts.
    """
    spans = []
    current_label = None
    current_start = None
    current_end = None

    for tag, (char_start, char_end) in zip(token_labels, offsets):
        # Skip special tokens (offset (0,0))
        if char_start == 0 and char_end == 0:
            # Flush any open span
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_label = None
            continue

        if tag == "O":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_label = None
        elif tag.startswith("B-"):
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
            current_label = tag[2:]
            current_start = char_start
            current_end = char_end
        elif tag.startswith("I-"):
            entity = tag[2:]
            if current_label == entity:
                current_end = char_end
            else:
                # Mismatch — flush and start new
                if current_label is not None:
                    spans.append((current_start, current_end, current_label))
                current_label = entity
                current_start = char_start
                current_end = char_end
        elif tag.startswith("E-"):
            entity = tag[2:]
            if current_label == entity:
                current_end = char_end
                spans.append((current_start, current_end, current_label))
                current_label = None
            else:
                if current_label is not None:
                    spans.append((current_start, current_end, current_label))
                # E- without matching B- → treat as single
                spans.append((char_start, char_end, entity))
                current_label = None
        elif tag.startswith("S-"):
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_label = None
            entity = tag[2:]
            spans.append((char_start, char_end, entity))

    # Flush remaining
    if current_label is not None:
        spans.append((current_start, current_end, current_label))

    # Map to common schema
    result = []
    for start, end, raw_label in spans:
        mapped = OPENAI_LABEL_MAP.get(raw_label)
        if mapped is None:
            continue
        result.append({"start": start, "end": end, "label": mapped, "score": 1.0})
    return result

def run_inference(model, tokenizer, id2label, samples):
    records = []
    model.eval()
    for i, sample in enumerate(samples):
        text = sample["text"]
        # Truncate very long texts to avoid OOM (max 512 tokens)
        inputs = tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=512,
        )
        offset_mapping = inputs.pop("offset_mapping")[0].tolist()
        with torch.no_grad():
            outputs = model(**inputs)
        pred_ids = outputs.logits.argmax(-1)[0].tolist()
        token_labels = [id2label[pid] for pid in pred_ids]
        predictions = decode_bies(token_labels, offset_mapping, text)
        records.append({
            "id": sample["id"],
            "language": sample.get("language", ""),
            "text": text,
            "predictions": predictions,
        })
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(samples)} processed")
    return records

def main():
    print("Loading openai/privacy-filter (trust_remote_code=True)...")
    tokenizer = AutoTokenizer.from_pretrained("openai/privacy-filter", trust_remote_code=True)
    model = AutoModelForTokenClassification.from_pretrained(
        "openai/privacy-filter",
        trust_remote_code=True,
        dtype=torch.float32,
    )
    id2label = model.config.id2label
    print(f"  Model loaded. Labels: {len(id2label)}")

    # English eval
    print("\nRunning openai/privacy-filter on English eval...")
    en_samples = load_jsonl(os.path.join(DATA_DIR, "eval_en_samples.jsonl"))
    en_records = run_inference(model, tokenizer, id2label, en_samples)
    save_jsonl(en_records, os.path.join(RESULTS_DIR, "openai_en_predictions.jsonl"))

    # Multilingual eval
    print("\nRunning openai/privacy-filter on Multilingual eval...")
    ml_samples = load_jsonl(os.path.join(DATA_DIR, "eval_ml_samples.jsonl"))
    ml_records = run_inference(model, tokenizer, id2label, ml_samples)
    save_jsonl(ml_records, os.path.join(RESULTS_DIR, "openai_ml_predictions.jsonl"))

    # Verify
    from collections import Counter
    en_labels = Counter(p["label"] for r in en_records for p in r["predictions"])
    ml_labels = Counter(p["label"] for r in ml_records for p in r["predictions"])
    en_nonempty = sum(1 for r in en_records if r["predictions"])
    ml_nonempty = sum(1 for r in ml_records if r["predictions"])
    print(f"\nEN: {en_nonempty}/{len(en_records)} samples with predictions, labels: {dict(en_labels)}")
    print(f"ML: {ml_nonempty}/{len(ml_records)} samples with predictions, labels: {dict(ml_labels)}")
    assert en_nonempty > len(en_records)*0.1, f"Too few EN predictions: {en_nonempty}"
    assert len(en_labels) >= 1, "No distinct EN labels"
    print("\n✓ openai/privacy-filter inference v2 complete!")

if __name__=="__main__":
    main()
