"""
Subtask 4: openai/privacy-filter inference
- Load openai/privacy-filter with AutoModelForTokenClassification + AutoTokenizer
- Run inference on all 300 texts
- Decode BIES tag sequences to character-level spans (B/I/E/S all handled)
- Map to common schema
- Save to /root/gliner_vs_openai/results/openai_predictions.jsonl
"""

import json
import os
import torch
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

# openai/privacy-filter label → common schema mapping
# BIES tags: B-<label>, I-<label>, E-<label>, S-<label>
OPENAI_LABEL_TO_COMMON = {
    "private_person": "PERSON",
    "private_address": "ADDRESS",
    "private_email": "EMAIL",
    "private_phone": "PHONE",
    "private_url": "URL",
    "private_date": "DATE",
    "account_number": None,   # not in common schema
    "secret": None,           # not in common schema
}


def decode_bies_to_spans(tokens, token_labels, tokenizer, text):
    """
    Decode BIES token-level labels to character-level spans.
    
    tokens: list of token strings (from tokenizer)
    token_labels: list of label strings per token (e.g. 'B-private_person', 'O')
    tokenizer: the tokenizer (for offset mapping)
    text: original text string
    
    Returns list of {start, end, label} dicts (character-level)
    """
    spans = []
    current_entity = None
    current_label = None
    current_start = None

    for i, (token, label) in enumerate(zip(tokens, token_labels)):
        if label == "O":
            # Close any open entity
            if current_entity is not None:
                spans.append(current_entity)
                current_entity = None
                current_label = None
                current_start = None
            continue

        # Parse BIES prefix and entity type
        if "-" in label:
            prefix, entity_type = label.split("-", 1)
        else:
            prefix = label
            entity_type = ""

        common_label = OPENAI_LABEL_TO_COMMON.get(entity_type, None)

        if prefix == "B":
            # Close any open entity first
            if current_entity is not None:
                spans.append(current_entity)
            current_label = common_label
            current_start = i
            current_entity = {"token_start": i, "token_end": i, "label": common_label, "raw_label": entity_type}

        elif prefix == "I":
            if current_entity is not None and entity_type == current_entity["raw_label"]:
                current_entity["token_end"] = i
            else:
                # Malformed sequence — treat as new entity
                if current_entity is not None:
                    spans.append(current_entity)
                current_entity = {"token_start": i, "token_end": i, "label": common_label, "raw_label": entity_type}

        elif prefix == "E":
            if current_entity is not None and entity_type == current_entity["raw_label"]:
                current_entity["token_end"] = i
                spans.append(current_entity)
                current_entity = None
                current_label = None
            else:
                # Malformed — close current and add this as single
                if current_entity is not None:
                    spans.append(current_entity)
                spans.append({"token_start": i, "token_end": i, "label": common_label, "raw_label": entity_type})
                current_entity = None

        elif prefix == "S":
            # Single-token entity
            if current_entity is not None:
                spans.append(current_entity)
                current_entity = None
            spans.append({"token_start": i, "token_end": i, "label": common_label, "raw_label": entity_type})

    # Close any remaining open entity
    if current_entity is not None:
        spans.append(current_entity)

    return spans


def run_inference_on_text(text, model, tokenizer, max_length=512):
    """Run openai/privacy-filter on a single text, return character-level spans."""
    # Tokenize with offset mapping
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )

    offset_mapping = encoding["offset_mapping"][0].tolist()  # [(start, end), ...]

    # Remove offset_mapping before passing to model
    model_inputs = {k: v for k, v in encoding.items() if k != "offset_mapping"}

    with torch.no_grad():
        outputs = model(**model_inputs)

    logits = outputs.logits[0]  # [seq_len, num_labels]
    predicted_ids = logits.argmax(dim=-1).tolist()
    predicted_labels = [model.config.id2label[pid] for pid in predicted_ids]

    # Get token strings for debugging
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0].tolist())

    # Decode BIES to token-level spans
    token_spans = decode_bies_to_spans(tokens, predicted_labels, tokenizer, text)

    # Convert token spans to character spans using offset_mapping
    char_spans = []
    for span in token_spans:
        if span["label"] is None:
            continue  # Skip unmapped labels (account_number, secret)

        t_start = span["token_start"]
        t_end = span["token_end"]

        # Get character offsets
        char_start = offset_mapping[t_start][0] if t_start < len(offset_mapping) else None
        char_end = offset_mapping[t_end][1] if t_end < len(offset_mapping) else None

        if char_start is None or char_end is None:
            continue
        if char_start == 0 and char_end == 0:
            continue  # Special tokens like [CLS], [SEP]

        char_spans.append({
            "start": char_start,
            "end": char_end,
            "label": span["label"],
            "raw_label": span["raw_label"],
            "text": text[char_start:char_end],
        })

    return char_spans


def main():
    os.makedirs("/root/gliner_vs_openai/results", exist_ok=True)

    print("Loading openai/privacy-filter...")
    tokenizer = AutoTokenizer.from_pretrained("openai/privacy-filter", trust_remote_code=True)
    model = AutoModelForTokenClassification.from_pretrained(
        "openai/privacy-filter",
        dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded. Labels: {list(model.config.id2label.values())[:10]}...")
    print(f"Total labels: {len(model.config.id2label)}")

    # Load eval samples
    input_path = "/root/gliner_vs_openai/data/eval_samples.jsonl"
    samples = []
    with open(input_path) as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"Loaded {len(samples)} eval samples.")

    # Run inference
    output_path = "/root/gliner_vs_openai/results/openai_predictions.jsonl"
    category_counts = {}
    total_entities = 0
    samples_with_preds = 0
    errors = 0

    with open(output_path, "w") as f_out:
        for sample in tqdm(samples, desc="openai/privacy-filter inference"):
            text = sample["text"]

            # Truncate very long texts (tokenizer will handle, but let's be safe)
            if len(text) > 4000:
                text = text[:4000]

            try:
                predictions = run_inference_on_text(text, model, tokenizer, max_length=512)
            except Exception as e:
                print(f"  ERROR on sample {sample['id']}: {e}")
                predictions = []
                errors += 1

            if predictions:
                samples_with_preds += 1
            total_entities += len(predictions)

            for p in predictions:
                category_counts[p["label"]] = category_counts.get(p["label"], 0) + 1

            record = {
                "id": sample["id"],
                "text": text,
                "predictions": predictions,
            }
            f_out.write(json.dumps(record) + "\n")

    print(f"\n✅ openai/privacy-filter inference complete!")
    print(f"  Samples with predictions: {samples_with_preds}/{len(samples)}")
    print(f"  Total entities predicted: {total_entities}")
    print(f"  Errors: {errors}")
    print(f"  Category distribution:")
    for cat, count in sorted(category_counts.items()):
        print(f"    {cat}: {count}")

    # Spot-check
    print("\n=== SPOT CHECK: First 5 samples with predictions ===")
    shown = 0
    with open(output_path) as f:
        for line in f:
            if shown >= 5:
                break
            rec = json.loads(line)
            if rec["predictions"]:
                print(f"\nSample {rec['id']} ({len(rec['predictions'])} entities):")
                for p in rec["predictions"][:3]:
                    print(f"  [{p['start']}:{p['end']}] {p['label']} = {p['text']!r}")
                shown += 1


if __name__ == "__main__":
    main()
