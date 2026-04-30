"""
Subtask 2: Dataset loading & gold span extraction
- Load ai4privacy/pii-masking-400k validation split
- Filter to English, sample 300 examples
- Parse privacy_mask JSON strings to character-level {start, end, label} dicts
- Apply 6-category label mapping
- Save to /root/gliner_vs_openai/data/eval_samples.jsonl
"""

import json
import random
import os
from datasets import load_dataset

# ── Label mapping: ai4privacy labels → common 6-category schema ──────────────
AI4PRIVACY_TO_COMMON = {
    # PERSON
    "GIVENNAME": "PERSON",
    "FIRSTNAME": "PERSON",
    "SURNAME": "PERSON",
    "LASTNAME": "PERSON",
    "MIDDLENAME": "PERSON",
    "PREFIX": "PERSON",
    "SUFFIX": "PERSON",
    "FULLNAME": "PERSON",
    "NAME": "PERSON",
    # EMAIL
    "EMAIL": "EMAIL",
    "EMAILADDRESS": "EMAIL",
    # PHONE
    "TELEPHONENUM": "PHONE",
    "PHONENUMBER": "PHONE",
    "PHONE": "PHONE",
    # ADDRESS
    "STREET": "ADDRESS",
    "STREETADDRESS": "ADDRESS",
    "CITY": "ADDRESS",
    "ZIPCODE": "ADDRESS",
    "POSTCODE": "ADDRESS",
    "ZIP": "ADDRESS",
    "STATE": "ADDRESS",
    "COUNTY": "ADDRESS",
    "COUNTRY": "ADDRESS",
    "BUILDINGNUM": "ADDRESS",
    "BUILDINGNUMBER": "ADDRESS",
    "SECONDARYADDRESS": "ADDRESS",
    "ADDRESS": "ADDRESS",
    # URL
    "URL": "URL",
    "DOMAIN": "URL",
    "DOMAINNAME": "URL",
    "IP": "URL",
    "IPADDRESS": "URL",
    # DATE
    "DATE": "DATE",
    "TIME": "DATE",
    "DATETIME": "DATE",
    "DOB": "DATE",
    "DATEOFBIRTH": "DATE",
    "BIRTHDATE": "DATE",
    "AGE": "DATE",
}

VALID_CATEGORIES = {"PERSON", "EMAIL", "PHONE", "ADDRESS", "URL", "DATE"}


def map_label(raw_label: str) -> str | None:
    """Map raw ai4privacy label to common schema. Returns None if not in scope."""
    normalized = raw_label.upper().replace(" ", "").replace("_", "").replace("-", "")
    return AI4PRIVACY_TO_COMMON.get(normalized, None)


def parse_privacy_mask(privacy_mask_raw) -> list[dict]:
    """Parse privacy_mask field (may be string or list) into list of span dicts."""
    if isinstance(privacy_mask_raw, str):
        try:
            spans = json.loads(privacy_mask_raw)
        except json.JSONDecodeError:
            return []
    elif isinstance(privacy_mask_raw, list):
        spans = privacy_mask_raw
    else:
        return []

    result = []
    for span in spans:
        if not isinstance(span, dict):
            continue
        raw_label = span.get("label", "")
        mapped = map_label(raw_label)
        if mapped is None:
            continue
        start = span.get("start")
        end = span.get("end")
        if start is None or end is None:
            continue
        result.append({
            "start": int(start),
            "end": int(end),
            "label": mapped,
            "original_label": raw_label,
            "value": span.get("value", ""),
        })
    return result


def main():
    random.seed(42)
    os.makedirs("/root/gliner_vs_openai/data", exist_ok=True)

    print("Loading ai4privacy/pii-masking-400k validation split...")
    dataset = load_dataset("ai4privacy/pii-masking-400k", split="validation")
    print(f"Total validation examples: {len(dataset)}")
    print(f"Columns: {dataset.column_names}")

    # Filter to English
    print("Filtering to English examples...")
    english_examples = [ex for ex in dataset if ex.get("language", "").lower() in ("en", "english")]
    print(f"English examples: {len(english_examples)}")

    # Sample 300
    if len(english_examples) > 300:
        sampled = random.sample(english_examples, 300)
    else:
        sampled = english_examples
    print(f"Sampled: {len(sampled)} examples")

    # Parse and save
    output_path = "/root/gliner_vs_openai/data/eval_samples.jsonl"
    saved = 0
    category_counts = {cat: 0 for cat in VALID_CATEGORIES}

    with open(output_path, "w") as f:
        for i, ex in enumerate(sampled):
            # Get source text - try different column names
            text = ex.get("source_text") or ex.get("unmasked_text") or ex.get("text") or ""
            if not text:
                print(f"  WARNING: Example {i} has no text, skipping")
                continue

            # Parse privacy mask
            privacy_mask_raw = ex.get("privacy_mask") or ex.get("span_labels") or "[]"
            gold_spans = parse_privacy_mask(privacy_mask_raw)

            # Count categories
            for span in gold_spans:
                category_counts[span["label"]] += 1

            record = {
                "id": i,
                "text": text,
                "gold_spans": gold_spans,
                "language": ex.get("language", ""),
                "locale": ex.get("locale", ""),
            }
            f.write(json.dumps(record) + "\n")
            saved += 1

    print(f"\nSaved {saved} examples to {output_path}")
    print("\nGold span category distribution:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count}")

    # Spot-check 5 examples
    print("\n=== SPOT CHECK: First 5 examples ===")
    with open(output_path) as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            rec = json.loads(line)
            print(f"\nExample {rec['id']}:")
            print(f"  Text (first 100 chars): {rec['text'][:100]!r}")
            print(f"  Gold spans ({len(rec['gold_spans'])} total):")
            for span in rec['gold_spans'][:3]:
                text_snippet = rec['text'][span['start']:span['end']]
                print(f"    [{span['start']}:{span['end']}] {span['label']} = {text_snippet!r}")

    print("\n✅ Dataset loading complete!")


if __name__ == "__main__":
    main()
