"""
load_dataset_v2.py — Scaled multilingual dataset loading for v2 evaluation pipeline.
"""
import json, os, random
from collections import Counter
from datasets import load_dataset

PROJECT_DIR = "/root/gliner_vs_openai"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

LABEL_MAP = {
    "GIVENNAME": "person name", "FIRSTNAME": "person name", "SURNAME": "person name",
    "LASTNAME": "person name", "MIDDLENAME": "person name", "PREFIX": "person name",
    "SUFFIX": "person name", "NAME": "person name", "FULLNAME": "person name",
    "PERSONTYPE": "person name", "TITLE": "person name",
    "EMAIL": "email address", "EMAILADDRESS": "email address",
    "TELEPHONENUM": "phone number", "PHONENUMBER": "phone number",
    "PHONE": "phone number", "TEL": "phone number", "FAX": "phone number",
    "STREET": "street address", "STREETADDRESS": "street address",
    "BUILDINGNUM": "street address", "CITY": "street address",
    "ZIPCODE": "street address", "POSTCODE": "street address",
    "STATE": "street address", "COUNTY": "street address",
    "COUNTRY": "street address", "SECONDARYADDRESS": "street address",
    "ADDRESS": "street address", "NEARBYGPSCOORDINATE": "street address",
    "URL": "url", "WEBSITE": "url", "DOMAIN": "url", "IP": "url", "IPADDRESS": "url",
    "DATE": "date", "TIME": "date", "DATETIME": "date", "DOB": "date",
    "AGE": "date", "DATEOFBIRTH": "date",
}

def parse_privacy_mask(raw):
    if isinstance(raw, str):
        try: spans = json.loads(raw)
        except: return []
    elif isinstance(raw, list): spans = raw
    else: return []
    result = []
    for span in spans:
        lbl = span.get("label","").upper().replace(" ","").replace("_","").replace("-","")
        mapped = LABEL_MAP.get(lbl)
        if not mapped:
            for k,v in LABEL_MAP.items():
                if k in lbl or lbl in k: mapped=v; break
        if not mapped: continue
        s,e = span.get("start"), span.get("end")
        if s is None or e is None: continue
        result.append({"start":int(s),"end":int(e),"label":mapped,"text":span.get("value","")})
    return result

def process_sample(row):
    text = row.get("source_text","") or row.get("unmasked_text","")
    return {"id":str(row.get("id","")),"language":row.get("language","en"),
            "text":text,"gold_spans":parse_privacy_mask(row.get("privacy_mask","[]"))}

def save_jsonl(samples, path):
    with open(path,"w",encoding="utf-8") as f:
        for s in samples: f.write(json.dumps(s,ensure_ascii=False)+"\n")
    print(f"  Saved {len(samples)} samples → {path}")

def main():
    print("Loading ai4privacy/pii-masking-400k validation split...")
    ds = load_dataset("ai4privacy/pii-masking-400k", split="validation")
    print(f"  Total: {len(ds)}, Columns: {ds.column_names}")

    en_indices, lang_indices = [], {}
    target_langs = {"fr","de","es","it","nl"}
    for i in range(len(ds)):
        row = ds[i]
        lang = row.get("language","").lower().strip()
        if lang in ("en","english"): en_indices.append(i)
        elif lang in target_langs: lang_indices.setdefault(lang,[]).append(i)

    print(f"  English: {len(en_indices)}")
    for lang in sorted(lang_indices): print(f"  {lang}: {len(lang_indices[lang])}")

    random.seed(42)
    random.shuffle(en_indices)
    assert len(en_indices) >= 450
    dev_indices = en_indices[:50]
    eval_en_indices = en_indices[50:450]

    eval_ml_indices = []
    for lang in ["fr","de","es","it","nl"]:
        idxs = lang_indices.get(lang,[])
        random.shuffle(idxs)
        sel = idxs[:40]
        eval_ml_indices.extend(sel)
        print(f"  Selected {len(sel)} {lang} samples")

    # Verify disjoint
    assert not set(dev_indices)&set(eval_en_indices)
    assert not set(dev_indices)&set(eval_ml_indices)
    assert not set(eval_en_indices)&set(eval_ml_indices)
    print("  ✓ All sets disjoint")

    dev_samples = [process_sample(ds[i]) for i in dev_indices]
    eval_en_samples = [process_sample(ds[i]) for i in eval_en_indices]
    eval_ml_samples = [process_sample(ds[i]) for i in eval_ml_indices]

    save_jsonl(dev_samples, os.path.join(DATA_DIR,"dev_samples.jsonl"))
    save_jsonl(eval_en_samples, os.path.join(DATA_DIR,"eval_en_samples.jsonl"))
    save_jsonl(eval_ml_samples, os.path.join(DATA_DIR,"eval_ml_samples.jsonl"))

    print("\n── Spot-check dev ──")
    for s in dev_samples[:3]:
        print(f"  [{s['language']}] {s['text'][:70]}...")
        for sp in s['gold_spans'][:2]:
            print(f"    [{sp['label']}] '{s['text'][sp['start']:sp['end']]}'")

    print("\n── Gold span dist (eval_en) ──")
    c = Counter(sp['label'] for s in eval_en_samples for sp in s['gold_spans'])
    for k,v in sorted(c.items()): print(f"  {k}: {v}")

    print("\n── Gold span dist (eval_ml) ──")
    c = Counter(sp['label'] for s in eval_ml_samples for sp in s['gold_spans'])
    for k,v in sorted(c.items()): print(f"  {k}: {v}")

    assert len(dev_samples)==50
    assert len(eval_en_samples)==400
    assert len(eval_ml_samples)==200
    print("\n✓ Dataset loading v2 complete!")

if __name__=="__main__":
    main()
