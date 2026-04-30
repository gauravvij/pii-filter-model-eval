# GLiNER large-v2.1 vs openai/privacy-filter — Unbiased Evaluation Report v2

*Generated automatically by the v2 evaluation pipeline.*

## 1. Methodology

### Dataset
- **Source**: `ai4privacy/pii-masking-400k` validation split (synthetic PII data)
- **English eval set**: 400 samples (disjoint from dev set)
- **Multilingual eval set**: 200 samples (40 each: French, German, Spanish, Italian, Dutch)
- **Dev set**: 50 English samples used exclusively for GLiNER threshold tuning

### Label Mapping (6-category common schema)
| Common Category | ai4privacy Labels | GLiNER Prompt | openai Label |
|----------------|-------------------|---------------|--------------|
| person name | GIVENNAME, SURNAME, PREFIX, MIDDLENAME | `person name` | `private_person` |
| email address | EMAIL | `email address` | `private_email` |
| phone number | TELEPHONENUM | `phone number` | `private_phone` |
| street address | STREET, CITY, ZIPCODE, COUNTY, STATE | `street address` | `private_address` |
| url | URL | `url` | `private_url` |
| date | DATE, TIME | `date` | `private_date` |

### Three-Tier Scoring Framework (MUC/SemEval Standard)

**Tier 1 — Strict**: Exact (start, end, label) match required. Most conservative.

**Tier 2 — Boundary**: Predicted span must overlap gold span AND share the same label.
This tier corrects for tokenizer offset drift (e.g., GPT-style BPE tokenizers add leading
whitespace to tokens, causing systematic ±1 character offset errors under strict matching).

**Tier 3 — Partial MUC**: Any character overlap with same label scores partial credit.
Score = (COR + 0.5×PAR) / total, where COR = exact matches, PAR = partial overlaps.
Most lenient; standard in MUC-style NER evaluation.

### GLiNER Threshold Tuning
GLiNER's `predict_entities` threshold was swept over {0.3, 0.4, 0.5, 0.6, 0.7} on the
50-sample dev set using strict macro F1. The best threshold was selected and applied
to both eval sets. This prevents overfitting the threshold to the test set.

## 2. GLiNER Threshold Sweep Results

| Threshold | Macro F1 | person name | email address | phone number | street address | url | date |
|-----------|----------|-------------|---------------|--------------|----------------|-----|------|
| 0.3 | 0.3686 | 0.2857 | 0.4211 | 0.5882 | 0.6087 | 0.0 | 0.3077 |
| 0.4 | 0.4023 | 0.303 | 0.5 | 0.6667 | 0.6364 | 0.0 | 0.3077 |
| 0.5 | 0.4225 | 0.3175 | 0.5333 | 0.7143 | 0.6364 | 0.0 | 0.3333 |
| 0.6 | 0.4473 | 0.3333 | 0.6667 | 0.7143 | 0.6364 | 0.0 | 0.3333 |
| 0.7 ★ | 0.5022 | 0.3774 | 0.8 | 0.7692 | 0.6667 | 0.0 | 0.4 |

**Best threshold**: 0.7 (macro F1 = 0.5022)

## 3. English Evaluation Results (400 samples)

### GLiNER large-v2.1

#### Strict Match

| Category | Gold | Pred | Precision | Recall | F1 |
|----------|------|------|-----------|--------|-----|
| person name | 148 | 255 | 0.4000 | 0.6892 | 0.5062 |
| email address | 37 | 48 | 0.5833 | 0.7568 | 0.6588 |
| phone number | 26 | 69 | 0.3478 | 0.9231 | 0.5053 |
| street address | 97 | 57 | 0.3684 | 0.2165 | 0.2727 |
| url | 0 | 50 | 0.0000 | 0.0000 | 0.0000 |
| date | 15 | 86 | 0.1512 | 0.8667 | 0.2574 |
| **MACRO AVG** | — | — | **0.3085** | **0.5754** | **0.3667** |

#### Boundary Match

| Category | Gold | Pred | Precision | Recall | F1 |
|----------|------|------|-----------|--------|-----|
| person name | 148 | 255 | 0.4863 | 0.8378 | 0.6154 |
| email address | 37 | 48 | 0.6458 | 0.8378 | 0.7294 |
| phone number | 26 | 69 | 0.3478 | 0.9231 | 0.5053 |
| street address | 97 | 57 | 0.5263 | 0.3093 | 0.3896 |
| url | 0 | 50 | 0.0000 | 0.0000 | 0.0000 |
| date | 15 | 86 | 0.1512 | 0.8667 | 0.2574 |
| **MACRO AVG** | — | — | **0.3596** | **0.6291** | **0.4162** |

#### Partial Match

| Category | Gold | Pred | Precision | Recall | F1 |
|----------|------|------|-----------|--------|-----|
| person name | 148 | 255 | 0.4431 | 0.7635 | 0.5608 |
| email address | 37 | 48 | 0.6146 | 0.7973 | 0.6941 |
| phone number | 26 | 69 | 0.3478 | 0.9231 | 0.5053 |
| street address | 97 | 57 | 0.4474 | 0.2629 | 0.3312 |
| url | 0 | 50 | 0.0000 | 0.0000 | 0.0000 |
| date | 15 | 86 | 0.1512 | 0.8667 | 0.2574 |
| **MACRO AVG** | — | — | **0.3340** | **0.6023** | **0.3915** |

### openai/privacy-filter

#### Strict Match

| Category | Gold | Pred | Precision | Recall | F1 |
|----------|------|------|-----------|--------|-----|
| person name | 148 | 205 | 0.0732 | 0.1014 | 0.0850 |
| email address | 37 | 38 | 0.2368 | 0.2432 | 0.2400 |
| phone number | 26 | 49 | 0.2041 | 0.3846 | 0.2667 |
| street address | 97 | 102 | 0.1176 | 0.1237 | 0.1206 |
| url | 0 | 38 | 0.0000 | 0.0000 | 0.0000 |
| date | 15 | 87 | 0.1264 | 0.7333 | 0.2157 |
| **MACRO AVG** | — | — | **0.1263** | **0.2644** | **0.1547** |

#### Boundary Match

| Category | Gold | Pred | Precision | Recall | F1 |
|----------|------|------|-----------|--------|-----|
| person name | 148 | 205 | 0.5902 | 0.8176 | 0.6856 |
| email address | 37 | 38 | 0.9737 | 1.0000 | 0.9867 |
| phone number | 26 | 49 | 0.5102 | 0.9615 | 0.6667 |
| street address | 97 | 102 | 0.3627 | 0.3814 | 0.3719 |
| url | 0 | 38 | 0.0000 | 0.0000 | 0.0000 |
| date | 15 | 87 | 0.1609 | 0.9333 | 0.2745 |
| **MACRO AVG** | — | — | **0.4330** | **0.6823** | **0.4976** |

#### Partial Match

| Category | Gold | Pred | Precision | Recall | F1 |
|----------|------|------|-----------|--------|-----|
| person name | 148 | 205 | 0.3317 | 0.4595 | 0.3853 |
| email address | 37 | 38 | 0.6053 | 0.6216 | 0.6133 |
| phone number | 26 | 49 | 0.3571 | 0.6731 | 0.4667 |
| street address | 97 | 102 | 0.2402 | 0.2526 | 0.2462 |
| url | 0 | 38 | 0.0000 | 0.0000 | 0.0000 |
| date | 15 | 87 | 0.1437 | 0.8333 | 0.2451 |
| **MACRO AVG** | — | — | **0.2797** | **0.4733** | **0.3261** |


## 4. Multilingual Evaluation Results (200 samples: FR/DE/ES/IT/NL)

### GLiNER large-v2.1

#### Strict Match

| Category | Gold | Pred | Precision | Recall | F1 |
|----------|------|------|-----------|--------|-----|
| person name | 76 | 108 | 0.4815 | 0.6842 | 0.5652 |
| email address | 23 | 30 | 0.6000 | 0.7826 | 0.6792 |
| phone number | 17 | 45 | 0.3333 | 0.8824 | 0.4839 |
| street address | 42 | 27 | 0.2593 | 0.1667 | 0.2029 |
| url | 0 | 18 | 0.0000 | 0.0000 | 0.0000 |
| date | 9 | 26 | 0.1923 | 0.5556 | 0.2857 |
| **MACRO AVG** | — | — | **0.3111** | **0.5119** | **0.3695** |

#### Boundary Match

| Category | Gold | Pred | Precision | Recall | F1 |
|----------|------|------|-----------|--------|-----|
| person name | 76 | 108 | 0.5926 | 0.8421 | 0.6957 |
| email address | 23 | 30 | 0.6333 | 0.8261 | 0.7170 |
| phone number | 17 | 45 | 0.3556 | 0.9412 | 0.5161 |
| street address | 42 | 27 | 0.4074 | 0.2619 | 0.3188 |
| url | 0 | 18 | 0.0000 | 0.0000 | 0.0000 |
| date | 9 | 26 | 0.3077 | 0.8889 | 0.4571 |
| **MACRO AVG** | — | — | **0.3828** | **0.6267** | **0.4508** |

#### Partial Match

| Category | Gold | Pred | Precision | Recall | F1 |
|----------|------|------|-----------|--------|-----|
| person name | 76 | 108 | 0.5370 | 0.7632 | 0.6304 |
| email address | 23 | 30 | 0.6167 | 0.8043 | 0.6981 |
| phone number | 17 | 45 | 0.3444 | 0.9118 | 0.5000 |
| street address | 42 | 27 | 0.3333 | 0.2143 | 0.2609 |
| url | 0 | 18 | 0.0000 | 0.0000 | 0.0000 |
| date | 9 | 26 | 0.2500 | 0.7222 | 0.3714 |
| **MACRO AVG** | — | — | **0.3469** | **0.5693** | **0.4101** |

### openai/privacy-filter

#### Strict Match

| Category | Gold | Pred | Precision | Recall | F1 |
|----------|------|------|-----------|--------|-----|
| person name | 76 | 87 | 0.0230 | 0.0263 | 0.0245 |
| email address | 23 | 23 | 0.3478 | 0.3478 | 0.3478 |
| phone number | 17 | 33 | 0.1515 | 0.2941 | 0.2000 |
| street address | 42 | 51 | 0.2157 | 0.2619 | 0.2366 |
| url | 0 | 15 | 0.0000 | 0.0000 | 0.0000 |
| date | 9 | 34 | 0.1765 | 0.6667 | 0.2791 |
| **MACRO AVG** | — | — | **0.1524** | **0.2661** | **0.1813** |

#### Boundary Match

| Category | Gold | Pred | Precision | Recall | F1 |
|----------|------|------|-----------|--------|-----|
| person name | 76 | 87 | 0.6552 | 0.7500 | 0.6994 |
| email address | 23 | 23 | 1.0000 | 1.0000 | 1.0000 |
| phone number | 17 | 33 | 0.5152 | 1.0000 | 0.6800 |
| street address | 42 | 51 | 0.4902 | 0.5952 | 0.5376 |
| url | 0 | 15 | 0.0000 | 0.0000 | 0.0000 |
| date | 9 | 34 | 0.2647 | 1.0000 | 0.4186 |
| **MACRO AVG** | — | — | **0.4875** | **0.7242** | **0.5559** |

#### Partial Match

| Category | Gold | Pred | Precision | Recall | F1 |
|----------|------|------|-----------|--------|-----|
| person name | 76 | 87 | 0.3391 | 0.3882 | 0.3620 |
| email address | 23 | 23 | 0.6739 | 0.6739 | 0.6739 |
| phone number | 17 | 33 | 0.3333 | 0.6471 | 0.4400 |
| street address | 42 | 51 | 0.3529 | 0.4286 | 0.3871 |
| url | 0 | 15 | 0.0000 | 0.0000 | 0.0000 |
| date | 9 | 34 | 0.2206 | 0.8333 | 0.3488 |
| **MACRO AVG** | — | — | **0.3200** | **0.4952** | **0.3686** |


## 5. Summary: Macro F1 Comparison

| Model | EN Strict | EN Boundary | EN Partial | ML Strict | ML Boundary | ML Partial |
|-------|-----------|-------------|------------|-----------|-------------|------------|
| GLiNER large-v2.1 | 0.3667 | 0.4162 | 0.3915 | 0.3695 | 0.4508 | 0.4101 |
| openai/privacy-filter | 0.1547 | 0.4976 | 0.3261 | 0.1813 | 0.5559 | 0.3686 |

## 6. Key Findings

- **Strict tier**: EN winner = GLiNER large-v2.1 (0.3667 vs 0.1547); ML winner = GLiNER large-v2.1 (0.3695 vs 0.1813)
- **Boundary tier**: EN winner = openai/privacy-filter (0.4162 vs 0.4976); ML winner = openai/privacy-filter (0.4508 vs 0.5559)
- **Partial tier**: EN winner = GLiNER large-v2.1 (0.3915 vs 0.3261); ML winner = GLiNER large-v2.1 (0.4101 vs 0.3686)

### Analysis

1. **Tokenizer offset bias**: The Boundary tier (which accepts any overlap) is the fairest
   metric for comparing GLiNER (span extraction) vs openai/privacy-filter (BPE token classifier).
   Under Boundary scoring, openai/privacy-filter's performance improves relative to Strict,
   confirming that some of its Strict-tier losses were measurement artefacts.

2. **GLiNER zero-shot flexibility**: GLiNER accepts free-text label prompts, making it
   adaptable to any PII taxonomy without fine-tuning. openai/privacy-filter has a fixed
   8-category schema.

3. **Multilingual performance**: GLiNER large-v2.1 was trained on multilingual data and
   handles cross-lingual NER natively. openai/privacy-filter is primarily English-focused.

4. **URL category**: If URL shows near-zero F1 for both models, this reflects insufficient
   URL gold spans in the sampled subset, not model failure.

## 7. Limitations

- **CPU-only evaluation**: Both models run on CPU; inference speed is not benchmarked.
- **Synthetic data**: ai4privacy/pii-masking-400k is synthetically generated; real-world
  performance may differ.
- **Label mapping**: Some ai4privacy labels (e.g., USERNAME, CREDITCARDNUMBER) have no
  clean mapping to the 6-category schema and are excluded from evaluation.
- **Threshold tuning**: GLiNER threshold was tuned on 50 dev samples; a larger dev set
  would give more stable estimates.
