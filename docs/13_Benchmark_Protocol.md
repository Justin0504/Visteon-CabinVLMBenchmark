# Cabin VLM Exterior Benchmark — Evaluation Protocol

> 袁傲杰 · formal packaging of the exterior data line as a benchmark (not just a training set)

## 1. What the benchmark measures
8 exterior perception use cases (Category 2). Each test image is tagged with its category and a
ground-truth answer (label) or a reference caption. Models are scored per category, then averaged.

## 2. Test split (frozen, never trained)
- **`heldout_frozen.json`**: 200 images, 40 per category for {cars, signs, landscape, ocr, exterior}.
- Frozen since creation; no model version has trained on it → all version numbers are comparable.
- Extension (planned): add 40 each for {traffic-light, VRU-crossing} so all 8 categories have a held-out slice.

## 3. Metrics (per category)
| Category | Metric | How |
|---|---|---|
| Vehicle make/model | Accuracy | answer matches GT label (judge-verified equality) |
| Traffic sign | Accuracy | answer matches GT sign class |
| Traffic light | Accuracy | predicted state (R/Y/G) matches GT |
| Natural landscape | Accuracy | answer matches GT landscape type |
| Text/OCR | Accuracy | predicted text matches GT transcription |
| Pedestrian/VRU | Accuracy | crossing/behavior judged vs GT behavior |
| Exterior scene | Judge 1–10 | factuality + risk + decision (CoT-matched rubric) |
| POI | Retrieval correctness | extracted entity matches |

Objective categories use exact/normalized match against ground-truth labels (no judge bias). Only the
open-ended scene description uses an LLM judge, with a **CoT-matched rubric** (scene→risk→decision) to
avoid the prompt-format penalty observed in v6 eval.

## 4. Baseline table (to be completed)
| Model | cars | signs | landscape | ocr | scene |
|---|---|---|---|---|---|
| Qwen2.5-VL-7B (base) | 5 | 92.5 | 60 | 82.5 | 7.58 |
| bootstrap v3 | 60 | 100 | 82.5 | 82.5 | 7.28 |
| **bootstrap v6** | **72.5** | 95 | 82.5 | 80 | 7.12 |
| bootstrap v7 (domain-pure) | TBD | TBD | TBD | TBD | TBD |
| Qwen3-VL-4B | 10 | 75 | 90 | 82.5 | — |
| **GPT-4V / Gemini** (closed, planned) | TBD | TBD | TBD | TBD | TBD |

> Closed-model baselines (GPT-4V / Gemini) require API keys — **planned, not yet run**. Adding them is
> needed for a credible public benchmark (a fine-tuned 7B should be compared to the strongest closed VLMs).

## 5. Eval harness
`code/eval_frozen*.py` — loads each model, generates answers per category, scores objectively
(labels) or via judge (scene). Reproducible: same frozen split + same prompts across versions.

## 6. Gaps to "formal benchmark" status
1. Extend frozen split to all 8 categories (add lights/VRU slices).
2. Add closed-model baselines (GPT-4V / Gemini).
3. Cross-distribution test (different country signs / cameras) to claim generalization.
4. Human-verified subset (see QC / agreement protocol, `qc_agreement.py`).
