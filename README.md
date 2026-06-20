# Cabin VLM Benchmark — Exterior Data Line

**Project:** B08 Visteon#1 — Cabin Vision Dataset Curation
**Module:** Category 2 / Exterior (out-of-cabin) data line
**Goal:** A multimodal **training dataset + fine-tuned model + evaluation suite** for in-cabin VLMs, covering 8 exterior use-case categories.

> Full technical report (中文, with figures): [`docs/TECH_REPORT_zh.md`](docs/TECH_REPORT_zh.md)

---

## Highlights

- **8,030 image-text pairs / ~34,578 QA** across 5 public sources, unified in **sharegpt** training format.
- **Label-grounded recaption** — inject ground-truth labels into the prompt → vehicle make/model accuracy **5% → 80%** (frozen test set).
- **GT-grounded captions (2,424)** — built from nuScenes detection annotations (class / count / 3D position / distance) → factual, zero-hallucination captions.
- **VLA Chain-of-Thought captions (2,424)** — `Scene → Risk → Decision`, camera-role conditioned (front / side / rear), grounded in GT objects. 100% three-part compliance, 6 cameras balanced (404 each).
- **Frozen 200-image test set** (40/category, never trained) for apples-to-apples, multi-version evaluation.
- **POI RAG MVP** — image → VLM entity extraction → Wikipedia retrieval → integrated answer.

## Frozen-test results (objective accuracy)

| Category | base | v6 | **v7 (FINAL)** | v8 |
|---|---|---|---|---|
| Vehicle make/model | 5% | 72.5% | **80.0%** | 72.5% |
| Traffic sign (EU) | 92.5% | 95% | **97.5%** | 75.0% |
| Natural landscape | 60% | 82.5% | 82.5% | 87.5% |
| Text OCR | 82.5% | 80% | 75.0% | 72.5% |
| Exterior scene (judge) | ~7.3 | 6.3 | 6.4 | 6.45 |

**v7 is the delivery model** — domain-pure + deduped training; best on the hardest recognition tasks
(vehicle 80, sign 97.5). v8 added Chinese signs (TT100K) + more OCR but **regressed** (sign 97.5→75)
due to multi-task interference + train/test domain mismatch (test signs are European). See
[docs/15_Final_Model_v7.md](docs/15_Final_Model_v7.md). Lesson: more data ≠ better; coverage additions
need matching test slices to show benefit.

---

## Repo layout

```
code/          Generation & training/eval pipeline (Qwen2-VL annotator, vLLM)
  gt_cot_full.py        VLA CoT caption + diversified QA, GT-grounded, camera-mapped
docs/          Technical docs (pipeline / prompt design / experiments / dataset card / results)
  TECH_REPORT_zh.md     Full report with figures
  images/               Showcase figures (contact sheets, montages, CoT comparison)
specs/         Data-format specs
  SHAREGPT_FORMAT_SPEC.md
data_samples/  Small samples (full datasets are non-commercial — not redistributed)
  cot_sample_30.jsonl   30 VLA CoT records
  sample_records_120.jsonl
```

## Data format (sharegpt)

```json
{"conversations":[
  {"from":"human","value":"<image>\nDescribe this exterior driving scene and give a driving decision."},
  {"from":"gpt","value":"Scene: ... Risk: ... Decision: ..."},
  {"from":"human","value":"How many pedestrians are crossing ahead?"},
  {"from":"gpt","value":"Six pedestrians within 40m."}],
 "images":["/.../nuscenes/.../CAM_FRONT/xxx.jpg"]}
```

## Tooling

- **Annotator:** Qwen2-VL-7B-Instruct (95% JSON compliance) via vLLM 0.6.6
- **Fine-tune base:** Qwen2.5-VL-7B-Instruct, LoRA (rank 8, 2 epoch) via LLaMA-Factory
- **Baseline comparison:** Qwen3-VL-4B-Instruct
- **Grounding:** nuScenes-devkit (`get_sample_data`, `BoxVisibility.ANY`)

## Data sources (non-commercial license — not redistributed)

nuScenes · Stanford Cars · GTSRB · TextVQA · SUN397. Delivery boundary to be confirmed with Visteon.

## License

Code: research/educational use within the capstone. Datasets retain their original licenses.
