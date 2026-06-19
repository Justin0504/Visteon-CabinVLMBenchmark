# Dataset Card — Cabin VLM Exterior (honest, post-cleaning)

> 袁傲杰 · supersedes optimistic raw counts. Numbers reflect dedup + domain filtering.

## Composition (per source)

| Source | Use case | Raw | After domain-filter | After dedup (clean) | Notes |
|---|---|---|---|---|---|
| nuScenes (CoT v2) | #2/#6/#7 scene+VRU+vehicle | 2,424 | 2,424 (in-domain) | ~2,400 | enhanced CoT, GT-grounded, bbox coords |
| Stanford Cars | #2 make/model | 1,997 | 1,997 (by-design) | ~1,990 | 196 classes, label-grounded |
| GTSRB | #4 signs | 1,497 | 1,497 (by-design) | ~1,050 | 43 classes; dedup removes ~450 near-dups |
| SUN397 | #3 landscape | 778 | outdoor-filtered (TBD) | TBD | indoor categories removed |
| TextVQA → road | #5 OCR | 1,500 | **227** | ~227 | **85% off-domain dropped** |
| road-traffic | #4 lights | 142 | 142 | 142 | label-grounded light state |
| JAAD | #6 VRU crossing | 270 | 270 | 270 | behavior-grounded |

**Honest totals:** ~8,600 raw unique → **~6,759 after dedup**; after domain filtering the OCR slice
shrinks dramatically (1500→227). Final exact counts to be regenerated after v7 build.

## 8-category coverage (honest)

| # | Category | Coverage | Note |
|---|---|---|---|
| 2 | Vehicle make/model | 🟢 strong | + nuScenes coarse class; **emergency vehicles missing** |
| 3 | Natural landscape | 🟢 (scene-level) | region-bucket segmentation not done |
| 4 | Traffic sign | 🟢 strong | Euro (GTSRB); multi-region (China/India) missing |
| 4 | Traffic light | 🟡 thin (142) | state R/Y/G; "flashing" + scale missing |
| 5 | Text/OCR | 🔴 **thin (227)** | road-text only after filter; **SVT planned to add storefront/sign + bbox** |
| 6 | Pedestrian/VRU | 🟢 | rider/vehicle described (nuScenes 1-box limit); JAAD crossing intent |
| 7 | Scene description | 🟢 strong | VLA CoT |
| 1 | Building/Landmark | 🔴 caption-only | GLDv2 available (deprioritized by advisor) |
| 8 | POI | 🔴 MVP only | RAG demo; SVT/ShopSign signage planned |

## Format
sharegpt (`conversations` + `images`). Captions are VLA CoT (Scene→Risk→Decision) where applicable;
QA diversified (Recognition/Reasoning/Decision). See `specs/SHAREGPT_FORMAT_SPEC.md`.

## Quality controls applied
- **Domain/relevance gate** (`relevance_filter.py`) — removes off-domain (e.g. TextVQA indoor products).
- **Technical gate** (`quality_gate.py`) — blur/exposure/resolution flags.
- **Dedup** (`clean_whitelist.json`) — dHash near-duplicate removal (−1,579).
- **Faithfulness QC** (`qc_agreement.py`) — multi-rater agreement (AI proxy) + minimal human-review set.

## ⚠️ Delivery boundary (critical)
All sources (nuScenes, JAAD, road-traffic, TextVQA, Stanford Cars, GTSRB, SUN397) are **non-commercial /
research licenses**. **Cannot be delivered to Visteon for commercial use as-is.** Either (a) confirm
research-only use with Visteon, or (b) rebuild on commercially-licensable data. Unresolved — escalate.

## Splits
- Frozen test: 200 (40×5 categories), never trained — see `docs/13_Benchmark_Protocol.md`.
- Train: remainder. Planned: extend frozen split to all 8 categories.
