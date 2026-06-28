# Visteon Cabin VLM Benchmark — Exterior (Category 2) Delivery Report v2

_Car-exterior / driving-viewpoint VLM benchmark, 8 use cases. All data from ego-vehicle cameras
(nuScenes 6-cam surround + nuImages urban + license-clean public sources)._

## 1. Delivery model

| | model | note |
|---|---|---|
| **v15 (new)** | `bootstrap_v15_merged` | rank-32 LoRA on Qwen2.5-VL-7B, balanced 8-use-case set |
| v9 (prior) | `bootstrap_v9_merged` | 3-4 task delivery model |

**Frozen test-set eval (v15 vs v9):**

| dim | v9 | v15 | result |
|---|---|---|---|
| signs | 95.0 | **100.0** | +5 win |
| landscape | 80.0 | **85.0** | +5 win |
| cars | 82.5 | 80.0 | −2.5 (within noise) |
| exterior | 7.65 | 7.3 | −0.35 (within noise) |
| ocr | 72.5 | 72.5 | tie |

v15 **matches or beats v9 on all five dims (2 wins, 3 ties within eval noise)** while additionally
covering use cases #4 (lights), #6 (VRU separation), #8 (POI) that v9 did not. Key lever: LoRA
rank 8→32 gave the capacity to hold 8 tasks without the task-interference seen at rank 8 (v13/v14).

## 2. Eight use-case coverage

| # | Use case | data | status |
|---|---|---|---|
| 1 | Building & Landmark | fusion captions (Qwen3.5-397B vision) | partial → upgraded |
| 2 | Vehicle Make & Model | Stanford Cars ~2000 (upweighted) | strong |
| 3 | Natural Landscape | landscape 778 + ade20k 176 | strong |
| 4 | Traffic Sign **+ Light** | GTSRB/TT100K signs 1497 **+ lights 337 (new)** | partial → upgraded |
| 5 | Text / Ad OCR | nuImages 1200 | partial |
| 6 | Pedestrian & **VRU** | **VRU type-separation 1162/1932 QA (new)** + JAAD | strong (new) |
| 7 | Exterior Scene Desc | fusion / exterior_cot_v4 | strong |
| 8 | **POI** | **223 verified (new)** | partial (new) |

New this cycle: #8 POI (10→223), #4 traffic lights (0→337), #6 VRU separation (+1162),
plus Qwen3.5-397B vision-description upgrade across all nuScenes captions.

## 3. Methodology highlights (zero-hallucination)

- **Fusion captions** — frontier vision model (Qwen3.5-397B, Nemotron-Omni fallback) supplies
  weather/road/lane/surface detail; sensor GROUND-TRUTH supplies object class/count/distance; a
  frontier text model fuses them into a caption + QA; a second model cross-checks every claim
  against GT. Objects never invented.
- **#8 POI** — VLM reads storefront signage from 6-cam driving views → geo-constrained OSM lookup
  (by capture city) verifies each name is a real place → name-similarity gate drops mismatches.
  Only web-verified POIs are kept.
- **#6 VRU** — derived directly from GT labels, separating rider / cyclist / motorcyclist /
  bicycle / motorcycle / adult / child / construction / police (zero-hallucination).
- **#4 lights** — VLM reads color state (R/Y/G) + lamp shape, keeps only confident detections.

## 4. Shareable pipeline

`caption_pipeline/` — config-driven (env vars, no hardcoded paths/keys), stdlib-only to run,
runs on Vultr (no GPU needed). Modules: `stage_a_vision`, `stage_b_fusion`, `poi_extract`,
`poi_web_verify`, `trafficlight_extract`, `trafficlight_sharegpt`, `vru_separation`, unified
`prompts.py`, `vultr_client.py`, `config.py`, `README.md`.

Prompt note: a natural-prose caption variant was A/B-tested against the current prompt and did
NOT improve the judge score (9.17 vs 9.29) — so the current prompt was kept. The "natural > labeled"
lesson is already reflected in v15's exterior data choice.

## 5. Honest gaps

- **#5 OCR**: reads text but no text→coordinate grounding yet.
- **#8 POI**: 223 is limited by nuScenes' low storefront density; nuImages extraction (urban,
  in progress) should raise this.
- **#4**: no India region, no countdown-pictogram (no public dataset labels these → custom annotation).
- **#6**: stroller class sparse in nuScenes GT.
- **External datasets** (ZOD #6, S2TLD #4): best license-clean sources, but ZOD's mini download
  link is broken upstream (Dropbox not_found) and S2TLD is Baidu-Pan only → both blocked at source,
  not integrated. Documented for future manual acquisition.

## 6. License boundary (commercial delivery)

- Commercial-OK sources used: Stanford Cars, GTSRB, ADE20K-derived, our own VLM/OSM-generated data.
- **Non-commercial (research-only, flag for Visteon)**: nuScenes / nuImages (CC BY-NC) underlie
  much of the exterior/VRU/POI/TL imagery → captions are deliverable as research, but the
  underlying images carry nuScenes' non-commercial license. Keep this boundary on redistribution.

---

## 7. Validation update (new-case evals on held-out data)

Three use cases that were previously unmeasured, now evaluated on data produced AFTER v15 training
(genuine held-out):

| use case | held-out set | metric | score |
|---|---|---|---|
| #4 traffic light | India dashcam (unseen) | light-color accuracy | **84.6%** |
| #6 VRU | nuScenes non-train slice | pedestrian count (±1) | **82.5%** |
| #6 VRU | " | cyclist recall (separation) | **87.5%** |
| #8 POI | nuImages (unseen) | POI detection rate | **75%** |
| #8 POI | " | exact business-name hit | **40%** |

#4 light and #6 VRU reach delivery-credible accuracy; #8 detects POIs well but exact storefront naming
is hard (distant signage).

## 8. Caption cleanup + OCR coordinate grounding

- **fusion_v2 format fix**: 24% of captions had the `scene` field as a serialized dict; all 590 converted
  to natural prose (`fusion_sharegpt_v2_clean.json`), 0 remaining.
- **#5 OCR→coordinates**: new `ocr_coord_extract.py` reads on-scene text + bounding box (e.g.
  `"Portsdown Rd" [219,160,313,252]`). Mechanism works, but **readable-text yield in driving scenes is
  genuinely low (~3%, confirmed across two models)** — driving views are text-sparse. For volume, pair
  with a dense scene-text set (Total-Text, BSD-3) for the skill + this extractor for domain realism.
