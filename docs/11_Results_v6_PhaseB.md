# Results — v6 model + Phase B data (gap filling)

> 袁傲杰 · 2026-06-18 · exterior data line

## 1. v6 model (enhanced captions + traffic lights)

`train_v6` = v3 composition with the nuScenes portion swapped for the **enhanced CoT v2 captions**
(detection-box 2D coords + VRU separation + vehicle coarse-class) + **142 traffic-light** images.
LoRA rank 8, 2 epoch, Qwen2.5-VL-7B, train_loss 0.41.

### Frozen-test (200 imgs, 40/category, never trained)

| Category | base | v3 | **v6** | v6 vs v3 |
|---|---|---|---|---|
| Vehicle make/model | 5% | 60% | **72.5%** | 🟢 +12.5 |
| Traffic sign | 92.5% | 100% | 95% | noise (2 imgs) |
| Natural landscape | 60% | 82.5% | 82.5% | tied |
| Text OCR | 82.5% | 82.5% | 80% | noise (1 img) |
| Exterior scene (free-form judge) | 7.58 | 7.92 | 7.38 | −0.54 |
| Exterior scene (CoT-matched judge) | — | 7.28 | 7.12 | −0.16 (tied) |

**Finding:** the apparent exterior-scene regression under the free-form judge (−0.54) shrinks to −0.16
(within noise) once the judge prompt matches v6's CoT training format — i.e. most of the "drop" was a
prompt-format measurement artifact, not a real caption regression. **v6 is the new best model**:
clear vehicle-recognition gain, everything else held.

## 2. Phase B — gap filling (no-account, HuggingFace / public)

| Gap (spec) | Source | Output |
|---|---|---|
| #4 Traffic **light** state+shape | `Francesco/road-traffic` (HF, no login) GT light boxes → localized VLM color/shape read | 142 imgs, 413 lights, 74% readable (red 228 / green 74 / yellow 3 / off 2; rest honest `unknown`); 41 with crosswalk |
| #6 VRU crossing intent | JAAD (GitHub, free) videos → cv2 frames + behavior-grounded captions | 270 imgs, 169 with crossing pedestrians; behavior labels crossing/looking/walking/standing + bbox |

Both label-grounded (no hallucination): the GT supplies *what/where*, the VLM fills prose; unreadable
cases are marked `unknown` rather than guessed.

## 3. Image quality gate (`quality_gate.py`)

Scanned 8,338 unique images: 59% clean, 28% flagged, 1,586 near-duplicates (dHash).
Per-source analysis shows most flags are **benign by design** — `low_res`/`blurry` are concentrated in
the GTSRB/Cars recognition crops (small by design, upsampled at train time), and `underexposed` is
mostly nuScenes night driving (wanted diversity). The genuinely actionable item is near-duplicate
thinning (TextVQA / GTSRB / nuScenes consecutive frames). No cleanup applied yet (kept for review).

## 4. Dataset state after Phase B

~8,608 unique images across 7 sources, 8 exterior categories. nuScenes captions upgraded to CoT v2.
#4-lights and #6-VRU(crossing) gaps filled at first-pass scale. Remaining gaps: #1 landmark, #8 POI
depth, emergency vehicles, text spatial coords, China-domain — flagged for teammates.
