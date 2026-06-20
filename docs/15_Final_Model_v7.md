# Final model selection — v7

> 袁傲杰 · model iteration verdict (v6 → v7 → v8) on the frozen test set

## Progression (frozen test, 40/category)

| Category | base | v6 | **v7 (FINAL)** | v8 |
|---|---|---|---|---|
| Vehicle make/model | 5 | 72.5 | **80.0** | 72.5 |
| Traffic sign (GTSRB/EU) | 92.5 | 95 | **97.5** | 75.0 |
| Natural landscape | 60 | 82.5 | 82.5 | 87.5 |
| Text OCR | 82.5 | 80 | 75.0 | 72.5 |
| Exterior scene (judge) | 7.3* | 6.3 | 6.4 | 6.45 |

\* judge score is LLM-rated and noisy across runs; treat as approximate.

## Verdict: **v7 is the delivery model.**

- **v7** = domain-pure + deduped training (CoT v2 captions + cars + GTSRB + road-OCR + outdoor-landscape
  + traffic-lights + JAAD-VRU). Best on the two hardest recognition tasks (vehicle 80, sign 97.5).
- **v8** added TT100K Chinese signs (600) + full road-OCR (227) → **regressed** (sign 97.5→75).
  Root cause: the frozen test signs are **GTSRB (European)**; mixing Chinese sign codes (pl40, …) into
  training **diluted/confused** EU-sign recognition, and the larger multi-task mix hurt vehicle too.
  Classic multi-task interference + train/test domain mismatch.

## Lesson (honest)
More data ≠ better. China-sign + extra OCR data **improve coverage** but show no gain on the current
test because there is **no China-sign / road-OCR test slice** — on the EU/general test they only dilute.
To credit them, the benchmark test split must be extended with matching slices.

## Action
- Ship **v7** as the fine-tuned model.
- Keep TT100K (`tt100k_china`, 600) in the dataset as **coverage**, flagged as not-yet-tested.
- Extend frozen test with China-sign + road-OCR slices in a future pass to measure their benefit.
