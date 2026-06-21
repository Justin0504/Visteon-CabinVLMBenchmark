# Canonical benchmark (one consistent run) — the trustworthy numbers

> 袁傲杰 · 2026-06-21 · base/v7/v9/v10 × 7 categories, single run, identical judge config (reproducible)

## Why this supersedes earlier per-model runs
Earlier comparisons used *different eval scripts* (different judge prompts / max_new_tokens), so numbers
weren't comparable across versions (e.g. v9 cars read 77.5 in one script, 62.5 in another). This table
runs **all models in one script with one judge config** — directly comparable and reproducible.

## Results (40 imgs/category)

| Category | base | v7 | v9 | v10 |
|---|---|---|---|---|
| Vehicle make/model | 2.5 | 62.5 | 62.5 | **70.0** |
| Traffic sign (EU) | 87.5 | **100** | 97.5 | 97.5 |
| Natural landscape | 70 | 80 | 80 | 80 |
| Text OCR | **85** | 70 | 67.5 | 65 |
| China sign | **40** | 37.5 | 30 | 20 |
| VRU crossing | **72.5** | 52.5 | 52.5 | 52.5 |
| Exterior scene (judge) | 6.55 | 6.12 | 6.58 | 6.65 |

## Honest verdict — fine-tuning is a trade, not a strict win

- **Fine-tuning wins:** vehicle make/model (2.5 → 70, huge), traffic sign (→100/97.5), landscape (→80).
- **Fine-tuning REGRESSED below base:** OCR (85 → 65-70), China sign (40 → 20-37.5),
  **VRU crossing (72.5 → 52.5 ≈ chance)**, exterior scene roughly tied.
- **VRU regression is a data-bias artifact:** all three FT models score an identical 52.5 — the JAAD
  training set is crossing-heavy (169/270), so the model learned to over-predict "crossing"; on a
  balanced 20/20 test that collapses to ~chance.
- Net: current FT model = **recognition specialist** (cars/signs/landscape) at the cost of general
  capability (OCR/VRU/China) — classic multi-task catastrophic forgetting + class imbalance.

## Implication for delivery
A single FT model does not dominate base across all 8 use cases. Options: (a) deliver FT for
recognition + keep base for OCR/VRU; (b) rebalance training (v11: balance VRU classes, fewer epochs,
restore OCR weight) to reduce forgetting. **The key methodological win is that a consistent benchmark
now makes these trade-offs visible and reproducible** — earlier optimistic numbers were eval-script noise.

Eval harness: `code/eval_full.py` (run all models in one invocation for comparability).
