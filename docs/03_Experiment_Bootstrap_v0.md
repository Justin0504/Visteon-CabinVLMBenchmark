# Experiment Report — Bootstrap v0 (Base vs Fine-tuned)

**Date:** 2026-06-10 · **Author:** Aojie Yuan (Justin)
Fulfills Task 2.2: "apply the IntelliCockpitBench paradigm; design an experiment studying the effect on nuScenes."

## 1. Setup
- **Base model:** Qwen2.5-VL-7B-Instruct.
- **Bootstrap v0:** Qwen2.5-VL-7B + LoRA (rank 8, 2 epochs) fine-tuned on **464 auto-annotated nuScenes exterior samples** (annotator = Qwen2-VL-7B), then merged.
  - Training loss: **1.21 → 0.36**.
- **Held-out test set:** 8 **CAM_BACK** images — a camera split NOT used in the seed (seed = front 3 cameras) → genuinely unseen.
- **Protocol:** for each image, generate 3 image-grounded questions (base + our `qa_prompt`); **both** base and bootstrap answer the *same* questions; an LLM judge (Qwen2.5-VL) scores each answer.
- **Metric:** weighted score (1–10) over IntelliCockpitBench dimensions — Factuality(3), Visual Location(3), Completeness(2), Responsibility(2).

## 2. Result
| Model | Weighted mean (1–10) | n answers |
|---|---|---|
| Base Qwen2.5-VL | **2.39** | 24 |
| **Bootstrap v0** | **2.53** | 24 |

→ Bootstrap v0 is **+0.14 (+6%)** over base.

**Qualitative:** on a "make/model of the car ahead?" question (rear camera, not determinable) both models correctly answered *"Sorry, I cannot answer"* — the anti-hallucination rejection works.

## 3. Honest analysis
1. **The loop works directionally** (small positive lift), and the full bootstrap pipeline (annotate → fine-tune → merge → evaluate) is validated end-to-end.
2. **The gain is marginal — expected.** The seed was auto-labeled by the *weaker* Qwen2-VL, so fine-tuning on it cannot dramatically surpass the *stronger* Qwen2.5-VL base. The +6% reflects format/behaviour alignment more than new capability.
3. **Absolute scores are low (~2.5/10)** because: the held-out CAM_BACK set has many genuinely unanswerable questions → correct "Sorry, I can't answer" still scores low on completeness/visual-location; and the judge is strict.
4. **Limitations:** (a) **self-judge** (Qwen2.5-VL judging) has a base-style preference bias; (b) small n (8 images / 24 QA); (c) judge ≠ the GPT-4o-mini judge IntelliCockpitBench uses.

## 4. What this tells us (actionable)
- To get a *meaningful* bootstrap gain, **improve the seed quality**, in priority order:
  1. **Use a stronger teacher for the seed** (Qwen2.5-VL itself, or a closed model GPT-4o/Gemini as teacher for the seed only — distillation; final model stays open).
  2. **Human-review the seed** (even light review of the 464).
  3. **Balance the seed across use cases** (add TT100K signs + Stanford Cars make/model — see `04_ADAS_Dataset_Sourcing.md`).
- **Upgrade the judge:** use a stronger / external judge to remove self-preference bias before trusting absolute numbers.
- **Scale the held-out eval** to ≥50 images with balanced use cases for a stable signal.

## 5. Reproduce
```bash
conda activate /data/haiyuez/visteon_cabin_vlm/envs/llamafactory
CUDA_VISIBLE_DEVICES=3 python /data/haiyuez/visteon_cabin_vlm/code/experiment_hf.py
# → data/exp_scores.json
```
