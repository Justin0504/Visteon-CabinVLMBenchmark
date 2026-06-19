# Domain purity + v7 (relevance gate)

> 袁傲杰 · exterior data line · the relevance/scope layer of the quality gate

## Finding — TextVQA was largely off-domain

A manual QC sheet surfaced TextVQA samples that were indoor product photos (beer/whisky bottles,
books), not road scenes. A VLM relevance check confirmed it:

| Source | total | driving-relevant | off-domain |
|---|---|---|---|
| TextVQA | 1500 | **227 (15%)** | **1273 (85%)** |

Spec #5 asks for OCR on **road signs and outdoor advertisements along the route** — so 85% of TextVQA
violated the domain. Root cause: the quality gate had a technical layer (blur/exposure/dedup) but **no
relevance/scope layer**, so off-domain images passed through.

## Fix — relevance filter (`relevance_filter.py`)

A VLM yes/no relevance classifier, applied per source:
- **TextVQA → `textvqa_road_sharegpt.json`**: 1500 → **227** kept (driving-relevant road text).
- **SUN397 landscape → `landscape_outdoor_sharegpt.json`**: filtered to outdoor natural landscape
  (SUN397 contains indoor categories too).
- Driving sources (nuScenes, road-traffic, JAAD) are in-domain by construction; recognition crops
  (Stanford Cars, GTSRB) are by-design and kept.

This is the missing **layer 1 (relevance/scope)** of the quality gate, complementing the technical
layer (`quality_gate.py`) and dedup (`clean_whitelist.json`).

## v7 — domain-pure + deduped retrain (`master_v7.sh` / `build_train_v7.py`)

`train_v7` = clean-whitelist (dedup) ∩ domain-filtered sources:
enhanced CoT v2 (nuScenes) + cars + signs + **textvqa_road** + **landscape_outdoor** + traffic lights + JAAD VRU.
Trains Qwen2.5-VL-7B LoRA, then evaluated on the frozen test set vs v6.

## Honest implications

- **Dataset shrinks and is more honest**: OCR is really ~227 in-domain images, not 1500. Headline
  counts were optimistic before the relevance gate.
- **#5 OCR coverage is genuinely thin** post-filter → needs a real road-text source (storefronts /
  billboards / road signs) — flagged for teammate sourcing.
- Lesson: for a benchmark, **relevance/scope filtering is mandatory**, not optional — convenient
  general-domain datasets (TextVQA) must be domain-gated before inclusion.
