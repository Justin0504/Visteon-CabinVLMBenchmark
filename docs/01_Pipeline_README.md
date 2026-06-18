# Cabin VLM Benchmark — Data Annotation Pipeline (README)

**Owner:** Aojie Yuan (Justin) · Task 2 (Exterior / NuScenes recaptioning)
**Server:** `haiyuez@10.136.20.188` (phe108-yuezhao-01, needs SJTU VPN) · 8× RTX 6000 Ada 48GB
**Project root:** `/data/haiyuez/visteon_cabin_vlm/`

> Goal (post-pivot 2026-06-10): build a **fine-tuning training set** (real + synthetic) for an intelligent-cockpit VLM, via an **auto-annotation + bootstrap** pipeline. 2000 image-text pairs by end-June, 8000 by end-July.

---

## 1. Directory layout
```
/data/haiyuez/visteon_cabin_vlm/
├── envs/
│   ├── cabin-vlm/         # inference (vLLM 0.6.6 + torch 2.5.1+cu124 + transformers 4.47.1) — Qwen2-VL
│   └── llamafactory/      # training (py3.11, torch 2.5.1+cu124, transformers 5.6) — LLaMA-Factory + Qwen2.5-VL
├── models/
│   ├── Qwen2-VL-7B-Instruct/        # annotator (seed generation)
│   ├── Qwen2.5-VL-7B-Instruct/      # fine-tune base
│   ├── bootstrap_v0_lora/           # LoRA adapter (trained)
│   └── bootstrap_v0_merged/         # merged full bootstrap model
├── data/
│   ├── nuscenes/                    # nuScenes mini (6 cams; CAM_FRONT 404 keyframes)
│   ├── seed_v0_sharegpt.json        # 464 SFT samples (LLaMA-Factory format)
│   ├── dataset_info.json            # LLaMA-Factory dataset registry
│   └── exp_state.json / exp_scores.json   # experiment artifacts
├── code/
│   ├── recaption_nuscenes.py        # annotation pipeline (image → caption + VQA → sharegpt)
│   ├── prompts_cabin.py             # high-fidelity prompts (IntelliCockpitBench paradigm)
│   ├── eval_config/                 # dimension_definition.json + dimension_set.json (from IntelliCockpitBench)
│   ├── experiment_hf.py             # base vs bootstrap evaluation
│   ├── train_bootstrap_v0.yaml      # LoRA training config
│   ├── export_bootstrap_v0.yaml     # LoRA merge config
│   ├── NuScenes-QA/                 # reference repo
│   └── IntelliCockpitBench/         # reference repo (real prompts studied)
└── logs/
```

## 2. The bootstrap loop
```
nuScenes image ──▶ Qwen2-VL (annotator) ──▶ caption + 5 VQA ──▶ sharegpt
                                                                  │
                              LLaMA-Factory LoRA (Qwen2.5-VL, 2ep)│
                                                                  ▼
                                                          bootstrap model
                                                                  │
                       (use it to annotate/filter more at scale) ◀┘
```

## 3. How to run each stage
Always `source /home/haiyuez/miniconda3/etc/profile.d/conda.sh` first. **Set `export PYTHONNOUSERSITE=1`** (env ignores haiyuez `~/.local`).

**(a) Annotate (generate training pairs):**
```bash
conda activate /data/haiyuez/visteon_cabin_vlm/envs/cabin-vlm
cd /data/haiyuez/visteon_cabin_vlm/code
CUDA_VISIBLE_DEVICES=1 python recaption_nuscenes.py --num 500 \
  --cams CAM_FRONT,CAM_FRONT_LEFT,CAM_FRONT_RIGHT \
  --raw ../data/seed_v0_raw.jsonl --sharegpt ../data/seed_v0_sharegpt.json
```
Output: `*_raw.jsonl` (inspect) + `*_sharegpt.json` (train).

**(b) Fine-tune (bootstrap):**
```bash
conda activate /data/haiyuez/visteon_cabin_vlm/envs/llamafactory
CUDA_VISIBLE_DEVICES=2 llamafactory-cli train code/train_bootstrap_v0.yaml
```

**(c) Merge LoRA → runnable model:**
```bash
CUDA_VISIBLE_DEVICES=3 llamafactory-cli export code/export_bootstrap_v0.yaml
```

**(d) Evaluate (base vs bootstrap):**
```bash
CUDA_VISIBLE_DEVICES=3 python code/experiment_hf.py   # in llamafactory env
```

## 4. Gotchas (hard-won — do not repeat)
- **Driver is CUDA 12.4.** Never let pip install cu130 wheels (torch 2.11 / vllm 0.22 / torchaudio 2.11) → `driver too old (12040)`. Pin **torch 2.5.1+cu124**.
- **transformers 5.x pulls cu13 nvidia libs** → conflicts. cabin-vlm pinned to **transformers 4.47.1**. llamafactory uses 5.6 but needed **`torchaudio==2.5.1`** fixed manually.
- **Qwen2.5-VL needs transformers ≥ 4.49** → only runs in the **llamafactory env**, NOT cabin-vlm (4.47). Qwen2-VL runs in cabin-vlm.
- **LLaMA-Factory requires Python ≥ 3.11.**
- `~/.local` (8.3G, haiyuez) leaks into envs → always `PYTHONNOUSERSITE=1` (set in each env's activate.d).
- Root `/` is 88% full → keep everything under `/data`.
- GPU 0 is usually used by others; use GPU 1–7.

## 5. nuScenes mini direct download (no login)
```bash
wget https://www.nuscenes.org/data/v1.0-mini.tgz && tar xzf v1.0-mini.tgz   # ~3.9GB, public
```
Full trainval (300GB+) still needs an account / per-blob URLs.
