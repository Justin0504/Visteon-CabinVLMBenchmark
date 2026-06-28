# Cabin-VLM Exterior Caption Pipeline

Turn driving images + sensor ground-truth into **causal-chain captions** and **diverse driver QA** for
training automotive vision-language models — with built-in hallucination control. Built for the
Visteon Cabin VLM Benchmark (car-exterior, Category 2).

👉 **All prompts are documented in [PROMPTS.md](PROMPTS.md)** — read that first.

## The idea (fusion)

A small VLM is visually shallow; a GT-only template has no picture detail. So we split the work:

```
            ┌─ VISION (Qwen3.5-397B) ──── weather, time, road, lane type, surface, buildings
 image ─────┤
            └─ GROUND-TRUTH (sensors) ─── object class / count / distance / bbox
                              │
                              ▼
        TEXT model (DeepSeek) : Chain-of-Causation caption + diverse QA
                              │
                              ▼
        second model cross-checks every claim against GT  ──►  caption + QA (ShareGPT)
```

**Rule that keeps it honest:** objects/counts/positions from GROUND-TRUTH only; visual context from VISION;
never invent; cross-check drops contradictions; every QA set has one `reject` ("not visible") item.

## Repo layout

```
caption_pipeline/
  config.py               all paths/keys/models from env vars (no hardcoding)
  prompts.py              every prompt in one place  (← documented in PROMPTS.md)
  vultr_client.py         API client: retry, reasoning-model fallback, JSON parse
  stage_a_vision.py       image → visual-context description
  stage_b_fusion.py       vision + GT → causal caption + QA + cross-check → ShareGPT
  poi_extract.py          #8 storefront/POI signage reader
  poi_web_verify.py       #8 OSM + name-gate verifier (zero-hallucination)
  trafficlight_extract.py #4 traffic-light state + shape reader
  trafficlight_sharegpt.py#4 build QA from extracted lights
  vru_separation.py       #6 rider/bike/moto/stroller separation from GT (no LLM)
  run_pipeline.sh         Stage A → Stage B
PROMPTS.md                the prompt book (caption / QA / answer / cross-check / POI / TL)
```

## Setup

Running needs **only the Python standard library** + a Vultr API key (models run on Vultr Inference,
OpenAI-compatible — no local GPU).

```bash
echo "VULTR_KEY_1=sk-..." > keys.env      # one or more keys, round-robined
export VULTR_KEYS=keys.env
```

## Input format (one JSON object per line)

```json
{"image":"/abs/path/frame.jpg","camera":"CAM_FRONT",
 "gt":"car:3, pedestrian:1, barrier:16",
 "vehicles":"truck right 11.9m bbox[1469,136,1600,727]",
 "vrus":"pedestrian(adult) ahead 16.8m bbox[543,410,...]"}
```
`gt`/`vehicles`/`vrus` are free-form strings built from your sensor labels (passed through verbatim as the
authoritative anchor). Only `image` is strictly required.

## Run

```bash
# whole pipeline (both stages, resumable — re-running skips finished images)
VULTR_KEYS=keys.env ./caption_pipeline/run_pipeline.sh input.jsonl out/

# or stage by stage
python -m caption_pipeline.stage_a_vision  --inp input.jsonl     --out out/vision.jsonl
python -m caption_pipeline.stage_b_fusion  --inp out/vision.jsonl --raw out/raw.jsonl --out out/sharegpt.json

# per-use-case extractors
python -m caption_pipeline.poi_extract          --inp input.jsonl --out out/poi_raw.jsonl
python    caption_pipeline/poi_web_verify.py    --inp out/poi_raw.jsonl --out out/poi_verified.jsonl --sg out/poi_sharegpt.json
python -m caption_pipeline.trafficlight_extract --inp input.jsonl --out out/tl_raw.jsonl
python -m caption_pipeline.trafficlight_sharegpt --inp out/tl_raw.jsonl --out out/tl_sharegpt.json
python -m caption_pipeline.vru_separation       --inp input.jsonl --out out/vru_sharegpt.json
```

## Tuning (all via env — see `config.py`)

| var | default | when to change |
|---|---|---|
| `VISION_MODEL` | `Qwen/Qwen3.5-397B-A17B` | best quality |
| `VISION_MODEL_FB` | `nvidia/Nemotron-...Omni...` | fast fallback on 504 |
| `GEN_MODEL` | `deepseek-ai/DeepSeek-V4-Flash` | clean JSON |
| `XCHECK_MODEL` | `deepseek-ai/DeepSeek-V3.2-NVFP4` | independent checker |
| `WORKERS` | `4` | lower to `2` if you hit 504/rate-limit |
| `VISION_MAX_TOKENS` | `2600` | lower to `1500` for reasoning fallback models |

**Gotcha learned the hard way:** reasoning-style vision models can spend the whole token budget "thinking"
and return empty `content` (the client salvages from the `reasoning` field), and an over-large budget makes
them slow enough to 504. Keep the fallback model at ~1500 tokens and `WORKERS=2` for clean recovery.

## What each use case is covered by

| # | use case | module |
|---|---|---|
| 1 | Building & Landmark | stage_a + stage_b |
| 3 | Natural Landscape | stage_a + stage_b |
| 4 | Traffic Sign **+ Light** | trafficlight_extract / trafficlight_sharegpt |
| 6 | Pedestrian & **VRU** | vru_separation |
| 7 | Exterior Scene | stage_a + stage_b |
| 8 | **POI** | poi_extract → poi_web_verify |

## License note

Code is shareable within the team. **Datasets generated with it inherit their source licenses** — e.g.
nuScenes/nuImages imagery is CC BY-NC (non-commercial). Keep that boundary on any redistribution.
