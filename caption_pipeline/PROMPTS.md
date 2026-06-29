# Prompt Book — Cabin VLM Exterior Caption Pipeline

All prompts used to generate the driving-scene **captions**, **QA pairs**, and **answers**, in one
place, with the design reasoning behind each. Source of truth is [`caption_pipeline/prompts.py`](caption_pipeline/prompts.py);
this doc explains them for humans.

**Core principle (what keeps every output honest):**
> Objects, counts and positions come from **GROUND-TRUTH** (sensor labels) only.
> Visual context (weather, time, road type, lane markings, surface) comes from the **VISION** model.
> The model may never invent. A second model cross-checks every claim against GT.

Two model roles:
- **VISION model** (Qwen3.5-397B, with Nemotron-Omni as fast fallback) — looks at the image, writes visual context.
- **TEXT model** (DeepSeek-V4-Flash generate, DeepSeek-V3.2 cross-check) — fuses VISION + GT into caption + QA.

---

## 1. Vision description (Stage A) — image → visual context

**System**
```
You describe ONLY the visual context of an exterior driving scene. You never count or list dynamic objects.
```
**User**
```
Describe the VISUAL CONTEXT of this exterior driving/street image in 2-3 sentences: weather, time of day,
road/setting type (urban street / highway / intersection / parking / rural), lane-marking type
(dashed / solid / double-yellow / none), road-surface condition (dry / wet / snow), lighting, and roadside
buildings / vegetation / terrain. Do NOT count or list vehicles or pedestrians — only the scene context.
Output ONLY the final description, no preamble.
```
**Why:** the VLM is explicitly told *not* to count vehicles/pedestrians — those come from sensor GT, so we never
let the model hallucinate object counts. It only supplies the rich context that GT labels lack.

---

## 2. Caption (Stage B) — VISION + GROUND-TRUTH → causal caption

**System**
```
You are the reasoning module of an autonomous-driving cockpit. You receive (1) authoritative GROUND-TRUTH
objects (sensor truth: class / count / distance / bbox) and (2) a VISION description of the same image.
Rule: for objects / counts / positions, trust GROUND-TRUTH only. For visual context not in GT (weather,
time-of-day, road type, lane markings, surface, lighting), you MAY use the VISION description.
Never invent anything. Output only JSON.
```
**User (schema)** — Chain-of-Causation, distilled from big-company VLA work (Alpamayo, MindVLA, etc.):
```
Use CHAIN-OF-CAUSATION. Output STRICT JSON {"scene","risk","decision","prediction"}:
scene = road type + weather/time/lane (from VISION) + each GT object with ego-relative position & distance (from GT);
risk = causal hazard, phrased "because <GT object> at <pos/dist> and <vision context>, it may <hazard>";
decision = therefore the ego action (proceed / slow / stop / yield / lane-change), justified by that cause;
prediction = 1-3s intent of the most safety-critical object, or "none".
```
**Why:** `scene → risk → decision → prediction` mirrors the perception→planning chain in production VLA stacks.

> ⚠️ **Tested lesson (authoritative):** the *labeled* form (`Scene: … Risk: …`) scored **lower** on the
> judge (6.45 vs 7.65) than natural prose. So **`stage_b_fusion._flatten()` emits label-free prose** — the
> model still *thinks* in the scene→risk→decision→prediction JSON schema, but the final ingested caption
> reads as flowing sentences. An A/B of a fully-rewritten "natural" *prompt* did **not** beat the current
> one (9.17 vs 9.29), so the prompt is kept. **Rule: causal content in the JSON schema, ingested as prose.**

---

## 3. QA generation (Stage B) — diverse driver Q&A

**System**
```
You are a DRIVER operating a vehicle, speaking about what you see THROUGH the windshield right now.
You build question-answer pairs for an automotive multimodal (VLM) training set. Output only JSON.
```
**User (schema)** — rigor adapted from IntelliCockpitBench:
```
Generate exactly N DIVERSE QA grounded in the GROUND-TRUTH objects (+ VISION context). RULES:
1. Each question must refer to concrete, perceivable content in THIS scene.
2. Must require MULTIMODAL perception — NOT answerable by an LLM alone or by a maps/weather/nav app.
3. Do NOT use phrases like "in the image" / "in the background" — speak as a driver.
4. Vary perspective across: Why, What, Where, When, Who/Which, How, How-many, Is/Can/Do.
5. Cover >=4 distinct capabilities from: counting, spatial, distance, recognition, risk, action,
   intent, weather/scene, reject.
6. Include exactly ONE "reject" item that asks about something NOT supported by GT/VISION; its answer
   must be "not visible" (teaches the model to refuse hallucination).
7. Every answer is grounded and concise, and carries a short reason.
Output STRICT JSON: {"qa":[{"q":"","a":"","reason":"","perspective":"","capability":""}]}
```
**Why each rule matters:**
- **Rule 2** (must be multimodal) keeps the benchmark testing *vision*, not language priors.
- **Rule 3** (no "in the image") makes the QA sound like a real driver, not an annotator.
- **Rule 6** (one forced `reject`) is the anti-hallucination control — every image teaches the model to say
  "not visible" when asked about something absent.
- `perspective` + `capability` tags let you audit coverage/diversity later.

---

## 4. Cross-check — drop anything that contradicts GT

**System**
```
You are a strict fact-checker. Remove any claim that contradicts the GROUND-TRUTH objects
(counts/classes/positions). Output only JSON.
```
**User**
```
Remove claims contradicting GT object counts/classes/positions. Keep everything consistent.
STRICT JSON {"corrected":{"scene","risk","decision","prediction"}}
```
**Why:** a *second, independent* model re-reads the caption against GT and strips hallucinated objects/counts.
This is the last gate before a caption is accepted.

---

## 5. POI extraction (#8) — storefront signage → POI

**System**
```
You read point-of-interest (POI) signage from street scenes. You never invent names.
```
**User**
```
Look ONLY at roadside buildings and storefronts. Is there any business/POI whose sign text is actually
READABLE? Output STRICT JSON: {"has_poi":true,"pois":[{"name":"<exact sign text>",
"category":"<restaurant/shop/fuel/hotel/bank/office/mall/other>","position":"<left/right/ahead>"}]}.
If no sign text is legible, output {"has_poi":false}. Do NOT guess or invent names — only transcribe
text you can actually read.
```
**Then verified** by [`poi_web_verify.py`](caption_pipeline/poi_web_verify.py): each read name is checked
against OpenStreetMap (geo-constrained to the capture city) + a name-similarity gate. Only names that match a
real place are kept — VLM mis-reads are dropped. **Zero-hallucination by construction.**

---

## 6. Traffic-light extraction (#4) — state + shape

**System**
```
You read traffic-light signal state from driving images. You report only what is clearly visible.
```
**User**
```
Look for TRAFFIC LIGHTS (signal heads) in this driving scene. For each clearly visible one report its
lit color state and lamp shape. Output STRICT JSON:
{"has_light":true,"lights":[{"state":"<red/yellow/green/off>","shape":"<circle/arrow/pedestrian/countdown>",
"position":"<left/right/ahead/overhead>","for_lane":"<ego/left-turn/right-turn/crosswalk/unknown>"}]}.
If no traffic light is clearly visible, output {"has_light":false}. Do NOT guess a color you cannot see.
```

---

## 7. VRU separation (#6) — NO prompt (derived from GT)

VRU type separation (rider vs bicycle/motorcycle, adult/child, construction/police) is generated
**deterministically from ground-truth labels** in [`vru_separation.py`](caption_pipeline/vru_separation.py) —
no LLM, so it is zero-hallucination by definition. Included here for completeness: when GT already carries
the distinction, parse it directly instead of asking a model.

---

## Output format (all stages → ShareGPT)

Every line trains as:
```json
{"conversations":[
   {"from":"human","value":"<image>\nDescribe this exterior driving scene and give a driving decision."},
   {"from":"gpt","value":"<caption>"},
   {"from":"human","value":"<question>"},
   {"from":"gpt","value":"<answer>"}],
 "images":["/path/frame.jpg"]}
```

## Model choices (and the one gotcha)

| role | model | note |
|---|---|---|
| vision | `Qwen/Qwen3.5-397B-A17B` | best visual detail |
| vision fallback | `nvidia/Nemotron-...Omni...` | use when primary 504s; **cap at ~1500 tokens or it over-thinks and 504s too** |
| generate | `deepseek-ai/DeepSeek-V4-Flash` | clean JSON |
| cross-check / judge | `deepseek-ai/DeepSeek-V3.2-NVFP4` (judge: use V4-Flash for clean integers) | independent checker |

> Reasoning-style models sometimes return empty `content` and put everything in a `reasoning` field — the
> client salvages the text from there. Give them enough tokens to finish, or they 504.
