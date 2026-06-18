# Prompt Design — Learned from IntelliCockpitBench, Adapted to Our Project

Source studied (verbatim from repo `Lane315/IntelliCockpitBench`):
- `DataConstruction/dataset/prompt/label_prompt.py`
- `DataConstruction/dataset/prompt/question_prompt.py`
- `DataConstruction/dataset/prompt/answer_prompt.py`
- `Evaluation/config/dimension_definition.json`, `dimension_set.json`

> This document fulfills Task 2.1 (add good use cases referencing IntelliCockpitBench prompts) and 2.2 (apply the IntelliCockpitBench paradigm to nuScenes).

---

## 1. The IntelliCockpitBench 3-stage prompt paradigm (what they actually do)

### Stage 1 — `label_prompt` (scene labeling)
GPT-4 labels each image on **4 axes**, output strict JSON, "no extra words":
- **Weather** (9): Clear / Cloudy / Overcast-or-Night / Light Rain / Moderate-Heavy Rain / Snowy / Foggy / Dust-Sandstorm / Unknown
- **Roadway** (6 primary × ~30 secondary): Urban / Rural / Highway / Special / Parking-Private / Other
- **Driving Status**: Moving / Stopped
- **Shooting Angle**: Inside / Front / Side / Rear

### Stage 2 — `question_prompt` (question generation) — the core
- **Persona:** "You are a **driver** operating a vehicle."
- **5 hard rules:** Clarity (refer to perceivable content) · Consistency with taxonomy · **Diverse perspectives** · Human alignment · **must require multimodal** (not solvable by LLM alone / maps / weather apps).
- **Forbidden:** phrases like "in the image" / "in the background".
- **10 question perspectives:** Why · What · Where · When · Who/Which · How · How-much/How-many · How-feel · Can/Have · Is/Do/Others.
- **Process:** randomly pick **3 sub-categories**, generate 1 question each (few-shot examples per subcategory drawn from a real-driver question pool).
- **Output:** `[{"Question","Perspective"}]`.

### Stage 3 — `answer_prompt` (answer generation)
- **Persona:** "You are an **in-car intelligent agent**."
- Generates **Primary Tag + Secondary Tag + Answer** (+ reason).
- **Rejection rule (anti-hallucination):** if the question is not multimodal-requiring, or contains "in the picture/background", → output exactly **"Sorry, I can't answer"**.

### Full taxonomy — 5 primary × 19 sub
1. **Description**
2. **Recognition:** Vehicle Model · Information Extraction · Object · Emotion · Human Activity
3. **World Knowledge:** Traffic Laws · Geospatial/Environmental · Socio-cultural · General
4. **Reasoning:** Quantitative · Distance · Angle · Area/Volume · Probabilistic/Intent · Driving Decisions
5. **Others:** Creation · Translation · Others

### Evaluation — 10 dimensions, **weighted per sub-category** (`dimension_set.json`)
10 dims: Factuality · User Satisfaction · Visual Location · Clarity · Naturalness · Richness · Completeness · Responsibility · Logical Coherence · Creativity.
Key weighting pattern (1–3):
- **Factuality, User Satisfaction, Visual Location = 3 for every category** (the non-negotiables).
- **Traffic/Geo/Driving-Decision → Responsibility 2–3.**
- **Probabilistic/Driving-Decision → Logical Coherence 2.**
- Creative → Creativity 2.

---

## 2. Our adaptation (`code/prompts_cabin.py`)
We kept the paradigm faithfully and specialized it to **exterior (out-of-vehicle)** + our 8 Category-2 use cases:
- Kept: driver persona, in-car-agent answer persona, 10 perspectives, "must be multimodal", reject-if-unanswerable, strict JSON.
- Specialized taxonomy keys to our exterior use cases (see mapping below).
- Output now also carries `answer` + `reason` + `perspective` + `category` so it converts directly to LLaMA-Factory sharegpt conversations (caption turn + QA turns).

## 3. Use-case mapping — our 0610 list ↔ IntelliCockpitBench taxonomy
| Our Category-2 (Exterior) use case | IntelliCockpitBench category used |
|---|---|
| Building & Landmark | WorldKnowledge / Geospatial + General |
| Vehicle Make & Model | Recognition / Vehicle Model |
| Natural Landscape | WorldKnowledge / Geospatial |
| Traffic Sign | WorldKnowledge / Traffic Laws (OCR via Information Extraction) |
| Text & Advertisement | Recognition / Information Extraction |
| Pedestrian & VRU | Recognition / Object + Human Activity |
| Exterior Scene Description | Description |
| POI Information Retrieval | WorldKnowledge / Geospatial + Socio-cultural |

**Our additions beyond IntelliCockpitBench (Task 2.1):** explicit **VehicleMakeModel**, **Traffic-Sign-by-region (China/India/Europe, OCR + non-OCR)**, **Text/Ad OCR**, **POI retrieval**, and **DrivingDecision/Intent** weighting for safety — these are under-represented in IntelliCockpitBench (which skews to general scene + recognition) and are core to Visteon's exterior use cases.

## 4. Key design takeaways we adopted
1. **Reject-if-unanswerable** ("Sorry, I can't answer") — essential anti-hallucination; verified working on rear-camera images.
2. **Must-require-multimodal** filter — keeps questions VLM-relevant.
3. **Per-category weighted scoring** — safety dims (Responsibility, Visual Location) weighted up for driving/traffic.
4. **Diverse perspectives + forced category coverage** — prevents the "all recognition, no reasoning" collapse (we enforce ≥2 Recognition, ≥2 Reasoning, ≥1 WorldKnowledge).
