# ADAS Dataset Sourcing — Finding Images for Our Use Cases

Fulfills Task: "how to find images from ADAS datasets that match our use cases."
Targets the 0610 use-case list: **Category 2 Exterior (3000 imgs)** + supports **Category 1 In-Cabin/OMS (4500 imgs)**.

---

## 1. Two complementary methods

**Method A — use category-specific datasets directly** (labels already exist → fastest).
**Method B — large pool + retrieval filtering** (scalable; also what the bootstrap model is for):
- Embed a large driving-image pool (e.g., BDD100K) with **CLIP**; embed text queries ("a traffic sign", "a pedestrian crossing", "a child car seat"); rank images by similarity → auto-bucket per use case.
- Once the **bootstrap VLM** is good enough, use it (instead of CLIP) to classify/filter — higher precision; this is the self-bootstrap flywheel.

## 2. Dataset map per use case

### Category 2 — Exterior (Justin's line)
| Use case | Recommended dataset(s) | Notes |
|---|---|---|
| Building & Landmark | nuScenes, Mapillary street-level | landmark labels sparse → caption via VLM |
| Vehicle Make & Model | **Stanford Cars**, CompCars | 196 classes / fine-grained |
| Natural Landscape | BDD100K (scene attr), Mapillary Vistas | filter by scene tag |
| Traffic Sign (CN/IN/EU) | **TT100K** (China), **GTSRB** (Germany/EU), **Mapillary Traffic Sign** (global), **IDD** (India) | OCR + non-OCR |
| Text & Advertisement | street-view OCR sets; crop from BDD100K/Mapillary | OCR |
| Pedestrian & VRU | nuScenes, **BDD100K**, EuroCity Persons, CityPersons | rich VRU labels |
| Exterior Scene Description | **BDD100K** (weather/scene/timeofday attrs), nuScenes | attribute-filterable |
| POI Information Retrieval | Mapillary + external POI APIs | retrieval, not a single dataset |

### Category 1 — In-Cabin + OMS (others / synthetic)
| Use case | Dataset(s) |
|---|---|
| Driver fatigue/emotion (DMS) | **NTHU-DDD**, State Farm / AUC Distracted Driver, **CK+ / AffectNet / FERPlus** (emotion) |
| Driver behavior / action | **Drive&Act**, **DMD** |
| Occupant / child / pet / seatbelt / OoP / left-behind | scarce real data → **synthetic** (Isaac Sim / AnyVerse), guided by **Euro NCAP OMS protocol** |

## 3. Access notes
- **nuScenes mini**: public direct URL (no login) — already downloaded.
- **Stanford Cars**: on Kaggle / HuggingFace mirrors → can `huggingface-cli download` on the server.
- **GTSRB**: public (Kaggle / official).
- **TT100K**: public download from cs.tsinghua.edu.cn project page.
- **BDD100K**: register at bdd-data.berkeley.edu (free).
- **Mapillary**: account + API key.
- ⚠️ Most are **non-commercial research** licenses → confirm Visteon-delivery boundary before shipping derived data.

## 4. Recommended sourcing plan (to hit balanced coverage)
1. **Exterior volume**: nuScenes (mini now; full trainval for scale) → scene/pedestrian/road.
2. **Fill gaps**: + **TT100K + GTSRB** (signs) + **Stanford Cars** (make/model) → ~250 + ~150 images into the seed.
3. **Build CLIP retrieval** over BDD100K to auto-bucket the remaining use cases at scale.
4. **In-cabin**: synthetic (Isaac Sim) + DMS/OMS public sets (separate workstream).

## 5. Suggested next download (server, no/low friction)
```bash
# Stanford Cars (HF mirror) — make/model
huggingface-cli download --repo-type dataset <stanford-cars-mirror> --local-dir data/stanford_cars
# GTSRB / TT100K — fetch from official/Kaggle, unzip into data/signs/
```
(Exact mirror repo IDs to be confirmed before download.)
