"""POI EXTRACTION (#8) — read storefront / business signage from street images with a frontier
vision model, as the first stage of the zero-hallucination POI pipeline:

    poi_extract (this) -> poi_web_verify (geo-constrained OSM + name gate) -> POI sharegpt

Uses the same Vultr vision client as Stage A. Prompts for readable sign text only and forbids
invented names; downstream web verification drops anything not confirmed to be a real place.

Input  jsonl rows: {image, camera, ...}  (only `image` is required)
Output jsonl rows: {image, pois:[{name,category,position}]}  (only rows with >=1 readable POI)

Run:
  VULTR_KEYS=keys.env python -m caption_pipeline.poi_extract --inp raw.jsonl --out poi_raw.jsonl
"""
import json, base64, os, argparse, threading
import concurrent.futures as cf
from . import config
from .vultr_client import chat_vision, parse_json, encode_image

_W = threading.Lock()
POI_SYS = "You read point-of-interest (POI) signage from street scenes. You never invent names."
POI_PROMPT = (
    "Look ONLY at roadside buildings and storefronts. Is there any business/POI whose sign text is "
    'actually READABLE? Output STRICT JSON: {"has_poi":true,"pois":[{"name":"<exact sign text>",'
    '"category":"<restaurant/shop/fuel/hotel/bank/office/mall/other>","position":"<left/right/ahead>"}]}. '
    'If no sign text is legible, output {"has_poi":false}. Do NOT guess or invent names — only transcribe '
    "text you can actually read."
)

def extract(img_path, key):
    b64 = encode_image(img_path)
    txt = chat_vision(config.VISION_MODEL, POI_SYS, POI_PROMPT, b64, key, config.VISION_MAX_TOKENS, 240)
    if not txt:
        txt = chat_vision(config.VISION_MODEL_FB, POI_SYS, POI_PROMPT, b64, key, 1500, 150)
    d = parse_json(txt)
    if not d or not d.get("has_poi"):
        return None
    pois = [p for p in (d.get("pois") or [])
            if p.get("name") and str(p["name"]).strip().lower() not in ("unknown", "sign", "store", "shop", "")]
    return pois or None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num", type=int, default=0)
    a = ap.parse_args()
    keys = config.load_keys()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    rows = [json.loads(l) for l in open(a.inp)]
    if a.num:
        rows = rows[:a.num]
    seen = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try: seen.add(json.loads(l)["image"])
            except Exception: pass
    rows = [r for r in rows if r["image"] not in seen]
    print(f"POI_EXTRACT rows={len(rows)} model={config.VISION_MODEL} workers={config.WORKERS}", flush=True)

    def work(ir):
        i, r = ir
        try:
            if not os.path.exists(r["image"]):
                return None
            pois = extract(r["image"], keys[i % len(keys)])
            return {"image": r["image"], "pois": pois} if pois else None
        except Exception:
            return None

    fr = open(a.out, "a" if seen else "w"); done = poi = 0
    with cf.ThreadPoolExecutor(max_workers=config.WORKERS) as ex:
        for res in ex.map(work, list(enumerate(rows))):
            done += 1
            if res:
                poi += 1
                with _W:
                    fr.write(json.dumps(res, ensure_ascii=False) + "\n"); fr.flush()
            if done % 50 == 0:
                print(f"done {done} with_poi {poi}", flush=True)
    fr.close()
    print(f"POI_EXTRACT_DONE images={done} with_poi={poi}", flush=True)

if __name__ == "__main__":
    main()
