"""Multi-use-case harvester: ONE vision call per frame extracts everything useful for the exterior
use cases — scene(#7) / traffic lights(#4) / signs(#4) / POI(#8) / on-scene text+coords(#5) /
VRU counts(#6). Far cheaper than running 5 separate extractor passes over the same frames.
Emits one combined raw jsonl + per-use-case sharegpt files. Only what is actually visible is kept.

Run:
  VULTR_KEYS=keys.env python -m caption_pipeline.multi_extract --inp frames.jsonl --out harvest
"""
import json, base64, os, argparse, threading
import concurrent.futures as cf
from . import config
from .vultr_client import chat_vision, parse_json, encode_image

_W = threading.Lock()
SYS = "You are a driving-scene perception module. Report ONLY what is clearly visible; never invent."
PROMPT = (
    "Analyze this driving (ego-camera) frame and extract everything clearly visible. STRICT JSON:\n"
    '{"scene":{"weather":"","time_of_day":"","road_type":""},'
    '"lights":[{"state":"red/yellow/green/off","shape":"circle/arrow/pedestrian/countdown","position":"left/right/ahead/overhead"}],'
    '"signs":[{"meaning":"<sign meaning or text>","position":""}],'
    '"poi":[{"name":"<readable business/sign name>","category":"","position":""}],'
    '"text":[{"text":"<exact readable words>","bbox":[x1,y1,x2,y2]}],'   # coords normalized 0-1000
    '"vru":{"pedestrians":0,"cyclists":0,"motorcyclists":0}}\n'
    "Use [] / 0 / \"\" when something is absent. Only transcribe text/names you can actually read. "
    "bbox normalized to 0-1000. Output only the JSON."
)

def extract(img_path, key):
    b64 = encode_image(img_path)
    txt = chat_vision(config.VISION_MODEL, SYS, PROMPT, b64, key, config.VISION_MAX_TOKENS, 240)
    if not txt:
        txt = chat_vision(config.VISION_MODEL_FB, SYS, PROMPT, b64, key, 1500, 150)
    return parse_json(txt)

def build_sharegpts(rows, outdir):
    poi, tl, ocr, scene, vru, sign = [], [], [], [], [], []
    for r in rows:
        img = r["image"]; d = r["data"]
        sc = d.get("scene", {})
        if sc.get("weather") or sc.get("road_type"):
            scene.append({"conversations": [
                {"from": "human", "value": "<image>\nDescribe this exterior driving scene."},
                {"from": "gpt", "value": f"A {sc.get('road_type','road')} during {sc.get('time_of_day','daytime')} under {sc.get('weather','clear')} weather."},
                {"from": "human", "value": "What is the weather and time of day?"},
                {"from": "gpt", "value": f"{sc.get('weather','clear')} weather, {sc.get('time_of_day','daytime')}."}], "images": [img]})
        L = [x for x in d.get("lights", []) if str(x.get("state","")).lower() in ("red","yellow","green","off")]
        if L:
            tl.append({"conversations": [
                {"from": "human", "value": "<image>\nWhat is the state of the traffic light ahead?"},
                {"from": "gpt", "value": f"The traffic light {L[0].get('position','ahead')} is {L[0]['state']} ({L[0].get('shape','circle')} signal)."}], "images": [img]})
        P = [x for x in d.get("poi", []) if x.get("name") and str(x["name"]).lower() not in ("","unknown")]
        if P:
            poi.append({"conversations": [
                {"from": "human", "value": "<image>\nList any points of interest (businesses/landmarks) visible."},
                {"from": "gpt", "value": "Visible POIs: " + ", ".join(f"{p['name']} ({p.get('category','poi')})" for p in P) + "."}], "images": [img]})
        T = [x for x in d.get("text", []) if x.get("text") and isinstance(x.get("bbox"), list) and len(x["bbox"]) == 4]
        if T:
            ocr.append({"conversations": [
                {"from": "human", "value": "<image>\nRead the visible text and give each item's location."},
                {"from": "gpt", "value": "; ".join(f"\"{t['text']}\" at {t['bbox']}" for t in T[:10]) + "."}], "images": [img]})
        S = [x for x in d.get("signs", []) if x.get("meaning")]
        if S:
            sign.append({"conversations": [
                {"from": "human", "value": "<image>\nWhat traffic signs are visible and what do they mean?"},
                {"from": "gpt", "value": "; ".join(f"{x['meaning']} ({x.get('position','')})" for x in S) + "."}], "images": [img]})
        v = d.get("vru", {})
        if (v.get("pedestrians", 0) or v.get("cyclists", 0) or v.get("motorcyclists", 0)):
            parts = []
            if v.get("pedestrians"): parts.append(f"{v['pedestrians']} pedestrian(s)")
            if v.get("cyclists"): parts.append(f"{v['cyclists']} cyclist(s)")
            if v.get("motorcyclists"): parts.append(f"{v['motorcyclists']} motorcyclist(s)")
            vru.append({"conversations": [
                {"from": "human", "value": "<image>\nBreak down the vulnerable road users by type."},
                {"from": "gpt", "value": "; ".join(parts) + "."}], "images": [img]})
    for name, arr in [("scene", scene), ("light", tl), ("poi", poi), ("ocr", ocr), ("sign", sign), ("vru", vru)]:
        dst = os.path.join(outdir, f"harvest_{name}_sharegpt.json")
        with open(dst + ".tmp", "w") as f:               # atomic: tmp then replace (no half-written artifact)
            json.dump(arr, f, ensure_ascii=False)
        os.replace(dst + ".tmp", dst)
        print(f"  harvest_{name}: {len(arr)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--num", type=int, default=0)
    a = ap.parse_args()
    keys = config.load_keys()
    os.makedirs(a.out, exist_ok=True)
    raw_path = os.path.join(a.out, "harvest_raw.jsonl")
    rows = [json.loads(l) for l in open(a.inp)]
    if a.num: rows = rows[:a.num]
    seen = set()
    if os.path.exists(raw_path):
        for l in open(raw_path):
            try: seen.add(json.loads(l)["image"])
            except Exception: pass
    todo = [r for r in rows if r["image"] not in seen]
    print(f"MULTI_EXTRACT rows={len(todo)} model={config.VISION_MODEL}", flush=True)
    def work(ir):
        i, r = ir
        try:
            if not os.path.exists(r["image"]): return None
            d = extract(r["image"], keys[i % len(keys)])
            return {"image": r["image"], "data": d} if d else None
        except Exception: return None
    fr = open(raw_path, "a" if seen else "w"); done = ok = 0
    with cf.ThreadPoolExecutor(max_workers=config.WORKERS) as ex:
        for res in ex.map(work, list(enumerate(todo))):
            done += 1
            if res:
                ok += 1
                with _W: fr.write(json.dumps(res, ensure_ascii=False) + "\n"); fr.flush()
            if done % 50 == 0: print(f"done {done} ok {ok}", flush=True)
    fr.close()
    allr = []
    for l in open(raw_path):          # guard per-line: a torn/empty line must not kill aggregation
        l = l.strip()
        if not l:
            continue
        try:
            r = json.loads(l)
            if r.get("data"):
                allr.append(r)
        except Exception:
            continue
    print(f"MULTI_EXTRACT_DONE frames={len(allr)}", flush=True)
    build_sharegpts(allr, a.out)

if __name__ == "__main__":
    main()
