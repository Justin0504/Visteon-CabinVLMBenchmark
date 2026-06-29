"""TRAFFIC-LIGHT EXTRACTION (#4 active light) — read traffic-light STATE and SHAPE from front-camera
street images with a frontier vision model. nuScenes (and most AV sets) label light *boxes* but not
the live color state, so we read it visually and keep only confident detections.

Captures: color state (red/yellow/green/off), luminaire shape (circle / arrow / pedestrian / countdown),
and ego-relative position. Forbids guessing — "none" if no light is clearly visible.

Input  jsonl rows: {image, camera, ...}
Output jsonl rows: {image, lights:[{state,shape,position,for_lane}]}  (only rows with >=1 light)

Run:
  VULTR_KEYS=keys.env python -m caption_pipeline.trafficlight_extract --inp raw.jsonl --out tl_raw.jsonl
"""
import json, base64, os, argparse, threading
import concurrent.futures as cf
from . import config
from .vultr_client import chat_vision, parse_json, encode_image

_W = threading.Lock()
TL_SYS = "You read traffic-light signal state from driving images. You report only what is clearly visible."
TL_PROMPT = (
    "Look for TRAFFIC LIGHTS (signal heads) in this driving scene. For each clearly visible one report its "
    "lit color state and lamp shape. Output STRICT JSON: "
    '{"has_light":true,"lights":[{"state":"<red/yellow/green/off>","shape":"<circle/arrow/pedestrian/countdown>",'
    '"position":"<left/right/ahead/overhead>","for_lane":"<ego/left-turn/right-turn/crosswalk/unknown>"}]}. '
    'If no traffic light is clearly visible, output {"has_light":false}. Do NOT guess a color you cannot see.'
)

def extract(img_path, key):
    b64 = encode_image(img_path)
    txt = chat_vision(config.VISION_MODEL, TL_SYS, TL_PROMPT, b64, key, config.VISION_MAX_TOKENS, 240)
    if not txt:
        txt = chat_vision(config.VISION_MODEL_FB, TL_SYS, TL_PROMPT, b64, key, 1500, 150)
    d = parse_json(txt)
    if not d or not d.get("has_light"):
        return None
    lights = [l for l in (d.get("lights") or [])
              if l.get("state") and str(l["state"]).lower() in ("red", "yellow", "green", "off")]
    return lights or None

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
    # traffic lights mostly appear in the FORWARD view; prefer front cameras if camera info present
    seen = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try: seen.add(json.loads(l)["image"])
            except Exception: pass
    rows = [r for r in rows if r["image"] not in seen]
    print(f"TL_EXTRACT rows={len(rows)} model={config.VISION_MODEL} workers={config.WORKERS}", flush=True)

    def work(ir):
        i, r = ir
        try:
            if not os.path.exists(r["image"]):
                return None
            lights = extract(r["image"], keys[i % len(keys)])
            return {"image": r["image"], "camera": r.get("camera", ""), "lights": lights} if lights else None
        except Exception:
            return None

    fr = open(a.out, "a" if seen else "w"); done = hit = 0
    with cf.ThreadPoolExecutor(max_workers=config.WORKERS) as ex:
        for res in ex.map(work, list(enumerate(rows))):
            done += 1
            if res:
                hit += 1
                with _W:
                    fr.write(json.dumps(res, ensure_ascii=False) + "\n"); fr.flush()
            if done % 50 == 0:
                print(f"done {done} with_light {hit}", flush=True)
    fr.close()
    print(f"TL_EXTRACT_DONE images={done} with_light={hit}", flush=True)

if __name__ == "__main__":
    main()
