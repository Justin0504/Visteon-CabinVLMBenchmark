"""STAGE A — rich visual-context description per image.

Replaces a small local VLM with a frontier multimodal model on Vultr (Qwen3.5-397B), falling back
to a fast vision model (Nemotron-Omni) when the primary times out (504 on long reasoning). Produces
the weather/time/road/lane/surface/building detail that ground-truth labels lack. No GPU needed —
inference is on Vultr; this process only reads images, base64-encodes them, and calls the API.

Input  jsonl rows: {image, camera, gt, vehicles, vrus}   (gt/vehicles/vrus are pass-through strings)
Output jsonl rows: same + {vision_desc}

Run:
  VULTR_KEYS=keys.env python -m caption_pipeline.stage_a_vision --inp raw.jsonl --out vision.jsonl
Tuning for the fast fallback model (avoids 504): VISION_MODEL=...Omni... VISION_MAX_TOKENS=1500 WORKERS=2
"""
import json, base64, os, argparse, threading
import concurrent.futures as cf
from . import config, prompts
from .vultr_client import chat_vision, encode_image

_W = threading.Lock()

def describe(img_path, key):
    b64 = encode_image(img_path)
    # primary (best quality)
    d = chat_vision(config.VISION_MODEL, prompts.VISION_SYS, prompts.VISION_PROMPT, b64, key, config.VISION_MAX_TOKENS, 240)
    if d:
        return d
    # fast fallback (handles primary 504s); capped tokens so it doesn't over-think
    return chat_vision(config.VISION_MODEL_FB, prompts.VISION_SYS, prompts.VISION_PROMPT, b64, key, 1500, 150)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="jsonl with rows {image,camera,gt,vehicles,vrus}")
    ap.add_argument("--out", required=True, help="output jsonl (resumable)")
    ap.add_argument("--num", type=int, default=0, help="limit rows (0 = all)")
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
    print(f"STAGE_A rows={len(rows)} model={config.VISION_MODEL} workers={config.WORKERS}", flush=True)

    def work(ir):
        i, r = ir
        try:
            if not os.path.exists(r["image"]):
                return None
            d = describe(r["image"], keys[i % len(keys)])
            if not d:
                return None
            return {"image": r["image"], "camera": r.get("camera", ""), "gt": r.get("gt", ""),
                    "vehicles": r.get("vehicles", "none"), "vrus": r.get("vrus", "none"), "vision_desc": d}
        except Exception:
            return None

    fr = open(a.out, "a" if seen else "w"); done = ok = 0
    with cf.ThreadPoolExecutor(max_workers=config.WORKERS) as ex:
        for res in ex.map(work, list(enumerate(rows))):
            done += 1
            if res:
                ok += 1
                with _W:
                    fr.write(json.dumps(res, ensure_ascii=False) + "\n"); fr.flush()
            if done % 50 == 0:
                print(f"done {done} ok {ok}", flush=True)
    fr.close()
    print(f"STAGE_A_DONE ok {ok}", flush=True)

if __name__ == "__main__":
    main()
