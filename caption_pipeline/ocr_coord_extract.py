"""OCR + COORDINATE grounding (#5) — read readable text in a driving scene AND give each text's
bounding box, so text is mapped to spatial coordinates (the use-case requirement). Same Vultr vision
client as the rest of the pipeline. Honest caveat: VLM box coordinates are approximate, not pixel-exact.

Input  jsonl rows: {image, ...}
Output jsonl rows: {image, texts:[{text, bbox:[x1,y1,x2,y2], position}]}  (normalized 0-1000 coords)
Output sharegpt: text->location QA.

Run:
  VULTR_KEYS=keys.env python -m caption_pipeline.ocr_coord_extract --inp raw.jsonl --out ocr_raw.jsonl --sg ocr_sharegpt.json
"""
import json, base64, os, argparse, threading, logging
import concurrent.futures as cf
from . import config
from .vultr_client import chat_vision, parse_json, encode_image

_W = threading.Lock()
_log = logging.getLogger("ocr")
OCR_SYS = "You read on-scene text (signs, billboards, shop names, road text) and report where each appears. You never invent text."
OCR_PROMPT = (
    "List every clearly READABLE text string in this driving scene and its bounding box. Use image coordinates "
    "normalized to 0-1000 (x1,y1 = top-left, x2,y2 = bottom-right). Output STRICT JSON: "
    '{"has_text":true,"texts":[{"text":"<exact words>","bbox":[x1,y1,x2,y2],"position":"<left/right/ahead/overhead>"}]}. '
    'If no text is legible, output {"has_text":false}. Transcribe only text you can actually read — do NOT invent.'
)

def extract(img_path, key):
    b64 = encode_image(img_path)
    txt = chat_vision(config.VISION_MODEL, OCR_SYS, OCR_PROMPT, b64, key, config.VISION_MAX_TOKENS, 240)
    if not txt:
        txt = chat_vision(config.VISION_MODEL_FB, OCR_SYS, OCR_PROMPT, b64, key, 1500, 150)
    d = parse_json(txt)
    if not d or not d.get("has_text"):
        return None
    out = []
    for t in (d.get("texts") or []):
        bb = t.get("bbox")
        if t.get("text") and isinstance(bb, list) and len(bb) == 4:
            out.append({"text": str(t["text"]), "bbox": [int(x) for x in bb], "position": t.get("position", "")})
    return out or None

def to_sharegpt(rec):
    texts = rec["texts"]
    listing = "; ".join(f'"{t["text"]}" at {t["bbox"]}' for t in texts)
    convs = [{"from": "human", "value": "<image>\nRead all visible text and give each one's bounding box (coords 0-1000)."},
             {"from": "gpt", "value": listing + "."}]
    t0 = texts[0]
    convs += [{"from": "human", "value": f'Where is the text "{t0["text"]}" located?'},
              {"from": "gpt", "value": f'At bounding box {t0["bbox"]} ({t0.get("position","")}).'}]
    return {"conversations": convs, "images": [rec["image"]]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--sg", required=True); ap.add_argument("--num", type=int, default=0)
    a = ap.parse_args()
    keys = config.load_keys()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    rows = [json.loads(l) for l in open(a.inp)]
    if a.num: rows = rows[:a.num]
    seen = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try: seen.add(json.loads(l)["image"])
            except Exception: pass
    rows = [r for r in rows if r["image"] not in seen]
    print(f"OCR_COORD rows={len(rows)} model={config.VISION_MODEL} workers={config.WORKERS}", flush=True)
    def work(ir):
        i, r = ir
        try:
            if not os.path.exists(r["image"]): return None
            ts = extract(r["image"], keys[i % len(keys)])
            return {"image": r["image"], "texts": ts} if ts else None
        except Exception as e:
            _log.warning("row failed %s: %s", r.get("image"), e); return None
    done = hit = 0
    with open(a.out, "a" if seen else "w") as fr, cf.ThreadPoolExecutor(max_workers=config.WORKERS) as ex:
        for res in ex.map(work, list(enumerate(rows))):
            done += 1
            if res:
                hit += 1
                with _W: fr.write(json.dumps(res, ensure_ascii=False) + "\n"); fr.flush()
            if done % 50 == 0: print(f"done {done} with_text {hit}", flush=True)
    allr = []
    for l in open(a.out):
        l = l.strip()
        if not l: continue
        try: allr.append(json.loads(l))
        except Exception: continue
    with open(a.sg + ".tmp", "w") as f:
        json.dump([to_sharegpt(r) for r in allr], f, ensure_ascii=False)
    os.replace(a.sg + ".tmp", a.sg)
    print(f"OCR_COORD_DONE images={done} with_text={hit} sharegpt={len(allr)}", flush=True)

if __name__ == "__main__":
    main()
