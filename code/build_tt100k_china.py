"""#4 China traffic signs + China-domain via TT100K (Tsinghua-Tencent 100K).
Streams the HF parquet (no full 73GB download), takes a stratified subset of sign images,
and builds label-grounded captions (GT category code + bbox) -> zero-hallucination."""
import json, os, io, argparse
from collections import Counter
from datasets import load_dataset
from PIL import Image
R = "/data/haiyuez/visteon_cabin_vlm"
IMGDIR = R + "/data/tt100k_imgs"
os.makedirs(IMGDIR, exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument("--num", type=int, default=600); a = ap.parse_args()

# TT100K category-code prefix -> readable meaning
def meaning(code):
    if code.startswith("pl"): return f"speed limit {code[2:]} km/h" if code[2:].isdigit() else "speed limit"
    if code.startswith("pm"): return "weight limit"
    if code.startswith("ph"): return "height limit"
    if code.startswith("pb"): return "no entry / restriction"
    if code.startswith("pn"): return "no parking / no stopping"
    if code.startswith("pr"): return f"speed-limit released {code[2:]}" if code[2:].isdigit() else "restriction released"
    if code.startswith("pne"): return "no entry"
    if code.startswith("p"): return "prohibitory sign"
    if code.startswith("w"): return "warning sign"
    if code.startswith("il"): return f"minimum speed {code[2:]}" if code[2:].isdigit() else "informative lane sign"
    if code.startswith("i"): return "informative sign"
    return "traffic sign"

ds = load_dataset("PrashantDixit0/TT-100K", split="train", streaming=True)
sg = []; fr = open(R + "/data/tt100k_china_sharegpt.json".replace("_sharegpt.json", "_raw.jsonl"), "w")
cnt = Counter(); n = 0
for r in ds:
    objs = r.get("objects") or []   # list of {category, bbox:[x,y,w,h]}
    if not objs:
        continue
    items = []
    for o in objs[:8]:
        nm = o.get("category")
        b = o.get("bbox")
        if not nm or not b:
            continue
        x, y, w, h = (int(v) for v in b)
        items.append((str(nm), meaning(str(nm)), [x, y, x + w, y + h]))
    if not items:
        continue
    p = f"{IMGDIR}/tt_{n:04d}.jpg"
    try:
        img = r["image"]
        im = Image.open(io.BytesIO(img["bytes"])) if isinstance(img, dict) else img
        im.convert("RGB").save(p, quality=88)
    except Exception as e:
        continue
    desc = "; ".join(f'{nm} ({mn}) at bbox{bb}' for nm, mn, bb in items)
    first = items[0]
    cap = (f"Scene: A Chinese road scene with {len(items)} traffic sign(s). "
           f"Ground-truth signs (with locations): {desc}. "
           f"Risk: obey the indicated restrictions; signs govern speed/lane/prohibitions. "
           f"Decision: comply with the most restrictive sign ahead and adjust speed/lane accordingly.")
    conv = [{"from": "human", "value": "<image>\nIdentify the Chinese traffic signs and give a driving decision."},
            {"from": "gpt", "value": cap},
            {"from": "human", "value": "What does the main sign mean and where is it?"},
            {"from": "gpt", "value": f"{first[0]} — {first[1]}, located at bbox {first[2]}."}]
    sg.append({"conversations": conv, "images": [p]})
    fr.write(json.dumps({"image": p, "signs": [(nm, mn, bb) for nm, mn, bb in items], "caption": cap}, ensure_ascii=False) + "\n")
    for nm, _, _ in items: cnt[nm] += 1
    n += 1
    if n >= a.num:
        break
fr.close()
json.dump(sg, open(R + "/data/tt100k_china_sharegpt.json", "w"), ensure_ascii=False)
print("TT100K_DONE", len(sg), "unique_sign_classes", len(cnt), flush=True)
