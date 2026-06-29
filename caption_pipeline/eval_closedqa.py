"""CLOSED-SET QA ACCURACY — the standalone, objective accuracy metric (NuScenes-QA / TIFA style).
Generate closed-set questions from GT (no LLM judge, exact/loose match) and measure accuracy per model.
Question types: object count, object presence (cyclist/truck), traffic-light color. Reports one overall
accuracy + per-type breakdown, baseline vs finetuned — the clean number the team asked for.

Run (llamafactory env): python eval_closedqa.py
"""
import os, json, re, gc, random, torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
R = "/data/haiyuez/visteon_cabin_vlm"
MODELS = {
    "v9_baseline":   R + "/models/bootstrap_v9_merged",
    "v15_finetuned": R + "/models/bootstrap_v15_merged",
    "v16_finetuned": R + "/models/bootstrap_v16_merged",
}
N = 60
random.seed(0)

def load(m):
    md = Qwen2_5_VLForConditionalGeneration.from_pretrained(m, torch_dtype=torch.bfloat16).to("cuda").eval()
    return md, AutoProcessor.from_pretrained(m, max_pixels=401408)
def gen(md, pr, img, t, mx=16):
    ms = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": t}]}]
    c = pr.apply_chat_template(ms, tokenize=False, add_generation_prompt=True); ii, _ = process_vision_info(ms)
    inp = pr(text=[c], images=ii, return_tensors="pt").to("cuda")
    with torch.no_grad(): o = md.generate(**inp, max_new_tokens=mx, do_sample=False)
    return pr.decode(o[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()

# ---- build closed-set QA from GT ----
rows = [json.loads(l) for l in open(R + "/data/exterior_cot_v2_raw.jsonl")]
rows = [r for r in rows if os.path.exists(r["image"])]
random.shuffle(rows)
def pc(s): return len(re.findall(r"pedestrian\(adult\)|pedestrian\(child\)", str(s).lower()))
qa = []  # (image, question, type, checker)
for r in rows[:N]:
    v = str(r.get("vrus", "")); veh = str(r.get("vehicles", "")).lower()
    n = pc(v)
    qa.append((r["image"], "How many pedestrians are ahead? Answer with just a number.", "count",
               lambda a, n=n: (lambda m: m and (int(m.group()) == n or abs(int(m.group()) - n) <= 1))(re.search(r"\d+", a))))
    hc = "cyclist" in v.lower()
    qa.append((r["image"], "Is there a cyclist (person riding a bicycle) ahead? Answer yes or no.", "presence_cyclist",
               lambda a, hc=hc: ("yes" in a) == hc))
    ht = "truck" in veh
    qa.append((r["image"], "Is there a truck ahead? Answer yes or no.", "presence_truck",
               lambda a, ht=ht: ("yes" in a) == ht))
# traffic-light color (closed-set) from extracted GT
for f in ["tl_raw.jsonl", "tl_india_raw.jsonl"]:
    p = R + "/data/" + f
    if not os.path.exists(p): continue
    L = [json.loads(l) for l in open(p)]
    random.shuffle(L)
    for r in L[:30]:
        if not r.get("lights") or not os.path.exists(r["image"]): continue
        st = str(r["lights"][0].get("state", "")).lower()
        if st not in ("red", "yellow", "green"): continue
        qa.append((r["image"], "What color is the traffic light ahead? Answer red, yellow, or green.", "light_color",
                   lambda a, st=st: st in a))

res = {}
for name, path in MODELS.items():
    if not os.path.exists(path + "/config.json"):
        print("SKIP", name); continue
    md, pr = load(path)
    by = {}
    for img, q, typ, chk in qa:
        ok = 1 if chk(gen(md, pr, img, q)) else 0
        by.setdefault(typ, [0, 0]); by[typ][0] += ok; by[typ][1] += 1
    acc = {t: round(100 * c[0] / c[1], 1) for t, c in by.items()}
    tot = sum(c[0] for c in by.values()); den = sum(c[1] for c in by.values())
    acc["OVERALL_ACC"] = round(100 * tot / den, 1)
    res[name] = acc
    del md; gc.collect(); torch.cuda.empty_cache()
    print("done", name, json.dumps(acc, ensure_ascii=False), flush=True)

json.dump(res, open(R + "/data/eval_closedqa.json", "w"), ensure_ascii=False, indent=1)
print("CLOSEDQA_EVAL", json.dumps(res, ensure_ascii=False))
print("CLOSEDQA_EVAL_DONE")
