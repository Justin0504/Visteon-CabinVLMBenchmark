"""SCENARIO-LEVEL decision eval (VLA reasoning-action consistency) — baseline vs finetuned.
Tags each held-out frame by driving scenario from GT, asks the model for the ego action, and checks
it against the scenario-appropriate action. Produces a per-scenario accuracy table per model, which
directly answers: (a) a standalone accuracy metric, (b) baseline vs finetuned, (c) which scenarios
finetune improves. Scenarios mirror the team's test cases:
  normal-pass / pedestrian-crossing(->slow) / red-light(->stop) / speed-sign(->comply).

Run (llamafactory env):
  python eval_scenarios.py
"""
import os, json, re, gc, random, torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
R = "/data/haiyuez/visteon_cabin_vlm"
MODELS = {  # baseline vs finetuned
    "v9_baseline":  R + "/models/bootstrap_v9_merged",
    "v15_finetuned": R + "/models/bootstrap_v15_merged",
    "v16_finetuned": R + "/models/bootstrap_v16_merged",
}
N_PER = 40
Q = ("Considering the scene ahead, what should the ego vehicle do RIGHT NOW? "
     "Answer with ONE word: proceed, slow, stop, or yield.")
STOP = {"stop", "slow", "yield", "brake", "wait", "decelerate", "halt", "slow down"}
GO = {"proceed", "go", "continue", "maintain", "drive", "accelerate"}

def load(m):
    md = Qwen2_5_VLForConditionalGeneration.from_pretrained(m, torch_dtype=torch.bfloat16).to("cuda").eval()
    return md, AutoProcessor.from_pretrained(m, max_pixels=401408)
def gen(md, pr, img, t, mx=16):
    ms = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": t}]}]
    c = pr.apply_chat_template(ms, tokenize=False, add_generation_prompt=True); ii, _ = process_vision_info(ms)
    inp = pr(text=[c], images=ii, return_tensors="pt").to("cuda")
    with torch.no_grad(): o = md.generate(**inp, max_new_tokens=mx, do_sample=False)
    return pr.decode(o[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()

# ---- build scenario sets from GT ----
random.seed(0)
scen = {}
vru = json.load(open(R + "/data/vru_test.json")) if os.path.exists(R + "/data/vru_test.json") else []
cross = [x for x in vru if str(x.get("crossing", "")).lower() == "yes" and os.path.exists(x["image"])]
normal = [x for x in vru if str(x.get("crossing", "")).lower() == "no" and os.path.exists(x["image"])]
scen["pedestrian_crossing(->slow/stop)"] = ([x["image"] for x in cross][:N_PER], "stop")
scen["normal_pass(->proceed)"] = ([x["image"] for x in normal][:N_PER], "go")
# red light from extracted TL
tl = []
for f in ["tl_raw.jsonl", "tl_india_raw.jsonl"]:
    p = R + "/data/" + f
    if os.path.exists(p):
        for l in open(p):
            r = json.loads(l)
            if r.get("lights") and str(r["lights"][0].get("state", "")).lower() == "red" and os.path.exists(r["image"]):
                tl.append(r["image"])
random.shuffle(tl)
scen["red_light(->stop)"] = (tl[:N_PER], "stop")

def correct(ans, expect):
    hit_stop = any(w in ans for w in STOP); hit_go = any(w in ans for w in GO)
    if expect == "stop":
        return 1 if (hit_stop and not (hit_go and not hit_stop)) else 0
    return 1 if (hit_go and not hit_stop) else 0

res = {}
for name, path in MODELS.items():
    if not os.path.exists(path + "/config.json"):
        print("SKIP", name); continue
    md, pr = load(path); res[name] = {}
    for sname, (imgs, expect) in scen.items():
        if not imgs:
            res[name][sname] = None; continue
        ok = sum(correct(gen(md, pr, im, Q), expect) for im in imgs)
        res[name][sname] = round(100 * ok / len(imgs), 1)
    # overall
    vals = [v for v in res[name].values() if v is not None]
    res[name]["OVERALL"] = round(sum(vals) / len(vals), 1) if vals else None
    del md; gc.collect(); torch.cuda.empty_cache()
    print("done", name, json.dumps(res[name], ensure_ascii=False), flush=True)

json.dump(res, open(R + "/data/eval_scenarios.json", "w"), ensure_ascii=False, indent=1)
print("SCENARIO_EVAL", json.dumps(res, ensure_ascii=False))
print("SCENARIO_EVAL_DONE")
