"""Demo builder: 3 scenarios x (baseline vs finetuned) driving-advice comparison.
Pipeline per scenario: pick a frame -> render perception overlay (GT bboxes) -> feed annotated frame
to BOTH models with the SAME general driving-assistant prompt -> collect advice.
Shows the mentor's point: with one general prompt, the finetuned model gives professional driving
advice while the baseline does not. Output: annotated frames + manifest.json (advice per model).
Run (llamafactory env): python demo_build.py
"""
import os, json, re, gc, random, torch
from PIL import Image, ImageDraw, ImageFont
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
R = "/data/haiyuez/visteon_cabin_vlm"
OUT = R + "/demo_assets"; os.makedirs(OUT, exist_ok=True)
MODELS = {"baseline_v9": R + "/models/bootstrap_v9_merged", "finetuned_v16": R + "/models/bootstrap_v16_merged"}
# ONE general prompt for both models (mentor: same general prompt, compare reasonableness)
PROMPT = ("You are an in-car driving assistant looking at the road ahead through the front camera. "
          "Detected objects are marked with boxes. In 1-2 sentences, give the driver clear, professional "
          "driving advice for this exact moment.")
random.seed(3)

def parse_boxes(s):
    out = []
    for m in re.finditer(r"(pedestrian\([a-z]+\)|cyclist|motorcyclist|truck|car|bus|barrier)[^\[]*\[(\d+),(\d+),(\d+),(\d+)\]", str(s)):
        out.append((m.group(1), [int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))]))
    return out

def render(img_path, boxes, dst):
    im = Image.open(img_path).convert("RGB")
    d = ImageDraw.Draw(im)
    for cls, b in boxes[:12]:
        color = (255, 60, 60) if "pedestrian" in cls or "cyclist" in cls or "motor" in cls else (60, 160, 255)
        d.rectangle(b, outline=color, width=3)
        d.text((b[0] + 2, max(0, b[1] - 14)), cls.split("(")[0], fill=color)
    im.save(dst)
    return dst

# ---- pick 3 scenario frames from GT ----
raw = {json.loads(l)["image"]: json.loads(l) for l in open(R + "/data/exterior_cot_v2_raw.jsonl")}
vru = json.load(open(R + "/data/vru_test.json")) if os.path.exists(R + "/data/vru_test.json") else []
scenarios = {}
# pedestrian-risk: crossing=yes and has bbox GT
for x in vru:
    if str(x.get("crossing", "")).lower() == "yes" and x["image"] in raw and os.path.exists(x["image"]):
        scenarios["pedestrian_risk"] = x["image"]; break
# normal: crossing=no, few objects
for x in vru:
    if str(x.get("crossing", "")).lower() == "no" and x["image"] in raw and os.path.exists(x["image"]):
        scenarios["normal"] = x["image"]; break
# red light: from extracted TL (india preferred, clear red)
for f in ["tl_india_raw.jsonl", "tl_raw.jsonl"]:
    p = R + "/data/" + f
    if os.path.exists(p):
        L = [json.loads(l) for l in open(p)]
        red = [r for r in L if r.get("lights") and str(r["lights"][0].get("state", "")).lower() == "red" and os.path.exists(r["image"])]
        if red:
            scenarios["red_light"] = red[0]["image"]; break

print("scenarios:", {k: v.split("/")[-1] for k, v in scenarios.items()}, flush=True)

# render overlays
annot = {}
for sc, img in scenarios.items():
    boxes = parse_boxes(raw[img].get("vehicles", "") + " " + raw[img].get("vrus", "")) if img in raw else []
    dst = os.path.join(OUT, f"{sc}.jpg")
    render(img, boxes, dst)
    annot[sc] = {"orig": img, "annotated": dst, "n_boxes": len(boxes)}

# dual-model advice
def load(m):
    md = Qwen2_5_VLForConditionalGeneration.from_pretrained(m, torch_dtype=torch.bfloat16).to("cuda").eval()
    return md, AutoProcessor.from_pretrained(m, max_pixels=401408)
def gen(md, pr, img, t, mx=80):
    ms = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": t}]}]
    c = pr.apply_chat_template(ms, tokenize=False, add_generation_prompt=True); ii, _ = process_vision_info(ms)
    inp = pr(text=[c], images=ii, return_tensors="pt").to("cuda")
    with torch.no_grad(): o = md.generate(**inp, max_new_tokens=mx, do_sample=False)
    return pr.decode(o[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()

for mname, path in MODELS.items():
    if not os.path.exists(path + "/config.json"): continue
    md, pr = load(path)
    for sc in scenarios:
        annot[sc][mname] = gen(md, pr, annot[sc]["annotated"], PROMPT)
    del md; gc.collect(); torch.cuda.empty_cache()
    print("advised:", mname, flush=True)

json.dump(annot, open(os.path.join(OUT, "manifest.json"), "w"), ensure_ascii=False, indent=1)
print("DEMO_BUILD_DONE")
for sc in scenarios:
    print("\n==", sc, "==")
    print(" baseline:", annot[sc].get("baseline_v9", "")[:120])
    print(" finetuned:", annot[sc].get("finetuned_v16", "")[:120])
