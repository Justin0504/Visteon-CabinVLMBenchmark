"""FUSION Stage-A (server, Qwen2-VL vision): produce a RICH visual-context description per image —
exactly the detail GT annotations lack (weather, time, road type, lane markings, surface, lighting,
roadside buildings/vegetation). Merged with GT facts -> input for vultr_fusion.py (Stage-B).
The model is told NOT to count/list dynamic objects (those come from sensor GT, to avoid hallucinated counts)."""
import json, argparse
from PIL import Image
from vllm import LLM, SamplingParams
R = "/data/haiyuez/visteon_cabin_vlm"
ap = argparse.ArgumentParser()
ap.add_argument("--inp", default=R + "/data/exterior_cot_v2_raw.jsonl")  # has image,camera,gt,vehicles,vrus
ap.add_argument("--out", default=R + "/data/fusion_input.jsonl")
ap.add_argument("--num", type=int, default=0)
a = ap.parse_args()
rows = [json.loads(l) for l in open(a.inp)]
if a.num: rows = rows[:a.num]
def qwen(q):
    return ("<|im_start|>system\nYou describe ONLY the visual context of a driving scene.<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>" + q + "<|im_end|>\n<|im_start|>assistant\n")
INSTR = ("Describe the VISUAL CONTEXT of this exterior driving image in 2-3 sentences: weather, time of day, "
         "road type (urban/highway/intersection/parking), lane markings, road-surface condition, lighting, and "
         "roadside buildings/vegetation. Do NOT count or list vehicles/pedestrians — only the scene context.")
llm = LLM(model=R + "/models/Qwen2-VL-7B-Instruct", max_model_len=4096, gpu_memory_utilization=0.55, limit_mm_per_prompt={"image": 1})
imgs = {r["image"]: Image.open(r["image"]).convert("RGB") for r in rows}
out = llm.generate([{"prompt": qwen(INSTR), "multi_modal_data": {"image": imgs[r["image"]]}} for r in rows], SamplingParams(max_tokens=120, temperature=0.3))
fw = open(a.out, "w")
for r, o in zip(rows, out):
    r2 = {"image": r["image"], "camera": r["camera"], "gt": r["gt"], "vehicles": r.get("vehicles", "none"), "vrus": r.get("vrus", "none"), "vision_desc": o.outputs[0].text.strip()}
    fw.write(json.dumps(r2, ensure_ascii=False) + "\n")
fw.close()
print("VISION_DESC_DONE", len(rows))
