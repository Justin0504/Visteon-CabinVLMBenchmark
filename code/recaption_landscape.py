import os,json,re
from PIL import Image
from vllm import LLM, SamplingParams
ROOT="/data/haiyuez/visteon_cabin_vlm"
def _resize(im,mx=768,mn=256):
    w,h=im.size; m=max(w,h)
    if m>mx: r=mx/m; im=im.resize((max(28,int(w*r)),max(28,int(h*r))))
    elif m<mn: r=mn/m; im=im.resize((max(28,int(w*r)),max(28,int(h*r))))
    return im
SYS="You are an in-car cockpit AI that describes scenery seen outside the vehicle."
def prm(label):
    return ('This image shows natural scenery. GROUND-TRUTH type: "'+label+'".\n'
     'Output STRICT JSON only: {"caption":"<one sentence>","qa":[{"question":"","answer":"","capability":"","use_case":""}]}\n'
     'Exactly 3 QA: 1) "What type of natural landscape is this?" -> answer based on ground truth (Recognition/NaturalLandscape); '
     '2) describe the key natural features visible (Recognition/NaturalLandscape); '
     '3) a driving or travel consideration for this terrain (Reasoning/DrivingDecision). Output ONLY JSON.')
def qwen(q): return f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{q}<|im_end|>\n<|im_start|>assistant\n"
def parse(t):
    t=t.replace("```json","").replace("```",""); s,e=t.find("{"),t.rfind("}")
    if s<0: return None
    try: return json.loads(t[s:e+1])
    except: return None
rows=[json.loads(l) for l in open(ROOT+"/data/landscape/labels.jsonl")]
llm=LLM(model=ROOT+"/models/Qwen2-VL-7B-Instruct",max_model_len=4096,gpu_memory_utilization=0.55,limit_mm_per_prompt={"image":1})
inputs=[{"prompt":qwen(prm(r["label"])),"multi_modal_data":{"image":_resize(Image.open(r["image"]).convert("RGB"))}} for r in rows]
outs=llm.generate(inputs,SamplingParams(max_tokens=500,temperature=0.3))
ok=0; sg=[]
for r,o in zip(rows,outs):
    rec=parse(o.outputs[0].text)
    if rec and "caption" in rec and isinstance(rec.get("qa"),list) and len(rec["qa"])>=2:
        ok+=1; conv=[{"from":"human","value":"<image>\nDescribe the natural landscape in this image."},{"from":"gpt","value":rec["caption"]}]
        for q in rec["qa"]:
            if q.get("question") and q.get("answer"): conv+=[{"from":"human","value":q["question"]},{"from":"gpt","value":q["answer"]}]
        sg.append({"conversations":conv,"images":[r["image"]]})
json.dump(sg,open(ROOT+"/data/landscape_sharegpt.json","w"),ensure_ascii=False)
print(f"landscape: OK {ok}/{len(rows)}"); print("LS_DONE")
