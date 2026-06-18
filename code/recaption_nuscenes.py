import os, json, glob, argparse, re
from collections import Counter
from PIL import Image
from vllm import LLM, SamplingParams

ap=argparse.ArgumentParser()
ap.add_argument("--nuscenes", default="/data/haiyuez/visteon_cabin_vlm/data/nuscenes")
ap.add_argument("--cams", default="CAM_FRONT,CAM_FRONT_LEFT,CAM_FRONT_RIGHT")
ap.add_argument("--num", type=int, default=500)
ap.add_argument("--raw", default="/data/haiyuez/visteon_cabin_vlm/data/seed_v0_raw.jsonl")
ap.add_argument("--sharegpt", default="/data/haiyuez/visteon_cabin_vlm/data/seed_v0_sharegpt.json")
ap.add_argument("--model", default="/data/haiyuez/visteon_cabin_vlm/models/Qwen2-VL-7B-Instruct")
a=ap.parse_args()

imgs=[]
for cam in a.cams.split(","):
    imgs+=sorted(glob.glob(os.path.join(a.nuscenes,"samples",cam,"*.jpg")))
imgs=imgs[:a.num]
print(f"{len(imgs)} images")

SYS=("You are an expert data annotator building a vision-language FINE-TUNING set for an "
     "automotive intelligent-cockpit AI that perceives OUTSIDE the vehicle via external cameras.")
INSTR=('Look at this exterior driving image and output STRICT JSON only. Schema:\n'
 '{"caption":"<one detailed sentence>","qa":[{"question":"","answer":"","capability":"","use_case":""}]}\n'
 'Generate exactly 1 caption and EXACTLY 5 QA pairs including: >=2 Recognition '
 '(vehicles/pedestrians/signs/weather/road), >=2 Reasoning (agent intent, may-ego-proceed, '
 'how close/many, a driving decision), >=1 WorldKnowledge (sign meaning, traffic rule, building/POI).\n'
 'capability in Recognition|Reasoning|WorldKnowledge. use_case in '
 'ExternalRecognition|VehicleMakeModel|TrafficSign|Pedestrian|DrivingDecision|SceneDescription|POI.\n'
 'Every question MUST be answerable from THIS image (no hallucination). Output ONLY JSON.')

def bp(q):
    return ("<|im_start|>system\n"+SYS+"<|im_end|>\n<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>"+q+"<|im_end|>\n<|im_start|>assistant\n")

llm=LLM(model=a.model,max_model_len=4096,gpu_memory_utilization=0.6,limit_mm_per_prompt={"image":1})
sp=SamplingParams(max_tokens=1024,temperature=0.5,top_p=0.9)
inputs=[{"prompt":bp(INSTR),"multi_modal_data":{"image":Image.open(p).convert("RGB")}} for p in imgs]
outs=llm.generate(inputs,sp)

def pj(t):
    t=t.strip().replace("```json","").replace("```","")
    s=t.find("{"); e=t.rfind("}")
    if s<0 or e<0: return None
    frag=t[s:e+1]
    try: return json.loads(frag)
    except:
        try: return json.loads(frag[:frag.rfind('},')+1]+']}')
        except: return None

ok=0; caps=Counter(); sgpt=[]
fr=open(a.raw,"w")
for p,o in zip(imgs,outs):
    rec=pj(o.outputs[0].text)
    if rec and "caption" in rec and isinstance(rec.get("qa"),list) and len(rec["qa"])>=3:
        ok+=1
        fr.write(json.dumps({"image":p,**rec},ensure_ascii=False)+"\n")
        caps.update(q.get("capability","") for q in rec["qa"])
        conv=[{"from":"human","value":"<image>\nDescribe this exterior driving scene in one sentence."},
              {"from":"gpt","value":rec["caption"]}]
        for q in rec["qa"]:
            if q.get("question") and q.get("answer"):
                conv.append({"from":"human","value":q["question"]})
                conv.append({"from":"gpt","value":q["answer"]})
        sgpt.append({"conversations":conv,"images":[p]})
    else:
        fr.write(json.dumps({"image":p,"parse_error":True},ensure_ascii=False)+"\n")
fr.close()
json.dump(sgpt, open(a.sharegpt,"w"), ensure_ascii=False, indent=1)
print(f"OK {ok}/{len(imgs)} | sharegpt samples: {len(sgpt)} | caps: {dict(caps)}")
print("DONE_MARKER")
