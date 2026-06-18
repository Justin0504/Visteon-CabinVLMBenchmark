import json,re,argparse
from collections import Counter
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility
from vllm import LLM, SamplingParams
ROOT="/data/haiyuez/visteon_cabin_vlm"
ap=argparse.ArgumentParser(); ap.add_argument("--cams",default="CAM_FRONT"); ap.add_argument("--num",type=int,default=20)
ap.add_argument("--out",default=ROOT+"/data/exterior_gt_sharegpt.json"); ap.add_argument("--raw",default=ROOT+"/data/exterior_gt_raw.jsonl"); a=ap.parse_args()
nusc=NuScenes(version='v1.0-mini',dataroot=ROOT+'/data/nuscenes',verbose=False)
SIMP={'vehicle.car':'car','vehicle.truck':'truck','vehicle.bus':'bus','vehicle.trailer':'trailer','vehicle.motorcycle':'motorcycle','vehicle.bicycle':'bicycle','human.pedestrian':'pedestrian','movable_object.barrier':'barrier','movable_object.trafficcone':'traffic cone','vehicle.construction':'construction vehicle'}
SALIENT={'car','truck','bus','pedestrian','bicycle','motorcycle','construction vehicle','traffic cone'}
def simp(n):
    for k,v in SIMP.items():
        if n.startswith(k): return v
    return n.split('.')[-1]
def gt(cam_token,maxd=40):
    path,boxes,K=nusc.get_sample_data(cam_token,box_vis_level=BoxVisibility.ANY)
    items=[]
    for b in boxes:
        x,_,z=b.center
        if z>maxd or z<=0: continue
        items.append((simp(b.name),"ahead" if abs(x)<5 else ("left" if x<0 else "right"),round(float(z),1)))
    items=[i for i in items if i[0] in SALIENT]
    cnt=Counter(i[0] for i in items)
    summ=", ".join(f"{n}:{c}" for n,c in cnt.most_common()) or "no salient objects within 40m"
    detail="; ".join(f"{n}({p},{d}m)" for n,p,d in sorted(items,key=lambda t:t[2])[:8])
    return path,summ,detail
def qwen(q): return f"<|im_start|>system\nYou are an in-car cockpit AI looking outside the vehicle.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{q}<|im_end|>\n<|im_start|>assistant\n"
def instr(summ,detail):
    return ('GROUND-TRUTH objects from the vehicle sensors in THIS image (authoritative, use to answer):\n'
     f'Counts (<=40m): {summ}\nNearest (type,position,distance): {detail}\n'
     'Output STRICT JSON {"caption":"<one sentence>","qa":[{"question":"","answer":"","capability":"","use_case":""}]} with caption + EXACTLY 5 QA, using the ground-truth to answer correctly:\n'
     '1) counting (how many cars/pedestrians) -> answer from GT counts (Reasoning/Quantitative)\n'
     '2) distance (how far is the nearest X) -> answer from GT distance (Reasoning/Distance)\n'
     '3) recognition (what notable objects) (Recognition/ExternalRecognition)\n'
     '4) spatial (what is on the left/right/ahead) -> from GT position (Reasoning/Spatial)\n'
     '5) driving decision grounded in the scene (Reasoning/DrivingDecision)\n'
     'Answers MUST match the ground-truth and image. Output ONLY JSON.')
def parse(t):
    t=t.replace("```json","").replace("```",""); s,e=t.find("{"),t.rfind("}")
    try: return json.loads(t[s:e+1])
    except: return None
samples=[]
for cam in a.cams.split(","):
    for s in nusc.sample:
        samples.append(s['data'][cam])
samples=samples[:a.num]
data=[gt(tok) for tok in samples]
llm=LLM(model=ROOT+"/models/Qwen2-VL-7B-Instruct",max_model_len=4096,gpu_memory_utilization=0.5,limit_mm_per_prompt={"image":1})
inputs=[{"prompt":qwen(instr(s,d)),"multi_modal_data":{"image":Image.open(p).convert("RGB")}} for p,s,d in data]
outs=llm.generate(inputs,SamplingParams(max_tokens=700,temperature=0.3))
ok=0; fr=open(a.raw,"w"); sg=[]
for (p,summ,detail),o in zip(data,outs):
    rec=parse(o.outputs[0].text)
    if rec and "caption" in rec and len(rec.get("qa",[]))>=3:
        ok+=1; fr.write(json.dumps({"image":p,"gt":summ,**rec},ensure_ascii=False)+"\n")
        conv=[{"from":"human","value":"<image>\nDescribe this exterior driving scene."},{"from":"gpt","value":rec["caption"]}]
        for q in rec["qa"]:
            if q.get("question") and q.get("answer"): conv+=[{"from":"human","value":q["question"]},{"from":"gpt","value":q["answer"]}]
        sg.append({"conversations":conv,"images":[p]})
fr.close(); json.dump(sg,open(a.out,"w"),ensure_ascii=False)
print(f"GT-recaption OK {ok}/{len(data)}"); print("GTREC_DONE")
