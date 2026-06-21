"""CoT v4 — AI cross-checked captions (raise faithfulness without degrading to list-dumps).
Pass 1: generate a natural Scene->Risk->Decision caption (v2 style, GT-grounded).
Pass 2: an independent cross-check that, given the caption + GT object list + image, DELETES/CORRECTS
any claim not supported by GT or clearly visible, keeping natural language. Output = verified caption.
"""
import json,argparse,re
from collections import Counter
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility, view_points
from vllm import LLM, SamplingParams
R="/data/haiyuez/visteon_cabin_vlm"
ap=argparse.ArgumentParser(); ap.add_argument("--num",type=int,default=0); ap.add_argument("--test",action="store_true")
ap.add_argument("--out",default=R+"/data/exterior_cot_v4_sharegpt.json"); ap.add_argument("--raw",default=R+"/data/exterior_cot_v4_raw.jsonl"); a=ap.parse_args()
nusc=NuScenes(version='v1.0-mini',dataroot=R+'/data/nuscenes',verbose=False)
def simp(x):
    if x.startswith('vehicle.car'): return 'car'
    if x.startswith('vehicle.truck'): return 'truck'
    if x.startswith('vehicle.bus'): return 'bus'
    if x.startswith('vehicle.trailer'): return 'trailer'
    if x.startswith('vehicle.construction'): return 'construction vehicle'
    if x.startswith('vehicle.emergency'): return 'emergency vehicle'
    if x.startswith('vehicle.motorcycle'): return 'motorcyclist (rider on motorcycle)'
    if x.startswith('vehicle.bicycle'): return 'cyclist (rider on bicycle)'
    if x.startswith('human.pedestrian.child'): return 'child pedestrian'
    if x.startswith('human.pedestrian.construction_worker'): return 'worker pedestrian'
    if x.startswith('human.pedestrian.police_officer'): return 'police officer'
    if x.startswith('human.pedestrian'): return 'pedestrian'
    if x=='movable_object.pushable_pullable': return 'stroller/cart'
    if x=='movable_object.trafficcone': return 'traffic cone'
    if x=='movable_object.barrier': return 'barrier'
    return None
ROLE={"CAM_FRONT":("front","stop/go, pedestrians ahead"),"CAM_FRONT_LEFT":("front-left/side","left blind-spot/lane-change"),
"CAM_FRONT_RIGHT":("front-right/side","right blind-spot/lane-change"),"CAM_BACK":("rear","vehicles from behind"),
"CAM_BACK_LEFT":("rear-left/side","rear-left blind-spot"),"CAM_BACK_RIGHT":("rear-right/side","rear-right blind-spot")}
def gt(tok,maxd=40):
    p,boxes,K=nusc.get_sample_data(tok,box_vis_level=BoxVisibility.ANY); items=[]
    for b in boxes:
        nm=simp(b.name)
        if nm is None: continue
        x,_,z=b.center
        if z<=0 or z>maxd: continue
        items.append((nm,"ahead" if abs(x)<5 else ("left" if x<0 else "right"),round(float(z),1)))
    items=sorted(items,key=lambda t:t[2]); cnt=Counter(i[0] for i in items)
    summ=", ".join(f"{n}:{c}" for n,c in cnt.most_common()) or "no salient objects within 40m"
    near="; ".join(f"{n} {pos} {d}m" for n,pos,d in items[:6]) or "none"
    return p,summ,near
def qwen(sysmsg,q): return f"<|im_start|>system\n{sysmsg}<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{q}<|im_end|>\n<|im_start|>assistant\n"
SYS1="You are the perception+reasoning module of an autonomous-driving cockpit."
def p1(role,focus,summ,near):
    return (f'This is the {role} camera. GROUND-TRUTH objects (authoritative): counts {summ}; nearest {near}.\n'
     f'Write a natural VLA caption in 3 parts: Scene (road + the listed objects), Risk (driving risk from them), '
     f'Decision (ego action for {focus}). Labeled "Scene:", "Risk:", "Decision:".')
SYS2="You are a strict fact-checker for driving captions. You delete or correct any claim not supported by ground-truth or not clearly visible, while keeping the text natural."
def p2(summ,near,cap):
    return (f'GROUND-TRUTH objects (the ONLY objects that may be asserted): counts {summ}; nearest {near}.\n'
     f'Draft caption: "{cap}".\n'
     f'Return a CORRECTED caption: remove any traffic light, sign, weather, lane-count, or object NOT in the ground-truth and not clearly visible; fix wrong counts; keep all ground-truth-supported content; keep the natural "Scene:/Risk:/Decision:" form and keep it concise. Output only the corrected caption.')
samples=nusc.sample[:5] if a.test else nusc.sample
tasks=[]
for s in samples:
    for cam in (["CAM_FRONT","CAM_FRONT_LEFT","CAM_BACK"] if a.test else ROLE):
        role,focus=ROLE[cam]; tok=s['data'][cam]; p,summ,near=gt(tok)
        tasks.append((p,role,focus,summ,near))
if a.test and a.num: tasks=tasks[:a.num]
print("TASKS",len(tasks),flush=True)
llm=LLM(model=R+"/models/Qwen2-VL-7B-Instruct",max_model_len=4096,gpu_memory_utilization=0.55,limit_mm_per_prompt={"image":1})
imgs={p:Image.open(p).convert("RGB") for p,*_ in tasks}
# pass1
o1=llm.generate([{"prompt":qwen(SYS1,p1(r,f,su,ne)),"multi_modal_data":{"image":imgs[p]}} for p,r,f,su,ne in tasks],SamplingParams(max_tokens=300,temperature=0.3))
draft=[x.outputs[0].text.strip() for x in o1]
# pass2 cross-check
o2=llm.generate([{"prompt":qwen(SYS2,p2(su,ne,d)),"multi_modal_data":{"image":imgs[p]}} for (p,r,f,su,ne),d in zip(tasks,draft)],SamplingParams(max_tokens=300,temperature=0.1))
final=[x.outputs[0].text.strip() for x in o2]
# diversified QA grounded in GT
def qa_instr(su,ne): return (f'GROUND-TRUTH: counts {su}; nearest {ne}. Generate 3 driving QA grounded ONLY in this (one counting, one risk, one action). Strict JSON: {{"qa":[{{"q":"","a":""}},{{"q":"","a":""}},{{"q":"","a":""}}]}}')
oq=llm.generate([{"prompt":qwen(SYS1,qa_instr(su,ne)),"multi_modal_data":{"image":imgs[p]}} for p,r,f,su,ne in tasks],SamplingParams(max_tokens=360,temperature=0.3))
def pqa(t):
    try: return json.loads(re.search(r'\{.*\}',t,re.S).group(0)).get("qa",[])[:3]
    except: return []
sg=[];fr=open(a.raw,"w")
for (p,role,focus,summ,near),cap,qo in zip(tasks,final,oq):
    qas=pqa(qo.outputs[0].text)
    conv=[{"from":"human","value":"<image>\nDescribe this exterior driving scene and give a driving decision."},{"from":"gpt","value":cap}]
    for x in qas:
        if x.get("q") and x.get("a"): conv+=[{"from":"human","value":str(x["q"])},{"from":"gpt","value":str(x["a"])}]
    sg.append({"conversations":conv,"images":[p]})
    fr.write(json.dumps({"image":p,"camera":role,"gt":summ,"caption":cap},ensure_ascii=False)+"\n")
fr.close(); json.dump(sg,open(a.out,"w"),ensure_ascii=False)
print("COT_V4_DONE",len(sg),flush=True)
