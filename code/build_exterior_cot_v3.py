"""CoT v3 — FAITHFULNESS-TIGHTENED captions (fix nuScenes CoT 53% faithful).
Same GT grounding as v2 (2D bbox + VRU separation + vehicle coarse-class) but a STRICT prompt:
describe ONLY ground-truth objects + plainly-visible facts; forbid inventing lights/signs/weather/
lane-counts/objects not in the GT list. Risk/Decision must reference only listed objects."""
import json,argparse,re
import numpy as np
from collections import Counter
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility, view_points
from vllm import LLM, SamplingParams
R="/data/haiyuez/visteon_cabin_vlm"
ap=argparse.ArgumentParser(); ap.add_argument("--num",type=int,default=0); ap.add_argument("--test",action="store_true")
ap.add_argument("--out",default=R+"/data/exterior_cot_v3_sharegpt.json"); ap.add_argument("--raw",default=R+"/data/exterior_cot_v3_raw.jsonl"); a=ap.parse_args()
nusc=NuScenes(version='v1.0-mini',dataroot=R+'/data/nuscenes',verbose=False)
def simp(x):
    if x.startswith('vehicle.car'): return 'car'
    if x.startswith('vehicle.truck'): return 'truck'
    if x.startswith('vehicle.bus'): return 'bus'
    if x.startswith('vehicle.trailer'): return 'trailer'
    if x.startswith('vehicle.construction'): return 'construction vehicle'
    if x.startswith('vehicle.emergency'): return 'emergency vehicle'
    if x.startswith('vehicle.motorcycle'): return 'motorcyclist (rider+motorcycle)'
    if x.startswith('vehicle.bicycle'): return 'cyclist (rider+bicycle)'
    if x=='human.pedestrian.adult': return 'pedestrian(adult)'
    if x=='human.pedestrian.child': return 'pedestrian(child)'
    if x=='human.pedestrian.construction_worker': return 'pedestrian(worker)'
    if x=='human.pedestrian.police_officer': return 'pedestrian(police)'
    if x=='human.pedestrian.personal_mobility': return 'person on personal-mobility device'
    if x.startswith('human.pedestrian'): return 'pedestrian'
    if x=='movable_object.pushable_pullable': return 'stroller/cart (pushable)'
    if x=='movable_object.trafficcone': return 'traffic cone'
    if x=='movable_object.barrier': return 'barrier'
    return None
VEH={'car','truck','bus','trailer','construction vehicle','emergency vehicle'}
VRU={'cyclist (rider+bicycle)','motorcyclist (rider+motorcycle)','person on personal-mobility device',
     'pedestrian(adult)','pedestrian(child)','pedestrian(worker)','pedestrian(police)','pedestrian'}
ROLE={"CAM_FRONT":("front","stop/go, pedestrians ahead"),"CAM_FRONT_LEFT":("front-left/side","left lane-change/blind-spot"),
"CAM_FRONT_RIGHT":("front-right/side","right lane-change/blind-spot"),"CAM_BACK":("rear","vehicles approaching from behind"),
"CAM_BACK_LEFT":("rear-left/side","rear-left blind-spot"),"CAM_BACK_RIGHT":("rear-right/side","rear-right blind-spot")}
def gt(tok,maxd=40):
    p,boxes,K=nusc.get_sample_data(tok,box_vis_level=BoxVisibility.ANY)
    W,H=Image.open(p).size; objs=[]
    for b in boxes:
        nm=simp(b.name)
        if nm is None: continue
        x,_,z=b.center
        if z<=0 or z>maxd: continue
        corners=view_points(b.corners(),K,normalize=True)[:2,:]
        x1,y1,x2,y2=corners[0].min(),corners[1].min(),corners[0].max(),corners[1].max()
        x1=max(0,int(x1));y1=max(0,int(y1));x2=min(W,int(x2));y2=min(H,int(y2))
        if x2<=x1 or y2<=y1: continue
        pos="ahead" if abs(x)<5 else ("left" if x<0 else "right")
        objs.append((nm,pos,round(float(z),1),[x1,y1,x2,y2]))
    objs=sorted(objs,key=lambda t:t[2]); cnt=Counter(o[0] for o in objs)
    veh=[o for o in objs if o[0] in VEH]; vru=[o for o in objs if o[0] in VRU]
    def fmt(lst): return "; ".join(f"{n} {pos} {d}m bbox[{b[0]},{b[1]},{b[2]},{b[3]}]" for n,pos,d,b in lst[:6]) or "none"
    summ=", ".join(f"{n}:{c}" for n,c in cnt.most_common()) or "no salient objects within 40m"
    return p,(W,H),summ,fmt(veh),fmt(vru)
def qwen(q): return f"<|im_start|>system\nYou are a precise driving-scene captioner. You ONLY state facts supported by the ground-truth object list or plainly visible in the image. You never invent objects, traffic lights, signs, weather, or lane counts.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{q}<|im_end|>\n<|im_start|>assistant\n"
def cot_instr(role,focus,wh,summ,veh,vru):
    return (f'This is the {role} camera. GROUND-TRUTH objects (authoritative, the ONLY objects you may assert): counts {summ}; vehicles {veh}; VRUs {vru}.\n'
     f'Write a SHORT, STRICTLY FACTUAL driving caption in 3 parts:\n'
     f'1) Scene: state ONLY the ground-truth objects above (type, rough position, distance). You may add road surface if clearly visible. DO NOT mention any traffic light, sign, weather, time-of-day, or lane count unless it is unmistakably visible — when unsure, OMIT it.\n'
     f'2) Risk: name the driving risk implied ONLY by the listed objects (e.g. pedestrian ahead, vehicle close). If no salient object, say "low risk".\n'
     f'3) Decision: ego action (proceed/slow/stop/yield) for this camera role ({focus}), justified only by the listed objects.\n'
     f'Be concise. Do not speculate. Output one caption labeled "Scene:", "Risk:", "Decision:".')
def qa_instr(summ,veh,vru):
    return (f'GROUND-TRUTH: counts {summ}; vehicles {veh}; VRUs {vru}.\n'
     f'Generate exactly 3 driving QA grounded ONLY in this ground-truth (one counting, one risk, one action). '
     f'Do not invent objects. Strict JSON: {{"qa":[{{"q":"","a":""}},{{"q":"","a":""}},{{"q":"","a":""}}]}}')
samples=nusc.sample[:5] if a.test else nusc.sample
tasks=[]
for s in samples:
    for cam in (["CAM_FRONT","CAM_FRONT_LEFT","CAM_BACK"] if a.test else ROLE):
        role,focus=ROLE[cam]; tok=s['data'][cam]; p,wh,summ,veh,vru=gt(tok)
        tasks.append((p,role,focus,wh,summ,veh,vru))
if a.test and a.num: tasks=tasks[:a.num]
print("TASKS",len(tasks),flush=True)
llm=LLM(model=R+"/models/Qwen2-VL-7B-Instruct",max_model_len=4608,gpu_memory_utilization=0.55,limit_mm_per_prompt={"image":1})
imgs={p:Image.open(p).convert("RGB") for p,*_ in tasks}
cap=llm.generate([{"prompt":qwen(cot_instr(r,f,wh,su,ve,vr)),"multi_modal_data":{"image":imgs[p]}} for p,r,f,wh,su,ve,vr in tasks],SamplingParams(max_tokens=300,temperature=0.2))
qa=llm.generate([{"prompt":qwen(qa_instr(su,ve,vr)),"multi_modal_data":{"image":imgs[p]}} for p,r,f,wh,su,ve,vr in tasks],SamplingParams(max_tokens=380,temperature=0.3))
def pqa(t):
    try: return json.loads(re.search(r'\{.*\}',t,re.S).group(0)).get("qa",[])[:3]
    except: return []
sg=[];fr=open(a.raw,"w")
for (p,role,focus,wh,summ,veh,vru),co,qo in zip(tasks,cap,qa):
    c=co.outputs[0].text.strip(); qas=pqa(qo.outputs[0].text)
    conv=[{"from":"human","value":"<image>\nDescribe this exterior driving scene and give a driving decision."},{"from":"gpt","value":c}]
    for x in qas:
        if x.get("q") and x.get("a"): conv+=[{"from":"human","value":str(x["q"])},{"from":"gpt","value":str(x["a"])}]
    sg.append({"conversations":conv,"images":[p]})
    fr.write(json.dumps({"image":p,"camera":role,"gt":summ,"caption":c},ensure_ascii=False)+"\n")
fr.close(); json.dump(sg,open(a.out,"w"),ensure_ascii=False)
print("COT_V3_DONE",len(sg),flush=True)
