"""Phase A: enhanced VLA CoT caption — adds 2D detection-box coords (Kevin 06-11 20:52),
VRU separation (#6), vehicle coarse-class (#2), traffic-light VLM pseudo-label (#4, Senna-style)."""
import json,argparse,re
import numpy as np
from collections import Counter
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility, view_points
from vllm import LLM, SamplingParams
R="/data/haiyuez/visteon_cabin_vlm"
ap=argparse.ArgumentParser(); ap.add_argument("--num",type=int,default=8); ap.add_argument("--test",action="store_true")
ap.add_argument("--out",default=R+"/data/exterior_cot_v2_sharegpt.json"); ap.add_argument("--raw",default=R+"/data/exterior_cot_v2_raw.jsonl"); a=ap.parse_args()
nusc=NuScenes(version='v1.0-mini',dataroot=R+'/data/nuscenes',verbose=False)
# 车型粗类 + VRU 细分
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
ROLE={"CAM_FRONT":("front","stop/go decisions, traffic lights & signs, pedestrians crossing ahead"),
"CAM_FRONT_LEFT":("front-left/side","lane-change safety and blind-spot vehicles on the left"),
"CAM_FRONT_RIGHT":("front-right/side","lane-change safety and blind-spot vehicles on the right"),
"CAM_BACK":("rear","vehicles approaching from behind, safe to reverse/yield"),
"CAM_BACK_LEFT":("rear-left/side","blind-spot and overtaking vehicles from rear-left"),
"CAM_BACK_RIGHT":("rear-right/side","blind-spot and overtaking vehicles from rear-right")}
def gt(tok,maxd=40):
    p,boxes,K=nusc.get_sample_data(tok,box_vis_level=BoxVisibility.ANY)
    W,H=Image.open(p).size; objs=[]
    for b in boxes:
        nm=simp(b.name)
        if nm is None: continue
        x,_,z=b.center
        if z<=0 or z>maxd: continue
        # 投影 3D 框 8 角点 -> 2D 像素 bbox
        corners=view_points(b.corners(),K,normalize=True)[:2,:]
        x1,y1=corners[0].min(),corners[1].min(); x2,y2=corners[0].max(),corners[1].max()
        x1=max(0,int(x1)); y1=max(0,int(y1)); x2=min(W,int(x2)); y2=min(H,int(y2))
        if x2<=x1 or y2<=y1: continue
        pos="ahead" if abs(x)<5 else ("left" if x<0 else "right")
        objs.append((nm,pos,round(float(z),1),[x1,y1,x2,y2]))
    objs=sorted(objs,key=lambda t:t[2])
    cnt=Counter(o[0] for o in objs)
    veh=[o for o in objs if o[0] in VEH]; vru=[o for o in objs if o[0] in VRU]
    def fmt(lst): return "; ".join(f"{n} {pos} {d}m bbox[{b[0]},{b[1]},{b[2]},{b[3]}]" for n,pos,d,b in lst[:6]) or "none"
    summ=", ".join(f"{n}:{c}" for n,c in cnt.most_common()) or "no salient objects within 40m"
    return p,(W,H),summ,fmt(veh),fmt(vru),len(vru)
def qwen(q,extra=""): return f"<|im_start|>system\nYou are the perception+reasoning module of an autonomous-driving cockpit.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{q}<|im_end|>\n<|im_start|>assistant\n"
def cot_instr(role,focus,wh,summ,veh,vru,scene):
    return (f'This is the {role} camera (image {wh[0]}x{wh[1]}). GROUND-TRUTH (sensors, authoritative):\n'
     f'- counts: {summ}\n- vehicles [class pos distance bbox(x1,y1,x2,y2)]: {veh}\n- VRUs (riders separated from two-wheelers): {vru}\n- scene note: "{scene}"\n'
     f'Generate a VLA chain-of-thought caption FOR DRIVING, 3 labeled parts:\n'
     f'1) Scene: road type, traffic elements (traffic LIGHT state if visible: red/yellow/green & shape circle/arrow; signs; lane markings), vehicles by type, pedestrians/VRUs, weather/time.\n'
     f'2) Risk: driving risks (pedestrian/cyclist crossing, vehicle cutting in, red light, blind spot), reference the bbox positions.\n'
     f'3) Decision: ego action (proceed/slow/stop/yield/lane-change) for this camera role: {focus}.\n'
     f'Use the GT classes/bboxes/distances to stay accurate. Output ONE caption labeled "Scene:", "Risk:", "Decision:".')
def qa_instr(role,summ,veh,vru):
    return (f'This is the {role} camera. GT counts: {summ}. vehicles: {veh}. VRUs: {vru}.\n'
     f'Generate exactly 3 diverse driving QA (one Recognition/counting referencing a bbox location, one Reasoning/risk, one Decision/action). '
     f'Use GT to stay accurate. Strict JSON: {{"qa":[{{"q":"","a":""}},{{"q":"","a":""}},{{"q":"","a":""}}]}}')
tasks=[]
samples=nusc.sample[:2] if a.test else nusc.sample
for s in samples:
    for cam in (["CAM_FRONT","CAM_FRONT_LEFT","CAM_BACK"] if a.test else ROLE):
        role,focus=ROLE[cam]; tok=s['data'][cam]; p,wh,summ,veh,vru,nvru=gt(tok); scene=nusc.get('scene',s['scene_token'])['description']
        tasks.append((p,cam,role,focus,wh,summ,veh,vru,scene))
if a.test: tasks=tasks[:a.num]
print("TASKS",len(tasks),flush=True)
llm=LLM(model=R+"/models/Qwen2-VL-7B-Instruct",max_model_len=4608,gpu_memory_utilization=0.55,limit_mm_per_prompt={"image":1})
imgs={p:Image.open(p).convert("RGB") for p,*_ in tasks}
cap=llm.generate([{"prompt":qwen(cot_instr(r,f,wh,su,ve,vr,sc)),"multi_modal_data":{"image":imgs[p]}} for p,c,r,f,wh,su,ve,vr,sc in tasks],SamplingParams(max_tokens=400,temperature=0.4))
qa=llm.generate([{"prompt":qwen(qa_instr(r,su,ve,vr)),"multi_modal_data":{"image":imgs[p]}} for p,c,r,f,wh,su,ve,vr,sc in tasks],SamplingParams(max_tokens=420,temperature=0.5))
def pqa(t):
    try: return json.loads(re.search(r'\{.*\}',t,re.S).group(0)).get("qa",[])[:3]
    except: return []
sg=[]; fr=open(a.raw,"w")
for (p,cam,role,focus,wh,summ,veh,vru,scene),co,qo in zip(tasks,cap,qa):
    c=co.outputs[0].text.strip(); qas=pqa(qo.outputs[0].text)
    conv=[{"from":"human","value":"<image>\nDescribe this exterior driving scene and give a driving decision."},{"from":"gpt","value":c}]
    for x in qas:
        if x.get("q") and x.get("a"): conv+=[{"from":"human","value":x["q"]},{"from":"gpt","value":x["a"]}]
    sg.append({"conversations":conv,"images":[p]})
    fr.write(json.dumps({"image":p,"camera":role,"gt":summ,"vehicles":veh,"vrus":vru,"cot_caption":c,"qa":qas},ensure_ascii=False)+"\n")
fr.close(); json.dump(sg,open(a.out,"w"),ensure_ascii=False)
print("V2_DONE",len(sg),flush=True)
