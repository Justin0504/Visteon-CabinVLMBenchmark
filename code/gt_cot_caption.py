import json,re,argparse
from collections import Counter
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility
from vllm import LLM, SamplingParams
R="/data/haiyuez/visteon_cabin_vlm"
ap=argparse.ArgumentParser(); ap.add_argument("--num",type=int,default=8); ap.add_argument("--out",default=R+"/data/exterior_cot_sharegpt.json"); ap.add_argument("--raw",default=R+"/data/exterior_cot_raw.jsonl"); a=ap.parse_args()
nusc=NuScenes(version='v1.0-mini',dataroot=R+'/data/nuscenes',verbose=False)
SIMP={'vehicle.car':'car','vehicle.truck':'truck','vehicle.bus':'bus','vehicle.motorcycle':'motorcycle','vehicle.bicycle':'bicycle','human.pedestrian':'pedestrian','movable_object.barrier':'barrier','movable_object.trafficcone':'traffic cone','vehicle.construction':'construction vehicle'}
SAL={'car','truck','bus','pedestrian','bicycle','motorcycle','construction vehicle','traffic cone'}
def simp(x):
    for k,v in SIMP.items():
        if x.startswith(k): return v
    return x.split('.')[-1]
# 摄像头映射 (Kevin)
ROLE={"CAM_FRONT":("front","stop/go decisions, traffic lights & signs, pedestrians crossing ahead"),
"CAM_FRONT_LEFT":("front-left/side","lane-change safety and blind-spot vehicles on the left"),
"CAM_FRONT_RIGHT":("front-right/side","lane-change safety and blind-spot vehicles on the right"),
"CAM_BACK":("rear","vehicles approaching from behind, safe to reverse/yield"),
"CAM_BACK_LEFT":("rear-left/side","blind-spot and overtaking vehicles from rear-left"),
"CAM_BACK_RIGHT":("rear-right/side","blind-spot and overtaking vehicles from rear-right")}
def gt(tok,maxd=40):
    p,boxes,_=nusc.get_sample_data(tok,box_vis_level=BoxVisibility.ANY); items=[]
    for b in boxes:
        x,_,z=b.center
        if z<=0 or z>maxd: continue
        nm=simp(b.name)
        if nm in SAL: items.append((nm,"ahead" if abs(x)<5 else ("left" if x<0 else "right"),round(float(z),1)))
    cnt=Counter(i[0] for i in items)
    summ=", ".join(f"{n}:{c}" for n,c in cnt.most_common()) or "no salient objects within 40m"
    detail="; ".join(f"{n}({p},{d}m)" for n,p,d in sorted(items,key=lambda t:t[2])[:8])
    return p,summ,detail
def qwen(q): return f"<|im_start|>system\nYou are the perception+reasoning module of an autonomous-driving cockpit.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{q}<|im_end|>\n<|im_start|>assistant\n"
def cot_instr(role,focus,summ,detail,scene):
    return (f'This is the {role} camera. GROUND-TRUTH objects (sensors, authoritative): counts {summ}; nearest {detail}. nuScenes scene note: "{scene}".\n'
     f'Generate a VLA chain-of-thought caption FOR DRIVING, as 3 labeled parts:\n'
     f'1) Scene Description: road type, traffic elements (lights/signs/lane markings), vehicles, pedestrians/VRU, weather.\n'
     f'2) Risk Analysis: identify driving risks from the above (e.g., pedestrian crossing, vehicle cutting in, red light).\n'
     f'3) Driving Decision: what the ego vehicle should do (proceed/slow/stop/yield/lane-change), focusing on this camera role: {focus}.\n'
     f'Use the ground-truth objects to stay accurate. Output ONE flowing caption with the 3 parts labeled "Scene:", "Risk:", "Decision:".')
samples=[]
for s in nusc.sample[:4]:
    for cam in ["CAM_FRONT","CAM_FRONT_LEFT","CAM_BACK"]: samples.append((cam,s))
samples=samples[:a.num]
llm=LLM(model=R+"/models/Qwen2-VL-7B-Instruct",max_model_len=4096,gpu_memory_utilization=0.5,limit_mm_per_prompt={"image":1})
data=[]
for cam,s in samples:
    role,focus=ROLE[cam]; tok=s['data'][cam]; p,summ,detail=gt(tok); scene=nusc.get('scene',s['scene_token'])['description']
    data.append((p,cam,role,focus,summ,detail,scene))
outs=llm.generate([{"prompt":qwen(cot_instr(r,f,su,de,sc)),"multi_modal_data":{"image":Image.open(p).convert("RGB")}} for p,c,r,f,su,de,sc in data],SamplingParams(max_tokens=400,temperature=0.4))
sg=[]; fr=open(a.raw,"w")
for (p,cam,role,focus,summ,detail,scene),o in zip(data,outs):
    cap=o.outputs[0].text.strip()
    fr.write(json.dumps({"image":p,"camera":role,"gt":summ,"cot_caption":cap},ensure_ascii=False)+"\n")
    sg.append({"conversations":[{"from":"human","value":"<image>\nDescribe this exterior driving scene and give a driving decision."},{"from":"gpt","value":cap}],"images":[p]})
fr.close(); json.dump(sg,open(a.out,"w"),ensure_ascii=False)
print("COT_DONE",len(sg))
