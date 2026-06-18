import json,argparse
from collections import Counter
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility
from vllm import LLM, SamplingParams
R="/data/haiyuez/visteon_cabin_vlm"
ap=argparse.ArgumentParser(); ap.add_argument("--out",default=R+"/data/exterior_cot_sharegpt.json"); ap.add_argument("--raw",default=R+"/data/exterior_cot_raw.jsonl"); a=ap.parse_args()
nusc=NuScenes(version='v1.0-mini',dataroot=R+'/data/nuscenes',verbose=False)
SIMP={'vehicle.car':'car','vehicle.truck':'truck','vehicle.bus':'bus','vehicle.motorcycle':'motorcycle','vehicle.bicycle':'bicycle','human.pedestrian':'pedestrian','movable_object.barrier':'barrier','movable_object.trafficcone':'traffic cone','vehicle.construction':'construction vehicle','vehicle.trailer':'trailer'}
SAL={'car','truck','bus','pedestrian','bicycle','motorcycle','construction vehicle','traffic cone','trailer'}
def simp(x):
    for k,v in SIMP.items():
        if x.startswith(k): return v
    return x.split('.')[-1]
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
    return p,summ,detail,cnt
def qwen(q): return f"<|im_start|>system\nYou are the perception+reasoning module of an autonomous-driving cockpit.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{q}<|im_end|>\n<|im_start|>assistant\n"
def cot_instr(role,focus,summ,detail,scene):
    return (f'This is the {role} camera. GROUND-TRUTH objects (sensors, authoritative): counts {summ}; nearest {detail}. nuScenes scene note: "{scene}".\n'
     f'Generate a VLA chain-of-thought caption FOR DRIVING, as 3 labeled parts:\n'
     f'1) Scene: road type, traffic elements (lights/signs/lane markings), vehicles, pedestrians/VRU, weather.\n'
     f'2) Risk: driving risks from the above (pedestrian crossing, vehicle cutting in, red light, blind spot).\n'
     f'3) Decision: what ego should do (proceed/slow/stop/yield/lane-change), focusing on this camera role: {focus}.\n'
     f'Use the ground-truth objects to stay accurate. Output ONE caption with parts labeled "Scene:", "Risk:", "Decision:".')
def qa_instr(role,summ,detail):
    return (f'This is the {role} camera. GROUND-TRUTH objects: counts {summ}; nearest {detail}.\n'
     f'Generate exactly 3 diverse driving QA pairs about THIS image (vary the type: one Recognition/counting, one Reasoning/risk, one Decision/action). '
     f'Use ground-truth to stay accurate. Output strict JSON: {{"qa":[{{"q":"","a":""}},{{"q":"","a":""}},{{"q":"","a":""}}]}}')
# 全相机, 前视优先(全保留), 侧/后各取部分以平衡
tasks=[]
for s in nusc.sample:
    for cam in ROLE:
        role,focus=ROLE[cam]; tok=s['data'][cam]; p,summ,detail,cnt=gt(tok); scene=nusc.get('scene',s['scene_token'])['description']
        tasks.append((p,cam,role,focus,summ,detail,scene))
print("TASKS",len(tasks),flush=True)
llm=LLM(model=R+"/models/Qwen2-VL-7B-Instruct",max_model_len=4096,gpu_memory_utilization=0.55,limit_mm_per_prompt={"image":1})
sp_cap=SamplingParams(max_tokens=360,temperature=0.4); sp_qa=SamplingParams(max_tokens=420,temperature=0.5)
imgs={p:Image.open(p).convert("RGB") for p,*_ in tasks}
cap_out=llm.generate([{"prompt":qwen(cot_instr(r,f,su,de,sc)),"multi_modal_data":{"image":imgs[p]}} for p,c,r,f,su,de,sc in tasks],sp_cap)
qa_out=llm.generate([{"prompt":qwen(qa_instr(r,su,de)),"multi_modal_data":{"image":imgs[p]}} for p,c,r,f,su,de,sc in tasks],sp_qa)
import re
def parse_qa(t):
    try:
        m=re.search(r'\{.*\}',t,re.S); o=json.loads(m.group(0)); return o.get("qa",[])[:3]
    except: return []
sg=[]; fr=open(a.raw,"w")
for (p,cam,role,focus,summ,detail,scene),co,qo in zip(tasks,cap_out,qa_out):
    cap=co.outputs[0].text.strip(); qas=parse_qa(qo.outputs[0].text)
    conv=[{"from":"human","value":"<image>\nDescribe this exterior driving scene and give a driving decision."},{"from":"gpt","value":cap}]
    for qa in qas:
        if qa.get("q") and qa.get("a"): conv+=[{"from":"human","value":qa["q"]},{"from":"gpt","value":qa["a"]}]
    sg.append({"conversations":conv,"images":[p]})
    fr.write(json.dumps({"image":p,"camera":role,"gt":summ,"cot_caption":cap,"qa":qas},ensure_ascii=False)+"\n")
fr.close(); json.dump(sg,open(a.out,"w"),ensure_ascii=False)
print("COT_FULL_DONE",len(sg),"avg_turns",round(sum(len(x["conversations"]) for x in sg)/len(sg),1),flush=True)
