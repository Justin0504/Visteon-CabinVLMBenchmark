"""#6 VRU caption from JAAD: pedestrians + crossing/looking/action behavior (GT) -> VLA caption."""
import os,json,argparse,glob,re
import xml.etree.ElementTree as ET
import cv2
from PIL import Image
from vllm import LLM, SamplingParams
R="/data/haiyuez/visteon_cabin_vlm"; J=R+"/data/JAAD"
ap=argparse.ArgumentParser(); ap.add_argument("--clips",type=int,default=50); ap.add_argument("--perclip",type=int,default=5); ap.add_argument("--test",action="store_true")
ap.add_argument("--out",default=R+"/data/jaad_vru_sharegpt.json"); ap.add_argument("--raw",default=R+"/data/jaad_vru_raw.jsonl"); a=ap.parse_args()
os.makedirs(R+"/data/jaad_frames",exist_ok=True)
def parse(xml):
    fr={}
    for tr in ET.parse(xml).getroot().findall(".//track"):
        if tr.attrib.get("label") not in ("pedestrian","ped"): continue
        for b in tr.findall("box"):
            if b.attrib.get("outside")=="1": continue
            f=int(b.attrib["frame"]); at={c.attrib.get("name"):c.text for c in b.findall("attribute")}
            bbox=[int(float(b.attrib[k])) for k in ("xtl","ytl","xbr","ybr")]
            fr.setdefault(f,[]).append({"bbox":bbox,"cross":at.get("cross"),"look":at.get("look"),"action":at.get("action")})
    return fr
def pick(fr,n):
    # 优先有横穿行人的帧,均匀采样
    cross=[f for f,ps in fr.items() if any(p["cross"]=="crossing" for p in ps)]
    base=cross if len(cross)>=n else sorted(fr.keys())
    if not base: return []
    step=max(1,len(base)//n); return sorted(base)[::step][:n]
clips=sorted(glob.glob(J+"/JAAD_clips/*.mp4"))[:(2 if a.test else a.clips)]
tasks=[]
for mp4 in clips:
    vid=os.path.basename(mp4).replace(".mp4","")
    xml=J+"/annotations/"+vid+".xml"
    if not os.path.exists(xml): continue
    fr=parse(xml); 
    for f in pick(fr,a.perclip):
        tasks.append((mp4,vid,f,fr[f]))
print("TASKS",len(tasks),flush=True)
# 抽帧
imgs={}
for mp4,vid,f,ps in tasks:
    cap=cv2.VideoCapture(mp4); cap.set(cv2.CAP_PROP_POS_FRAMES,f); ok,fr_img=cap.read(); cap.release()
    if not ok: continue
    p=f"{R}/data/jaad_frames/{vid}_{f}.jpg"; cv2.imwrite(p,fr_img); imgs[(vid,f)]=p
def vru_str(ps):
    out=[]
    for p in ps[:6]:
        beh=[]
        if p["cross"]=="crossing": beh.append("crossing")
        if p["look"]=="looking": beh.append("looking at ego")
        if p["action"]: beh.append(p["action"])
        out.append(f"pedestrian {'/'.join(beh) or 'present'} bbox{p['bbox']}")
    return "; ".join(out)
def qwen(q): return f"<|im_start|>system\nYou are the perception+reasoning module of an autonomous-driving cockpit.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{q}<|im_end|>\n<|im_start|>assistant\n"
def instr(vs):
    return (f"Front camera. GROUND-TRUTH pedestrians (with behavior, authoritative): {vs}.\n"
     f"Generate a VLA chain-of-thought caption FOR DRIVING, 3 labeled parts:\n"
     f"1) Scene: road + pedestrians/VRUs and what they are doing (crossing/looking/walking/standing).\n"
     f"2) Risk: crossing or about-to-cross pedestrians = high risk; reference bbox positions.\n"
     f"3) Decision: ego action (stop/yield/slow/proceed) prioritizing pedestrian safety.\n"
     f'Use the GT behavior to stay accurate. Output ONE caption labeled "Scene:", "Risk:", "Decision:".')
data=[(imgs[(vid,f)],ps) for mp4,vid,f,ps in tasks if (vid,f) in imgs]
llm=LLM(model=R+"/models/Qwen2-VL-7B-Instruct",max_model_len=3072,gpu_memory_utilization=0.5,limit_mm_per_prompt={"image":1})
outs=llm.generate([{"prompt":qwen(instr(vru_str(ps))),"multi_modal_data":{"image":Image.open(p).convert("RGB")}} for p,ps in data],SamplingParams(max_tokens=320,temperature=0.4))
sg=[];fr=open(a.raw,"w")
for (p,ps),o in zip(data,outs):
    c=o.outputs[0].text.strip()
    sg.append({"conversations":[{"from":"human","value":"<image>\nDescribe the pedestrians and give a driving decision."},{"from":"gpt","value":c}],"images":[p]})
    fr.write(json.dumps({"image":p,"vru":vru_str(ps),"caption":c},ensure_ascii=False)+"\n")
fr.close(); json.dump(sg,open(a.out,"w"),ensure_ascii=False)
print("JAAD_DONE",len(sg),flush=True)
