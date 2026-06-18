"""#4 Traffic-light caption via label-grounded color reading.
road-traffic (HF, no login) gives GT traffic-light bboxes; we crop each light and let the VLM
read ONLY its color+shape (localized -> near-zero hallucination), then assemble a driving caption."""
import json,argparse,re,io
from PIL import Image
from datasets import load_dataset
from vllm import LLM, SamplingParams
R="/data/haiyuez/visteon_cabin_vlm"
ap=argparse.ArgumentParser(); ap.add_argument("--num",type=int,default=0); ap.add_argument("--test",action="store_true")
ap.add_argument("--out",default=R+"/data/trafficlight_sharegpt.json"); ap.add_argument("--raw",default=R+"/data/trafficlight_raw.jsonl"); a=ap.parse_args()
CATS=['road-traffic','bicycles','buses','crosswalks','fire hydrants','motorcycles','traffic lights','vehicles']
LIGHT=CATS.index('traffic lights'); CW=CATS.index('crosswalks')
ds=load_dataset('Francesco/road-traffic',split='train')
imgdir=R+"/data/road_traffic_imgs"; import os; os.makedirs(imgdir,exist_ok=True)
def crop(im,b,pad=6):
    x,y,w,h=b; W,H=im.size
    x1=max(0,int(x-pad)); y1=max(0,int(y-pad)); x2=min(W,int(x+w+pad)); y2=min(H,int(y+h+pad))
    c=im.crop((x1,y1,x2,y2))
    if min(c.size)<64:  # 放大小灯
        s=64/min(c.size); c=c.resize((int(c.size[0]*s),int(c.size[1]*s)))
    return c
def qwen(q): return f"<|im_start|>system\nYou read traffic-light state from a tight crop.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{q}<|im_end|>\n<|im_start|>assistant\n"
READ='This crop is a single traffic light. Output strict JSON: {"color":"red|yellow|green|off|unknown","shape":"circle|left-arrow|right-arrow|forward-arrow|pedestrian|unknown"}'
# 收集含灯的图
rows=[]
for i,r in enumerate(ds):
    cats=r['objects']['category']; boxes=r['objects']['bbox']
    lights=[boxes[j] for j,c in enumerate(cats) if c==LIGHT]
    if not lights: continue
    has_cw=any(c==CW for c in cats)
    rows.append((i,r,lights,has_cw))
if a.test: rows=rows[:a.num or 4]
elif a.num: rows=rows[:a.num]
print("IMGS_WITH_LIGHTS",len(rows),flush=True)
llm=LLM(model=R+"/models/Qwen2-VL-7B-Instruct",max_model_len=2048,gpu_memory_utilization=0.5,limit_mm_per_prompt={"image":1})
# 一次性读所有灯 crop
crops=[]; idx=[]
imgs={}
for k,(i,r,lights,cw) in enumerate(rows):
    im=r['image'].convert("RGB"); imgs[i]=im
    for b in lights[:5]:
        crops.append(crop(im,b)); idx.append(k)
outs=llm.generate([{"prompt":qwen(READ),"multi_modal_data":{"image":c}} for c in crops],SamplingParams(max_tokens=40,temperature=0.0))
COLORS={"red","yellow","green","off"}; SHAPES={"circle","left-arrow","right-arrow","forward-arrow","pedestrian"}
def pr(t):
    try: o=json.loads(re.search(r"\{.*\}",t,re.S).group(0))
    except: o={}
    c=o.get("color","unknown"); s=o.get("shape","unknown")
    return {"color":c if c in COLORS else "unknown","shape":s if s in SHAPES else "unknown"}
# 聚合每图的灯态
per={}
for j,o in zip(idx,outs): per.setdefault(j,[]).append(pr(o.outputs[0].text))
DEC={'red':'stop and wait behind the line','green':'proceed with caution','yellow':'prepare to stop','off':'treat as uncontrolled, yield','unknown':'approach cautiously'}
sg=[]; fr=open(a.raw,"w")
for k,(i,r,lights,cw) in enumerate(rows):
    states=per.get(k,[]); 
    desc=", ".join(f"{s.get('color','unknown')} {s.get('shape','')}".strip() for s in states) or "traffic light present"
    prim=next((s['color'] for s in states if s.get('color') in DEC and s['color']!='unknown'),'unknown')
    p=f"{imgdir}/rt_{i:04d}.jpg"; imgs[i].save(p,quality=90)
    cap=(f"Scene: An exterior road scene with {len(lights)} traffic light(s) ahead"+(", and a pedestrian crosswalk" if cw else "")+f". The active light state(s): {desc}. "
         f"Risk: Traffic-light controlled intersection"+(", pedestrians may cross at the crosswalk" if cw else "")+f". Decision: The ego vehicle should {DEC.get(prim,'approach cautiously')}.")
    conv=[{"from":"human","value":"<image>\nDescribe the traffic light state in this driving scene and give a driving decision."},{"from":"gpt","value":cap},
          {"from":"human","value":"What color is the active traffic light?"},{"from":"gpt","value":(prim if prim!='unknown' else 'not clearly visible')+("" if prim=='unknown' else f" — the ego should {DEC[prim]}.")}]
    sg.append({"conversations":conv,"images":[p]})
    fr.write(json.dumps({"image":p,"n_lights":len(lights),"crosswalk":cw,"states":states,"caption":cap},ensure_ascii=False)+"\n")
fr.close(); json.dump(sg,open(a.out,"w"),ensure_ascii=False)
print("TL_DONE",len(sg),flush=True)
