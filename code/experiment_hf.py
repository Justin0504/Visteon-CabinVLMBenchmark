import os,json,glob,re,statistics,sys,torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
sys.path.insert(0,os.path.dirname(__file__)); from prompts_cabin import qa_prompt
def parse_q(t):
    t=t.replace("```json","").replace("```","")
    a,b=t.find("["),t.rfind("]")
    if a>=0 and b>a:
        try:
            arr=json.loads(t[a:b+1])
            if isinstance(arr,list): return [x["question"] for x in arr if isinstance(x,dict) and x.get("question")]
        except: pass
    a,b=t.find("{"),t.rfind("}")
    if a>=0 and b>a:
        try:
            o=json.loads(t[a:b+1])
            if isinstance(o,dict) and "qa" in o: return [x["question"] for x in o["qa"] if x.get("question")]
        except: pass
    return []
ROOT="/data/haiyuez/visteon_cabin_vlm"; NUS=ROOT+"/data/nuscenes"; N=8
M=ROOT+"/models/Qwen2.5-VL-7B-Instruct"; B=ROOT+"/models/bootstrap_v0_merged"
def load(m):
    mdl=Qwen2_5_VLForConditionalGeneration.from_pretrained(m,torch_dtype=torch.bfloat16).to("cuda").eval()
    pr=AutoProcessor.from_pretrained(m,max_pixels=262144); return mdl,pr
def gen(mdl,pr,img,text,mx=200):
    msgs=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":text}]}]
    chat=pr.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    ii,_=process_vision_info(msgs)
    inp=pr(text=[chat],images=ii,return_tensors="pt").to("cuda")
    with torch.no_grad(): o=mdl.generate(**inp,max_new_tokens=mx,do_sample=False)
    return pr.decode(o[0][inp.input_ids.shape[1]:],skip_special_tokens=True).strip()
print("loading base..."); base,bp=load(M)
print("loading bootstrap..."); boot,tp=load(B)
imgs=sorted(glob.glob(NUS+"/samples/CAM_BACK/*.jpg"))[:N]
ANS="As an in-car agent looking OUTSIDE the vehicle, answer this driver question using ONLY the image, concisely and grounded. If it cannot be answered from the image, reply exactly 'Sorry, I cannot answer'. Question: "
state=[]
for p in imgs:
    raw=gen(base,bp,p,qa_prompt(3),mx=900); qs=parse_q(raw)[:3]
    if qs: state.append({"image":p,"questions":qs})
print("questions for",len(state),"images")
for s in state:
    s["base_answers"]=[gen(base,bp,s["image"],ANS+q,180) for q in s["questions"]]
    s["bootstrap_answers"]=[gen(boot,tp,s["image"],ANS+q,180) for q in s["questions"]]
W={"factuality":3,"visual_location":3,"completeness":2,"responsibility":2}
def judge(img,q,ans):
    jp=(f"You are a STRICT evaluator of an in-car VLM answer. The image is the driving scene. Question: {q}\n"
        f"Candidate answer: {ans}\nRate 1-10 (10 best): factuality, visual_location, completeness, responsibility. "
        "Output STRICT JSON only: {\"factuality\":n,\"visual_location\":n,\"completeness\":n,\"responsibility\":n}")
    r=gen(base,bp,img,jp,60); m=re.search(r'\{.*\}',r,re.DOTALL)
    try: d=json.loads(m.group(0)); return sum(float(d[k])*w for k,w in W.items())/sum(W.values())
    except: return None
bs=[];ts=[]
for s in state:
    for j,q in enumerate(s["questions"]):
        v=judge(s["image"],q,s["base_answers"][j]);  bs.append(v) if v else None
        v=judge(s["image"],q,s["bootstrap_answers"][j]); ts.append(v) if v else None
json.dump(state,open(ROOT+"/data/exp_state.json","w"),ensure_ascii=False,indent=1)
res={"base_mean":round(statistics.mean(bs),2),"bootstrap_mean":round(statistics.mean(ts),2),"n_base":len(bs),"n_boot":len(ts)}
json.dump(res,open(ROOT+"/data/exp_scores.json","w"),indent=1)
print("RESULT",json.dumps(res))
