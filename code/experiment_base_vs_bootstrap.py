import os,json,glob,argparse,re,statistics,sys
from PIL import Image
from vllm import LLM, SamplingParams
sys.path.insert(0,os.path.dirname(__file__)); from prompts_cabin import qa_prompt
ap=argparse.ArgumentParser()
ap.add_argument("--stage",required=True); ap.add_argument("--model",required=True)
ap.add_argument("--state",default="/data/haiyuez/visteon_cabin_vlm/data/exp_state.json")
ap.add_argument("--nuscenes",default="/data/haiyuez/visteon_cabin_vlm/data/nuscenes")
ap.add_argument("--num",type=int,default=10)
a=ap.parse_args()
def load(m): return LLM(model=m,max_model_len=4096,gpu_memory_utilization=0.55,limit_mm_per_prompt={"image":1})
def IM(p): return Image.open(p).convert("RGB")
def qwen(q): return ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"+q+"<|im_end|>\n<|im_start|>assistant\n")
ANSP=lambda q:qwen("As an in-car agent looking OUTSIDE the vehicle, answer this driver question using ONLY the image, concisely and grounded. If it cannot be answered from the image, reply exactly 'Sorry, I can't answer'. Question: "+q)
def answers(llm,state):
    flat=[(i,q) for i,s in enumerate(state) for q in s["questions"]]
    o=llm.generate([{"prompt":ANSP(q),"multi_modal_data":{"image":IM(state[i]["image"])}} for i,q in flat],SamplingParams(max_tokens=200,temperature=0.2))
    res={}
    for (i,q),oo in zip(flat,o): res.setdefault(i,[]).append(oo.outputs[0].text.strip())
    return [res.get(i,[]) for i in range(len(state))]

if a.stage=="questions":
    imgs=sorted(glob.glob(os.path.join(a.nuscenes,"samples","CAM_BACK","*.jpg")))[:a.num]
    llm=load(a.model)
    o=llm.generate([{"prompt":qwen(qa_prompt(3)),"multi_modal_data":{"image":IM(p)}} for p in imgs],SamplingParams(max_tokens=800,temperature=0.3))
    state=[]
    for p,oo in zip(imgs,o):
        m=re.search(r'\{.*\}',oo.outputs[0].text,re.DOTALL); qs=[]
        try: qs=[x["question"] for x in json.loads(m.group(0))["qa"] if x.get("question")][:3]
        except: pass
        state.append({"image":p,"questions":qs})
    state=[s for s in state if s["questions"]]
    for s,ans in zip(state,answers(llm,state)): s["base_answers"]=ans
    json.dump(state,open(a.state,"w"),ensure_ascii=False,indent=1); print("QUESTIONS_DONE",len(state))
elif a.stage=="bootstrap":
    state=json.load(open(a.state)); llm=load(a.model)
    for s,ans in zip(state,answers(llm,state)): s["bootstrap_answers"]=ans
    json.dump(state,open(a.state,"w"),ensure_ascii=False,indent=1); print("BOOTSTRAP_DONE")
elif a.stage=="judge":
    state=json.load(open(a.state)); llm=load(a.model)
    W={"factuality":3,"visual_location":3,"completeness":2,"responsibility":2}
    JP=lambda q,ans:qwen(f"You are a STRICT evaluator of an in-car VLM. The image is the driving scene. Question: {q}\nCandidate answer: {ans}\nRate 1-10 (10=best): factuality(matches image), visual_location(spatial correctness), completeness, responsibility(safe/sensible). Output STRICT JSON only: {{\"factuality\":n,\"visual_location\":n,\"completeness\":n,\"responsibility\":n}}")
    def sc(items):
        o=llm.generate([{"prompt":JP(q,an),"multi_modal_data":{"image":IM(im)}} for im,q,an in items],SamplingParams(max_tokens=60,temperature=0)); ws=[]
        for oo in o:
            m=re.search(r'\{.*\}',oo.outputs[0].text,re.DOTALL)
            try: d=json.loads(m.group(0)); ws.append(sum(float(d[k])*w for k,w in W.items())/sum(W.values()))
            except: pass
        return ws
    bi=[(s["image"],q,s["base_answers"][j]) for s in state for j,q in enumerate(s["questions"]) if j<len(s.get("base_answers",[]))]
    ti=[(s["image"],q,s["bootstrap_answers"][j]) for s in state for j,q in enumerate(s["questions"]) if j<len(s.get("bootstrap_answers",[]))]
    bs=sc(bi); ts=sc(ti)
    out={"base_mean":round(statistics.mean(bs),2),"bootstrap_mean":round(statistics.mean(ts),2),"n_base":len(bs),"n_boot":len(ts)}
    json.dump({**out,"base_scores":bs,"bootstrap_scores":ts},open(a.state.replace(".json","_scores.json"),"w"),indent=1)
    print("RESULT",json.dumps(out)); print("JUDGE_DONE")
