"""Comprehensive benchmark eval: standard 5 categories + China-sign slice + VRU-crossing slice.
MODELS configurable. Objective where possible; LLM-judge for scene + sign-meaning."""
import os,json,torch,re,statistics,gc
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
R="/data/haiyuez/visteon_cabin_vlm"
held=json.load(open(R+"/data/heldout_frozen.json"))
carlab={json.loads(l)["image"]:json.loads(l)["label"] for l in open(R+"/data/stanford_cars/labels.jsonl")}
signlab={json.loads(l)["image"]:json.loads(l)["label"] for l in open(R+"/data/gtsrb/labels.jsonl")}
lslab={json.loads(l)["image"]:json.loads(l)["label"] for l in open(R+"/data/landscape/labels.jsonl")}
china=json.load(open(R+"/data/china_test.json")); vru=json.load(open(R+"/data/vru_test.json"))
MODELS={"v9":R+"/models/bootstrap_v9_merged","v10":R+"/models/bootstrap_v10_merged"}
Q={"cars":"What is the make and model of this vehicle? Answer concisely.",
   "signs":"What does this traffic sign mean? Answer concisely.",
   "landscape":"What type of natural landscape is this? Answer concisely.",
   "exterior":"Describe this exterior driving scene and give a driving decision."}
def load(m):
    md=Qwen2_5_VLForConditionalGeneration.from_pretrained(m,torch_dtype=torch.bfloat16).to("cuda").eval()
    return md,AutoProcessor.from_pretrained(m,max_pixels=401408)
def gen(md,pr,img,t,mx=96):
    ms=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":t}]}]
    c=pr.apply_chat_template(ms,tokenize=False,add_generation_prompt=True); ii,_=process_vision_info(ms)
    inp=pr(text=[c],images=ii,return_tensors="pt").to("cuda")
    with torch.no_grad(): o=md.generate(**inp,max_new_tokens=mx,do_sample=False)
    return pr.decode(o[0][inp.input_ids.shape[1]:],skip_special_tokens=True).strip()
ans={}
for nm,p in MODELS.items():
    md,pr=load(p); a={}
    for cat in ["cars","signs","landscape","exterior"]:
        a[cat]=[(im,gen(md,pr,im,Q[cat])) for im in held[cat]]
    a["ocr"]=[(o["image"],o["answer"],gen(md,pr,o["image"],o["question"])) for o in held["ocr"]]
    a["china"]=[(x,gen(md,pr,x["image"],"What does this traffic sign mean? Answer concisely.")) for x in china]
    a["vru"]=[(x,gen(md,pr,x["image"],"Is any pedestrian crossing or about to cross the road ahead? Answer only yes or no.")) for x in vru]
    ans[nm]=a; del md;gc.collect();torch.cuda.empty_cache();print("answered",nm,flush=True)
jm,jp=load(R+"/models/Qwen2.5-VL-7B-Instruct")
def ok(img,gt,a):
    r=gen(jm,jp,img,f'Ground-truth: "{gt}". Answer: "{a}". Consistent? Reply ONLY yes or no.',6).lower(); return 1 if "yes" in r else 0
def sc(img,a):
    r=gen(jm,jp,img,f'Driving answer: "{a}". Rate 1-10 factuality+risk+decision. Reply ONLY an integer.',6); m=re.search(r"\d+",r); return int(m.group()) if m else None
res={}; labmap={"cars":carlab,"signs":signlab,"landscape":lslab}
for nm in ans:
    r={}
    for cat in ["cars","signs","landscape"]:
        v=[ok(im,labmap[cat].get(im,"?"),x) for im,x in ans[nm][cat]]; r[cat]=round(100*sum(v)/len(v),1)
    v=[ok(im,gt,x) for im,gt,x in ans[nm]["ocr"]]; r["ocr"]=round(100*sum(v)/len(v),1)
    v=[ok(x["image"],x["meaning"],a) for x,a in ans[nm]["china"]]; r["china_sign"]=round(100*sum(v)/len(v),1)
    v=[1 if x["crossing"] in a.lower() else 0 for x,a in ans[nm]["vru"]]; r["vru_crossing"]=round(100*sum(v)/len(v),1)
    e=[sc(im,x) for im,x in ans[nm]["exterior"]]; e=[z for z in e if z]; r["exterior"]=round(statistics.mean(e),2)
    res[nm]=r
json.dump(res,open(R+"/data/eval_full.json","w"),indent=1); print("EVAL_FULL",json.dumps(res)); print("EVAL_FULL_DONE")
