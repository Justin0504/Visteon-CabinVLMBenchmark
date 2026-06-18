import os,json,torch,re,statistics,gc
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
R="/data/haiyuez/visteon_cabin_vlm"
held=json.load(open(R+"/data/heldout_frozen.json"))
carlab={json.loads(l)["image"]:json.loads(l)["label"] for l in open(R+"/data/stanford_cars/labels.jsonl")}
signlab={json.loads(l)["image"]:json.loads(l)["label"] for l in open(R+"/data/gtsrb/labels.jsonl")}
lslab={json.loads(l)["image"]:json.loads(l)["label"] for l in open(R+"/data/landscape/labels.jsonl")}
MODELS={"base":R+"/models/Qwen2.5-VL-7B-Instruct","v1":R+"/models/bootstrap_v1_merged","v2":R+"/models/bootstrap_v2_merged","v3":R+"/models/bootstrap_v3_merged","v4":R+"/models/bootstrap_v4_merged"}
Q={"cars":"What is the make and model of this vehicle? Answer concisely.",
   "signs":"What does this traffic sign mean? Answer concisely.",
   "landscape":"What type of natural landscape is this? Answer concisely.",
   "exterior":"Describe this exterior driving scene and key objects, and state one driving consideration."}
def load(m):
    md=Qwen2_5_VLForConditionalGeneration.from_pretrained(m,torch_dtype=torch.bfloat16).to("cuda").eval()
    pr=AutoProcessor.from_pretrained(m,max_pixels=401408); return md,pr
def gen(md,pr,img,t,mx=96):
    ms=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":t}]}]
    c=pr.apply_chat_template(ms,tokenize=False,add_generation_prompt=True); ii,_=process_vision_info(ms)
    inp=pr(text=[c],images=ii,return_tensors="pt").to("cuda")
    with torch.no_grad(): o=md.generate(**inp,max_new_tokens=mx,do_sample=False)
    return pr.decode(o[0][inp.input_ids.shape[1]:],skip_special_tokens=True).strip()
ans={}
for name,path in MODELS.items():
    if not os.path.exists(path+"/config.json"): print("SKIP",name); continue
    md,pr=load(path); ans[name]={}
    for cat in ["cars","signs","landscape","exterior"]:
        ans[name][cat]=[(im,gen(md,pr,im,Q[cat])) for im in held[cat]]
    ans[name]["ocr"]=[(o["image"],o["answer"],gen(md,pr,o["image"],o["question"])) for o in held["ocr"]]
    del md; gc.collect(); torch.cuda.empty_cache(); print("answered:",name)
jm,jp=load(MODELS["base"])
def ok(img,gt,a):
    r=gen(jm,jp,img,f'Ground-truth: "{gt}". Candidate answer: "{a}". Do they mean the same / is candidate consistent with ground-truth? Reply ONLY yes or no.',8).lower()
    return 1 if "yes" in r else 0
def sc(img,a):
    r=gen(jm,jp,img,f'Driving scene answer: "{a}". Rate 1-10 factuality+completeness+responsibility. Reply ONLY an integer.',8); m=re.search(r"\d+",r); return int(m.group()) if m else None
res={}
labmap={"cars":carlab,"signs":signlab,"landscape":lslab}
for name in ans:
    res[name]={}
    for cat in ["cars","signs","landscape"]:
        a=[ok(im,labmap[cat].get(im,"?"),ansr) for im,ansr in ans[name][cat]]; res[name][cat]=round(100*sum(a)/len(a),1)
    a=[ok(im,gt,ansr) for im,gt,ansr in ans[name]["ocr"]]; res[name]["ocr"]=round(100*sum(a)/len(a),1)
    e=[sc(im,ansr) for im,ansr in ans[name]["exterior"]]; e=[x for x in e if x]; res[name]["exterior"]=round(statistics.mean(e),2)
json.dump(res,open(R+"/data/eval_frozen_v4.json","w"),indent=1); print("EVAL_FROZEN",json.dumps(res)); print("EVAL_FROZEN_DONE")
