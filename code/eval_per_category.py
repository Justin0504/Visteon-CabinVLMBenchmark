import os,json,torch,re,statistics,gc
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
ROOT="/data/haiyuez/visteon_cabin_vlm"
held=json.load(open(ROOT+"/data/heldout_test.json"))
carlab={json.loads(l)["image"]:json.loads(l)["label"] for l in open(ROOT+"/data/stanford_cars/labels.jsonl")}
signlab={json.loads(l)["image"]:json.loads(l)["label"] for l in open(ROOT+"/data/gtsrb/labels.jsonl")}
MODELS={"base":ROOT+"/models/Qwen2.5-VL-7B-Instruct","v0":ROOT+"/models/bootstrap_v0_merged","v1":ROOT+"/models/bootstrap_v1_merged"}
Q={"cars":"What is the make and model of this vehicle? Answer concisely.",
   "signs":"What does this traffic sign mean? Answer concisely.",
   "exterior":"Describe this exterior driving scene and the key objects, and state one driving consideration."}
def load(m):
    md=Qwen2_5_VLForConditionalGeneration.from_pretrained(m,torch_dtype=torch.bfloat16).to("cuda").eval()
    pr=AutoProcessor.from_pretrained(m,max_pixels=401408); return md,pr
def gen(md,pr,img,text,mx=128):
    msgs=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":text}]}]
    chat=pr.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    ii,_=process_vision_info(msgs); inp=pr(text=[chat],images=ii,return_tensors="pt").to("cuda")
    with torch.no_grad(): o=md.generate(**inp,max_new_tokens=mx,do_sample=False)
    return pr.decode(o[0][inp.input_ids.shape[1]:],skip_special_tokens=True).strip()
# 1) collect answers per model
ans={}
for name,path in MODELS.items():
    if not os.path.exists(path+"/config.json"): print("SKIP",name); continue
    md,pr=load(path); ans[name]={}
    for cat,imgs in held.items():
        ans[name][cat]=[(im,gen(md,pr,im,Q[cat])) for im in imgs]
    del md; gc.collect(); torch.cuda.empty_cache(); print("answered:",name)
# 2) judge with base
jm,jp=load(MODELS["base"])
def correct(img,gtlabel,answer):
    q=(f'Ground-truth label: "{gtlabel}". Candidate answer: "{answer}". '
       'Is the candidate answer consistent with the ground-truth label (same vehicle make/model OR same sign meaning)? Reply ONLY "yes" or "no".')
    r=gen(jm,jp,img,q,8).lower(); return 1 if "yes" in r else 0
def score_ext(img,answer):
    q=(f'Driving scene answer: "{answer}". Rate 1-10 on factuality+completeness+responsibility combined. Reply ONLY an integer.')
    r=gen(jm,jp,img,q,8); m=re.search(r"\d+",r); return int(m.group()) if m else None
res={}
for name in ans:
    res[name]={}
    # cars/signs accuracy vs ground truth
    for cat,lab in [("cars",carlab),("signs",signlab)]:
        accs=[correct(im,lab.get(im,"?"),a) for im,a in ans[name][cat]]
        res[name][cat]=round(100*sum(accs)/len(accs),1)
    exts=[score_ext(im,a) for im,a in ans[name]["exterior"]]; exts=[x for x in exts if x]
    res[name]["exterior"]=round(statistics.mean(exts),2)
json.dump(res,open(ROOT+"/data/eval_per_category.json","w"),indent=1)
print("EVAL_RESULT",json.dumps(res))
print("EVAL_DONE")
