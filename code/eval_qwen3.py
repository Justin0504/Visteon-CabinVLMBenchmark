import os,json,torch,re,statistics,gc
from transformers import Qwen3VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
R="/data/haiyuez/visteon_cabin_vlm"
held=json.load(open(R+"/data/heldout_frozen.json"))
carlab={json.loads(l)["image"]:json.loads(l)["label"] for l in open(R+"/data/stanford_cars/labels.jsonl")}
signlab={json.loads(l)["image"]:json.loads(l)["label"] for l in open(R+"/data/gtsrb/labels.jsonl")}
lslab={json.loads(l)["image"]:json.loads(l)["label"] for l in open(R+"/data/landscape/labels.jsonl")}
Q={"cars":"What is the make and model of this vehicle? Answer concisely.","signs":"What does this traffic sign mean? Answer concisely.","landscape":"What type of natural landscape is this? Answer concisely.","exterior":"Describe this exterior driving scene and key objects, and state one driving consideration."}
def gen(md,pr,img,t,mx=96):
    ms=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":t}]}]
    c=pr.apply_chat_template(ms,tokenize=False,add_generation_prompt=True); ii,_=process_vision_info(ms)
    inp=pr(text=[c],images=ii,return_tensors="pt").to("cuda")
    with torch.no_grad(): o=md.generate(**inp,max_new_tokens=mx,do_sample=False)
    return pr.decode(o[0][inp.input_ids.shape[1]:],skip_special_tokens=True).strip()
# Qwen3-VL-4B answers
M=R+"/models/Qwen3-VL-4B-Instruct"
md=Qwen3VLForConditionalGeneration.from_pretrained(M,torch_dtype=torch.bfloat16).to("cuda").eval()
pr=AutoProcessor.from_pretrained(M,max_pixels=401408)
ans={}
for cat in ["cars","signs","landscape","exterior"]: ans[cat]=[(im,gen(md,pr,im,Q[cat])) for im in held[cat]]
ans["ocr"]=[(o["image"],o["answer"],gen(md,pr,o["image"],o["question"])) for o in held["ocr"]]
del md; gc.collect(); torch.cuda.empty_cache(); print("qwen3-4b answered")
# judge with base Qwen2.5-VL
jm=Qwen2_5_VLForConditionalGeneration.from_pretrained(R+"/models/Qwen2.5-VL-7B-Instruct",torch_dtype=torch.bfloat16).to("cuda").eval()
jp=AutoProcessor.from_pretrained(R+"/models/Qwen2.5-VL-7B-Instruct",max_pixels=401408)
def ok(img,gt,a):
    r=gen(jm,jp,img,f'Ground-truth: "{gt}". Candidate answer: "{a}". Consistent? Reply ONLY yes or no.',8).lower(); return 1 if "yes" in r else 0
def sc(img,a):
    r=gen(jm,jp,img,f'Driving scene answer: "{a}". Rate 1-10 factuality+completeness+responsibility. Reply ONLY an integer.',8); m=re.search(r"\d+",r); return int(m.group()) if m else None
res={}; lab={"cars":carlab,"signs":signlab,"landscape":lslab}
for cat in ["cars","signs","landscape"]:
    a=[ok(im,lab[cat].get(im,"?"),x) for im,x in ans[cat]]; res[cat]=round(100*sum(a)/len(a),1)
a=[ok(im,gt,x) for im,gt,x in ans["ocr"]]; res["ocr"]=round(100*sum(a)/len(a),1)
e=[sc(im,x) for im,x in ans["exterior"]]; e=[x for x in e if x]; res["exterior"]=round(statistics.mean(e),2)
json.dump({"qwen3_4b":res},open(R+"/data/eval_qwen3.json","w"),indent=1)
print("QWEN3_RESULT",json.dumps(res)); print("QWEN3_EVAL_DONE")
