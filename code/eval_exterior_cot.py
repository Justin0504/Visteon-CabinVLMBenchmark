import json,torch,re,statistics,gc
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
R="/data/haiyuez/visteon_cabin_vlm"
held=json.load(open(R+"/data/heldout_frozen.json"))["exterior"]
MODELS={"v3":R+"/models/bootstrap_v3_merged","v6":R+"/models/bootstrap_v6_merged"}
QC="Describe this exterior driving scene and give a driving decision."
def load(m):
    md=Qwen2_5_VLForConditionalGeneration.from_pretrained(m,torch_dtype=torch.bfloat16).to("cuda").eval()
    pr=AutoProcessor.from_pretrained(m,max_pixels=401408); return md,pr
def gen(md,pr,img,t,mx=160):
    ms=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":t}]}]
    c=pr.apply_chat_template(ms,tokenize=False,add_generation_prompt=True); ii,_=process_vision_info(ms)
    inp=pr(text=[c],images=ii,return_tensors="pt").to("cuda")
    with torch.no_grad(): o=md.generate(**inp,max_new_tokens=mx,do_sample=False)
    return pr.decode(o[0][inp.input_ids.shape[1]:],skip_special_tokens=True).strip()
ans={}
for n,p in MODELS.items():
    md,pr=load(p); ans[n]=[(im,gen(md,pr,im,QC)) for im in held]; del md;gc.collect();torch.cuda.empty_cache(); print("answered",n,flush=True)
jm,jp=load(R+"/models/Qwen2.5-VL-7B-Instruct")
def sc(img,a):
    q=(f'A driving assistant answered for this scene: "{a}". Rate 1-10 on factual accuracy of traffic elements, risk analysis quality, and soundness of the driving decision. A good answer reasons scene->risk->decision. Reply ONLY an integer.')
    r=gen(jm,jp,img,q,8); m=re.search(r"\d+",r); return int(m.group()) if m else None
res={}
for n in ans:
    e=[sc(im,a) for im,a in ans[n]]; e=[x for x in e if x]; res[n]=round(statistics.mean(e),2)
print("EXT_COT",json.dumps(res)); print("EXT_COT_DONE")
