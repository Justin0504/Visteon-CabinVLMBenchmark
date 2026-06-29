"""Run baseline(v9) vs finetuned(v16) driving advice on given demo frames (same general prompt)."""
import os,json,gc,torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
R="/data/haiyuez/visteon_cabin_vlm"
FRAMES=R+"/demo_frames"; OUT=R+"/demo_frames/advice.json"
MODELS={"baseline_v9":R+"/models/bootstrap_v9_merged","finetuned_v16":R+"/models/bootstrap_v16_merged"}
PROMPT=("You are an in-car driving assistant looking at the road ahead. In 1-2 sentences, give the "
        "driver clear, professional driving advice for this exact moment.")
def load(m):
    md=Qwen2_5_VLForConditionalGeneration.from_pretrained(m,torch_dtype=torch.bfloat16).to("cuda").eval()
    return md,AutoProcessor.from_pretrained(m,max_pixels=401408)
def gen(md,pr,img,t,mx=80):
    ms=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":t}]}]
    c=pr.apply_chat_template(ms,tokenize=False,add_generation_prompt=True); ii,_=process_vision_info(ms)
    inp=pr(text=[c],images=ii,return_tensors="pt").to("cuda")
    with torch.no_grad(): o=md.generate(**inp,max_new_tokens=mx,do_sample=False)
    return pr.decode(o[0][inp.input_ids.shape[1]:],skip_special_tokens=True).strip()
frames=[f for f in os.listdir(FRAMES) if f.endswith(".jpg")]
res={f:{} for f in frames}
for name,path in MODELS.items():
    if not os.path.exists(path+"/config.json"): continue
    md,pr=load(path)
    for f in frames: res[f][name]=gen(md,pr,os.path.join(FRAMES,f),PROMPT)
    del md;gc.collect();torch.cuda.empty_cache(); print("advised",name,flush=True)
json.dump(res,open(OUT,"w"),ensure_ascii=False,indent=1)
print("DEMO_ADVICE_DONE")
for f in frames:
    print("\n==",f,"=="); 
    for m in res[f]: print(f"  {m}: {res[f][m][:120]}")
