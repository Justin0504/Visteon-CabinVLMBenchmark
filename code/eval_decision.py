"""Driving-DECISION eval (the proper test for causal-chain/VLA value).
GT action derived from VRU crossing label: crossing -> {stop,slow,yield,brake,wait}; not -> {proceed,go,continue}.
Tests whether the model picks the RIGHT ACTION, not description fluency. v9 vs v12."""
import os,json,torch,gc
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
R="/data/haiyuez/visteon_cabin_vlm"
vru=json.load(open(R+"/data/vru_test.json"))
MODELS={"v9":R+"/models/bootstrap_v9_merged","v12":R+"/models/bootstrap_v12_merged"}
STOP={"stop","slow","yield","brake","wait","decelerate","halt"}
GO={"proceed","go","continue","maintain","drive"}
Q="Considering pedestrians/VRUs ahead, what should the ego vehicle do RIGHT NOW? Answer with ONE action word: proceed, slow, stop, or yield."
def load(m):
    md=Qwen2_5_VLForConditionalGeneration.from_pretrained(m,torch_dtype=torch.bfloat16).to("cuda").eval()
    return md,AutoProcessor.from_pretrained(m,max_pixels=401408)
def gen(md,pr,img,t,mx=24):
    ms=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":t}]}]
    c=pr.apply_chat_template(ms,tokenize=False,add_generation_prompt=True); ii,_=process_vision_info(ms)
    inp=pr(text=[c],images=ii,return_tensors="pt").to("cuda")
    with torch.no_grad(): o=md.generate(**inp,max_new_tokens=mx,do_sample=False)
    return pr.decode(o[0][inp.input_ids.shape[1]:],skip_special_tokens=True).strip().lower()
def correct(ans,crossing):
    hit_stop=any(w in ans for w in STOP); hit_go=any(w in ans for w in GO)
    if crossing=="yes": return 1 if (hit_stop and not (hit_go and not hit_stop)) else 0
    else: return 1 if (hit_go and not hit_stop) else 0
res={}
for name,p in MODELS.items():
    md,pr=load(p); ok=0
    for x in vru:
        a=gen(md,pr,x["image"],Q); ok+=correct(a,x["crossing"])
    res[name]=round(100*ok/len(vru),1); del md;gc.collect();torch.cuda.empty_cache(); print("done",name,res[name],flush=True)
print("DECISION_EVAL",json.dumps(res)); print("DECISION_DONE")
