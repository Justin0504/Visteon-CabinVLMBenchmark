import os,json,argparse
from PIL import Image
ROOT="/data/haiyuez/visteon_cabin_vlm"
ap=argparse.ArgumentParser(); ap.add_argument("--backend",required=True); ap.add_argument("--model",required=True); ap.add_argument("--out",required=True); ap.add_argument("--n",type=int,default=20); a=ap.parse_args()
held=json.load(open(ROOT+"/data/heldout_frozen.json"))["exterior"][:a.n]
SYS="You are an in-car cockpit AI looking outside the vehicle."
INSTR='Output STRICT JSON {"caption":"<one sentence>","qa":[{"question":"","answer":""}]} with exactly 4 grounded QA about vehicles/pedestrians/signs/scene. Every answer must be specific and from the image (avoid "unknown"). Output ONLY JSON.'
def parse(t):
    t=t.replace("```json","").replace("```",""); s,e=t.find("{"),t.rfind("}")
    try: return json.loads(t[s:e+1])
    except: return None
res=[]
if a.backend=="vllm":
    from vllm import LLM, SamplingParams
    def qwen(q): return f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{q}<|im_end|>\n<|im_start|>assistant\n"
    llm=LLM(model=a.model,max_model_len=4096,gpu_memory_utilization=0.5,limit_mm_per_prompt={"image":1})
    outs=llm.generate([{"prompt":qwen(INSTR),"multi_modal_data":{"image":Image.open(p).convert("RGB")}} for p in held],SamplingParams(max_tokens=500,temperature=0.3))
    for p,o in zip(held,outs): res.append({"image":p,"rec":parse(o.outputs[0].text)})
else:
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    md=Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model,torch_dtype=torch.bfloat16).to("cuda").eval()
    pr=AutoProcessor.from_pretrained(a.model,max_pixels=401408)
    for p in held:
        ms=[{"role":"user","content":[{"type":"image","image":p},{"type":"text","text":INSTR}]}]
        c=pr.apply_chat_template(ms,tokenize=False,add_generation_prompt=True); ii,_=process_vision_info(ms)
        inp=pr(text=[c],images=ii,return_tensors="pt").to("cuda")
        with torch.no_grad(): o=md.generate(**inp,max_new_tokens=500,do_sample=False)
        res.append({"image":p,"rec":parse(pr.decode(o[0][inp.input_ids.shape[1]:],skip_special_tokens=True))})
json.dump(res,open(a.out,"w"),ensure_ascii=False)
ok=[r for r in res if r["rec"] and r["rec"].get("qa")]
ans=[q.get("answer","") for r in ok for q in r["rec"]["qa"]]
unk=sum(1 for x in ans if any(w in x.lower() for w in["unknown","cannot","n/a","not sure","unclear","not visible"]))
al=sum(len(x.split()) for x in ans)/max(1,len(ans))
print(f"RESULT backend={a.backend} parsed={len(ok)}/{len(res)} answers={len(ans)} unknown_rate={100*unk/max(1,len(ans)):.1f}% avg_ans_words={al:.1f}")
