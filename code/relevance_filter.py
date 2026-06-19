import json,argparse
from PIL import Image
from vllm import LLM, SamplingParams
ap=argparse.ArgumentParser()
ap.add_argument("--inp"); ap.add_argument("--out"); ap.add_argument("--q"); ap.add_argument("--tag",default="FILTER")
a=ap.parse_args()
d=json.load(open(a.inp))
def qwen(q): return f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{q}<|im_end|>\n<|im_start|>assistant\n"
def load(p):
    im=Image.open(p).convert("RGB")
    if max(im.size)>768:
        s=768/max(im.size); im=im.resize((int(im.size[0]*s),int(im.size[1]*s)))
    return im
llm=LLM(model="/data/haiyuez/visteon_cabin_vlm/models/Qwen2-VL-7B-Instruct",max_model_len=4096,gpu_memory_utilization=0.5,limit_mm_per_prompt={"image":1})
o=llm.generate([{"prompt":qwen(a.q),"multi_modal_data":{"image":load(x["images"][0])}} for x in d],SamplingParams(max_tokens=4,temperature=0))
keep=[x for x,r in zip(d,o) if "yes" in r.outputs[0].text.lower()]
json.dump(keep,open(a.out,"w"),ensure_ascii=False)
print(a.tag+" total="+str(len(d))+" kept="+str(len(keep))+" dropped="+str(len(d)-len(keep)))
