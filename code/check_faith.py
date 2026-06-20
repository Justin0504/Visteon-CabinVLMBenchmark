"""Quick 3-rater faithfulness check on a raw jsonl ({image, caption})."""
import json,argparse
from PIL import Image
from vllm import LLM, SamplingParams
ap=argparse.ArgumentParser(); ap.add_argument("--raw"); a=ap.parse_args()
d=[json.loads(l) for l in open(a.raw)]
def load(p):
    im=Image.open(p).convert("RGB")
    if max(im.size)>768:
        s=768/max(im.size); im=im.resize((int(im.size[0]*s),int(im.size[1]*s)))
    return im
def qwen(q): return f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{q}<|im_end|>\n<|im_start|>assistant\n"
RUBRICS=[
 'Does this caption faithfully describe the image with NO hallucinated objects or wrong facts? Caption: "{c}". Answer strictly yes or no.',
 'Is every concrete claim (objects, counts, colors, actions) in this caption actually supported by what is visible? Caption: "{c}". Reply strictly yes or no.',
 'Is this driving caption accurate and free of made-up details for this image? Caption: "{c}". Output strictly yes or no.']
llm=LLM(model="/data/haiyuez/visteon_cabin_vlm/models/Qwen2-VL-7B-Instruct",max_model_len=4096,gpu_memory_utilization=0.5,limit_mm_per_prompt={"image":1})
imgs={r["image"]:load(r["image"]) for r in d}
key="caption" if "caption" in d[0] else "cot_caption"
votes=[]
for ri,rub in enumerate(RUBRICS):
    o=llm.generate([{"prompt":qwen(rub.format(c=r[key][:600])),"multi_modal_data":{"image":imgs[r["image"]]}} for r in d],SamplingParams(max_tokens=4,temperature=0.3,seed=ri+1))
    votes.append([1 if "yes" in x.outputs[0].text.lower() else 0 for x in o])
N=len(d); maj=sum(1 for i in range(N) if votes[0][i]+votes[1][i]+votes[2][i]>=2)
print("FAITH N="+str(N)+" faithful="+str(maj)+" ("+str(round(100*maj/N))+"%)")
