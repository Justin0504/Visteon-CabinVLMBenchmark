"""AI multi-rater faithfulness QC (proxy for inter-annotator agreement).
3 independent rater-prompts judge each sampled caption faithful/unfaithful to its image;
reports faithfulness rate + Fleiss kappa, and exports disagreed/unfaithful items for human review."""
import json,random,os
from PIL import Image
from vllm import LLM, SamplingParams
R="/data/haiyuez/visteon_cabin_vlm/data"
PER=30
prefer=['exterior_cot_v2_sharegpt.json','trafficlight_sharegpt.json','jaad_vru_sharegpt.json',
        'cars_sharegpt.json','signs_sharegpt.json','landscape_outdoor_sharegpt.json','textvqa_road_sharegpt.json']
def src(p):
    for k in ['nuscenes','stanford_cars','gtsrb','textvqa','sun397','road_traffic','jaad_frames']:
        if k in p: return k
    return 'other'
samp=[]; cnt={}
for f in prefer:
    p=R+"/"+f
    if not os.path.exists(p): continue
    d=json.load(open(p)); random.seed(7); random.shuffle(d)
    for x in d:
        s=src(x['images'][0])
        if cnt.get(s,0)>=PER: continue
        cap=x['conversations'][1]['value'] if len(x['conversations'])>1 else ''
        samp.append((x['images'][0],s,cap)); cnt[s]=cnt.get(s,0)+1
print("SAMPLE",len(samp),cnt,flush=True)
def load(p):
    im=Image.open(p).convert("RGB")
    if max(im.size)>768:
        sc=768/max(im.size); im=im.resize((int(im.size[0]*sc),int(im.size[1]*sc)))
    return im
def qwen(q): return f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{q}<|im_end|>\n<|im_start|>assistant\n"
# 3 个独立评审 rubric
RUBRICS=[
 'Does this caption faithfully describe the image with NO hallucinated objects or wrong facts? Caption: "{c}". Answer strictly yes or no.',
 'Read the caption and look at the image. Is every concrete claim (objects, counts, colors, actions) actually supported by what is visible? Caption: "{c}". Reply strictly yes or no.',
 'A reviewer checks driving captions for accuracy. Is this caption accurate and free of made-up details for this image? Caption: "{c}". Output strictly yes or no.']
llm=LLM(model="/data/haiyuez/visteon_cabin_vlm/models/Qwen2-VL-7B-Instruct",max_model_len=4096,gpu_memory_utilization=0.5,limit_mm_per_prompt={"image":1})
imgs={p:load(p) for p,_,_ in samp}
votes=[]
for ri,rub in enumerate(RUBRICS):
    o=llm.generate([{"prompt":qwen(rub.format(c=c[:600])),"multi_modal_data":{"image":imgs[p]}} for p,s,c in samp],SamplingParams(max_tokens=4,temperature=0.3,seed=ri+1))
    votes.append([1 if "yes" in r.outputs[0].text.lower() else 0 for r in o])
# 聚合
N=len(samp); R3=3
maj=[1 if (votes[0][i]+votes[1][i]+votes[2][i])>=2 else 0 for i in range(N)]
unanimous=sum(1 for i in range(N) if votes[0][i]==votes[1][i]==votes[2][i])
faithful=sum(maj)
# Fleiss kappa (3 raters, 2 cats)
def fleiss(votes,N,n=3):
    Pi=[]; tot1=0
    for i in range(N):
        n1=votes[0][i]+votes[1][i]+votes[2][i]; n0=n-n1; tot1+=n1
        Pi.append((n1*n1+n0*n0-n)/(n*(n-1)))
    Pbar=sum(Pi)/N; p1=tot1/(N*n); p0=1-p1; Pe=p1*p1+p0*p0
    return (Pbar-Pe)/(1-Pe) if (1-Pe)>0 else 1.0
kappa=fleiss(votes,N)
# 导出需人工复核(分歧 或 判不忠实)
review=[{"image":p,"source":s,"caption":c,"votes":[votes[0][i],votes[1][i],votes[2][i]],"majority":"faithful" if maj[i] else "UNFAITHFUL"}
        for i,(p,s,c) in enumerate(samp) if not(votes[0][i]==votes[1][i]==votes[2][i]==1)]
json.dump(review,open(R+"/qc_agreement_review.json","w"),ensure_ascii=False,indent=1)
# 分源忠实率
from collections import defaultdict,Counter
bysrc=defaultdict(lambda:[0,0])
for i,(p,s,c) in enumerate(samp): bysrc[s][0]+=maj[i]; bysrc[s][1]+=1
persrc={s:str(round(100*a/b))+"%" for s,(a,b) in bysrc.items()}
print("QC_AGREEMENT N="+str(N)+" faithful="+str(faithful)+" ("+str(round(100*faithful/N))+"%) unanimous="+str(round(100*unanimous/N))+"% fleiss_kappa="+str(round(kappa,3)))
print("PER_SOURCE_FAITHFUL "+json.dumps(persrc))
print("NEED_HUMAN_REVIEW "+str(len(review)))
print("QC_AGREEMENT_DONE")
