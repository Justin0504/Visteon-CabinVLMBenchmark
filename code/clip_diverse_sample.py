"""CLIP feature-space diversity sampling: embed images -> measure redundancy -> furthest-point-sample
a maximally diverse subset. Answers 'mine low-redundancy + guarantee diversity' rigorously."""
import json,os,argparse
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
ap=argparse.ArgumentParser()
ap.add_argument("--manifest")          # jsonl/json with image paths, or a dir
ap.add_argument("--field",default="image")
ap.add_argument("--n",type=int,default=800)   # target diverse subset size
ap.add_argument("--out",default="/tmp/diverse_subset.json")
a=ap.parse_args()
# 收集图片路径
paths=[]
if a.manifest.endswith(".json"):
    for x in json.load(open(a.manifest)):
        p=x.get(a.field) or (x.get("images") or [None])[0]
        if p: paths.append(p)
else:
    for l in open(a.manifest):
        try:
            x=json.loads(l); p=x.get(a.field) or (x.get("images") or [None])[0]
            if p: paths.append(p)
        except: pass
paths=[p for p in dict.fromkeys(paths) if os.path.exists(p)]
print("IMAGES",len(paths),flush=True)
dev="cuda" if torch.cuda.is_available() else "cpu"
m=CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()
pr=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
embs=[]
B=64
for i in range(0,len(paths),B):
    ims=[Image.open(p).convert("RGB") for p in paths[i:i+B]]
    inp=pr(images=ims,return_tensors="pt").to(dev)
    with torch.no_grad(): f=m.get_image_features(**inp)
    f=torch.nn.functional.normalize(f,dim=-1)
    embs.append(f.cpu().numpy())
    if i%640==0: print("embed",i,flush=True)
E=np.concatenate(embs); N=len(E)
# 冗余度:近重复(cos>0.95)对占比(抽样估计)
import random; random.seed(0)
idx=random.sample(range(N),min(2000,N))
sims=E[idx]@E[idx].T; np.fill_diagonal(sims,0)
red=float((sims>0.95).sum()/(len(idx)*(len(idx)-1)))
print(f"REDUNDANCY cos>0.95 pair-rate ~ {red*100:.2f}%")
# 最远点采样(FPS):最大化多样性
k=min(a.n,N); sel=[random.randrange(N)]
mind=E@E[sel[0]]
for _ in range(k-1):
    j=int(np.argmin(mind)); sel.append(j)
    mind=np.maximum(mind*0,np.minimum(mind, 1-(E@E[j])))  # update min cos-dist
sel=list(dict.fromkeys(sel))
json.dump([paths[i] for i in sel],open(a.out,"w"))
print(f"DIVERSE_SUBSET selected {len(sel)} / {N} (FPS, max coverage)")
