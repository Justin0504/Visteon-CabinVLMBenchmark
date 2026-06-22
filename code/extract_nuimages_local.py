"""nuImages (graid-nuimages, 6-cam surround) — ingest native question/answer as sharegpt directly (no API)."""
import json,os
from collections import defaultdict
from datasets import load_dataset
IMG="/Users/justin/SJTU-450/网盘_数据集/_imgs/nuimages"; os.makedirs(IMG,exist_ok=True)
OUT="/Users/justin/SJTU-450/网盘_数据集/_gen/nuimages_sharegpt.json"; os.makedirs(os.path.dirname(OUT),exist_ok=True)
N=int(os.environ.get("N","1500"))
ds=load_dataset("kd7/graid-nuimages",split="train",streaming=True)
byimg=defaultdict(list); saved={}; n=0; seen=0
for r in ds:
    seen+=1
    q=r.get("question"); a=r.get("answer")
    if not q or not a: continue
    sid=r.get("source_id") or r.get("id") or seen
    if sid not in saved:
        if len(saved)>=N: break
        try:
            p=f"{IMG}/nuim_{len(saved):05d}.jpg"; r["image"].convert("RGB").save(p,quality=88); saved[sid]=p
        except: continue
    byimg[saved[sid]].append((str(q),str(a),r.get("question_type")))
sg=[]
for img,qas in byimg.items():
    conv=[]
    for i,(q,a,qt) in enumerate(qas[:6]):
        qtext=("<image>\n"+q) if i==0 else q
        conv+=[{"from":"human","value":qtext},{"from":"gpt","value":a}]
    sg.append({"conversations":conv,"images":[img]})
json.dump(sg,open(OUT,"w"),ensure_ascii=False)
print("NUIMAGES_DONE images",len(sg),"qa",sum(len(v) for v in byimg.values()))
