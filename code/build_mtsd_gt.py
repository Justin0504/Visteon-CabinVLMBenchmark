import json,os
from collections import Counter
from datasets import load_dataset
R="/data/haiyuez/visteon_cabin_vlm"; D=R+"/data/mtsd_imgs"; os.makedirs(D,exist_ok=True)
N=int(os.environ.get("N","300"))
ds=load_dataset("sparshgarg57/mapillary_traffic_signs",split="train",streaming=True)
out=open(R+"/data/mtsd_gt.jsonl","w"); n=0
for r in ds:
    try:
        o=r["objects"]; cats=o.get("category_name") or []; boxes=o.get("bbox") or []
        if not cats: continue
        im=r["image"]; p=f"{D}/mtsd_{n:04d}.jpg"; im.convert("RGB").save(p,quality=88)
        shapes=Counter(cats); W,Hh=im.size
        sig=", ".join(f"{c}:{v}" for c,v in shapes.most_common())
        det="; ".join(f"{cats[i]} bbox[{int(boxes[i][0])},{int(boxes[i][1])},{int(boxes[i][0]+boxes[i][2])},{int(boxes[i][1]+boxes[i][3])}]" for i in range(min(6,len(cats))))
        gt=f"global street-view scene with {len(cats)} traffic sign(s); shapes: {sig}"
        out.write(json.dumps({"image":p,"camera":"front","gt":gt,"vehicles":det,"vrus":"none"})+"\n")
        n+=1
        if n>=N: break
    except Exception: continue
out.close(); print("MTSD_GT_DONE",n)
