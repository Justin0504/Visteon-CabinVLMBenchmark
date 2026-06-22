"""ADE20K -> facts for #1/#3/#6, RESILIENT: re-create stream + resume on client-closed crashes."""
import json,os,time
from collections import Counter
from datasets import load_dataset
IMG="/Users/justin/SJTU-450/网盘_数据集/_imgs/ade20k"; os.makedirs(IMG,exist_ok=True)
OUT="/tmp/ade20k_gt.jsonl"; N=int(os.environ.get("N","200"))
KEEP={"building","road","sky","tree","grass","sidewalk","car","person","rider","bicycle","motorbike","truck","bus","wall","fence","streetlight","signboard","traffic light","earth","mountain","plant","path","van","pole"}
def names_of(o):
    if isinstance(o,dict):
        for k in ("name","names","class","category","label"):
            if k in o and o[k]: return [str(x) for x in (o[k] if isinstance(o[k],list) else [o[k]])]
    if isinstance(o,list):
        return [str(it.get("name") or it.get("class") or it.get("label") or "") if isinstance(it,dict) else str(it) for it in o]
    return []
done=0
if os.path.exists(OUT): done=sum(1 for _ in open(OUT))
out=open(OUT,"a")
scanned_target=done
while done<N:
    try:
        ds=load_dataset("1aurent/ADE20K",split="train",streaming=True)
        seen=0; kept_this=0
        for r in ds:
            seen+=1
            if seen<=scanned_target: continue   # resume: skip already-scanned
            scanned_target=seen
            try:
                nm=[x.lower() for x in names_of(r.get("objects")) if x]
                blob=" ".join(nm)+" "+str(r.get("scene","")).lower()
                if not ("road" in blob or "building" in blob or "street" in blob): continue
                im=r.get("image")
                if im is None: continue
                p=f"{IMG}/ade_{done:04d}.jpg"; im.convert("RGB").save(p,quality=88)
                c=Counter(x for x in nm if any(k in x for k in KEEP))
                gt=f"outdoor street scene ({r.get('scene','')}); semantic regions: "+", ".join(f"{k}:{v}" for k,v in c.most_common(12))
                out.write(json.dumps({"image":p,"camera":"front","gt":gt,"vehicles":"none","vrus":"none"})+"\n"); out.flush()
                done+=1
                if done>=N: break
            except Exception: continue
        if done<N: time.sleep(3)  # stream exhausted or partial; retry
    except Exception:
        time.sleep(5)  # client closed -> recreate stream, resume from scanned_target
out.close(); print("ADE20K_DONE",done)
