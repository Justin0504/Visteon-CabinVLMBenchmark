import os,json
from datasets import load_dataset
ROOT="/data/haiyuez/visteon_cabin_vlm/data"
def dump(repo, split, outdir, n, fmt="jpg"):
    os.makedirs(outdir+"/images",exist_ok=True)
    ds=load_dataset(repo, split=split)
    names=None
    try: names=ds.features["label"].names
    except: pass
    meta=open(outdir+"/labels.jsonl","w")
    cnt=0
    for i,ex in enumerate(ds):
        if cnt>=n: break
        img=ex.get("image")
        if img is None: continue
        lab=ex.get("label")
        lname=names[lab] if names and isinstance(lab,int) else str(lab)
        p=f"{outdir}/images/{cnt:05d}.{fmt}"
        try: img.convert("RGB").save(p)
        except Exception as e: continue
        meta.write(json.dumps({"image":p,"label":lname},ensure_ascii=False)+"\n"); cnt+=1
    meta.close()
    print(f"{repo}: saved {cnt} imgs -> {outdir}; classes={len(names) if names else '?'}")
print("[stanford_cars]")
dump("tanganke/stanford_cars","train",ROOT+"/stanford_cars",2000,"jpg")
print("[gtsrb]")
dump("tanganke/gtsrb","train",ROOT+"/gtsrb",1500,"png")
print("FETCH_DONE")
