import os,json,random
from datasets import load_dataset, load_dataset_builder
random.seed(0); ROOT="/data/haiyuez/visteon_cabin_vlm/data"

# ---- OCR: TextVQA -> sharegpt directly (image + text-reading QA) ----
def textvqa(n=1500):
    outdir=ROOT+"/textvqa"; os.makedirs(outdir+"/images",exist_ok=True)
    ds=load_dataset("lmms-lab/textvqa", split="validation")
    sg=[]; cnt=0
    for ex in ds:
        if cnt>=n: break
        img=ex.get("image"); q=ex.get("question")
        ans=ex.get("answers") or []
        if img is None or not q or not ans: continue
        a=max(set(ans),key=ans.count)  # majority answer
        if a.strip().lower() in ("","unanswerable","answering does not require reading text in the image"): continue
        p=f"{outdir}/images/{cnt:05d}.jpg"
        try: img.convert("RGB").save(p)
        except: continue
        sg.append({"conversations":[{"from":"human","value":"<image>\n"+q},{"from":"gpt","value":a}],"images":[p]})
        cnt+=1
    json.dump(sg,open(ROOT+"/textvqa_sharegpt.json","w"),ensure_ascii=False)
    print(f"textvqa: {len(sg)} OCR pairs")

# ---- Landscape: SUN397 natural classes -> images + labels (for later recaption) ----
def sun397(per_class=60):
    outdir=ROOT+"/landscape"; os.makedirs(outdir+"/images",exist_ok=True)
    names=load_dataset_builder("tanganke/sun397").info.features["label"].names
    KW=["mountain","forest","coast","beach","desert","field","valley","lake","river","ocean","canyon","cliff","hill","waterfall","snow","glacier","wetland","sky","rainforest","badlands"]
    keep={i:n for i,n in enumerate(names) if any(k in n.lower() for k in KW)}
    print("landscape classes kept:",len(keep))
    ds=load_dataset("tanganke/sun397", split="train", streaming=True)
    cap={i:0 for i in keep}; meta=open(outdir+"/labels.jsonl","w"); cnt=0
    for ex in ds:
        lab=ex.get("label")
        if lab not in keep or cap[lab]>=per_class: continue
        img=ex.get("image")
        if img is None: continue
        nm=keep[lab].strip("/").replace("/","_")
        p=f"{outdir}/images/{cnt:05d}.jpg"
        try: img.convert("RGB").save(p)
        except: continue
        meta.write(json.dumps({"image":p,"label":"natural landscape: "+nm},ensure_ascii=False)+"\n")
        cap[lab]+=1; cnt+=1
        if cnt>=800: break
    meta.close(); print(f"landscape: {cnt} imgs")

print("[OCR]"); textvqa(1500)
print("[Landscape]"); sun397(60)
print("GAPS_DONE")
