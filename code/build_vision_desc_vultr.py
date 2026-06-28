"""FUSION Stage-A UPGRADE: replace local Qwen2-VL-7B with Vultr Qwen3.5-397B-A17B (vision).
Far stronger visual understanding; runs via Vultr (inference offloaded, only CPU+internet needed).
base64-encodes each image, asks for a rich visual-context description (NOT object counts).
Reasoning model -> give large max_tokens, take final `content` (fallback to last paragraph of reasoning).
5-key round-robin, threading, retry/backoff, resume, incremental write.
Input rows: {image,camera,gt,vehicles,vrus}  Output: same + upgraded vision_desc."""
import json,urllib.request,base64,argparse,concurrent.futures as cf,os,threading,time
_W=threading.Lock()
KEYS=[l.split("=")[1].strip() for l in open(os.environ.get("VK","/Users/justin/SJTU-450/.secrets/vultr_keys.env")) if l.startswith("VULTR_KEY")]
URL="https://api.vultrinference.com/v1/chat/completions"
M=os.environ.get("VDESC_MODEL","Qwen/Qwen3.5-397B-A17B")    # override primary (gap-fill uses fast Nemotron)
MFB="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"   # fast vision fallback when Qwen3.5 times out (504)
ap=argparse.ArgumentParser()
ap.add_argument("--inp"); ap.add_argument("--out"); ap.add_argument("--num",type=int,default=0)
a=ap.parse_args(); os.makedirs(os.path.dirname(a.out),exist_ok=True)
PROMPT=("Describe the VISUAL CONTEXT of this exterior driving/street scene in 2-3 sentences: weather, time of day, "
        "road/setting type (urban street / highway / intersection / parking / rural), lane markings, road-surface "
        "condition, lighting, and roadside buildings/vegetation/terrain. Do NOT count or list vehicles/pedestrians "
        "— only the scene context. Output ONLY the final description, no preamble.")
import re as _re
def _from_reasoning(r):
    # greedy: strip markdown, pull descriptive sentences (long reasoning often holds the answer in bullets)
    r=_re.sub(r"\*\*|`|#+","",r)
    sents=_re.split(r"(?<=[.!?])\s+",r.replace("\n"," "))
    desc=[s.strip(" -*") for s in sents if len(s.strip())>40 and any(w in s.lower() for w in
          ("scene","road","street","weather","sky","building","lane","pavement","asphalt","daytime","night","sunny","cloudy","overcast","urban","highway","lighting","surface","vegetation","intersection","parking"))]
    if desc: return " ".join(desc[-3:])[:600]               # last few descriptive sentences
    return " ".join(s.strip(" -*") for s in sents[-3:])[:600] if sents else ""
def _call(model,img_b64,key,mx,to):
    body=json.dumps({"model":model,"messages":[{"role":"user","content":[
        {"type":"text","text":PROMPT},
        {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+img_b64}}]}],
        "max_tokens":mx,"temperature":0.3}).encode()
    req=urllib.request.Request(URL,data=body,headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"})
    m=json.loads(urllib.request.urlopen(req,timeout=to).read())["choices"][0]["message"]
    c=(m.get("content") or "").strip()
    if c: return c
    r=(m.get("reasoning") or "").strip()
    return _from_reasoning(r) if r else ""
PMX=int(os.environ.get("VDESC_MX","2600"))        # primary max_tokens (Nemotron needs ~1500 or it 504s)
def desc(img_b64,key):
    for att in range(3):                          # primary (Qwen3.5-397B best, or Nemotron for gap)
        try:
            d=_call(M,img_b64,key,PMX,240)
            if d: return d
        except Exception: time.sleep(2*(att+1))
    for att in range(3):                          # fallback: fast Nemotron-Omni (handles 504 cases)
        try:
            d=_call(MFB,img_b64,key,1500,150)
            if d: return d
        except Exception: time.sleep(2*(att+1))
    return ""
rows=[json.loads(l) for l in open(a.inp)]
if a.num: rows=rows[:a.num]
seen=set()
if os.path.exists(a.out):
    for l in open(a.out):
        try: seen.add(json.loads(l)["image"])
        except: pass
rows=[r for r in rows if r["image"] not in seen]
print("VDESC_V2 rows",len(rows),"model",M,flush=True)
def work(ir):
    i,r=ir
    try:
        if not os.path.exists(r["image"]): return None
        b64=base64.b64encode(open(r["image"],"rb").read()).decode()
        d=desc(b64,KEYS[i%len(KEYS)])
        if not d: return None
        return {"image":r["image"],"camera":r["camera"],"gt":r["gt"],
                "vehicles":r.get("vehicles","none"),"vrus":r.get("vrus","none"),"vision_desc":d}
    except Exception: return None
fr=open(a.out,"a" if seen else "w"); done=ok=0
with cf.ThreadPoolExecutor(max_workers=int(os.environ.get("WORKERS","5"))) as ex:
    for res in ex.map(work,list(enumerate(rows))):
        done+=1
        if res:
            ok+=1
            with _W: fr.write(json.dumps(res,ensure_ascii=False)+"\n"); fr.flush()
        if done%50==0: print("done",done,"ok",ok,flush=True)
fr.close(); print("VDESC_V2_DONE ok",ok,flush=True)
