"""Scene/terrain taxonomy tagging via Vultr DeepSeek-V4 (text), classifying from Qwen2-VL vision_desc.
Multi-axis labels: environment, road_type, surface, landscape_regions. Fully local (Vultr cloud)."""
import json,urllib.request,argparse,concurrent.futures as cf,os,threading,time
_W=threading.Lock()
KEYS=[l.split("=")[1].strip() for l in open("/Users/justin/SJTU-450/.secrets/vultr_keys.env") if l.startswith("VULTR_KEY")]
URL="https://api.vultrinference.com/v1/chat/completions"; M="deepseek-ai/DeepSeek-V4-Flash"
ap=argparse.ArgumentParser(); ap.add_argument("--inp",default="/tmp/fusion_input.jsonl"); ap.add_argument("--out",default="/Users/justin/SJTU-450/网盘_数据集/_gen/scene_tags.jsonl"); ap.add_argument("--num",type=int,default=0); a=ap.parse_args()
os.makedirs(os.path.dirname(a.out),exist_ok=True)
ENV="downtown,commercial,residential,suburb,rural,wilderness,industrial,port,coastal,mountain,forest,plain,desert,farmland"
ROAD="highway,arterial,street,intersection,roundabout,ramp,toll,tunnel,bridge,one-way,narrow-lane,parking,construction,crosswalk-zone"
SURF="asphalt,concrete,cobblestone,gravel,dirt,wet,snow,flat,sloped"
LAND="dense-vegetation,open-sky,flat-ground,water,mountain-terrain"
SYS="You are a driving-scene classifier. Given a VISION description, assign multi-axis labels ONLY from the allowed lists. Output only JSON."
def usr(vd):
    return (f"VISION description: {vd}\nAssign labels (multi-select, only from lists; use [] if none):\n"
    f"environment: [{ENV}]\nroad_type: [{ROAD}]\nsurface: [{SURF}]\nlandscape_regions: [{LAND}]\n"
    'Also day_night (day/night/dawn/dusk) and weather (clear/cloudy/rain/fog/snow). '
    'STRICT JSON: {"environment":[],"road_type":[],"surface":[],"landscape_regions":[],"day_night":"","weather":""}')
def chat(u,key):
    body=json.dumps({"model":M,"messages":[{"role":"system","content":SYS},{"role":"user","content":u}],"max_tokens":400,"temperature":0.1}).encode()
    for att in range(5):
        try:
            req=urllib.request.Request(URL,data=body,headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"})
            m=json.loads(urllib.request.urlopen(req,timeout=90).read())["choices"][0]["message"]; return m.get("content") or m.get("reasoning") or ""
        except Exception: time.sleep(2*(att+1))
    return ""
def jp(t):
    if not t: return None
    t=t.replace("```json","").replace("```","").strip(); i=t.find("{");j=t.rfind("}")
    try: return json.loads(t[i:j+1]) if i>=0 else None
    except: return None
rows=[json.loads(l) for l in open(a.inp)]
if a.num: rows=rows[:a.num]
seen=set()
if os.path.exists(a.out):
    for l in open(a.out):
        try: seen.add(json.loads(l)["image"])
        except: pass
rows=[r for r in rows if r["image"] not in seen]
print("TAG rows",len(rows),flush=True)
def work(ir):
    i,r=ir
    try:
        d=jp(chat(usr(r.get("vision_desc","")),KEYS[i%len(KEYS)]))
        if not d: return None
        d["image"]=r["image"]; return d
    except: return None
fr=open(a.out,"a" if seen else "w"); done=ok=0
with cf.ThreadPoolExecutor(max_workers=5) as ex:
    for res in ex.map(work,list(enumerate(rows))):
        done+=1
        if res:
            ok+=1
            with _W: fr.write(json.dumps(res,ensure_ascii=False)+"\n"); fr.flush()
        if done%100==0: print("done",done,"ok",ok,flush=True)
fr.close(); print("SCENE_TAG_DONE ok",ok,flush=True)
