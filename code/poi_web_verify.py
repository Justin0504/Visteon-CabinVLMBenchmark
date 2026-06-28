"""POI WEB-VERIFICATION GATE (#8). Input: poi_raw.jsonl from VLM extraction (image, pois[{name,category,position}]).
For each VLM-read sign name, query the web (Wikipedia OpenSearch + Nominatim/OSM, both free no-key) to verify
it's a REAL place/business and enrich category. KEEP only verified POIs (zero-hallucination gate); drop unverifiable.
Output: poi_verified.jsonl + poi_sharegpt.json (QA grounded in verified POIs + map-layer fields)."""
import json,urllib.request,urllib.parse,argparse,time,os
ap=argparse.ArgumentParser()
ap.add_argument("--inp",default="/Users/justin/SJTU-450/网盘_数据集/_gen/poi_raw.jsonl")
ap.add_argument("--out",default="/Users/justin/SJTU-450/网盘_数据集/_gen/poi_verified.jsonl")
ap.add_argument("--sg",default="/Users/justin/SJTU-450/网盘_数据集/_gen/poi_sharegpt.json")
a=ap.parse_args()
UA={"User-Agent":"VisteonCabinVLM/1.0 (research; aojieyua@usc.edu)"}
def getj(url):
    try:
        req=urllib.request.Request(url,headers=UA); return json.loads(urllib.request.urlopen(req,timeout=15).read())
    except Exception: return None
def wiki(name):
    # Wikipedia OpenSearch: returns [query,[titles],[descs],[urls]]
    d=getj("https://en.wikipedia.org/w/api.php?action=opensearch&limit=1&format=json&search="+urllib.parse.quote(name))
    if d and len(d)>=4 and d[1]:
        return {"source":"wikipedia","title":d[1][0],"url":d[3][0] if d[3] else "","desc":(d[2][0] if d[2] else "")}
    return None
def geo_of(image):
    # nuScenes scene prefix -> capture city (true map-grounding constraint)
    b=image.split("/")[-1]
    if b.startswith("n015"): return "sg"          # Singapore (One-North/Queenstown)
    if b.startswith("n008"): return "us"          # Boston Seaport
    return None
def osm(name,cc=None):
    # Nominatim: name -> real place w/ category, constrained to capture city when known
    u="https://nominatim.openstreetmap.org/search?format=json&limit=1&q="+urllib.parse.quote(name)
    if cc: u+="&countrycodes="+cc
    d=getj(u)
    if d:
        x=d[0]; return {"source":"osm","title":x.get("display_name",""),"category":x.get("type") or x.get("class",""),"url":""}
    return None
import difflib,re as _re
def name_match(name,title):
    # guard against fuzzy false-positives: the read name must actually appear in the matched title
    n=_re.sub(r"[^a-z0-9 ]","",name.lower()).strip(); t=title.lower()
    if not n: return False
    if n in t: return True                                   # substring (Subway in "Subway, Jurong...")
    cand=t.split(",")[0]                                     # primary name token of the place
    return difflib.SequenceMatcher(None,n,cand).ratio()>=0.6  # near-match misread (Synbiosis~Symbiosis)
def verify(name,cc=None):
    if not name or name.lower() in ("unknown","sign","store","shop",""): return None
    w=osm(name,cc); time.sleep(1.1)         # geo-constrained first (rate limit 1 req/s)
    if not w and cc: w=osm(name,None); time.sleep(1.1)   # fallback: global chains (Subway etc.)
    if not w: w=wiki(name); time.sleep(0.5)
    if w and not name_match(name,w.get("title","")): return None   # reject fuzzy mismatch
    return w
rows=[json.loads(l) for l in open(a.inp)] if os.path.exists(a.inp) else []
print("POI_VERIFY input images",len(rows),flush=True)
seen=set()
if os.path.exists(a.out):
    for l in open(a.out):
        try: seen.add(json.loads(l)["image"])
        except: pass
fw=open(a.out,"a" if seen else "w"); kept=tot=0; allv=[]
for r in rows:
    if r["image"] in seen: continue
    vp=[]; cc=geo_of(r["image"])
    for p in r.get("pois",[]):
        tot+=1; ev=verify(p.get("name",""),cc)
        if ev:
            kept+=1; p["verified"]=True; p["evidence"]=ev
            if ev.get("category"): p["category"]=ev["category"]
            vp.append(p)
    if vp:
        rec={"image":r["image"],"pois":vp,"map_layer":"POI"}; allv.append(rec)
        fw.write(json.dumps(rec,ensure_ascii=False)+"\n"); fw.flush()
    print(f"  {r['image'].split('/')[-1][:30]} kept {len(vp)}/{len(r.get('pois',[]))}",flush=True)
fw.close()
# build sharegpt QA grounded in verified POIs
sg=[]
for r in allv:
    names=", ".join(p["name"] for p in r["pois"])
    cats=", ".join(f"{p['name']} ({p.get('category','poi')})" for p in r["pois"])
    convs=[{"from":"human","value":"<image>\nList any points of interest (businesses/landmarks) visible and their type."},
           {"from":"gpt","value":f"Visible POIs: {cats}."}]
    p0=r["pois"][0]
    convs+=[{"from":"human","value":f"Where is {p0['name']} located in the view?"},
            {"from":"gpt","value":f"{p0['name']} is on the {p0.get('position','roadside')}."}]
    sg.append({"conversations":convs,"images":[r["image"]]})
json.dump(sg,open(a.sg,"w"),ensure_ascii=False)
print(f"POI_VERIFY_DONE verified {kept}/{tot} pois across {len(allv)} images, sharegpt {len(sg)}",flush=True)
