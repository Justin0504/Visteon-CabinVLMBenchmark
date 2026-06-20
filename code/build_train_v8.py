"""train_v8 = domain-pure + deduped, adds TT100K China signs + FULL textvqa_road (recover OCR)."""
import json,os,random
R="/data/haiyuez/visteon_cabin_vlm/data"
wl=set(json.load(open(R+"/clean_whitelist.json")))
def load(f): return json.load(open(R+"/"+f)) if os.path.exists(R+"/"+f) else []
def cap(d,n,seed=0,dedup=True):
    if dedup: d=[x for x in d if x.get("images") and x["images"][0] in wl]
    random.seed(seed); random.shuffle(d); return d[:n] if n else d
parts=[]
parts+=cap(load("exterior_cot_v2_sharegpt.json"),1200,1)          # nuScenes enhanced CoT
parts+=cap(load("cars_sharegpt.json"),1200,2)                     # vehicle make/model
parts+=cap(load("signs_sharegpt.json"),1200,3)                    # GTSRB signs (EU)
parts+=cap(load("tt100k_china_sharegpt.json"),0,4,dedup=False)    # NEW China signs (full ~600)
parts+=cap(load("textvqa_road_sharegpt.json"),0,5,dedup=False)    # FULL road OCR (227, recover OCR)
parts+=cap(load("landscape_outdoor_sharegpt.json"),0,6,dedup=False) # outdoor landscape (~770)
parts+=cap(load("trafficlight_sharegpt.json"),0,7,dedup=False)    # traffic lights (142)
parts+=cap(load("jaad_vru_sharegpt.json"),0,8,dedup=False)        # VRU crossing (270)
for x in parts:
    for t in x["conversations"]:
        if not isinstance(t["value"],str):
            t["value"]=json.dumps(t["value"],ensure_ascii=False) if isinstance(t["value"],(dict,list)) else str(t["value"])
    if len(x["conversations"])%2!=0: x["conversations"]=x["conversations"][:-1]
random.seed(42); random.shuffle(parts)
json.dump(parts,open(R+"/train_v8_sharegpt.json","w"),ensure_ascii=False)
from collections import Counter
def src(p):
    for k in ["nuscenes","stanford_cars","gtsrb","textvqa","sun397","road_traffic","jaad_frames","tt100k"]:
        if k in p: return k
    return "other"
print("TRAIN_V8",len(parts),dict(Counter(src(x["images"][0]) for x in parts)))
p=R+"/dataset_info.json"; di=json.load(open(p))
for k,f in [("tt100k_china","tt100k_china_sharegpt.json"),("train_v8","train_v8_sharegpt.json")]:
    di[k]={"file_name":f,"formatting":"sharegpt","columns":{"messages":"conversations","images":"images"},"tags":{"role_tag":"from","content_tag":"value","user_tag":"human","assistant_tag":"gpt"}}
json.dump(di,open(p,"w"),ensure_ascii=False,indent=2); print("registered tt100k_china, train_v8")
