"""train_v9 = v7 recipe (best) but nuScenes captions upgraded to cross-checked cot_v4."""
import json,os,random
R="/data/haiyuez/visteon_cabin_vlm/data"
wl=set(json.load(open(R+"/clean_whitelist.json")))
def load(f): return json.load(open(R+"/"+f)) if os.path.exists(R+"/"+f) else []
def cap(d,n,seed=0,dedup=True):
    if dedup: d=[x for x in d if x.get("images") and x["images"][0] in wl]
    random.seed(seed); random.shuffle(d); return d[:n] if n else d
parts=[]
parts+=cap(load("exterior_cot_v4_sharegpt.json"),1200,1,dedup=False)  # cross-checked nuScenes CoT
parts+=cap(load("cars_sharegpt.json"),1200,2)
parts+=cap(load("signs_sharegpt.json"),1200,3)
parts+=cap(load("textvqa_road_sharegpt.json"),0,5,dedup=False)
parts+=cap(load("landscape_outdoor_sharegpt.json"),0,6,dedup=False)
parts+=cap(load("trafficlight_sharegpt.json"),0,7,dedup=False)
parts+=cap(load("jaad_vru_sharegpt.json"),0,8,dedup=False)
for x in parts:
    for t in x["conversations"]:
        if not isinstance(t["value"],str):
            t["value"]=json.dumps(t["value"],ensure_ascii=False) if isinstance(t["value"],(dict,list)) else str(t["value"])
    if len(x["conversations"])%2!=0: x["conversations"]=x["conversations"][:-1]
random.seed(42); random.shuffle(parts)
json.dump(parts,open(R+"/train_v9_sharegpt.json","w"),ensure_ascii=False)
print("TRAIN_V9",len(parts))
p=R+"/dataset_info.json"; di=json.load(open(p))
for k,f in [("exterior_cot_v4","exterior_cot_v4_sharegpt.json"),("train_v9","train_v9_sharegpt.json")]:
    di[k]={"file_name":f,"formatting":"sharegpt","columns":{"messages":"conversations","images":"images"},"tags":{"role_tag":"from","content_tag":"value","user_tag":"human","assistant_tag":"gpt"}}
json.dump(di,open(p,"w"),ensure_ascii=False,indent=2); print("registered")
