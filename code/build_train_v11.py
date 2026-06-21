"""train_v11 = v9 recipe but BALANCED VRU (1:1 crossing/non) to fix the over-predict-crossing bias."""
import json,os,random
R="/data/haiyuez/visteon_cabin_vlm/data"
wl=set(json.load(open(R+"/clean_whitelist.json")))
def load(f): return json.load(open(R+"/"+f)) if os.path.exists(R+"/"+f) else []
def cap(d,n,seed=0,dedup=True):
    if dedup: d=[x for x in d if x.get("images") and x["images"][0] in wl]
    random.seed(seed); random.shuffle(d); return d[:n] if n else d
# 均衡 JAAD:按 raw 的 crossing 标记 1:1
raw={json.loads(l)["image"]:json.loads(l) for l in open(R+"/jaad_vru_raw.jsonl")}
jaad=load("jaad_vru_sharegpt.json")
cross=[x for x in jaad if "crossing" in raw.get(x["images"][0],{}).get("vru","")]
noncross=[x for x in jaad if "crossing" not in raw.get(x["images"][0],{}).get("vru","")]
random.seed(11); random.shuffle(cross)
k=min(len(cross),len(noncross)); jaad_bal=cross[:k]+noncross[:k]
print("JAAD balanced:",len(cross),"cross +",len(noncross),"non ->",len(jaad_bal),"(",k,"each )")
parts=[]
parts+=cap(load("exterior_cot_v4_sharegpt.json"),1200,1,dedup=False)
parts+=cap(load("cars_sharegpt.json"),1200,2)
parts+=cap(load("signs_sharegpt.json"),1200,3)
parts+=cap(load("textvqa_road_sharegpt.json"),0,5,dedup=False)
parts+=cap(load("landscape_outdoor_sharegpt.json"),0,6,dedup=False)
parts+=cap(load("trafficlight_sharegpt.json"),0,7,dedup=False)
parts+=jaad_bal
for x in parts:
    for t in x["conversations"]:
        if not isinstance(t["value"],str): t["value"]=str(t["value"])
    if len(x["conversations"])%2!=0: x["conversations"]=x["conversations"][:-1]
random.seed(42); random.shuffle(parts)
json.dump(parts,open(R+"/train_v11_sharegpt.json","w"),ensure_ascii=False)
print("TRAIN_V11",len(parts))
p=R+"/dataset_info.json"; di=json.load(open(p))
di["train_v11"]={"file_name":"train_v11_sharegpt.json","formatting":"sharegpt","columns":{"messages":"conversations","images":"images"},"tags":{"role_tag":"from","content_tag":"value","user_tag":"human","assistant_tag":"gpt"}}
json.dump(di,open(p,"w"),ensure_ascii=False,indent=2); print("registered train_v11")
