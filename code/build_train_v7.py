import json,os,random
R="/data/haiyuez/visteon_cabin_vlm/data"
wl=set(json.load(open(R+"/clean_whitelist.json")))  # dedup 白名单
def load(f): return json.load(open(R+"/"+f)) if os.path.exists(R+"/"+f) else []
def cap(d,n,seed=0):
    d=[x for x in d if x.get("images") and x["images"][0] in wl]  # 仅保留去重白名单内
    random.seed(seed); random.shuffle(d); return d[:n]
parts=[]
parts+=cap(load("exterior_cot_v2_sharegpt.json"),1200,1)   # nuScenes 增强CoT
parts+=cap(load("cars_sharegpt.json"),1200,2)              # 车型
parts+=cap(load("signs_sharegpt.json"),1200,3)             # 标志
parts+=cap(load("textvqa_road_sharegpt.json"),9999,4)      # OCR(已域过滤,全收)
parts+=cap(load("landscape_outdoor_sharegpt.json"),9999,5) # 景观(已域过滤,全收)
parts+=cap(load("trafficlight_sharegpt.json"),9999,6)      # 交通灯
parts+=cap(load("jaad_vru_sharegpt.json"),9999,7)          # VRU
# 清洗非字符串值 + 去奇数轮
for x in parts:
    for t in x["conversations"]:
        if not isinstance(t["value"],str):
            t["value"]=json.dumps(t["value"],ensure_ascii=False) if isinstance(t["value"],(dict,list)) else str(t["value"])
    if len(x["conversations"])%2!=0: x["conversations"]=x["conversations"][:-1]
random.seed(42); random.shuffle(parts)
json.dump(parts,open(R+"/train_v7_sharegpt.json","w"),ensure_ascii=False)
from collections import Counter
def src(p):
    for k in ["nuscenes","stanford_cars","gtsrb","textvqa","sun397","road_traffic","jaad_frames"]:
        if k in p: return k
    return "other"
print("TRAIN_V7",len(parts),dict(Counter(src(x["images"][0]) for x in parts)))
p=R+"/dataset_info.json"; di=json.load(open(p))
di["train_v7"]={"file_name":"train_v7_sharegpt.json","formatting":"sharegpt","columns":{"messages":"conversations","images":"images"},"tags":{"role_tag":"from","content_tag":"value","user_tag":"human","assistant_tag":"gpt"}}
json.dump(di,open(p,"w"),ensure_ascii=False,indent=2); print("registered train_v7")
