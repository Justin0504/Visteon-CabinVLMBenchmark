import json,os
from collections import Counter
from datasets import load_dataset
IMG="/Users/justin/SJTU-450/网盘_数据集/_imgs/carla"; os.makedirs(IMG,exist_ok=True)
OUT="/tmp/carla_gt.jsonl"; N=int(os.environ.get("N","300"))
def weather(c,p,f,sun):
    w=["rainy" if (p and p>20) else ("overcast" if (c and c>60) else "clear")]
    if f and f>30: w.append("foggy")
    w.append("night" if (sun is not None and sun<5) else "daytime")
    return ", ".join(w)
ds=load_dataset("immanuelpeter/carla-autopilot-multimodal-dataset",split="train",streaming=True)
out=open(OUT,"w"); n=0
for r in ds:
    try:
        im=r.get("image_front")
        if im is None: continue
        p=f"{IMG}/carla_{n:04d}.jpg"; im.convert("RGB").save(p,quality=88)
        labs=Counter(r.get("box_labels") or [])
        veh=labs.get("vehicle",0); walk=labs.get("walker",0)+labs.get("pedestrian",0)
        spd=round(r.get("speed_kmh") or 0,1); near=r.get("nearby_vehicles_50m") or 0
        wx=weather(r.get("weather_cloudiness"),r.get("weather_precipitation"),r.get("weather_fog_density"),r.get("weather_sun_altitude"))
        gt=f"ego speed {spd} km/h; weather {wx}; nearby vehicles within 50m: {near}; detected vehicles: {veh}, pedestrians/VRU: {walk}"
        vehs=", ".join(f"{k}:{v}" for k,v in labs.most_common() if k!="walker") or "none"
        out.write(json.dumps({"image":p,"camera":"front","gt":gt,"vehicles":vehs,"vrus":(f"pedestrian:{walk}" if walk else "none")})+"\n")
        n+=1
        if n>=N: break
    except Exception: continue
out.close(); print("CARLA_LOCAL_DONE",n)
