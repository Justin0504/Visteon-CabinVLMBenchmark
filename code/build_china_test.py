"""Held-out China-sign test slice from TT100K VAL split (disjoint from train)."""
import json,os,io
from datasets import load_dataset
from PIL import Image
R="/data/haiyuez/visteon_cabin_vlm"; D=R+"/data/china_test_imgs"; os.makedirs(D,exist_ok=True)
def meaning(c):
    if c.startswith("pl"): return f"speed limit {c[2:]} km/h" if c[2:].isdigit() else "speed limit"
    if c.startswith("pm"): return "weight limit"
    if c.startswith("ph"): return "height limit"
    if c.startswith("pn"): return "no parking/stopping"
    if c.startswith("pne"): return "no entry"
    if c.startswith("pr"): return "speed restriction"
    if c.startswith("p"): return "prohibitory sign"
    if c.startswith("w"): return "warning sign"
    if c.startswith("il"): return "minimum speed"
    if c.startswith("i"): return "informative sign"
    return "traffic sign"
ds=load_dataset("PrashantDixit0/TT-100K",split="val",streaming=True)
rows=[]; n=0
for r in ds:
    objs=r.get("objects") or []
    if not objs: continue
    big=max(objs,key=lambda o:(o["bbox"][2]*o["bbox"][3]))  # 最大的标志
    cat=big.get("category");
    if not cat: continue
    img=r["image"]; im=Image.open(io.BytesIO(img["bytes"])) if isinstance(img,dict) else im
    p=f"{D}/ct_{n:03d}.jpg"; im.convert("RGB").save(p,quality=88)
    rows.append({"image":p,"code":cat,"meaning":meaning(str(cat))})
    n+=1
    if n>=40: break
json.dump(rows,open(R+"/data/china_test.json","w"),ensure_ascii=False)
print("CHINA_TEST_BUILT",len(rows))
