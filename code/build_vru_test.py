"""Held-out VRU-crossing test from JAAD clips NOT used in training (clips 60+)."""
import os,json,glob
import xml.etree.ElementTree as ET
import cv2
R="/data/haiyuez/visteon_cabin_vlm"; J=R+"/data/JAAD"; D=R+"/data/vru_test_imgs"; os.makedirs(D,exist_ok=True)
def parse(xml):
    fr={}
    for tr in ET.parse(xml).getroot().findall(".//track"):
        if tr.attrib.get("label") not in ("pedestrian","ped"): continue
        for b in tr.findall("box"):
            if b.attrib.get("outside")=="1": continue
            f=int(b.attrib["frame"]); at={c.attrib.get("name"):c.text for c in b.findall("attribute")}
            fr.setdefault(f,[]).append(at.get("cross"))
    return fr
clips=sorted(glob.glob(J+"/JAAD_clips/*.mp4"))[60:120]  # held-out (train used [:60])
rows=[]; n=0
for mp4 in clips:
    vid=os.path.basename(mp4).replace(".mp4",""); xml=J+"/annotations/"+vid+".xml"
    if not os.path.exists(xml): continue
    fr=parse(xml)
    cross_frames=[f for f,cs in fr.items() if "crossing" in cs]
    nocross=[f for f,cs in fr.items() if cs and "crossing" not in cs]
    pick=[]
    if cross_frames: pick.append((sorted(cross_frames)[len(cross_frames)//2],"yes"))
    if nocross: pick.append((sorted(nocross)[len(nocross)//2],"no"))
    for f,lab in pick:
        cap=cv2.VideoCapture(mp4); cap.set(cv2.CAP_PROP_POS_FRAMES,f); ok,im=cap.read(); cap.release()
        if not ok: continue
        p=f"{D}/vru_{n:03d}.jpg"; cv2.imwrite(p,im)
        rows.append({"image":p,"crossing":lab}); n+=1
    if n>=40: break
json.dump(rows,open(R+"/data/vru_test.json","w"),ensure_ascii=False)
from collections import Counter
print("VRU_TEST_BUILT",len(rows),dict(Counter(r["crossing"] for r in rows)))
