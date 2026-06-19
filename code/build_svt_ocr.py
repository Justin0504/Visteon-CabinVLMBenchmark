"""#5 road-text + spatial coordinates (and #8 storefront signage) from SVT (Street View Text).
SVT = Google Street View images annotated with word-level bounding boxes of business signage.
GT text + bbox are authoritative -> template captions (GPU-free, zero hallucination) that map
each transcribed word to its spatial coordinates, exactly per spec #5."""
import os, json, glob
import xml.etree.ElementTree as ET
R = "/data/haiyuez/visteon_cabin_vlm"
S = R + "/data/svt1"
OUT = R + "/data/svt_ocr_sharegpt.json"
RAW = R + "/data/svt_ocr_raw.jsonl"

def parse(xml):
    rows = []
    for im in ET.parse(xml).getroot().findall("image"):
        name = im.find("imageName").text  # e.g. img/00_00.jpg
        words = []
        for r in im.findall(".//taggedRectangle"):
            t = r.find("tag").text
            x, y, w, h = (int(float(r.attrib[k])) for k in ("x", "y", "width", "height"))
            words.append((t, [x, y, x + w, y + h]))
        if words:
            rows.append((name, words))
    return rows

rows = []
for xml in glob.glob(S + "/*.xml"):
    rows += parse(xml)
print("SVT images with text:", len(rows), flush=True)

sg = []
fr = open(RAW, "w")
for name, words in rows:
    p = os.path.join(S, name)
    if not os.path.exists(p):
        continue
    txt = ", ".join(f'"{t}" at bbox{b}' for t, b in words[:8])
    reads = ", ".join(f'"{t}"' for t, _ in words[:8])
    cap = (f"Scene: A street-level view containing storefront/road signage. "
           f"Detected text (ground-truth, with locations): {txt}. "
           f"Risk: signage and storefronts indicate a populated roadside area; watch for entering/exiting traffic and pedestrians. "
           f"Decision: proceed with normal caution, using the readable signs for navigation.")
    conv = [{"from": "human", "value": "<image>\nRead the text/signage in this street scene and give their locations."},
            {"from": "gpt", "value": cap},
            {"from": "human", "value": "What text appears on the signs?"},
            {"from": "gpt", "value": f"The signage reads {reads}."},
            {"from": "human", "value": f'Where is the text {words[0][0]!r} located?'},
            {"from": "gpt", "value": f"At bounding box {words[0][1]} (x1,y1,x2,y2)."}]
    sg.append({"conversations": conv, "images": [p]})
    fr.write(json.dumps({"image": p, "words": [(t, b) for t, b in words], "caption": cap}, ensure_ascii=False) + "\n")
fr.close()
json.dump(sg, open(OUT, "w"), ensure_ascii=False)
print("SVT_DONE", len(sg), flush=True)
