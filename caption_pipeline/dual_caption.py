"""Dual-version captions: every image gets TWO captions for two purposes —
  (1) DESCRIPTION  — rich, comprehensive scene description (general scene understanding)
  (2) DRIVING      — concise causal scene->risk->decision (driving assistant)
Assembled from data already computed (Stage-A vision_desc = description; Stage-B fusion = driving),
so NO new inference is needed. Emits a combined sharegpt where each image appears as two samples with
different first-turn prompts, teaching the model to switch register by instruction.

Inputs:
  --vision   fusion_input_v2.jsonl  (has image + vision_desc = the detailed description)
  --driving  fusion_raw_v2.jsonl    (has image + caption{scene,risk,decision,prediction} = driving)
Output:
  --out      dual_caption_sharegpt.json
"""
import json, os, argparse

DESC_Q = "<image>\nDescribe this scene in detail — layout, buildings, vegetation, sky, road, signage, and all visible objects with their colors and materials."
DRIVE_Q = "<image>\nYou are an in-car driving assistant. Describe the driving scene and give a driving decision."
KEYS = ["scene", "risk", "decision", "prediction"]

def driving_text(cap):
    parts = []
    for k in KEYS:
        v = str(cap.get(k, "")).strip()
        if v and v.lower() != "none":
            parts.append(v if v.endswith((".", "!", "?")) else v + ".")
    return " ".join(parts).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision", required=True)
    ap.add_argument("--driving", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    desc = {}
    for l in open(a.vision):
        try:
            r = json.loads(l)
            if r.get("vision_desc"):
                desc[r["image"]] = r["vision_desc"].strip()
        except Exception:
            continue
    drive = {}
    for l in open(a.driving):
        try:
            r = json.loads(l)
            t = driving_text(r.get("caption", {}))
            if t:
                drive[r["image"]] = (t, r.get("qa", []))
        except Exception:
            continue
    sg, nd, ndr = [], 0, 0
    for img in set(list(desc) + list(drive)):
        if img in desc:
            sg.append({"conversations": [{"from": "human", "value": DESC_Q},
                                         {"from": "gpt", "value": desc[img]}], "images": [img], "version": "description"})
            nd += 1
        if img in drive:
            t, qa = drive[img]
            convs = [{"from": "human", "value": DRIVE_Q}, {"from": "gpt", "value": t}]
            for q in qa:
                if q.get("q") and q.get("a"):
                    convs += [{"from": "human", "value": str(q["q"])}, {"from": "gpt", "value": str(q["a"])}]
            sg.append({"conversations": convs, "images": [img], "version": "driving"})
            ndr += 1
    with open(a.out + ".tmp", "w") as f:
        json.dump(sg, f, ensure_ascii=False)
    os.replace(a.out + ".tmp", a.out)
    print(f"DUAL_CAPTION_DONE total={len(sg)} description={nd} driving={ndr} -> {a.out}")

if __name__ == "__main__":
    main()
