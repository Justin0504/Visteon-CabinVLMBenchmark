"""VRU SEPARATION QA (#6) — build grounded QA that SEPARATES vulnerable-road-user types from
ground-truth labels: pedestrian(adult/child), cyclist vs bicycle, motorcyclist vs motorcycle,
construction/police personnel. Because it is derived directly from GT counts, it is zero-hallucination
(no VLM) — exactly the "rider vs the bike/motorcycle they ride" distinction the use case asks for.

Input  jsonl rows: {image, gt, vehicles, vrus, ...}  (vrus/vehicles are free-form GT strings)
Output sharegpt json: VRU-separation QA per image with >=1 VRU.

Run:
  python -m caption_pipeline.vru_separation --inp fusion_input.jsonl --out vru_sharegpt.json
"""
import json, re, os, argparse

# canonical VRU types and the regex that finds them in the GT strings
TYPES = {
    "pedestrian_adult":  r"pedestrian\(adult\)|adult pedestrian",
    "pedestrian_child":  r"pedestrian\(child\)|child",
    "cyclist":           r"cyclist|\brider\b.*bicycle|bicycle.*rider",
    "bicycle":           r"\bbicycle\b",
    "motorcyclist":      r"motorcyclist",
    "motorcycle":        r"\bmotorcycle\b",
    "construction_worker": r"construction worker|construction",
    "police_officer":    r"police",
    "stroller":          r"stroller|pram",
}

def count_types(text):
    t = (text or "").lower()
    out = {}
    for name, pat in TYPES.items():
        n = len(re.findall(pat, t))
        if n:
            out[name] = n
    return out

def build_qa(vru):
    """Deterministic, GT-grounded QA emphasising type SEPARATION."""
    qa = []
    peds = vru.get("pedestrian_adult", 0) + vru.get("pedestrian_child", 0)
    cyc, bike = vru.get("cyclist", 0), vru.get("bicycle", 0)
    moto_r, moto = vru.get("motorcyclist", 0), vru.get("motorcycle", 0)

    if peds:
        ans = f"{peds} pedestrian(s)"
        if vru.get("pedestrian_child"):
            ans += f", including {vru['pedestrian_child']} child(ren)"
        qa.append(("How many pedestrians are there, and are any of them children?", ans + "."))
    # the key separation: rider vs the vehicle
    if cyc or bike:
        qa.append(("Is the cyclist a person riding, or just a parked bicycle?",
                   (f"There is a person actively riding a bicycle (cyclist)." if cyc
                    else f"There is a bicycle present without a visible rider.")))
    if moto_r or moto:
        qa.append(("Is there a motorcyclist (rider) or an unattended motorcycle?",
                   (f"There is a motorcyclist actively riding." if moto_r
                    else f"There is a motorcycle present without a visible rider.")))
    # cross-type counting
    groups = []
    if peds: groups.append(f"{peds} pedestrian(s)")
    if cyc:  groups.append(f"{cyc} cyclist(s)")
    if moto_r: groups.append(f"{moto_r} motorcyclist(s)")
    if vru.get("construction_worker"): groups.append(f"{vru['construction_worker']} construction worker(s)")
    if vru.get("police_officer"): groups.append(f"{vru['police_officer']} police officer(s)")
    if len(groups) >= 2:
        qa.append(("Break down the vulnerable road users by type.", "; ".join(groups) + "."))
    # stroller (rare; honest when absent handled by caller not adding)
    if vru.get("stroller"):
        qa.append(("Is anyone pushing a stroller?", f"Yes, {vru['stroller']} stroller(s) visible."))
    return qa

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    rows = [json.loads(l) for l in open(a.inp)]
    sg = []
    stats = {}
    for r in rows:
        vru = count_types(str(r.get("vrus", "")) + " " + str(r.get("vehicles", "")))
        if not vru:
            continue
        for k, v in vru.items():
            stats[k] = stats.get(k, 0) + v
        qa = build_qa(vru)
        if not qa:
            continue
        convs = []
        for q, ans in qa:
            convs.append({"from": "human", "value": (("<image>\n" if not convs else "") + q)})
            convs.append({"from": "gpt", "value": ans})
        sg.append({"conversations": convs, "images": [r["image"]]})
    json.dump(sg, open(a.out, "w"), ensure_ascii=False)
    print(f"VRU_SEPARATION_DONE images={len(sg)} qa_pairs={sum(len(s['conversations'])//2 for s in sg)}")
    print("type totals:", json.dumps(stats, ensure_ascii=False))

if __name__ == "__main__":
    main()
