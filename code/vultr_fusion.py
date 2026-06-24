"""FUSION caption pipeline (fixes the 'visually-shallow' limit of GT-only generation).
Stage A (server, vision): Qwen2-VL writes a RICH free-form visual description per image
   (weather, time, road type, lane markings, building/road surface, lighting) -> vision_desc.
Stage B (local, Vultr frontier LLM, THIS script): given BOTH the GT object facts (authoritative:
   class/count/distance/bbox) AND the vision_desc (rich visual nuance), produce the final
   causal-chain CoT caption + diverse QA. GT anchors objects/counts (no hallucination on them);
   vision_desc supplies the visual detail GT lacks; cross-check removes vision_desc claims that
   contradict GT. = vision richness + GT grounding + frontier reasoning + cross-check.

Input jsonl rows: {image, camera, gt, vehicles, vrus, vision_desc}
"""
import json, urllib.request, argparse, concurrent.futures as cf, os, threading, time
_W = threading.Lock()
KEYS = [l.split("=")[1].strip() for l in open("/Users/justin/SJTU-450/.secrets/vultr_keys.env") if l.startswith("VULTR_KEY")]
URL = "https://api.vultrinference.com/v1/chat/completions"
GEN = "deepseek-ai/DeepSeek-V4-Flash"; XC = "deepseek-ai/DeepSeek-V3.2-NVFP4"
ap = argparse.ArgumentParser()
ap.add_argument("--inp"); ap.add_argument("--out"); ap.add_argument("--raw"); ap.add_argument("--num", type=int, default=0)
a = ap.parse_args(); os.makedirs(os.path.dirname(a.out), exist_ok=True); NL = "\n"

def chat(model, sys, usr, key, mx=1100):
    body = json.dumps({"model": model, "messages": [{"role": "system", "content": sys}, {"role": "user", "content": usr}], "max_tokens": mx, "temperature": 0.3}).encode()
    for att in range(5):
        try:
            req = urllib.request.Request(URL, data=body, headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
            m = json.loads(urllib.request.urlopen(req, timeout=120).read())["choices"][0]["message"]
            return m.get("content") or m.get("reasoning") or ""
        except Exception:
            time.sleep(2 * (att + 1))
    return ""

def jp(t):
    if not t: return None
    t = t.replace("```json", "").replace("```", "").strip(); i = t.find("{"); j = t.rfind("}")
    try: return json.loads(t[i:j + 1]) if i >= 0 else None
    except: return None

CSYS = ("You are the reasoning module of an autonomous-driving cockpit. You receive (1) authoritative GROUND-TRUTH "
        "objects (sensor truth: class/count/distance/bbox) and (2) a VISION description of the same image. "
        "Rule: for objects/counts/positions, trust GROUND-TRUTH only. For visual context not in GT (weather, time-of-day, "
        "road type, lane markings, surface, lighting), you MAY use the VISION description. Never invent. Output only JSON.")
CSCHEMA = ('Use CHAIN-OF-CAUSATION. Output STRICT JSON {"scene","risk","decision","prediction"}: '
           'scene = road type + weather/time/lane (from vision) + each GT object with ego-relative position & distance (from GT); '
           'risk = causal hazard ("because <GT object> at <pos/dist> and <vision context>, it may <hazard>"); '
           'decision = therefore ego action (proceed/slow/stop/yield/lane-change), justified by the cause; '
           'prediction = 1-3s intent of the most safety-critical object, or "none".')
QSCHEMA = ('Generate 4 DIVERSE driving QA grounded in GT (+vision context), >=4 types of: counting, spatial, risk, '
           'action, intent, weather/scene, reject(ask about something NOT supported -> "not visible"). '
           'Output STRICT JSON: {"qa":[{"q":"","a":"","capability":"","type":""}]}')
XSYS = "You are a strict fact-checker. Remove any claim contradicting GROUND-TRUTH objects (counts/classes/positions). Output only JSON."
def cusr(r): return ("Camera: " + r["camera"] + NL + "GROUND-TRUTH objects: counts " + r["gt"] + "; vehicles " + r.get("vehicles", "none") + "; VRUs " + r.get("vrus", "none") + NL + "VISION description: " + r.get("vision_desc", "") + NL + CSCHEMA)
def qusr(r): return ("GROUND-TRUTH: " + r["gt"] + " | VISION: " + r.get("vision_desc", "")[:300] + NL + QSCHEMA)
def xusr(r, cap): return ("GROUND-TRUTH objects: counts " + r["gt"] + NL + "Caption: " + json.dumps(cap, ensure_ascii=False) + NL + 'Remove claims contradicting GT object counts/classes/positions. STRICT JSON {"corrected":{"scene","risk","decision","prediction"}}')

def work(ir):
    i, r = ir
    try:
        cap = jp(chat(GEN, CSYS, cusr(r), KEYS[i % len(KEYS)]))
        qa = jp(chat(GEN, CSYS, qusr(r), KEYS[(i + 1) % len(KEYS)]))
        xc = jp(chat(XC, XSYS, xusr(r, cap), KEYS[(i + 2) % len(KEYS)])) if cap else None
        final = (xc.get("corrected") if (xc and xc.get("corrected")) else cap) or {}
        scene = " ".join(k.capitalize() + ": " + str(final.get(k, "")) for k in ["scene", "risk", "decision", "prediction"] if final.get(k))
        if not scene: return None
        return {"image": r["image"], "camera": r["camera"], "gt": r["gt"], "caption": final, "qa": qa.get("qa", []) if qa else []}
    except Exception: return None

rows = [json.loads(l) for l in open(a.inp)]
if a.num: rows = rows[:a.num]
seen = set()
if os.path.exists(a.raw):
    for l in open(a.raw):
        try: seen.add(json.loads(l)["image"])
        except: pass
rows = [r for r in rows if r["image"] not in seen]
print("FUSION rows", len(rows), flush=True)
fr = open(a.raw, "a" if seen else "w"); done = ok = 0
with cf.ThreadPoolExecutor(max_workers=5) as ex:
    for res in ex.map(work, list(enumerate(rows))):
        done += 1
        if res:
            ok += 1
            with _W: fr.write(json.dumps(res, ensure_ascii=False) + NL); fr.flush()
        if done % 50 == 0: print("done", done, "ok", ok, flush=True)
fr.close()
allr = []
for l in open(a.raw):
    try: allr.append(json.loads(l))
    except: pass
sg = [{"conversations": [{"from": "human", "value": "<image>\nDescribe this exterior driving scene and give a driving decision."}, {"from": "gpt", "value": " ".join(k.capitalize() + ": " + str(x["caption"].get(k, "")) for k in ["scene", "risk", "decision", "prediction"] if x["caption"].get(k))}] + sum(([{"from": "human", "value": str(q["q"])}, {"from": "gpt", "value": str(q["a"])}] for q in x["qa"] if q.get("q") and q.get("a")), []), "images": [x["image"]]} for x in allr]
json.dump(sg, open(a.out, "w"), ensure_ascii=False)
print("FUSION_DONE ok", ok, "total_sg", len(sg), flush=True)
