"""Frontier-LLM reasoning layer (production): regenerate CoT caption + diversified QA from GT facts,
cross-checked. Runs locally vs Vultr (DeepSeek-V4 gen + DeepSeek-V3.2 cross-check), 5-key concurrency."""
import json, urllib.request, argparse, concurrent.futures as cf, os, threading
_WLOCK=threading.Lock()
KEYS = [l.split("=")[1].strip() for l in open("/Users/justin/SJTU-450/.secrets/vultr_keys.env") if l.startswith("VULTR_KEY")]
URL = "https://api.vultrinference.com/v1/chat/completions"
GEN = "deepseek-ai/DeepSeek-V4-Flash"; XC = "deepseek-ai/DeepSeek-V3.2-NVFP4"
ap = argparse.ArgumentParser()
ap.add_argument("--inp", default="/tmp/ex2/data/exterior_cot_v2_raw.jsonl")
ap.add_argument("--out", default="/Users/justin/SJTU-450/网盘_数据集/_gen/exterior_cot_v5_sharegpt.json")
ap.add_argument("--raw", default="/Users/justin/SJTU-450/网盘_数据集/_gen/exterior_cot_v5_raw.jsonl")
ap.add_argument("--num", type=int, default=0)
a = ap.parse_args()
os.makedirs(os.path.dirname(a.out), exist_ok=True)
NL = "\n"

import time
def chat(model, sys, usr, key, maxtok=1100, temp=0.3):
    body = json.dumps({"model": model, "messages": [{"role": "system", "content": sys}, {"role": "user", "content": usr}], "max_tokens": maxtok, "temperature": temp}).encode()
    for att in range(5):
        try:
            req = urllib.request.Request(URL, data=body, headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
            m = json.loads(urllib.request.urlopen(req, timeout=120).read())["choices"][0]["message"]
            return m.get("content") or m.get("reasoning") or ""
        except Exception:
            time.sleep(2*(att+1))
    return ""

def jp(t):
    if not t: return None
    t = t.replace("```json", "").replace("```", "").strip()
    i = t.find("{"); j = t.rfind("}")
    if i < 0 or j < 0: return None
    try: return json.loads(t[i:j+1])
    except: return None

ROLE = {"front": "stop/go, lights & signs, pedestrians ahead", "front-left/side": "left lane-change & blind-spot", "front-right/side": "right lane-change & blind-spot", "rear": "vehicles from behind", "rear-left/side": "rear-left blind-spot", "rear-right/side": "rear-right blind-spot"}
CSYS = "You are the reasoning module of an autonomous-driving cockpit. You ONLY assert facts in the GROUND-TRUTH. Never invent objects/signs/lights/weather. Output only JSON."
QSYS = CSYS
XSYS = "You are a strict driving-caption fact-checker. Output only JSON."
CSCHEMA = 'Use CHAIN-OF-CAUSATION reasoning (NVIDIA Alpamayo style): each step must causally follow from the previous. Output STRICT JSON {"scene","risk","decision","prediction"}: scene=road type + each GT object with ego-relative position & distance + ego state if given (GT only); risk=the CAUSAL hazard — "because <object> is at <position/distance> and ego is <state>, it may <hazard>"; decision=therefore the ego action (proceed/slow/stop/yield/lane-change) for this camera role, explicitly justified by the risk cause; prediction=1-3s future intent of the most safety-critical dynamic object, or "none". Be causal and concrete, GT-grounded only.'
QSCHEMA = 'Generate 4 DIVERSE driving QA grounded ONLY in GT, covering at least 4 of: counting, spatial, risk, action, intent, reject (ask about something NOT in GT -> answer "not visible"). Output STRICT JSON: {"qa":[{"q":"","a":"","capability":"","type":""}]}'
XSCHEMA = 'Delete or correct EVERY claim not in GT. Output STRICT JSON: {"faithful":true,"corrected":{"scene":"","risk":"","decision":"","prediction":""}}'

def cusr(cam, gt, veh, vru):
    return ("Camera: " + cam + " (focus: " + ROLE.get(cam, "driving") + ")." + NL + "GROUND-TRUTH:" + NL + "- counts: " + gt + NL + "- vehicles[class pos dist bbox]: " + veh + NL + "- VRUs(riders separated): " + vru + NL + CSCHEMA)

def qusr(gt, veh, vru):
    return ("GROUND-TRUTH: counts " + gt + "; vehicles " + veh + "; VRUs " + vru + "." + NL + QSCHEMA)

def xusr(gt, veh, vru, cap):
    return ("GROUND-TRUTH: counts " + gt + "; vehicles " + veh + "; VRUs " + vru + "." + NL + "Caption: " + json.dumps(cap, ensure_ascii=False) + NL + XSCHEMA)

def work(idx_r):
    idx, r = idx_r
    gt, veh, vru, cam = r["gt"], r.get("vehicles", "none"), r.get("vrus", "none"), r["camera"]
    try:
        cap = jp(chat(GEN, CSYS, cusr(cam, gt, veh, vru), KEYS[idx % len(KEYS)]))
        qa = jp(chat(GEN, QSYS, qusr(gt, veh, vru), KEYS[(idx+1) % len(KEYS)]))
        xc = jp(chat(XC, XSYS, xusr(gt, veh, vru, cap), KEYS[(idx+2) % len(KEYS)])) if cap else None
        final = (xc.get("corrected") if (xc and xc.get("corrected")) else cap) or {}
        scene = " ".join(k.capitalize() + ": " + str(final.get(k, "")) for k in ["scene", "risk", "decision", "prediction"] if final.get(k))
        if not scene: return None
        conv = [{"from": "human", "value": "<image>\nDescribe this exterior driving scene and give a driving decision."}, {"from": "gpt", "value": scene}]
        for q in (qa.get("qa", []) if qa else []):
            if q.get("q") and q.get("a"): conv += [{"from": "human", "value": str(q["q"])}, {"from": "gpt", "value": str(q["a"])}]
        return {"sg": {"conversations": conv, "images": [r["image"]]}, "raw": {"image": r["image"], "camera": cam, "gt": gt, "caption": final, "qa": qa.get("qa", []) if qa else [], "faithful": xc.get("faithful") if xc else None}}
    except Exception:
        return None

rows = [json.loads(l) for l in open(a.inp)]
if a.num: rows = rows[:a.num]
seen=set()
if os.path.exists(a.raw):
    for l in open(a.raw):
        try: seen.add(json.loads(l)["image"])
        except: pass
rows=[r for r in rows if r["image"] not in seen]
print("RESUME skip",len(seen),"todo",len(rows),flush=True)
print("ROWS", len(rows), flush=True)
done = 0; okc = 0
fr=open(a.raw,"a" if seen else "w")
with cf.ThreadPoolExecutor(max_workers=5) as ex:
    for res in ex.map(work, list(enumerate(rows))):
        done += 1
        if res:
            okc += 1
            with _WLOCK:
                fr.write(json.dumps(res["raw"],ensure_ascii=False)+NL); fr.flush()
        if done % 50 == 0: print("done", done, "ok", okc, flush=True)
fr.close()
allraw=[]
for l in open(a.raw):
    l=l.strip()
    if not l: continue
    try: allraw.append(json.loads(l))
    except: pass
sgall=[{"conversations":[{"from":"human","value":"<image>\nDescribe this exterior driving scene and give a driving decision."},{"from":"gpt","value":" ".join(k.capitalize()+": "+str(x["caption"].get(k,"")) for k in ["scene","risk","decision","prediction"] if x["caption"].get(k))}]+sum(([{"from":"human","value":str(q["q"])},{"from":"gpt","value":str(q["a"])}] for q in x["qa"] if q.get("q") and q.get("a")),[]),"images":[x["image"]]} for x in allraw]
json.dump(sgall, open(a.out, "w"), ensure_ascii=False)
print("VULTR_DONE total", len(rows), "ok", okc, flush=True)
