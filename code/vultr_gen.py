"""Frontier-LLM reasoning layer (production): regenerate CoT caption + diversified QA from GT facts,
cross-checked. Runs locally vs Vultr (DeepSeek-V4 gen + DeepSeek-V3.2 cross-check), 5-key concurrency."""
import json, urllib.request, argparse, concurrent.futures as cf, os
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

def chat(model, sys, usr, key, maxtok=1100, temp=0.3):
    body = json.dumps({"model": model, "messages": [{"role": "system", "content": sys}, {"role": "user", "content": usr}], "max_tokens": maxtok, "temperature": temp}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    m = json.loads(urllib.request.urlopen(req, timeout=120).read())["choices"][0]["message"]
    return m.get("content") or m.get("reasoning") or ""

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
CSCHEMA = 'Output STRICT JSON {"scene","risk","decision","prediction"}: scene=road + each GT object with ego-relative position and distance (GT only); risk=risk implied ONLY by listed objects, name object and position; decision=ego action (proceed/slow/stop/yield/lane-change) for this camera role; prediction=1-3s intent of most safety-critical dynamic object or "none".'
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
print("ROWS", len(rows), flush=True)
sg = []; raw = []; done = 0
with cf.ThreadPoolExecutor(max_workers=10) as ex:
    for res in ex.map(work, list(enumerate(rows))):
        done += 1
        if res: sg.append(res["sg"]); raw.append(res["raw"])
        if done % 50 == 0: print("done", done, "ok", len(sg), flush=True)
json.dump(sg, open(a.out, "w"), ensure_ascii=False)
open(a.raw, "w").write(NL.join(json.dumps(x, ensure_ascii=False) for x in raw))
print("VULTR_DONE total", len(rows), "ok", len(sg), flush=True)
