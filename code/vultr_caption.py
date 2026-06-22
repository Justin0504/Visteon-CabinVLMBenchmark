import json,urllib.request,os
KEYS=[l.split("=")[1].strip() for l in open("/Users/justin/SJTU-450/.secrets/vultr_keys.env") if l.startswith("VULTR_KEY")]
URL="https://api.vultrinference.com/v1/chat/completions"
def chat(model,sys,usr,key,maxtok=1200,temp=0.3):
    body=json.dumps({"model":model,"messages":[{"role":"system","content":sys},{"role":"user","content":usr}],"max_tokens":maxtok,"temperature":temp}).encode()
    req=urllib.request.Request(URL,data=body,headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"})
    r=json.loads(urllib.request.urlopen(req,timeout=90).read())
    m=r["choices"][0]["message"]; return m.get("content") or m.get("reasoning") or ""
def jparse(t):
    if not t: return None
    t=t.replace("```json","").replace("```","").strip()
    a=t.find("{"); b=t.rfind("}")
    if a<0 or b<0: return None
    try: return json.loads(t[a:b+1])
    except: return None
ROLE={"front":"stop/go, traffic lights & signs, pedestrians crossing ahead","front-left/side":"left lane-change & blind-spot",
"front-right/side":"right lane-change & blind-spot","rear":"vehicles approaching from behind","rear-left/side":"rear-left blind-spot","rear-right/side":"rear-right blind-spot"}
CAP_SYS="You are the reasoning module of an autonomous-driving cockpit. You ONLY assert facts present in the GROUND-TRUTH. Never invent objects, signs, lights, or weather."
def cap_usr(cam,gt,veh,vru):
    return (f"Camera: {cam} (focus: {ROLE.get(cam,'driving')}).\nGROUND-TRUTH (authoritative perception):\n- counts: {gt}\n- vehicles [class pos dist bbox]: {veh}\n- VRUs (riders separated): {vru}\n"
    'Write a VLA chain-of-thought driving caption. Output STRICT JSON {"scene","risk","decision","prediction"}:\n'
    "scene: road + each GT object with ego-relative position & distance (only GT).\n"
    "risk: driving risk implied ONLY by listed objects, name the object + position.\n"
    "decision: ego action (proceed/slow/stop/yield/lane-change) for this camera role, justified by risk.\n"
    "prediction: 1-3s intent of the most safety-critical dynamic object (or 'none').")
def qa_usr(gt,veh,vru):
    return (f"GROUND-TRUTH: counts {gt}; vehicles {veh}; VRUs {vru}.\n"
    "Generate 4 DIVERSE driving QA grounded ONLY in GT, covering >=4 of: counting, spatial, risk, action, intent, reject(ask about something NOT in GT -> 'not visible').\n"
    'Each tagged. STRICT JSON: {"qa":[{"q","a","capability","type"}]}')
XC_SYS="You are a strict driving-caption fact-checker."
def xc_usr(gt,veh,vru,cap):
    return (f"GROUND-TRUTH: counts {gt}; vehicles {veh}; VRUs {vru}.\nCaption: {json.dumps(cap,ensure_ascii=False)}.\n"
    'Delete/correct EVERY claim not in GT. Output STRICT JSON {"faithful":true/false,"removed_claims":[],"corrected":{"scene","risk","decision","prediction"}}')
GEN="deepseek-ai/DeepSeek-V4-Flash"
XC="deepseek-ai/DeepSeek-V3.2-NVFP4"
d=[json.loads(l) for l in open("/tmp/ex2/data/exterior_cot_v2_raw.jsonl")][:3]
for i,r in enumerate(d):
    gt,veh,vru,cam=r["gt"],r.get("vehicles","none"),r.get("vrus","none"),r["camera"]
    cap=jparse(chat(GEN,CAP_SYS,cap_usr(cam,gt,veh,vru),KEYS[0]))
    qa=jparse(chat("deepseek-ai/DeepSeek-V4-Flash",CAP_SYS,qa_usr(gt,veh,vru),KEYS[1]))
    xc=jparse(chat(XC,XC_SYS,xc_usr(gt,veh,vru,cap),KEYS[2])) if cap else None
    print(f"\n===== 样例{i+1} [{cam}] GT: {gt[:60]} =====")
    print("CAPTION:",json.dumps(cap,ensure_ascii=False)[:420] if cap else "FAIL")
    print("QA:",json.dumps(qa.get("qa",[])[:2],ensure_ascii=False)[:300] if qa else "FAIL")
    print("XCHECK faithful:",xc.get("faithful") if xc else "?","removed:",(xc.get("removed_claims") if xc else [])[:2])
