"""STAGE B — fuse VISION description + GROUND-TRUTH objects into a causal caption + diverse QA.

For each image:
  1. generate a Chain-of-Causation caption (scene -> risk -> decision -> prediction),
     objects anchored to GT, context from VISION;
  2. generate diverse driver QA (IntelliCockpitBench rigor: multimodal-required, reason field,
     perspective + capability diversity, one reject item);
  3. cross-check the caption with a second model, dropping any claim that contradicts GT.
Outputs a resumable raw jsonl and a ready-to-train sharegpt json.

Input  jsonl rows: {image, camera, gt, vehicles, vrus, vision_desc}   (Stage A output)
Output raw jsonl : {image, camera, gt, caption{scene,risk,decision,prediction}, qa[...]}
Output sharegpt  : [{conversations:[...], images:[path]}]

Run:
  VULTR_KEYS=keys.env python -m caption_pipeline.stage_b_fusion \
      --inp vision.jsonl --raw fusion_raw.jsonl --out fusion_sharegpt.json
"""
import json, os, argparse, threading
import concurrent.futures as cf
from . import config, prompts
from .vultr_client import chat_text, parse_json

_W = threading.Lock()
NL = "\n"
KEYS_OF = ["scene", "risk", "decision", "prediction"]

def _cap_user(r):
    return ("Camera: " + r.get("camera", "") + NL +
            "GROUND-TRUTH objects: counts " + r.get("gt", "") +
            "; vehicles " + r.get("vehicles", "none") + "; VRUs " + r.get("vrus", "none") + NL +
            "VISION description: " + r.get("vision_desc", "") + NL + prompts.CAPTION_SCHEMA)

def _qa_user(r):
    return ("GROUND-TRUTH: " + r.get("gt", "") + " | VISION: " + r.get("vision_desc", "")[:300] + NL +
            prompts.qa_schema(4))

def _xc_user(r, cap):
    return ("GROUND-TRUTH objects: counts " + r.get("gt", "") + NL +
            "Caption: " + json.dumps(cap, ensure_ascii=False) + NL + prompts.xcheck_schema())

def _flatten(cap):
    return " ".join(k.capitalize() + ": " + str(cap.get(k, "")) for k in KEYS_OF if cap.get(k))

def work(ir, keys):
    i, r = ir
    try:
        cap = parse_json(chat_text(config.GEN_MODEL, prompts.CAPTION_SYS, _cap_user(r), keys[i % len(keys)], config.GEN_MAX_TOKENS))
        qa = parse_json(chat_text(config.GEN_MODEL, prompts.QA_SYS, _qa_user(r), keys[(i + 1) % len(keys)], config.GEN_MAX_TOKENS))
        xc = parse_json(chat_text(config.XCHECK_MODEL, prompts.XCHECK_SYS, _xc_user(r, cap), keys[(i + 2) % len(keys)], config.GEN_MAX_TOKENS)) if cap else None
        final = (xc.get("corrected") if (xc and xc.get("corrected")) else cap) or {}
        if not _flatten(final):
            return None
        return {"image": r["image"], "camera": r.get("camera", ""), "gt": r.get("gt", ""),
                "caption": final, "qa": (qa.get("qa", []) if qa else [])}
    except Exception:
        return None

def to_sharegpt(rec):
    convs = [{"from": "human", "value": "<image>\nDescribe this exterior driving scene and give a driving decision."},
             {"from": "gpt", "value": _flatten(rec["caption"])}]
    for q in rec["qa"]:
        if q.get("q") and q.get("a"):
            convs.append({"from": "human", "value": str(q["q"])})
            convs.append({"from": "gpt", "value": str(q["a"])})
    return {"conversations": convs, "images": [rec["image"]]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="Stage A output jsonl (has vision_desc)")
    ap.add_argument("--raw", required=True, help="resumable raw jsonl output")
    ap.add_argument("--out", required=True, help="sharegpt json output")
    ap.add_argument("--num", type=int, default=0)
    a = ap.parse_args()
    keys = config.load_keys()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    rows = [json.loads(l) for l in open(a.inp)]
    if a.num:
        rows = rows[:a.num]
    seen = set()
    if os.path.exists(a.raw):
        for l in open(a.raw):
            try: seen.add(json.loads(l)["image"])
            except Exception: pass
    todo = [r for r in rows if r["image"] not in seen]
    print(f"STAGE_B rows={len(todo)} gen={config.GEN_MODEL} xcheck={config.XCHECK_MODEL} workers={config.WORKERS}", flush=True)

    fr = open(a.raw, "a" if seen else "w"); done = ok = 0
    with cf.ThreadPoolExecutor(max_workers=config.WORKERS) as ex:
        for res in ex.map(lambda ir: work(ir, keys), list(enumerate(todo))):
            done += 1
            if res:
                ok += 1
                with _W:
                    fr.write(json.dumps(res, ensure_ascii=False) + NL); fr.flush()
            if done % 50 == 0:
                print(f"done {done} ok {ok}", flush=True)
    fr.close()

    allr = []
    for l in open(a.raw):
        try: allr.append(json.loads(l))
        except Exception: pass
    sg = [to_sharegpt(x) for x in allr]
    json.dump(sg, open(a.out, "w"), ensure_ascii=False)
    print(f"STAGE_B_DONE ok {ok} total_sharegpt {len(sg)}", flush=True)

if __name__ == "__main__":
    main()
