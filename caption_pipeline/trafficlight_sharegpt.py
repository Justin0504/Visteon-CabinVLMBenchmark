"""Build #4 traffic-light sharegpt QA from tl_raw.jsonl (output of trafficlight_extract).
Grounded in the extracted state/shape/position; teaches state recognition + the lawful ego action.

Run: python -m caption_pipeline.trafficlight_sharegpt --inp tl_raw.jsonl --out tl_sharegpt.json
"""
import json, os, argparse

ACTION = {"red": "stop and wait", "yellow": "prepare to stop", "green": "proceed with caution", "off": "treat as uncontrolled and yield"}

def build(rec):
    lights = rec.get("lights", [])
    if not lights:
        return None
    primary = lights[0]
    state = str(primary.get("state", "")).lower()
    shape = primary.get("shape", "circle")
    pos = primary.get("position", "ahead")
    convs = [{"from": "human", "value": "<image>\nWhat is the state of the traffic light ahead?"},
             {"from": "gpt", "value": f"The traffic light {pos} is {state}" + (f" ({shape} signal)." if shape else ".")}]
    # lawful action
    if state in ACTION:
        convs += [{"from": "human", "value": "Given that signal, what should I do?"},
                  {"from": "gpt", "value": f"You should {ACTION[state]}."}]
    # shape / multi-light
    if shape and shape != "circle":
        convs += [{"from": "human", "value": "What shape is the signal lamp?"},
                  {"from": "gpt", "value": f"It is a {shape} signal" + (f" for {primary.get('for_lane')}." if primary.get("for_lane", "unknown") not in ("unknown", "") else ".")}]
    if len(lights) > 1:
        summ = "; ".join(f"{l.get('state')} {l.get('shape','')}".strip() + f" ({l.get('position','')})" for l in lights[:4])
        convs += [{"from": "human", "value": "List all the traffic signals you can see."},
                  {"from": "gpt", "value": summ + "."}]
    return {"conversations": convs, "images": [rec["image"]]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    rows = [json.loads(l) for l in open(a.inp)]
    sg, states = [], {}
    for r in rows:
        s = build(r)
        if s:
            sg.append(s)
            for l in r.get("lights", []):
                st = str(l.get("state", "")).lower(); states[st] = states.get(st, 0) + 1
    json.dump(sg, open(a.out, "w"), ensure_ascii=False)
    print(f"TL_SHAREGPT_DONE images={len(sg)} qa_pairs={sum(len(s['conversations'])//2 for s in sg)}")
    print("state distribution:", json.dumps(states, ensure_ascii=False))

if __name__ == "__main__":
    main()
