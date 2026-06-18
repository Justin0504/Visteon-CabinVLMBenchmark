import os,json,argparse,re
from PIL import Image
from vllm import LLM, SamplingParams
ROOT="/data/haiyuez/visteon_cabin_vlm"
M=ROOT+"/models/Qwen2-VL-7B-Instruct"
def _resize(im,maxside=768,minside=256):
    w,h=im.size; m=max(w,h)
    if m>maxside:
        r=maxside/m; im=im.resize((max(28,int(w*r)),max(28,int(h*r))))
    elif m<minside:
        r=minside/m; im=im.resize((max(28,int(w*r)),max(28,int(h*r))))
    return im

def qwen(sys,q):
    return (f"<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>"+q+"<|im_end|>\n<|im_start|>assistant\n")

CAR_SYS="You are an in-car cockpit AI that recognizes vehicles seen on the road."
def car_prompt(label):
    return ('This image shows a single vehicle. GROUND-TRUTH make/model: "'+label+'".\n'
     'Output STRICT JSON only: {"caption":"<one sentence describing the vehicle>","qa":[{"question":"","answer":"","capability":"","use_case":""}]}\n'
     'Make EXACTLY 4 QA pairs:\n'
     '1) "What is the make and model of this vehicle?" -> answer MUST be exactly "'+label+'" (capability=Recognition, use_case=VehicleMakeModel)\n'
     '2) body type (sedan/SUV/truck/...) (Recognition/VehicleMakeModel)\n'
     '3) color of the vehicle (Recognition/ExternalRecognition)\n'
     '4) is it a passenger or commercial vehicle, and one driving consideration (Reasoning/DrivingDecision)\n'
     'Answers must be consistent with the image and the ground truth. Output ONLY JSON.')

SIGN_SYS="You are an in-car cockpit AI that recognizes traffic signs."
def sign_prompt(label):
    return ('This image shows a traffic sign. GROUND-TRUTH meaning: "'+label+'".\n'
     'Output STRICT JSON only: {"caption":"<one sentence describing the sign>","qa":[{"question":"","answer":"","capability":"","use_case":""}]}\n'
     'Make EXACTLY 3 QA pairs:\n'
     '1) "What does this traffic sign mean?" -> answer based on ground truth "'+label+'" (capability=WorldKnowledge, use_case=TrafficSign)\n'
     '2) shape and color of the sign (Recognition/TrafficSign)\n'
     '3) what should the driver do given this sign (Reasoning/DrivingDecision)\n'
     'Answers must match the ground truth. Output ONLY JSON.')

def parse(t):
    t=t.strip().replace("```json","").replace("```","")
    s,e=t.find("{"),t.rfind("}")
    if s<0 or e<0: return None
    try: return json.loads(t[s:e+1])
    except:
        try: return json.loads(t[s:t.rfind('},')+1]+']}')
        except: return None

def run(llm,labels_file,kind,outshare,outraw,first_instr):
    rows=[json.loads(l) for l in open(labels_file)]
    sysp=CAR_SYS if kind=="car" else SIGN_SYS
    pf=car_prompt if kind=="car" else sign_prompt
    inputs=[{"prompt":qwen(sysp,pf(r["label"])),"multi_modal_data":{"image":_resize(Image.open(r["image"]).convert("RGB"))}} for r in rows]
    outs=llm.generate(inputs,SamplingParams(max_tokens=600,temperature=0.3))
    ok=0; fr=open(outraw,"w"); sg=[]
    for r,o in zip(rows,outs):
        rec=parse(o.outputs[0].text)
        if rec and "caption" in rec and isinstance(rec.get("qa"),list) and len(rec["qa"])>=2:
            ok+=1; fr.write(json.dumps({"image":r["image"],"label":r["label"],**rec},ensure_ascii=False)+"\n")
            conv=[{"from":"human","value":"<image>\n"+first_instr},{"from":"gpt","value":rec["caption"]}]
            for q in rec["qa"]:
                if q.get("question") and q.get("answer"):
                    conv+=[{"from":"human","value":q["question"]},{"from":"gpt","value":q["answer"]}]
            sg.append({"conversations":conv,"images":[r["image"]]})
        else: fr.write(json.dumps({"image":r["image"],"parse_error":True},ensure_ascii=False)+"\n")
    fr.close(); json.dump(sg,open(outshare,"w"),ensure_ascii=False,indent=1)
    print(f"{kind}: OK {ok}/{len(rows)} -> {outshare}")

llm=LLM(model=M,max_model_len=8192,gpu_memory_utilization=0.6,limit_mm_per_prompt={"image":1})
# CARS ALREADY DONE (1997)
if False: run(llm,ROOT+"/data/stanford_cars/labels.jsonl","car",
    ROOT+"/data/cars_sharegpt.json",ROOT+"/data/cars_raw.jsonl",
    "Identify the vehicle in this image.")
run(llm,ROOT+"/data/gtsrb/labels.jsonl","sign",
    ROOT+"/data/signs_sharegpt.json",ROOT+"/data/signs_raw.jsonl",
    "Identify the traffic sign in this image.")
print("LABELED_DONE")
