import sys,json,requests
from PIL import Image
from vllm import LLM, SamplingParams
ROOT="/data/haiyuez/visteon_cabin_vlm"
def qwen(q): return f"<|im_start|>system\nYou are an in-car cockpit AI.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{q}<|im_end|>\n<|im_start|>assistant\n"
llm=LLM(model=ROOT+"/models/Qwen2-VL-7B-Instruct",max_model_len=4096,gpu_memory_utilization=0.5,limit_mm_per_prompt={"image":1})
def vlm(img,q,mx=120):
    o=llm.generate([{"prompt":qwen(q),"multi_modal_data":{"image":img}}],SamplingParams(max_tokens=mx,temperature=0))
    return o[0].outputs[0].text.strip()
def wiki(query):
    H={"User-Agent":"CabinVLM/1.0"}
    def summary(title):
        try:
            r=requests.get("https://en.wikipedia.org/api/rest_v1/page/summary/"+title.replace(" ","_"),headers=H,timeout=8)
            if r.status_code==200:
                j=r.json()
                if j.get("type")!="disambiguation" and j.get("extract"): return {"title":j.get("title"),"extract":j["extract"]}
        except: pass
        return None
    d=summary(query)
    if d: return d
    try:
        s=requests.get("https://en.wikipedia.org/w/api.php",params={"action":"query","list":"search","srsearch":query,"format":"json","srlimit":1},headers=H,timeout=8).json()
        hits=s.get("query",{}).get("search",[])
        if hits: return summary(hits[0]["title"])
    except: pass
    return None
def poi_answer(img):
    ent=vlm(img,"Extract the most prominent business name, brand, building, or landmark text visible in this image. Reply ONLY the name, or 'NONE' if none.",30).strip().strip('".')
    if ent.upper()=="NONE" or len(ent)<2:
        return {"entity":None,"answer":"未识别到可检索的店招/地标。"}
    kb=wiki(ent)
    if not kb: return {"entity":ent,"answer":f"识别到 '{ent}'，但知识库未找到相关信息。"}
    ans=vlm(img,f"The user asks about the place in this image. We identified '{ent}'. Retrieved info: \"{kb['extract'][:400]}\". Give a concise, friendly POI introduction for the driver based on this.",150)
    return {"entity":ent,"kb_title":kb["title"],"answer":ans}
if __name__=="__main__":
    import glob
    tests=[ROOT+"/data/nuscenes/samples/CAM_FRONT/"+__import__("os").path.basename(p) for p in []]
    # 用样例图(含 Starbucks 招牌)
    cand=ROOT+"/data/textvqa/images/00000.jpg"
    import os
    imgs=[ "/Users/justin" ] # placeholder
    # 直接用 deliverables 里的 starbucks 示例对应的 nuscenes 原图
    p=ROOT+"/data/nuscenes/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg"
    print("TEST1 (street w/ Starbucks):", json.dumps(poi_answer(Image.open(p).convert("RGB")),ensure_ascii=False))
    p2=sorted(glob.glob(ROOT+"/data/textvqa/images/*.jpg"))[2]
    print("TEST2 (textvqa):", json.dumps(poi_answer(Image.open(p2).convert("RGB")),ensure_ascii=False))
