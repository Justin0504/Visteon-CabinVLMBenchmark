import torch, gradio as gr, glob, os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
R="/data/haiyuez/visteon_cabin_vlm"
MODEL=os.environ.get("DEMO_MODEL", R+"/models/bootstrap_v1_merged")
print("loading",MODEL)
model=Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL,torch_dtype=torch.bfloat16).to("cuda").eval()
proc=AutoProcessor.from_pretrained(MODEL,max_pixels=401408)
def answer(image, question):
    if image is None: return "请上传一张图片"
    q=question.strip() or "Describe this driving scene and any important details for the driver."
    ms=[{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":q}]}]
    c=proc.apply_chat_template(ms,tokenize=False,add_generation_prompt=True); ii,_=process_vision_info(ms)
    inp=proc(text=[c],images=ii,return_tensors="pt").to("cuda")
    with torch.no_grad(): o=model.generate(**inp,max_new_tokens=256,do_sample=False)
    return proc.decode(o[0][inp.input_ids.shape[1]:],skip_special_tokens=True).strip()
exs=sorted(glob.glob(R+"/data/nuscenes/samples/CAM_FRONT/*.jpg"))[:3]
demo=gr.Interface(fn=answer,
    inputs=[gr.Image(type="pil",label="座舱/车外图像"),gr.Textbox(label="问题(留空=自动场景描述)")],
    outputs=gr.Textbox(label="模型回答"),
    title="Cabin VLM — 智能座舱多模态识别 Demo",
    description="上传车内/车外图像，模型识别场景、车型、标志、行人等并问答（Visteon Capstone）",
    examples=[[e,"What is happening in this scene?"] for e in exs])
if __name__=="__main__":
    demo.launch(server_name="0.0.0.0",server_port=7860,share=False)
