from PIL import Image, ImageDraw
from vllm import LLM, SamplingParams
# make a test image: red circle + STOP text on white
img = Image.new("RGB",(480,480),"white")
d=ImageDraw.Draw(img); d.ellipse([90,90,390,390],fill="red")
d.text((200,225),"STOP",fill="white")
img.save("/data/haiyuez/visteon_cabin_vlm/code/_test.png")
llm = LLM(model="/data/haiyuez/visteon_cabin_vlm/models/Qwen2-VL-7B-Instruct",
          max_model_len=4096, gpu_memory_utilization=0.45, limit_mm_per_prompt={"image":1})
q="Describe this image. What color and shape is the main object, and what text does it show?"
prompt=("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"+q+"<|im_end|>\n"
        "<|im_start|>assistant\n")
o=llm.generate({"prompt":prompt,"multi_modal_data":{"image":img}}, SamplingParams(max_tokens=120,temperature=0))
print("=== MODEL OUTPUT ===")
print(o[0].outputs[0].text)
