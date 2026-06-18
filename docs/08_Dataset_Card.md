# 📦 Dataset Card — Cabin VLM Benchmark · Exterior (车外)

> 更新 2026-06-11 · 车外（Category 2）训练/评测数据集。8,030 图 · ~34,578 QA · 5 源。

## 1. 概览
| 源 | 图 | QA | 覆盖能力 | 标注方式 | 许可 |
|---|---|---|---|---|---|
| nuScenes (CAM 全6相机) | 2,258 | 18,275 | 场景描述/行人/路况/天气 | Qwen2-VL 生成 | CC BY-NC-SA |
| Stanford Cars | 1,997 | 7,982 | 车型 make/model | 标签引导(真值) | research |
| GTSRB | 1,497 | 4,491 | 交通标志(德/欧) | 标签引导(真值) | research |
| SUN397 | 778 | 2,330 | 自然景观 | 标签引导(真值) | research |
| TextVQA | 1,500 | 1,500 | 文字/广告 OCR | 现成 QA | CC BY |
| **合计** | **8,030** | **~34,578** | 车外 7/8 类 + POI(RAG) | | |

## 2. 切分
- **训练集**：`train_v3_sharegpt.json` — 5,538（每源上限 1200，均衡，修 v2 混合退化）
- **冻结测试集**：`heldout_frozen.json` — 200（每类 40，seed=123，**永不进训练**，所有版本同集评测）

## 3. 格式
**训练（sharegpt，LLaMA-Factory）**
```json
{"conversations":[
  {"from":"human","value":"<image>\nIdentify the vehicle in this image."},
  {"from":"gpt","value":"A silver AM General Hummer SUV 2000 ..."},
  {"from":"human","value":"What is the make and model of this vehicle?"},
  {"from":"gpt","value":"AM General Hummer SUV 2000"}],
 "images":["/.../stanford_cars/images/00000.jpg"]}
```
**可读（raw jsonl）**：每行 `{image, caption, qa:[{question,answer,capability,use_case}]}`。

## 4. 标注方法：标签引导 recaption（核心创新）
对有真值标签的源（车型/标志/景观），把**真值灌进 prompt 当 ground truth** → VLM 生成**保证正确**的问答，根治"Unknown"幻觉。实证：车型准确率 base 7.5% → 微调后 90%+。

## 5. 8 类覆盖
| 用例 | 数据来源 | 状态 |
|---|---|---|
| 行人/VRU | nuScenes | ✅ |
| 场景描述 | nuScenes | ✅ |
| 建筑 | nuScenes | ✅ |
| 车型 | Stanford Cars | ✅ 90% |
| 交通标志 | GTSRB | ✅ 100% |
| 自然景观 | SUN397 | ✅ 77.5% |
| 文字 OCR | TextVQA | ✅ 82.5% |
| POI | RAG MVP(图→实体→Wikipedia) | ✅ demo |

## 6. 已知局限（诚实）
- nuScenes 部分答案为 "Unknown"（未用标签引导时 VLM 看不清）；可用更强 prompt 重标提升；
- GTSRB 为德/欧标志（缺中/印）、为小裁剪图（已放大）；
- TextVQA 非驾驶域（通用读字）；
- 数据集多为非商业许可，**交付/商用边界需与 Visteon 确认**。

## 7. 文件位置（服务器 `/data/haiyuez/visteon_cabin_vlm/data/`）
`*_sharegpt.json`（训练）、`*_raw.jsonl`（可读）、`train_v3_sharegpt.json`、`heldout_frozen.json`、`eval_*.json`（评测结果）。本地副本在 `SJTU-450/deliverables/results/`。
