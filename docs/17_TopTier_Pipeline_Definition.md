# 顶级车外 Caption/QA Pipeline 定义(工业级)

> 袁傲杰 · 2026-06 · 综合 6 篇前沿 VLA(MindVLA/Xiaomi/DeepRoute/ZYT世界模型/ByteDance/Qwen-Omni)蒸馏

## 0. 核心决策:统一 schema + 模块化 prompt(不是全分、也不是全统一)

**结论:一套统一输出 schema + 统一 CoT 框架 + 按 use case 的模块化 prompt 适配器。**
- ❌ 每源完全独立 pipeline → 口径不一、无法合并评测。
- ❌ 一个 prompt 套所有类 → 丢失类别专属信息(车型要 make/model,灯要状态+形状)。
- ✅ **统一**:输出 JSON schema、CoT 结构(感知→风险→决策)、能力/用例标签、质检与交叉核查流程。
- ✅ **模块化**:每个 use case 一个 prompt 适配器(注入该类专属字段 + 该类 QA 题型)。

## 1. 架构:视觉感知 + 前沿LLM推理 + 多模型交叉核查(三层)

借鉴 VLA "感知模块 + 推理模块" 分工(Vultr 前沿模型是纯文本,正好做推理层):

```
① 感知接地层(视觉):Qwen2-VL + 数据集 GT 标注
   → 抽取结构化事实:物体类别/数量/2D&3D框/距离/相对位置/速度/OCR文字/属性
② 推理生成层(前沿文本LLM:GLM-5.1 / DeepSeek-V4 / Kimi-K2.6 / MiniMax-M2.7 / Qwen3.5-397B)
   → 基于①的事实,生成 VLA-CoT caption + 多元 QA(纯文本推理,质量远超 7B)
③ 交叉核查层(多模型投票):换 2-3 个不同前沿模型核查忠实度 + Fleiss κ,分歧→人工
```
**优势**:感知用视觉模型(GT 接地、零幻觉);推理/语言质量用前沿大模型;核查用"真·多个不同 AI"(真正满足"多问几个 AI 梳理")。

## 2. 统一输出 schema(所有源一致)

```json
{
  "image": "<path>", "source": "<dataset>", "use_case": "<1-8>", "camera": "front|front-left|...",
  "ego": {"speed_kmh": null, "lane": null, "navigation": null},
  "caption": {
    "scene": "<交通要素+物体+自车相对空间+天气/时段,只述GT可见>",
    "risk": "<由列出物体推出的风险:横穿/切入/红灯/盲区,带相对位置>",
    "decision": "<proceed|slow|stop|yield|lane-change + 理由,按相机角色>",
    "prediction": "<关键动态目标未来1-3s意图,可空>"
  },
  "qa": [{"q":"","a":"","capability":"Recognition|Reasoning|WorldKnowledge|Prediction|Planning","use_case":"","type":"counting|attribute|spatial|risk|action|ocr|intent|reject"}],
  "grounding": {"objects":[{"cls":"","pos":"","dist_m":0,"bbox":[0,0,0,0]}], "ocr":[{"text":"","bbox":[0,0,0,0]}]},
  "provenance": {"perception_model":"Qwen2-VL","reasoning_model":"GLM-5.1","verified_by":["DeepSeek-V4","Kimi-K2.6"],"faithful":true}
}
```
要点(综合 VLA 共识):**自车状态 + 时序意图预测 + 自车相对空间 + 细粒度属性 + 文字坐标 + provenance(可追溯)**。

## 3. Prompt 全套(重点)

### 3.1 CoT caption — 生成(推理层,前沿LLM,输入=感知层GT事实)
```
SYSTEM: You are the reasoning module of an autonomous-driving cockpit. You ONLY assert facts present
in the GROUND-TRUTH below or explicitly marked visible. Never invent objects/signs/lights/weather.

USER:
Camera: {role} (focus: {focus}).
GROUND-TRUTH (authoritative perception):
- objects: {cls, rel-position, distance(m), bbox, velocity?}
- ocr text: {text @ bbox}
- ego: speed/lane/navi if available
- scene tag: {nuScenes/source note}
Write a VLA chain-of-thought driving caption with EXACTLY these labeled parts:
  Scene:   road type + traffic elements (lights/signs/lane lines if in GT) + each GT object with its
           ego-relative position & distance + weather/time only if clearly visible.
  Risk:    the driving risk implied ONLY by the listed objects (crossing VRU, cut-in, red light,
           blind-spot), naming the responsible object + its position.
  Decision: ego action (proceed/slow/stop/yield/lane-change) for THIS camera role, justified by Risk.
  Prediction: short-horizon (1-3s) intent of the most safety-critical dynamic object (omit if none).
Be concise, factual, no speculation beyond GT. Output JSON {"scene","risk","decision","prediction"}.
```

### 3.2 多元 QA — 生成(覆盖 8 用例题型)
```
Given the GROUND-TRUTH facts above, generate {N} DIVERSE driving QA, each grounded ONLY in GT.
Cover ≥4 distinct types from: counting, attribute, spatial(relative position/distance),
risk, action(decision), ocr(read text), intent(predict behavior), reject(if asked about
something NOT in GT → answer "not visible / cannot determine").
Each QA tag: capability(Recognition/Reasoning/WorldKnowledge/Prediction/Planning) + type + use_case.
Answers must be verifiable from GT; for any unanswerable question, use the reject pattern.
Strict JSON: {"qa":[{"q","a","capability","type","use_case"}]}
```
> 新增题型(综合 VLA):**spatial(自车相对空间)、intent(意图预测)、reject(拒答不可答)**——比原来"识别/推理/决策"更全。

### 3.3 交叉核查 — 验证(换不同前沿模型)
```
SYSTEM: You are a strict driving-caption fact-checker.
USER: GROUND-TRUTH: {facts}. Caption: {draft}.
Delete or correct EVERY claim not supported by GT or not clearly visible (invented lights/signs/
weather/lane-counts, wrong counts/positions). Keep all GT-supported content and the Scene/Risk/
Decision/Prediction structure. Output: {"corrected": {...}, "removed_claims":[...], "faithful": true/false}
```
> 用 2 个**不同**前沿模型各核查一次 → 多数票 + 记录 removed_claims(可审计)。

### 3.4 按 use case 的模块化适配(注入到 3.1/3.2 的 GT 段)
| 用例 | 注入字段 / 专属题型 |
|---|---|
| ②车型 | make/model + **28类车型taxonomy** + 真实路况(停放/行驶/对向)+ 颜色/车身 |
| ④标志 | 多地区(中/欧/印)+ 类型(限速/禁令/警告)+ 文字(OCR) |
| ④灯 | 状态(红/黄/绿)+ 形状(圆/箭头/**闪烁**)+ 决策(红→停) |
| ⑤文字 | 文字内容 + **空间坐标 bbox** + 类型(路牌/广告) |
| ⑥VRU | **骑手 vs 车 分离** + 行人细分 + 婴儿车 + **横穿意图预测** |
| ①建筑/地标 | 建筑立面 vs 可行驶路面(分割接地)+ 地标名(若世界级) |
| ③景观 | **区域分桶**(植被/天空/地面)+ 整体类型 |
| ⑦场景 | 完整 CoT + 天气/时段/车流密度 |
| ⑧POI | 店招文字 → 实体 → 类别(餐饮/加油…)|

## 4. 数据集丰富计划(补不足)
1. **车辆真实路况标注**:不只 make/model,加"停放/行驶/对向/同向 + 所在车道 + 与自车距离"(用 nuScenes GT 接地)。
2. **补 8 类缺口**:ADE20K(#1立面/#3区域分桶/#6骑手分离)、TextOCR(#5坐标)、应急车、灯量。
3. **高价值场景挖掘**(DeepRoute 洞见):优先夜/雨/路口/近危/横穿等罕见场景,而非通用直路。
4. **时序样本**:用 nuScenes 连续帧加"未来 1-3s 意图"标注。

## 5. 实施步骤
1. 感知层:扩 `gt_extract`(加速度/相对车道/属性/OCR坐标),输出统一 GT JSON。
2. 推理层:`vultr_caption.py` 调前沿 LLM 按 3.1/3.2 生成(多 key 并发)。
3. 核查层:`vultr_xcheck.py` 多模型投票 + Fleiss κ。
4. 按 schema 落库 → 质检 → 合并 → 训练 → 权威评测。

## 6. 一句话定义
**统一 schema + 模块化 prompt;视觉模型接地感知、前沿 LLM 做 CoT 推理与多元 QA、多个不同前沿模型交叉核查;输出带自车状态/时序意图/相对空间/文字坐标/可追溯 provenance —— 一套工业级、可合并、可审计的车外 VLA caption/QA 生产线。**
