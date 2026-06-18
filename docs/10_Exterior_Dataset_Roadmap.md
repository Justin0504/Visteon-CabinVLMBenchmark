# 车外数据集构建 Roadmap（工业级 · 一步到位）

> 袁傲杰 · Category 2 / Exterior General Perception · 目标 3,000 图，8 类全覆盖
> 主线：严格对齐 Category-2 八类规范；方法：**caption 优先（CoT + GT 接地 + 检测框坐标 + 摄像头映射）**，QA 为辅。

---

## 0. 方法准则（Kevin 对齐 + Senna 范式）

1. **caption 优先于 QA**：注入车路场景认知，LoRA 微调偏置内部权重；QA finetune 为后续可选。
2. **GT 接地**：用 nuScenes（及其他数据集）已有标注——**检测分类 + 检测框坐标 + 距离 + 速度**——写进 caption，杜绝幻觉。
3. **VLA CoT**：场景描述（交通要素）→ 风险 → 驾驶判断，"为开车服务"。
4. **摄像头映射**：前视 stop-go/路牌/行人；侧视变道盲区；后视后方来车。
5. **优先级**：交通要素 + 整体描述首要；自然景观要；地标放过（除非占比大/世界级）；车型/广告/POI 酌情。
6. **Senna 范式对齐**：6 相机、class/3D 框/距离/相对位置/速度/未来轨迹；交通灯无 GT → VLM 伪标注。

---

## 1. 八类 × 数据源 × 缺口 总表

| # | 类别 | 现有数据源 | 状态 | 补法（数据源）|
|---|---|---|---|---|
| 1 | Building & Landmark | nuScenes（caption 描述）| ⚠️ 描述级 | 地标按优先级放过；建筑保留在场景 caption |
| 2 | Vehicle Make & Model | Stanford Cars 196 类 + nuScenes 粗类(car/truck/bus/trailer/construction) | ⚠️ 缺应急车 | **加 nuScenes 粗类接地**；应急车待补（BDD/SODA）|
| 3 | Natural Landscape | SUN397 778（场景级）| ⚠️ 场景级 | 保留；CoT 提植被/天空/地形 |
| 4 | Traffic Sign + **Light** | GTSRB 43 类（标志）| 🔴 **缺交通灯** | **Bosch STL**（灯状态+形状）+ **BDD100K**（灯色）|
| 5 | Text & Advertisement | TextVQA 1500 | ⚠️ 无坐标 | 保留读字；坐标后续（TextOCR 可选）|
| 6 | Pedestrian & VRU | nuScenes（行人/骑手/车分别标注）| ⚠️ 未显式分离 | **加 VRU 分离接地** + **JAAD**（横穿意图）|
| 7 | Exterior Scene Desc | nuScenes + CoT | ✅ | 加天气/时段（BDD 属性 / A*3D 夜雨）|
| 8 | POI Retrieval | POI RAG MVP | ⚠️ MVP | 保留 MVP；不接 HD map（超范围）|

---

## 2. 实施阶段（依次完成）

### Phase A — nuScenes 主力强化（已就位，零下载，立即做）
- **A1**：CoT caption 升级 = 加**检测框 2D 坐标**（Kevin 06-11 20:52）+ 速度/相对位置（Senna）。
- **A2**：**VRU 分离接地**（#6）——骑手 vs 自行车/摩托分开、行人细分（成人/儿童/工人/警察/personal_mobility）、婴儿车(pushable_pullable)。
- **A3**：**车型粗类接地**（#2）——truck/bus.rigid/bus.bendy/trailer/construction 计数与位置写入 caption。
- **A4**：交通灯 **VLM 伪标注**（#4，Senna 同法，nuScenes 无 GT）。
- 产出：升级版 `exterior_cot_v2_sharegpt.json`（替换/合入现 2424）。

### Phase B — 补三个真缺口（下载 + 标注）
- **B1 Bosch STL**：交通灯状态(红黄绿)+形状(圆/箭头) → 标签引导 caption（#4 灯）。
- **B2 JAAD**：行人横穿/意图 → VRU 行为 caption（#6）。
- **B3 BDD100K**：灯色+标志+天气/时段属性 → 多类接地 caption（#4/#7，并补应急车/中国域以外的多样性）。

### Phase C — 中国域补充（可选，提升泛化）
- **C1 SODA10M / D2-City**：中国道路车/人/标志 → 域多样性。

### Phase D — 评测闭环（每阶段后做）
- 基线 case analysis（base 未微调：>70% 强 / <60% 弱，已部分做）。
- 弱类补 caption → LoRA 微调 → 冻结集复评（每类客观准确率）。
- Qwen3-VL-4B 作 baseline 对比（已做）。

---

## 3. 交付口径
- 主交付 = **benchmark 数据集（图文对，sharegpt）** + 微调模型 + 评测；3,000 图覆盖 8 类。
- 许可：nuScenes/BDD/Bosch/JAAD 等均非商业，交付边界与 Visteon 确认。

---

## 4. 当前进度锚点（2026-06-16）
- 已就位：8,030 图 / 5 源 + nuScenes CoT 2,424（100% 三段、6 相机均衡）。
- 进行中：**Phase A**（检测框坐标 + VRU/车型粗类接地）。
- 待办：Phase B 三数据集下载 + 标注。
