# sharegpt 字段定义（车外数据线 · 对齐用）

> 袁傲杰 · 给负责 prompt 范式的同学对齐。**格式已定，照此产出即可直接合并训练。**

## 1. 一条样本的结构
```json
{
  "conversations": [
    {"from": "human", "value": "<image>\n<指令或问题>"},
    {"from": "gpt",   "value": "<caption 或答案>"},
    {"from": "human", "value": "<问题2>"},
    {"from": "gpt",   "value": "<答案2>"}
  ],
  "images": ["<图片绝对路径>"]
}
```

## 2. 字段规则（必须遵守）
| 字段 | 规则 |
|---|---|
| `conversations` | 列表，**human/gpt 交替**，human 先、gpt 后 |
| `from` | 只能是 `"human"` 或 `"gpt"` |
| `value` | **第一条 human 必须含 `<image>\n`**（图像占位符），后续轮不要再放 |
| `images` | 列表，**1 张图 = 1 个绝对路径**（与 `<image>` 占位符一一对应）|
| 轮次 | caption 作第 1 轮（human:指令 → gpt:caption），其后每个 QA 各占 2 条 |

## 3. CoT caption 推荐填法（按 Kevin 要求）
- 第 1 轮 human：`<image>\n描述这个车外驾驶场景并给出驾驶判断。`
- 第 1 轮 gpt（CoT，三段）：
  > 场景描述（交通要素：红绿灯/路牌/车道线/车辆/行人/天气）→ 风险分析 → 驾驶判断。
- 其后接多元化 QA（每个 2 条）。

## 4. 文件级约定
- 一个 `.json` = 一个列表 `[ {样本}, {样本}, ... ]`（不是 jsonl）；
- 我这边按数据源分文件（如 `exterior_cot_sharegpt.json`），最后统一 merge；
- 图片放服务器 `/data/.../data/<source>/`，路径用绝对路径。

## 5. 可选元字段（不进训练，便于质检/分析）
若想带标签，单独存一份 raw jsonl（每行）：
```json
{"image":"...","caption":"...","camera":"front|side|rear","qa":[{"question":"","answer":"","capability":"Recognition|Reasoning|WorldKnowledge","use_case":"..."}]}
```
sharegpt 训练文件里**不放**这些元字段（LLaMA-Factory 只读 conversations + images）。

## 6. 对齐结论
- 你：prompt 范式（CoT / 多元化 QA / 摄像头映射）；
- 我：nuScenes 实跑 + 合并 + 训练评测；
- 你产出的样本**按上面结构**给我，我直接合进训练集。
