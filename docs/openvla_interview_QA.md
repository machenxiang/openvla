# OpenVLA 面试常见问题

> 基于 OpenVLA 推理流程、模型架构、训练机制的常见面试题整理

---

## 一、模型架构

### Q1：OpenVLA 的整体架构是什么样的？

OpenVLA 是一个基于 **PrismaticVLM** 的视觉-语言-动作模型，由三个核心组件构成：

```
图像输入
  └─► Vision Backbone (TIMM ViT)  → 图像 patch features [B, 256, vision_dim]
        │
        ▼
  ┌─► Projector (MLP)            → 投影到 LLM 嵌入空间 [B, 256, llm_dim]
  │
文本输入 → Tokenizer → input_ids
  │
  └─► Multimodal Embedding 拼接： [CLS] + 图像 patches + 文本 tokens
        │
        ▼
  ┌─► LLM (Llama/Mistral)         → 自回归生成 action tokens
```

**继承链：**
```
OpenVLAForActionPrediction
  └─ PrismaticForConditionalGeneration
      └─ PrismaticPreTrainedModel
          └─ PreTrainedModel  [transformers]
              └─ GenerationMixin  [transformers]
```

### Q2：图像和文本是怎么融合的？

**拼接位置**：在 embedding 空间拼接，而非 attention 阶段。

```python
multimodal_embeddings = torch.cat([
    input_embeddings[:, :1, :],      # [CLS] token
    projected_patch_embeddings,         # 图像 patches [B, 256, hidden]
    input_embeddings[:, 1:, :],       # 文本 tokens
], dim=1)
```

**为什么这样设计**：
- [CLS] 作为序列起始标记
- 图像夹在 [CLS] 和文本之间，让 LLM 在做 self-attention 时能同时看到图像和文本
- 直接用 `inputs_embeds` 传入，跳过 input_ids 的 embedding 查找

### Q3：Vision Backbone 用的是什么？图像尺寸是多少？

- **ViT**：来自 TIMM 库，通常是 ViT-L/DINOv2
- **图像尺寸**：224×224
- **Fused Backbone**：有时用双 ViT（`use_fused_vision_backbone=True`）

### Q4：Projector 的作用是什么？

将 Vision Backbone 输出的 `vision_dim`（如 1024）投影到 LLM 的 `hidden_dim`（如 4096），使得图像特征和文本特征可以在同一个 embedding 空间拼接。

---

## 二、推理流程

### Q5：一次推理的完整流程是什么？

```
obs["full_image"]  →  PIL Image → processor(prompt, image)
  └─► BatchFeature({input_ids, attention_mask, pixel_values})

vla.predict_action(**inputs, unnorm_key=unnorm_key)
  │
  ├─► self.generate(input_ids, max_new_tokens=7)  ← 自回归循环 7 次
  │     └─► self.forward() × 7  →  7 个 action tokens
  │
  └─► 反归一化 → 7 维动作向量 [position(3) + rotation(3) + gripper(1)]
```

### Q6：generate 和 forward 的区别是什么？

| | forward | generate |
|---|---|---|
| **调用次数** | 1 次 | 循环 N 次 |
| **输出** | 所有位置的 logits | 整个生成的序列 |
| **用途** | 训练 / 一次前向传播 | 自回归生成（推理） |

`generate()` 内部循环调用 `forward()`：
```
while 未达到 max_new_tokens:
    outputs = forward()      # 获取当前 step 的 logits
    next_token = sample()    # 采样
    input_ids += next_token  # 更新序列
```

### Q7：为什么生成 7 个动作需要循环 7 次？

每个 action token 只预测 **1 个维度的动作**（7 个自由度 = 7 个 token）。`generate(max_new_tokens=7)` 会循环 7 次，每次生成 1 个 token。

### Q8：predict_action 中 token 怎么变成实际动作的？

```python
# 1. 从 vocab 中找最高概率的 token
predicted_action_token_ids = generated_ids[0, -7:]  # 最后 7 个 token
discretized_actions = vocab_size - predicted_action_token_ids  # 离散化
discretized_actions = clip(discretized_actions - 1, 0, n_bins-1)

# 2. 查 bin_centers 变成连续值
normalized_actions = bin_centers[discretized_actions]  # [-1, 1] 范围

# 3. 反归一化到实际动作范围
actions = 0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low
```

### Q9：推理时 labels=None 会影响结果吗？

**不会**。`labels` 只在训练时用于计算 loss，推理时不传 labels，模型只输出 logits，完全不影响。

### Q10：center_crop 的作用是什么？

**模拟训练时的数据增强**：训练时用了随机裁剪（crop_scale=0.9），推理时做 center crop 保持数据分布一致。

```python
crop_scale = 0.9
# 裁剪面积 = 0.9 × 原面积
# 裁剪尺寸 = √0.9 × 原尺寸 ≈ 0.949 × 原尺寸
# 然后 resize 回 224×224
```

---

## 三、Action Tokenization

### Q11：连续动作怎么变成 token 的？

将连续动作空间离散化到 `n_action_bins` 个 bins：

```python
bins = np.linspace(-1, 1, n_action_bins)  # 如 256 个 bin
bin_centers = (bins[:-1] + bins[1:]) / 2.0  # 每个 bin 的中心值

# 训练时：连续动作 → 找到最近的 bin → token id
action_bin_idx = np.digitize(raw_action, bins)  # 返回 bin 索引（0~255）
```

### Q12：为什么要用 bin_centers 而不是直接映射？

因为 bins 之间的边界值没有物理意义，使用 bin 中心值作为回归目标更合理：
- 避免边界剧烈变化
- 模型学习的是"在这个 bin 范围内应该取什么代表值"

---

## 四、图像旋转 180 度

### Q13：img[::-1, ::-1] 是什么操作？

```python
img = img[::-1, ::-1]  # H 和 W 都翻转，即旋转 180 度
```

**原因**：训练时的数据预处理和仿真环境中的图像采集角度不一致，旋转 180 度用来对齐两者。推理时也必须做同样的旋转才能保证一致。

---

## 五、Gripper Action 处理

### Q14：gripper action 为什么需要 invert 和 normalize？

| 处理 | 原因 |
|---|---|
| **normalize**：`[0,1] → [-1,+1]` | 环境期望 gripper 动作为 `[-1, +1]` |
| **invert**：乘以 `-1` | RLDS dataloader 定义 `0=close, 1=open`，与环境 `+1=close, -1=open` 相反 |

---

## 六、模型加载与 AutoClass

### Q15：AutoModelForVision2Seq 怎么知道用 OpenVLAForActionPrediction？

```python
# 注册
AutoModelForVision2Seq.register("openvla", OpenVLAForActionPrediction)

# 加载时
config = AutoConfig.from_pretrained(checkpoint)  # 读取 config.json，model_type="openvla"
# AutoModelForVision2Seq 根据 model_type 找到注册的 OpenVLAForActionPrediction
vla = AutoModelForVision2Seq.from_pretrained(checkpoint)
```

### Q16：processor 加载了什么？

```python
processor = AutoProcessor.from_pretrained(checkpoint)
# 实际类型：PrismaticProcessor
# 包含：image_processor (TIMM transforms) + tokenizer (LlamaTokenizerFast)
```

---

## 七、Python 语法（__call__）

### Q17：processor(prompt, image) 为什么能调用？

**因为定义了 `__call__` 方法**：

```python
class PrismaticProcessor(ProcessorMixin):
    def __call__(self, text, images, ...):
        # 逻辑...
        return BatchFeature(...)

processor = PrismaticProcessor(...)
processor(prompt, image)  # 等价于 processor.__call__(prompt, image)
```

Python 发现对象不是函数时，会自动调用其 `__call__` 方法。

---

## 八、多态与 MRO

### Q18：generate 方法在 GenerationMixin 中，self 是子类实例时怎么调用到子类的 forward？

```python
class GenerationMixin:
    def generate(self, ...):
        outputs = self.forward(...)  # self 的类型决定调用哪个 forward

obj = OpenVLAForActionPrediction(...)
obj.generate(...)  # self 是 OpenVLAForActionPrediction 实例
                   # self.forward() 会沿着 MRO 找到子类的实现
```

Python **MRO（方法解析顺序）**：从实例的实际类型开始，沿着继承链向上找，**第一个匹配的方法就是被调用的**。

```python
OpenVLAForActionPrediction → ... → GenerationMixin
# self.forward() 会找到 PrismaticForConditionalGeneration.forward()
```

---

## 九、训练相关

### Q19：训练和推理的 forward 有什么区别？

| | 训练 | 推理 |
|---|---|---|
| **labels** | 有值，计算 loss | None |
| **use_cache** | False | 可用 KV cache 加速 |
| **图像处理** | 可能有多次随机裁剪 | center crop 一次 |
| **输出** | logits + loss | logits |

### Q20：LoRA 微调时推理怎么处理？

LoRA 权重通过 `model.merge_and_unload()` 合并到原始权重中，推理时和全量权重推理完全一样，无需特殊处理。

---

## 十、数据预处理

### Q21：图像预处理流程是什么？

```
原始图像 (H, W, 3)
  │
  ├─ 可选 center_crop: 裁剪到 √0.9 面积 → resize 224×224
  │
  ├─ TIMM transform:
  │     Resize(224) → CenterCrop(224) → ToTensor → Normalize(mean=0.5, std=0.5)
  │
  └─ 输出: [3, 224, 224] tensor
```

### Q22：PrismaticImageProcessor 的 transform 来自哪里？

来自 **TIMM** 库：
```python
transform = timm.data.create_transform(
    input_size=(3, 224, 224),
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5],
    interpolation="bicubic",
)
```

---

## 十一、高频追问

### Q23：为什么用 bfloat16 而不是 float16？

- `bfloat16`：动态范围大（8 bit 指数），适合训练大模型，和模型权重一致
- `float16`：精度更高但动态范围小，容易 overflow

### Q24：flash_attention_2 是什么？带来了什么？

Flash Attention 是一种 **IO-aware 的 attention 实现**，通过减少 GPU HBM 和 SRAM 之间的数据搬运，将 attention 计算从 O(N²) 显存降到 O(N)。

**优势**：显存减少 2~4 倍，速度提升 1.5~2 倍。

### Q25：为什么不直接在 image token 上做 action regression，而要用 tokenization？

1. **统一格式**：动作变成 token 后，和文本 token 一起由 LLM 自回归生成，架构统一
2. **利用 LLM 的能力**：LLM 预训练时见过大量 token 预测任务，迁移过来
3. **离散化优势**：连续动作空间大，直接回归难以优化；离散化后变成分类问题，更稳定
