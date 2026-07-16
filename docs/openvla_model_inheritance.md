# OpenVLA 模型继承体系与推理调用栈

## 1. 模型加载路径

```
run_libero_eval.py:104    model = get_model(cfg)
  └─► robot_utils.py:43   model = get_vla(cfg)
        └─► openvla_utils.py:43  vla = AutoModelForVision2Seq.from_pretrained(...)
```

---

## 2. 类继承体系

`vla` 对象的类型为 `OpenVLAForActionPrediction`，其完整继承链如下：

```
nn.Module
 └─ PreTrainedModel                              [transformers]
     └─ PrismaticPreTrainedModel                 [prismatic/extern/hf/modeling_prismatic.py:176]
         └─ PrismaticForConditionalGeneration    [prismatic/extern/hf/modeling_prismatic.py:213]
             └─ PrismaticVLM                     [prismatic/models/vlms/prismatic.py:38]
                 └─ VLM                          [prismatic/models/vlms/base_vlm.py:31]
                     └─ nn.Module + GenerationMixin + ABC
```

### 2.1 各层详解

| 类名 | 定义文件 | 作用 |
|---|---|---|
| `nn.Module` | PyTorch 内置 | 神经网络基类，提供参数管理、GPU 迁移等基础能力 |
| `PreTrainedModel` | `transformers` | HuggingFace 模型基类，提供权重加载、save_pretrained 等 |
| `PrismaticPreTrainedModel` | `modeling_prismatic.py:176` | Prismatic 系列模型的基类 |
| `PrismaticForConditionalGeneration` | `modeling_prismatic.py:213` | 条件生成模型封装，持有 `vision_backbone`、`llm_backbone`、`projector` |
| `PrismaticVLM` | `prismatic.py:38` | 视觉-语言模型核心封装，重载 `forward()` 实现多模态融合 |
| `VLM` | `base_vlm.py:31` | 抽象基类，继承 `nn.Module` + `GenerationMixin`，定义 VLM 统一接口 |

### 2.2 关键类定义

**VLM（基类）** — `prismatic/models/vlms/base_vlm.py:31`
```python
class VLM(nn.Module, GenerationMixin, ABC):
```
- `GenerationMixin` 来自 `transformers`，提供 `.generate()` 方法

**PrismaticVLM** — `prismatic/models/vlms/prismatic.py:38`
```python
class PrismaticVLM(VLM):
```
- 持有 `vision_backbone`、`llm_backbone`、`projector`
- 实现 `forward()` — 核心多模态融合逻辑

**OpenVLAForActionPrediction** — `prismatic/extern/hf/modeling_prismatic.py:492`
```python
class OpenVLAForActionPrediction(PrismaticForConditionalGeneration):
    config_class: PretrainedConfig = OpenVLAConfig

    def __init__(self, config: OpenVLAConfig) -> None:
        super().__init__(config)
        self.norm_stats = config.norm_stats
        self.bins = np.linspace(-1, 1, config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.vocab_size = self.config.text_config.vocab_size - self.config.pad_to_multiple_of

    def predict_action(self, ...) -> np.ndarray: ...
```

---

## 3. 模型推理调用栈

`run_libero_eval.py` 中每个 timestep 的推理路径：

```
run_libero_eval.py:222    get_action()
  └─► robot_utils.py:66    get_vla_action()
        └─► openvla_utils.py:169  vla.predict_action(**inputs, ...)
              └─► modeling_prismatic.py:506  OpenVLAForActionPrediction.predict_action()
                    └─► modeling_prismatic.py:518  self.generate(input_ids, max_new_tokens=...)
                          └─► GenerationMixin.generate()  [through MRO]
                                └─► prismatic.py:312  PrismaticVLM.forward()
                                      ├─► vision_backbone()         [图像特征提取]
                                      ├─► projector()               [投影到 LLM 嵌入空间]
                                      └─► llm_backbone()            [LLM 自回归生成]
```

### 3.1 调用细节

| 步骤 | 文件:行号 | 函数 | 说明 |
|---|---|---|---|
| 1 | `robot_utils.py:66` | `get_vla_action(model, processor, ...)` | 分发到 VLA 专用 action 生成 |
| 2 | `openvla_utils.py:166` | `processor(prompt, image)` | 将文本 prompt 和 PIL Image 处理为 `input_ids` + `pixel_values` |
| 3 | `openvla_utils.py:169` | `vla.predict_action(**inputs, ...)` | 调用模型预测动作 |
| 4 | `modeling_prismatic.py:518` | `self.generate(...)` | 继承自 `GenerationMixin`，触发自回归生成 |
| 5 | `prismatic.py:372` | `self.vision_backbone(...)` | ViT 图像编码 |
| 6 | `prismatic.py:375` | `self.projector(...)` | 图像特征投影到 LLM 嵌入维度 |
| 7 | `prismatic.py:391` | `self.llm_backbone(...)` | LLM 自回归前向传播 |

### 3.2 predict_action 参数处理

`openvla_utils.py:169` 调用：
```python
action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
```

`inputs` 来自 `processor(prompt, image)` 返回的字典，包含：
- `input_ids`: tokenized prompt
- `pixel_values`: 预处理后的图像 tensor
- `attention_mask`: 注意力掩码

---

## 4. OpenVLA v0.1 与 v1.0 的 predict_action 差异

项目中存在两套 `predict_action` 实现：

### 4.1 OpenVLA v0.1（`prismatic/models/vlas/openvla.py:36`）

通过 `OpenVLA` 类（**非 HF 模型**）实现：
```
OpenVLA  →  PrismaticVLM  →  VLM(nn.Module + GenerationMixin + ABC)
```
- `predict_action` 构建 prompt 并调用 `super().generate()`
- 最终走到 `PrismaticVLM.forward()`

### 4.2 OpenVLA v1.0 / HuggingFace（`prismatic/extern/hf/modeling_prismatic.py:506`）

通过 `OpenVLAForActionPrediction` 实现：
```
OpenVLAForActionPrediction  →  PrismaticForConditionalGeneration  →  ...  →  PrismaticVLM  →  VLM
```
- `predict_action` 直接调用 `self.generate()`
- 同样最终走到 `PrismaticVLM.forward()`

**两者最终都汇聚到 `PrismaticVLM.forward()`**，差异在于：
1. v0.1 在 Python 侧构建 prompt；v1.0 通过 HuggingFace AutoClass 加载，processor 统一预处理
2. v0.1 使用 `ActionTokenizer` 解码；v1.0 使用 `bin_centers` 数组解码

---

## 5. 相关文件索引

| 文件路径 | 关键内容 |
|---|---|
| `experiments/robot/robot_utils.py:40` | `get_model()` |
| `experiments/robot/openvla_utils.py:31` | `get_vla()` — 使用 `AutoModelForVision2Seq.from_pretrained()` |
| `experiments/robot/openvla_utils.py:127` | `get_vla_action()` — 预处理图像 + 构建 prompt |
| `experiments/robot/openvla_utils.py:169` | `vla.predict_action()` — 触发推理 |
| `prismatic/extern/hf/modeling_prismatic.py:492` | `OpenVLAForActionPrediction` 类定义 |
| `prismatic/extern/hf/modeling_prismatic.py:506` | `predict_action()` HF 版实现 |
| `prismatic/models/vlms/prismatic.py:38` | `PrismaticVLM` 类定义 |
| `prismatic/models/vlms/prismatic.py:312` | `PrismaticVLM.forward()` — 多模态融合核心 |
| `prismatic/models/vlms/prismatic.py:594` | `PrismaticVLM.generate()` — 对话生成用（非 action） |
| `prismatic/models/vlms/base_vlm.py:31` | `VLM` 抽象基类，继承 `GenerationMixin` |
| `prismatic/extern/hf/configuration_prismatic.py:72` | `PrismaticConfig` |
| `prismatic/extern/hf/configuration_prismatic.py:129` | `OpenVLAConfig` |
| `prismatic/extern/hf/processing_prismatic.py:175` | `PrismaticProcessor` |
