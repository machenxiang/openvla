# OpenVLA 推理完整调用栈

> 入口：`python experiments/robot/libero/run_libero_eval.py`
> 入口函数：`eval_libero()` — `run_libero_eval.py:91`

---

## 一、顶层入口

### 1.1 模型加载

```
run_libero_eval.py:104      model = get_model(cfg)
  └─► robot_utils.py:43     model = get_vla(cfg)
        └─► openvla_utils.py:43   vla = AutoModelForVision2Seq.from_pretrained(...)
```

#### AutoClass 注册映射（openvla_utils.py:38-41）

| HF Auto类 | 注册标识 | 实际类型 |
|---|---|---|
| `AutoConfig` | `"openvla"` | `OpenVLAConfig` |
| `AutoImageProcessor` | `"openvla"` | `PrismaticImageProcessor` |
| `AutoProcessor` | `"openvla"` | `PrismaticProcessor` |
| `AutoModelForVision2Seq` | `"openvla"` | `OpenVLAForActionPrediction` |

`cfg.pretrained_checkpoint` 目录下的 `config.json` 包含 `"model_type": "openvla"`，HuggingFace Auto 系统据此找到上述映射。

#### model 对象的真实类型

```
vla 的类型: OpenVLAForActionPrediction
  └─ PrismaticForConditionalGeneration
      └─ PrismaticPreTrainedModel
          └─ PreTrainedModel  [transformers]
              └─ GenerationMixin  [transformers]  ← 提供 .generate()
              └─ ...
```

---

### 1.2 Processor 加载

```
run_libero_eval.py:117      processor = get_processor(cfg)
  └─► openvla_utils.py:77   processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)
                                ↑
                                │
                          AutoProcessor 自动路由到 PrismaticProcessor
```

---

## 二、推理主循环（每个 timestep 执行一次）

### 2.1 完整调用栈总览

```
run_libero_eval.py:207      img = get_libero_image(obs, resize_size)
run_libero_eval.py:222      action = get_action(cfg, model, observation, task_description, processor=processor)
  └─► robot_utils.py:66    get_vla_action(model, processor, cfg.pretrained_checkpoint, obs, task_label, cfg.unnorm_key, ...)
        └─► openvla_utils.py:166   inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
        |     └─► [详见 §3 — processor 调用栈]
        └─► openvla_utils.py:169   action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
              └─► [详见 §4 — predict_action → generate → forward]
```

---

## 三、Processor 调用栈

```
openvla_utils.py:166   processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
  └─► processing_prismatic.py:187   PrismaticProcessor.__call__(text=prompt, images=image)
        │
        ├─► processing_prismatic.py:207   self.image_processor(images, ...)
        │     └─► processing_prismatic.py:147   PrismaticImageProcessor.preprocess(images, return_tensors="pt")
        │           └─► timm.data.create_transform(...)  # 创建 TIMM transform pipeline
        │                 │
        │                 ├─► Resize      (TIMM / torchvision)
        │                 ├─► CenterCrop  (TIMM / torchvision)
        │                 ├─► ToTensor    (TIMM / torchvision)
        │                 └─► Normalize   (TIMM / torchvision)
        │           返回: pixel_values tensor [1, 3, 224, 224]
        │
        └─► processing_prismatic.py:208   self.tokenizer(text, padding=..., truncation=..., return_tensors=...)
              └─► HuggingFace PreTrainedTokenizer.encode(text)
                    返回: {input_ids, attention_mask}

  返回: BatchFeature({input_ids, attention_mask, pixel_values})
```

### PrismaticProcessor.__call__ 源码（processing_prismatic.py:187-216）

```python
def __call__(self, text, images, ...):
    # 1. 图像 → pixel_values
    pixel_values = self.image_processor(images, return_tensors=return_tensors)["pixel_values"]
    #   └─► PrismaticImageProcessor.preprocess() → TIMM transform

    # 2. 文本 → input_ids + attention_mask
    text_inputs = self.tokenizer(text, ...)

    # 3. 合并返回
    return BatchFeature(data={**text_inputs, "pixel_values": pixel_values})
```

---

## 四、predict_action → generate → forward 完整调用栈

### 4.1 predict_action 入口

```
openvla_utils.py:169   vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
  └─► modeling_prismatic.py:506   OpenVLAForActionPrediction.predict_action(input_ids, pixel_values, ...)
```

**OpenVLAForActionPrediction.predict_action()** 源码（modeling_prismatic.py:506-531）：

```python
def predict_action(self, input_ids=None, unnorm_key=None, **kwargs):
    # [可选] 在 prompt 末尾追加 empty token 以匹配训练时的输入格式
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat([input_ids, ...], dim=1)

    # 核心：调用 generate() 自回归生成 action token
    generated_ids = self.generate(
        input_ids, max_new_tokens=self.get_action_dim(unnorm_key), **kwargs
    )   # ← 第 518 行

    # 提取 token 并反归一化为连续动作
    predicted_action_token_ids = generated_ids[0, -self.get_action_dim(unnorm_key):]
    discretized_actions = self.vocab_size - predicted_action_token_ids
    normalized_actions = self.bin_centers[discretized_actions]
    actions = unnormalize(normalized_actions, unnorm_key)

    return actions   # ← 返回 np.ndarray shape=(7,)
```

---

### 4.2 generate() 调用

```
modeling_prismatic.py:518   self.generate(input_ids, max_new_tokens=...)
  └─► GenerationMixin.generate()  [继承自 PreTrainedModel → transformers.generation.utils.GenerationMixin]
```

**关键继承链：**

```
OpenVLAForActionPrediction
  └─ PrismaticForConditionalGeneration
      └─ PrismaticPreTrainedModel
          └─ PreTrainedModel
              └─ GenerationMixin  ◄── .generate() 定义在此
```

**GenerationMixin.generate() 内部核心逻辑：**

源码文件：`/home/vcar/anaconda3/envs/openvla/lib/python3.10/site-packages/transformers/generation/utils.py`

```python
# transformers.generation.utils.GenerationMixin.generate()  # 第 1284 行
# 签名：def generate(self, inputs=None, generation_config=None, logits_processor=...,
#                  stopping_criteria=None, max_new_tokens=None, ...)
def generate(self, inputs, generation_config=None, **kwargs) -> torch.LongTensor:
    # 1. 从 generation_config 获取 max_new_tokens（或其他 max_length 配置）
    #    kwargs 中的 max_new_tokens 会被合并到 generation_config 中
    generation_config = self._resolve_generation_config(max_new_tokens, **kwargs)

    # 2. 自回归生成主循环  # ~第 3131 行
    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)  # ~第 3132 行

        # 调用 forward() 获取当前 step 的 logits  # ~第 3171 行
        outputs = self(**model_inputs, ...)
        #   └─► PrismaticForConditionalGeneration.forward()  ← 多模态前向传播

        next_token_logits = outputs.logits[:, -1, :]  # ~第 3182 行
        next_token_scores = log_softmax(next_token_logits, dim=-1)
        next_token_scores = logits_processor(input_ids, next_token_scores)  # ~第 3187 行

        # 采样并更新
        input_ids, cur_len = self._update(input_ids, next_token_scores, ...)
        #   └─► 内部执行: input_ids = torch.cat([input_ids, next_token_id], dim=1)
        #       cur_len += 1  # ~第 3179 / 3256 行

    return input_ids
```

> **注意**：`max_new_tokens` 不是 `generate()` 的直接形参，而是通过 `kwargs` 传入后由
> `generation_config` 持有，最终在 `stopping_criteria`（停止条件）中控制生成长度上限。

---

### 4.3 forward() 调用

```
GenerationMixin.generate() 内部调用
  └─► modeling_prismatic.py:291   PrismaticForConditionalGeneration.forward(input_ids, pixel_values, ...)
```

**forward() 内部执行流程：**

```
PrismaticForConditionalGeneration.forward()
  │
  ├─► modeling_prismatic.py:366   patch_features = self.vision_backbone(pixel_values)
  │     └─► timm ViT 模型（由 config.timm_model_ids 指定）
  │           返回: [batch, num_patches, vision_dim]
  │
  ├─► modeling_prismatic.py:369   projected_patch_embeddings = self.projector(patch_features)
  │     └─► PrismaticProjector (MLP 或 FusedMLP)
  │           返回: [batch, num_patches, llm_dim]
  │
  ├─► modeling_prismatic.py:380   input_embeddings = self.get_input_embeddings()(input_ids)
  │
  ├─► modeling_prismatic.py:383-385   multimodal_embeddings = torch.cat([...], dim=1)
  │     拼接顺序: [CLS embedding, 图像 patch embeddings, 剩余文本 embeddings]
  │
  └─► modeling_prismatic.py:404   language_model_output = self.language_model(inputs_embeds=multimodal_embeddings, ...)
        └─► AutoModelForCausalLM.from_config(config.text_config)
              即 LlamaForCausalLM 或类似 LLM
              返回: CausalLMOutputWithPast(logits, past_key_values, ...)
```

**forward() 源码核心片段（modeling_prismatic.py:362-416）：**

```python
# 第 362 行：多模态分支
elif (input_ids.shape[0] == pixel_values.shape[0]) or (inputs_embeds.shape[0] == pixel_values.shape[0]):
    # [图像分支] ViT 图像编码
    patch_features = self.vision_backbone(pixel_values)          # 第 366 行

    # [投影分支] 投影到 LLM 嵌入维度
    projected_patch_embeddings = self.projector(patch_features)  # 第 369 行

    # 扩展 attention mask 以覆盖新增的 patch tokens
    projected_patch_attention_mask = torch.full(...)             # 第 372-377 行

    # 获取文本 input embeddings
    input_embeddings = self.get_input_embeddings()(input_ids)    # 第 380 行

    # 拼接图像和文本 embedding: [CLS] + patch embeddings + 剩余文本
    multimodal_embeddings = torch.cat([                          # 第 383-385 行
        input_embeddings[:, :1, :],
        projected_patch_embeddings,
        input_embeddings[:, 1:, :],
    ], dim=1)

    # 拼接后的 attention mask
    multimodal_attention_mask = torch.cat([...])                 # 第 388-390 行

    # 送入 LLM（使用 inputs_embeds 绕过 input_ids）
    language_model_output = self.language_model(                 # 第 404 行
        input_ids=None,
        attention_mask=multimodal_attention_mask,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=multimodal_embeddings,     # ← 关键：直接传入 embedding
        labels=multimodal_labels,
        use_cache=False,
        ...
    )

    return PrismaticCausalLMOutputWithPast(logits=language_model_output.logits, ...)
```

---

## 五、所有涉及的文件汇总

| 文件路径 | 关键行号 | 内容 |
|---|---|---|
| `experiments/robot/libero/run_libero_eval.py` | 91 | `eval_libero()` 入口 |
| `experiments/robot/libero/run_libero_eval.py` | 104 | `get_model(cfg)` 调用 |
| `experiments/robot/libero/run_libero_eval.py` | 117 | `get_processor(cfg)` 调用 |
| `experiments/robot/libero/run_libero_eval.py` | 207 | `get_libero_image(obs, resize_size)` |
| `experiments/robot/libero/run_libero_eval.py` | 222 | `get_action()` 调用 |
| `experiments/robot/robot_utils.py` | 40 | `get_model()` |
| `experiments/robot/robot_utils.py` | 63 | `get_action()` |
| `experiments/robot/robot_utils.py` | 66 | `get_vla_action()` 调用 |
| `experiments/robot/openvla_utils.py` | 31 | `get_vla()` |
| `experiments/robot/openvla_utils.py` | 38-41 | AutoClass 注册 |
| `experiments/robot/openvla_utils.py` | 43 | `AutoModelForVision2Seq.from_pretrained()` |
| `experiments/robot/openvla_utils.py` | 75 | `get_processor()` |
| `experiments/robot/openvla_utils.py` | 127 | `get_vla_action()` |
| `experiments/robot/openvla_utils.py` | 158-163 | 构建 prompt 字符串 |
| `experiments/robot/openvla_utils.py` | 166 | `processor(prompt, image)` |
| `experiments/robot/openvla_utils.py` | 169 | `vla.predict_action()` 调用 |
| `prismatic/extern/hf/processing_prismatic.py` | 32 | `PrismaticImageProcessor` 类定义 |
| `prismatic/extern/hf/processing_prismatic.py` | 147 | `PrismaticImageProcessor.preprocess()` |
| `prismatic/extern/hf/processing_prismatic.py` | 175 | `PrismaticProcessor` 类定义 |
| `prismatic/extern/hf/processing_prismatic.py` | 187 | `PrismaticProcessor.__call__()` |
| `prismatic/extern/hf/processing_prismatic.py` | 207-210 | 图像+文本预处理 |
| `prismatic/extern/hf/modeling_prismatic.py` | 176 | `PrismaticPreTrainedModel` 类定义 |
| `prismatic/extern/hf/modeling_prismatic.py` | 213 | `PrismaticForConditionalGeneration` 类定义 |
| `prismatic/extern/hf/modeling_prismatic.py` | 236-250 | `__init__()` 中实例化 vision_backbone / projector / language_model |
| `prismatic/extern/hf/modeling_prismatic.py` | 291 | `PrismaticForConditionalGeneration.forward()` |
| `prismatic/extern/hf/modeling_prismatic.py` | 366 | `self.vision_backbone(pixel_values)` |
| `prismatic/extern/hf/modeling_prismatic.py` | 369 | `self.projector(patch_features)` |
| `prismatic/extern/hf/modeling_prismatic.py` | 404 | `self.language_model(inputs_embeds=...)` |
| `prismatic/extern/hf/modeling_prismatic.py` | 492 | `OpenVLAForActionPrediction` 类定义 |
| `prismatic/extern/hf/modeling_prismatic.py` | 506 | `predict_action()` |
| `prismatic/extern/hf/modeling_prismatic.py` | 518 | `self.generate()` 调用 |
| `prismatic/extern/hf/configuration_prismatic.py` | 72 | `PrismaticConfig` |
| `prismatic/extern/hf/configuration_prismatic.py` | 129 | `OpenVLAConfig` |

---

## 六、类继承体系全图

```
nn.Module
 │
PreTrainedModel  [transformers]
 │
PrismaticPreTrainedModel  [modeling_prismatic.py:176]
 │
PrismaticForConditionalGeneration  [modeling_prismatic.py:213]
 │   ├── self.vision_backbone   = PrismaticVisionBackbone     (timm ViT)
 │   ├── self.projector         = PrismaticProjector          (MLP)
 │   └── self.language_model    = AutoModelForCausalLM        (Llama/Mistral)
 │
     OpenVLAForActionPrediction  [modeling_prismatic.py:492]
         ├── self.norm_stats     = dataset statistics (for un-normalization)
         ├── self.bins           = linspace(-1, 1, n_action_bins)
         └── self.bin_centers   = (bins[:-1] + bins[1:]) / 2.0

─────────────────────────────────────────────────────────────────────────
GenerationMixin  [transformers]  ← 所有 .generate() 方法来自此 mixin
```

---

## 七、单步推理数据流

```
obs (dict)
  └─ obs["full_image"]: np.array [H, W, 3]
        │
        ▼ get_libero_image() — openvla_utils.py:207
  PIL.Image
        │
        ▼ processor(prompt, image) — openvla_utils.py:166
  BatchFeature({
      input_ids:       Tensor [1, seq_len]
      attention_mask:  Tensor [1, seq_len]
      pixel_values:    Tensor [1, 3, 224, 224]
  })
        │
        ▼ vla.predict_action() — openvla_utils.py:169
  np.ndarray (7,)  ← 归一化后的 7D 动作向量 (位置+旋转+夹爪)
        │
        ▼ normalize_gripper_action() / invert_gripper_action() — run_libero_eval.py:231-260
  env.step(action) — run_libero_eval.py:263
```
