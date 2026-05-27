# OpenVLA Fine-tune 调用栈与数据流文档

## 1. 入口点

| 文件 | 行号 | 说明 |
|------|------|------|
| `vla-scripts/finetune.py` | 391-392 | `if __name__ == "__main__": finetune()` 启动入口 |
| `vla-scripts/finetune.py` | 116-117 | `@draccus.wrap()` 装饰的 `finetune(cfg: FinetuneConfig)` 函数 |

## 2. 完整调用栈（线性流程）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 入口：my_libero_train.py                                                     │
│   importlib 动态加载 finetune.py                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ finetune.py:117 @draccus.wrap()装饰的 finetune(cfg)                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. 注册 OpenVLA 模型到 HF Auto Classes (行 154-157)                           │
│    AutoConfig.register("openvla", OpenVLAConfig)                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. 加载 Processor (行 160)                                                   │
│    processor = AutoProcessor.from_pretrained(cfg.vla_path)                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. 加载 VLA Model (行 161-167)                                                │
│    vla = AutoModelForVision2Seq.from_pretrained(cfg.vla_path)                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. LoRA 包装 (行 176-185)                                                     │
│    lora_config = LoraConfig(...)                                             │
│    vla = get_peft_model(vla, lora_config)                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. DDP 包装 (行 188-189) — 仅在多 GPU 时启用                                   │
│    vla = DDP(vla, device_ids=[device_id])                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 6. 创建 ActionTokenizer (行 198)                                             │
│    action_tokenizer = ActionTokenizer(processor.tokenizer)                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 7. 创建 RLDSDataset (行 215-231)                                             │
│    batch_transform = RLDSBatchTransform(...)                                │
│    vla_dataset = RLDSDataset(cfg.data_root_dir, cfg.dataset_name, ...)      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 8. 创建 DataLoader (行 241-247)                                               │
│    dataloader = DataLoader(vla_dataset, batch_size=cfg.batch_size, ...)     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 9. 训练循环 (行 263-389)                                                      │
│    for batch_idx, batch in enumerate(dataloader):                          │
│        output = vla(input_ids, attention_mask, pixel_values, labels)        │
│        loss = output.loss                                                    │
│        loss.backward()                                                       │
│        optimizer.step()                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 10. 保存 Checkpoint (行 340-384)                                             │
│     vla.save_pretrained(run_dir)                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 3. 关键函数详解

### 3.1 模型加载 (finetune.py:159-173)

```python
# 行 160: Processor 加载
processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)

# 行 161-167: Model 加载
vla = AutoModelForVision2Seq.from_pretrained(
    cfg.vla_path,
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

# 行 173: 模型移到 GPU
vla = vla.to(device_id)
```

### 3.2 LoRA 配置 (finetune.py:176-185)

```python
lora_config = LoraConfig(
    r=cfg.lora_rank,                    # 默认 32
    lora_alpha=min(cfg.lora_rank, 16),
    lora_dropout=cfg.lora_dropout,
    target_modules="all-linear",       # 所有 Linear 层
    init_lora_weights="gaussian",
)
vla = get_peft_model(vla, lora_config) # PEFT 包装
```

### 3.3 数据集加载 (datasets.py:70-143)

```python
# 行 92-100: 获取数据集配置
per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
    self.data_root_dir,
    mixture_spec,
    load_camera_views=("primary",),
    load_depth=False,
    load_proprio=False,
    load_language=True,
    action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
)

# 行 101-119: 构建 RLDS 配置
rlds_config = dict(
    traj_transform_kwargs=dict(window_size=1, future_action_window_size=0, ...),
    frame_transform_kwargs=dict(resize_size=resize_resolution, ...),
    dataset_kwargs_list=per_dataset_kwargs,
    shuffle_buffer_size=shuffle_buffer_size,
    sample_weights=weights,
    ...
)

# 行 140: 创建数据集
self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)
```

### 3.4 Batch Transform (datasets.py:38-67)

```python
def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
    # 1. 提取图像和语言指令
    img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
    lang = rlds_batch["task"]["language_instruction"].decode().lower()

    # 2. 构建 Prompt
    prompt_builder = self.prompt_builder_fn("openvla")
    conversation = [
        {"from": "human", "value": f"What action should the robot take to {lang}?"},
        {"from": "gpt", "value": self.action_tokenizer(action)},
    ]
    for turn in conversation:
        prompt_builder.add_turn(turn["from"], turn["value"])

    # 3. Tokenize
    input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids

    # 4. 图像预处理
    pixel_values = self.image_transform(img)

    # 5. Labels: 只保留 action token 位置的 loss
    labels[: -(len(action) + 1)] = IGNORE_INDEX

    return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
```

### 3.5 VLA Forward (modeling_prismatic.py:361-415)

```python
# 行 366: Vision Backbone 处理图像
patch_features = self.vision_backbone(pixel_values)

# 行 369: Projector 投影
projected_patch_embeddings = self.projector(patch_features)

# 行 380-384: 拼接 multimodal embeddings
multimodal_embeddings = torch.cat([
    input_embeddings[:, :1, :],           # BOS token
    projected_patch_embeddings,           # 图像特征
    input_embeddings[:, 1:, :],           # 文本 token
], dim=1)

# 行 404-415: Llama Forward
language_model_output = self.language_model(
    input_ids=None,
    attention_mask=multimodal_attention_mask,
    inputs_embeds=multimodal_embeddings,
    labels=multimodal_labels,  # 用于计算 loss
)
```

### 3.6 Action Tokenizer (action_tokenizer.py)

```python
# 行 38-47: 将连续动作离散化
def __call__(self, action: np.ndarray) -> str:
    action = np.clip(action, a_min=self.min_action, a_max=self.max_action)
    discretized_action = np.digitize(action, self.bins)
    return self.tokenizer.decode(list(self.tokenizer.vocab_size - discretized_action))

# 行 49-68: 将 token ID 转回连续动作
def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
    discretized_actions = self.tokenizer.vocab_size - action_token_ids
    discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
    return self.bin_centers[discretized_actions]
```

## 4. 数据流框图

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           RLDS TFRecord Dataset                              │
│                    (/data_root_dir/dataset_name/1.0.0/)                     │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        make_interleaved_dataset()                            │
│                        (rlds/dataset.py:make_interleaved_dataset)            │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  traj_transform: 窗口化、跳过无标签、目标重标记                           │    │
│  │  frame_transform: resize、并行处理、图像增强                           │    │
│  │  standardize_fn: 数据集特定转换 (libero_dataset_transform等)            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          RLDSBatchTransform.__call__()                        │
│                              (datasets.py:38-67)                               │
│                                                                               │
│   输入: {"observation": {"image_primary": ...}, "action": [...],            │
│         "task": {"language_instruction": "..."}}                            │
│                                                                               │
│   处理步骤:                                                                  │
│   1. Image.fromarray() → PIL Image                                          │
│   2. image_transform(img) → pixel_values (224x224)                          │
│   3. PromptBuilder 构建对话                                                   │
│   4. action_tokenizer(action) → action tokens string                         │
│   5. base_tokenizer(text) → input_ids                                        │
│   6. labels[:-action_len] = IGNORE_INDEX                                      │
│                                                                               │
│   输出: {"pixel_values": Tensor, "input_ids": Tensor, "labels": Tensor}     │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              DataLoader                                       │
│                         (torch.utils.data.DataLoader)                         │
│                                                                               │
│   batch_size: 16 (默认), num_workers: 0                                      │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          VLA Forward Pass                                     │
│                       (finetune.py:269-275)                                   │
│                                                                               │
│   output = vla(                                                              │
│       input_ids=batch["input_ids"].to(device),                               │
│       attention_mask=batch["attention_mask"].to(device),                      │
│       pixel_values=batch["pixel_values"].bfloat16().to(device),              │
│       labels=batch["labels"],                                                │
│   )                                                                          │
│   loss = output.loss                                                         │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
        ┌─────────────────────────────┴─────────────────────────────┐
        │                                                       │
        ▼                                                       ▼
┌───────────────────────┐                       ┌───────────────────────────────────┐
│ Vision Backbone       │                       │ LlamaForCausalLM                  │
│ (TIMM/DINOv2+SigLIP)  │                       │ (transformers.LlamaForCausalLM)    │
│                       │                       │                                   │
│ 输入: pixel_values    │                       │ 输入: inputs_embeds (multimodal)   │
│ 输出: patch_features  │                       │ 输出: logits, loss                 │
└───────────────────────┘                       └───────────────────────────────────┘
        │                                                       │
        ▼                                                       │
┌───────────────────────┐                                       │
│ Projector             │                                       │
│ (Linear Layer)        │                                       │
│                       │                                       │
│ 输入: patch_features  │                                       │
│ 输出: projected_     │                                       │
│      patch_embeddings │                                       │
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│ Multimodal Concatenation
│                       │
│ [BOS] + img_emb +     │
│ text_emb              │
└───────────────────────┘
```

## 5. 训练循环详解 (finetune.py:259-388)

```python
# 初始化
vla.train()
optimizer.zero_grad()

for batch_idx, batch in enumerate(dataloader):
    # 1. Forward
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = vla(
            input_ids=batch["input_ids"].to(device_id),
            attention_mask=batch["attention_mask"].to(device_id),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
            labels=batch["labels"],
        )
        loss = output.loss

    # 2. Backward
    normalized_loss = loss / grad_accumulation_steps
    normalized_loss.backward()

    # 3. 计算 Metrics (仅 action token 位置)
    # 获取 vision patch 数量
    vision_model = vla.module if hasattr(vla, 'module') else vla.base_model
    num_patches = vision_model.vision_backbone.featurizer.patch_embed.num_patches

    # 提取 action logits (跳过 vision patch + BOS token)
    action_logits = output.logits[:, num_patches:-1]
    action_preds = action_logits.argmax(dim=2)
    action_gt = batch["labels"][:, 1:].to(action_preds.device)

    # mask: 只保留 action token 位置 (token_id > action_token_begin_idx)
    mask = action_gt > action_tokenizer.action_token_begin_idx

    # Action Accuracy
    correct_preds = (action_preds == action_gt) & mask
    action_accuracy = correct_preds.sum().float() / mask.sum().float()

    # L1 Loss (将离散 action token 转回连续动作)
    continuous_actions_pred = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
    )
    continuous_actions_gt = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
    )
    action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

    # 存储最近 metrics (用于 gradient accumulation 平滑)
    recent_losses.append(loss.item())
    recent_action_accuracies.append(action_accuracy.item())
    recent_l1_losses.append(action_l1_loss.item())

    # 4. 日志记录 (每10步)
    if gradient_step_idx % 10 == 0:
        tb_writer.add_scalar("train/train_loss", smoothened_loss)
        tb_writer.add_scalar("train/action_accuracy", smoothened_action_accuracy)
        tb_writer.add_scalar("train/l1_loss", smoothened_l1_loss)

    # 5. Optimizer Step
    if (batch_idx + 1) % grad_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
        progress.update()

    # 6. 保存 Checkpoint (每 save_steps 步)
    if gradient_step_idx % save_steps == 0:
        # 保存 processor 和 adapter weights
        # 合并 LoRA 权重到 base model
        merged_vla.save_pretrained(run_dir)
```

### 5.1 Loss 计算详解

**总损失** = `output.loss`（来自 transformers 库）

VLA forward 时会将 `labels` 传给 LlamaForCausalLM，transformers 库自动计算 Cross Entropy Loss：

```python
# modeling_prismatic.py:404-415
language_model_output = self.language_model(
    input_ids=None,
    attention_mask=multimodal_attention_mask,
    inputs_embeds=multimodal_embeddings,
    labels=multimodal_labels,   # ← labels 传入位置
)
loss = language_model_output.loss  # Cross Entropy Loss
```

**为什么只计算 action token 的 loss？**

在 `RLDSBatchTransform.__call__` 中，labels 被处理成：
- `action token` 位置：保留真实 token id
- `其他位置`：设为 `IGNORE_INDEX`（-100），在计算 loss 时被忽略

```python
# datasets.py:181
labels[: -(len(action) + 1)] = IGNORE_INDEX  # 非 action token 位置 mask 掉
```

### 5.2 Action Accuracy 计算

```
action_accuracy = 预测正确的 action token 数 / 有效的 action token 总数
```

- `action_preds = action_logits.argmax(dim=2)` — 取每个位置概率最高的 token
- `mask = action_gt > action_token_begin_idx` — 只考虑 action token 范围（最后 256 个 token）

### 5.3 L1 Loss 计算

```
action_l1_loss = mean(|continuous_actions_pred - continuous_actions_gt|)
```

将离散的 action token ID 映射回连续动作值（bin centers），然后计算 L1 距离。

## 6. 文件结构索引

| 模块 | 文件路径 | 关键类/函数 |
|------|----------|------------|
| 入口 | `vla-scripts/finetune.py` | `finetune()`, `FinetuneConfig` |
| 数据集 | `prismatic/vla/datasets/datasets.py` | `RLDSDataset`, `RLDSBatchTransform` |
| 数据加载 | `prismatic/vla/datasets/rlds/dataset.py` | `make_interleaved_dataset()`, `make_dataset_from_rlds()` |
| 数据配置 | `prismatic/vla/datasets/rlds/oxe/configs.py` | `OXE_DATASET_CONFIGS` |
| 数据转换 | `prismatic/vla/datasets/rlds/oxe/transforms.py` | `OXE_STANDARDIZATION_TRANSFORMS` |
| 模型 | `prismatic/extern/hf/modeling_prismatic.py` | `OpenVLAForActionPrediction`, `PrismaticForConditionalGeneration` |
| 动作分词 | `prismatic/vla/action_tokenizer.py` | `ActionTokenizer` |
| LoRA | `vla-scripts/finetune.py:176-185` | `LoraConfig`, `get_peft_model()` |

## 8. 完整预训练流程（OpenVLA-7B 如何训练的）

OpenVLA 完整训练分 **两个阶段**，涉及两个代码库：

### 阶段1: VLM 预训练（Prismatic 框架）

**代码位置**: `scripts/pretrain.py`（使用 FSDP 分布式训练）

```
torchrun --standalone --nnodes 1 --nproc-per-node 64 scripts/pretrain.py \
    --model.type prismatic_dinosiglip_7b \
    --dataset.type llava_v15 \
    --stage finetune
```

**完整调用栈**:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ 入口：scripts/pretrain.py:238                                                │
│    pretrain()                                                                  │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 1. 加载 Vision Backbone (行 146-149)                                          │
│    get_vision_backbone_and_transform(vision_backbone_id)                      │
│    → TIMM/DINOv2 + SigLIP                                                    │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 2. 加载 LLM Backbone (行 152-155)                                              │
│    get_llm_backbone_and_tokenizer(llm_backbone_id)                           │
│    → Llama-2-7B (from HuggingFace)                                           │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 3. 构建 VLM (行 159-165)                                                       │
│    get_vlm(model_id, arch_specifier, vision_backbone, llm_backbone)         │
│    → PrismaticVLM (vision + llm + projector)                                │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 4. Freeze Backbones (行 169)                                                 │
│    vlm.freeze_backbones(stage)                                               │
│    → stage="align": 只训练 projector                                          │
│    → stage="finetune": 训练 projector + LLM                                   │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 5. 加载 Checkpoint (行 173)                                                     │
│    vlm.load_from_checkpoint(stage, run_dir)                                 │
│    → 加载 align 阶段权重作为 finetune 起点                                     │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 6. 获取数据集 (行 177-185)                                                      │
│    get_dataset_and_collator(stage, dataset, image_transform, tokenizer)       │
│    → LLaVA 图像-文本数据集                                                     │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 7. 创建 FSDP Train Strategy (行 189-207)                                      │
│    get_train_strategy(fsdp-shard-grad-op, vlm, ...)                          │
│    → FSDPStrategy (全分片数据并行)                                             │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 8. 训练循环 (行 225)                                                           │
│    train_strategy.run_training(train_dataset, collator, metrics)            │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 阶段2: VLA 微调（Action Head 训练）

**代码位置**: `vla-scripts/train.py`（或 `finetune.py` 的前身）

```
torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/train.py \
    --data_root_dir datasets/open-x-embodiment \
    --dataset_name oxe_magic_soup_plus_minus \
    --run_root_dir runs
```

**训练内容**:
- 在 970K 机器人轨迹数据上训练
- 添加 action prediction head（一个线性层）
- 训练 projector + action head
- 冻结 vision backbone 和 LLM

### 硬件需求对比

| 阶段 | GPU 数量 | 显存/GPU | Batch Size |
|------|---------|---------|-----------|
| VLM Pretrain (align) | 32-64 | ~80GB | 2048 global |
| VLM Pretrain (finetune) | 32-64 | ~80GB | 2048 global |
| VLA Fine-tune (action head) | 1-8 | ~48GB | 16 |
| LoRA Fine-tune | 1 | ~24GB | 1-16 |

### 产出模型

```
openvla/openvla-7b (HuggingFace)
├── config.json
├── model.safetensors (7B params)
├── processor/
│   ├── tokenizer.json
│   └── image_processor.json
└── README.md
```

### 相关文件索引

| 文件 | 作用 |
|------|------|
| `scripts/pretrain.py` | VLM 预训练入口（FSDP） |
| `prismatic/conf/models.py` | 模型配置（PRISM_DINOSIGLIP_CONTROLLED_7B） |
| `prismatic/training/strategies/fsdp.py` | FSDP 分布式训练策略 |
| `prismatic/models/vlms.py` | PrismaticVLM 模型定义 |
| `prismatic/models/backbones/` | Vision/LLM 加载 |
| `vla-scripts/train.py` | VLA action head 训练 |
| `vla-scripts/finetune.py` | LoRA 微调（你用的） |

---

## 9. 快速参考：四类训练对比

| 训练类型 | 入口脚本 | 用途 | 可用 GPU |
|---------|---------|------|---------|
| VLM align | `scripts/pretrain.py --stage align` | 训练 projector | 32+ |
| VLM finetune | `scripts/pretrain.py --stage finetune` | 训练 projector + LLM | 32+ |
| VLA train | `vla-scripts/train.py` | 训练 action head | 8+ |
| LoRA finetune | `vla-scripts/finetune.py` | LoRA 微调 | 1+ |