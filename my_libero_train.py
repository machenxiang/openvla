"""
Debug script for training OpenVLA on LIBERO
Usage: python my_libero_train.py
Then use debugger to set breakpoints and step through.
"""

import os
import sys
from pathlib import Path

# Set environment variables BEFORE importing anything else
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTHONPATH"] = "/home/vcar/LIBERO:" + os.environ.get("PYTHONPATH", "")
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Debug mode: Starting training...")
print(f"Python: {sys.executable}")
print(f"Working directory: {os.getcwd()}")

if __name__ == "__main__":
    # 动态导入 finetune 模块（因为目录名有连字符，不能直接 import）
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "finetune_module",
        "/home/vcar/openvla/vla-scripts/finetune.py"
    )
    finetune_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(finetune_module)

    FinetuneConfig = finetune_module.FinetuneConfig
    finetune_decorated = finetune_module.finetune

    cfg = FinetuneConfig(
        vla_path="openvla/openvla-7b",
        data_root_dir=Path("/home/vcar/LIBERO/modified_libero_rlds"),
        dataset_name="libero_spatial_no_noops",
        run_root_dir=Path("./runs"),
        adapter_tmp_dir=Path("./adapter-tmp"),
        batch_size=2,
        max_steps=30000,
        save_steps=3000,
        learning_rate=5e-4,
        grad_accumulation_steps=16,
        image_aug=True,
        shuffle_buffer_size=100_00,
        save_latest_checkpoint_only=False,
        use_lora=True,
        lora_rank=32,
        lora_dropout=0.0,
        use_quantization=False,
        wandb_project="openvla-libero",
        wandb_entity="YOUR_ENTITY",
        use_wandb=False,
        run_id_note=None,
    )

    # 调用被装饰的内部函数，绕过 draccus 命令行解析
    finetune_inner = finetune_decorated.__wrapped__
    finetune_inner(cfg)
