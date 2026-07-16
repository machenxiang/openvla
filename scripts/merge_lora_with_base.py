#!/usr/bin/env python3
"""
合并LoRA adapter和base模型 - 参考finetune.py的实现

Usage:
    python scripts/merge_lora_with_base.py --step 12000
"""

import os
import argparse
import torch
from transformers import AutoModelForVision2Seq
from peft import PeftModel
import gc

# 本地模型路径
LOCAL_VLA_PATH = "/home/vcar/.cache/huggingface/hub/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f"

def merge_single_step(adapter_path, output_path):
    """合并单个step的adapter，参考finetune.py"""

    # 清理内存
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Loading base model from: {LOCAL_VLA_PATH}")
    base_vla = AutoModelForVision2Seq.from_pretrained(
        LOCAL_VLA_PATH,
        torch_dtype=torch.bfloat16,  # 和finetune.py一致
        low_cpu_mem_usage=True,      # 减少内存峰值
        trust_remote_code=True,
    )

    print(f"Loading adapter from: {adapter_path}")
    merged_vla = PeftModel.from_pretrained(base_vla, adapter_path)

    print("Merging LoRA weights...")
    merged_vla = merged_vla.merge_and_unload()

    # 保存
    print(f"Saving to: {output_path}")
    merged_vla.save_pretrained(output_path)

    print(f"Done! Saved to: {output_path}")

    # 清理
    del merged_vla
    del base_vla
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, required=True, help="Step number to merge (e.g., 12000)")
    parser.add_argument("--adapter_dir", type=str,
        default="/home/vcar/openvla/adapter-tmp/openvla-7b+libero_spatial_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug+20260615-102128",
        help="Adapter directory")
    parser.add_argument("--output_dir", type=str,
        default="/home/vcar/openvla/merged_checkpoints",
        help="Output directory")

    args = parser.parse_args()

    adapter_path = os.path.join(args.adapter_dir, f"step_{args.step}")
    output_path = os.path.join(args.output_dir, f"step_{args.step}")

    if not os.path.exists(adapter_path):
        print(f"Error: Adapter path not found: {adapter_path}")
        return

    os.makedirs(output_path, exist_ok=True)

    print(f"{'='*60}")
    print(f"Merging step {args.step}")
    print(f"Adapter: {adapter_path}")
    print(f"Output: {output_path}")
    print(f"{'='*60}")

    try:
        merge_single_step(adapter_path, output_path)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
