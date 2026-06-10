"""
分析训练数据集中的指令
查找类似 "bowl on the ramekin" 的指令

Usage:
    python scripts/analyze_instructions.py --keyword ramekin
"""

import tensorflow as tf
from pathlib import Path
import argparse


def load_tfrecord_dataset(data_root: Path, dataset_name: str):
    """加载 TFRecord 数据集"""
    dataset_path = data_root / dataset_name / "1.0.0"
    tfrecord_files = sorted(dataset_path.glob("*-train.tfrecord-*"))

    if not tfrecord_files:
        raise FileNotFoundError(f"No tfrecord files found in {dataset_path}")

    dataset = tf.data.TFRecordDataset(tfrecord_files)

    def parse_example(example):
        return tf.io.parse_single_example(
            example,
            {
                "steps/language_instruction": tf.io.VarLenFeature(tf.string),
            }
        )

    parsed_dataset = dataset.map(parse_example)
    return parsed_dataset


def analyze_instructions(
    dataset_name: str = "libero_spatial_no_noops",
    data_root: str = "/home/vcar/LIBERO/modified_libero_rlds",
    keyword: str = "ramekin",
):
    """分析训练数据集中的指令"""
    data_root = Path(data_root)
    dataset = load_tfrecord_dataset(data_root, dataset_name)

    print(f"Dataset: {dataset_name}")
    print(f"Data root: {data_root}")
    print(f"Keyword: '{keyword}'")
    print()

    all_instructions = set()
    matching_instructions = set()
    total_samples = 0

    print("Scanning dataset...")

    for sample in dataset:
        instructions = tf.sparse.to_dense(sample["steps/language_instruction"]).numpy()
        for instr in instructions:
            if len(instr) > 0:
                text = instr.decode('utf-8')
                all_instructions.add(text)
                if keyword.lower() in text.lower():
                    matching_instructions.add(text)

        total_samples += 1
        if total_samples % 1000 == 0:
            print(f"Processed {total_samples} samples...")

    print(f"\nTotal unique instructions: {len(all_instructions)}")
    print(f"Instructions containing '{keyword}': {len(matching_instructions)}")

    # 打印所有包含关键词的指令
    if matching_instructions:
        print(f"\n=== Instructions containing '{keyword}' ===")
        for i, instr in enumerate(sorted(matching_instructions), 1):
            print(f"{i}. {instr}")

    # 打印所有唯一指令（如果数量不多）
    if len(all_instructions) <= 50:
        print(f"\n=== All unique instructions ({len(all_instructions)}) ===")
        for i, instr in enumerate(sorted(all_instructions), 1):
            print(f"{i}. {instr}")
    else:
        print(f"\n=== Sample instructions (first 20) ===")
        for i, instr in enumerate(sorted(all_instructions)[:20], 1):
            print(f"{i}. {instr}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze instructions in training data")
    parser.add_argument("--data_root", type=str,
                        default="/home/vcar/LIBERO/modified_libero_rlds")
    parser.add_argument("--dataset", type=str,
                        default="libero_spatial_no_noops")
    parser.add_argument("--keyword", type=str,
                        default="ramekin")
    args = parser.parse_args()

    analyze_instructions(
        dataset_name=args.dataset,
        data_root=args.data_root,
        keyword=args.keyword
    )
