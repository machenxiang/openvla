"""
分析训练数据中夹爪第一次闭合的step
统计训练数据中夹爪关闭的时机

Usage:
    python scripts/analyze_training_gripper.py
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf


def load_tfrecord_dataset(data_root: Path, dataset_name: str):
    """加载 TFRecord 数据集"""
    dataset_path = data_root / dataset_name / "1.0.0"
    tfrecord_files = sorted(dataset_path.glob("libero_spatial-train.tfrecord-*"))

    if not tfrecord_files:
        raise FileNotFoundError(f"No tfrecord files found in {dataset_path}")

    dataset = tf.data.TFRecordDataset(tfrecord_files)

    # 解析 example - 扁平化结构
    def parse_example(example):
        return tf.io.parse_single_example(
            example,
            {
                "steps/is_first": tf.io.VarLenFeature(tf.int64),
                "steps/action": tf.io.VarLenFeature(tf.float32),
                "steps/discount": tf.io.VarLenFeature(tf.float32),
                "steps/is_last": tf.io.VarLenFeature(tf.int64),
                "steps/language_instruction": tf.io.VarLenFeature(tf.string),
                "steps/observation/wrist_image": tf.io.VarLenFeature(tf.string),
                "steps/reward": tf.io.VarLenFeature(tf.float32),
                "steps/is_terminal": tf.io.VarLenFeature(tf.int64),
                "steps/observation/state": tf.io.VarLenFeature(tf.float32),
                "steps/observation/joint_state": tf.io.VarLenFeature(tf.float32),
                "steps/observation/image": tf.io.VarLenFeature(tf.string),
            }
        )

    parsed_dataset = dataset.map(parse_example)
    return parsed_dataset


def analyze_training_gripper(
    dataset_name: str = "libero_spatial_no_noops",
    data_root: str = "/home/vcar/LIBERO/modified_libero_rlds",
):
    """分析训练数据中夹爪关闭的时机"""
    data_root = Path(data_root)
    dataset = load_tfrecord_dataset(data_root, dataset_name)

    print(f"Dataset: {dataset_name}")
    print(f"Data root: {data_root}")

    gripper_close_steps = []  # 每个trajectory第一次关闭夹爪的step
    trajectory_lengths = []  # 每个trajectory的长度
    total_trajectories = 0

    print("Processing trajectories...")

    for i, sample in enumerate(dataset):
        # 获取动作数据
        actions = tf.sparse.to_dense(sample["steps/action"]).numpy()
        instructions = tf.sparse.to_dense(sample["steps/language_instruction"])

        num_elements = actions.shape[0]
        action_dim = 7
        num_steps = num_elements // action_dim

        # Reshape actions: [num_steps, 7]
        actions = actions[:num_steps * action_dim].reshape(num_steps, action_dim)

        # Gripper action is the last dimension (index 6)
        # In RLDS format: -1 = close, 1 = open
        gripper_actions = actions[:, 6]

        # 打印gripper的唯一值，统计是否只有-1和1
        unique_gripper = set(gripper_actions.tolist())
        print(f"Trajectory {i}: unique gripper values: {unique_gripper}")
        print(f"  gripper first 10: {gripper_actions[:10]}")
        print(f"  gripper last 10: {gripper_actions[-10:]}")

        total_trajectories += 1

        if total_trajectories % 500 == 0:
            print(f"Processed {total_trajectories} trajectories...")

    # Statistics
    print("\n" + "=" * 80)
    print("统计结果")
    print("=" * 80)

    if gripper_close_steps:
        gripper_close_steps = np.array(gripper_close_steps)
        trajectory_lengths = np.array(trajectory_lengths)

        print(f"\n总轨迹数: {total_trajectories}")
        print(f"有夹爪关闭动作的轨迹数: {len(gripper_close_steps)}")

        print(f"\n第一次关闭夹爪的step统计:")
        print(f"  平均值: {np.mean(gripper_close_steps):.1f}")
        print(f"  中位数: {np.median(gripper_close_steps):.1f}")
        print(f"  标准差: {np.std(gripper_close_steps):.1f}")
        print(f"  最小值: {np.min(gripper_close_steps)}")
        print(f"  最大值: {np.max(gripper_close_steps)}")

        print(f"\n分位数:")
        for p in [10, 25, 50, 75, 90]:
            val = np.percentile(gripper_close_steps, p)
            print(f"  {p}%: Step {val:.0f}")

        # Distribution
        print(f"\n分布:")
        bins = [0, 20, 40, 60, 80, 100, 150, 200, 300, 1000]
        for i in range(len(bins) - 1):
            if bins[i + 1] == 1000:
                label = f"{bins[i]}+"
            else:
                label = f"{bins[i]}-{bins[i + 1] - 1}"
            count = np.sum(
                (gripper_close_steps >= bins[i]) &
                (gripper_close_steps < bins[i + 1])
            )
            pct = count / len(gripper_close_steps) * 100
            bar = "#" * int(pct / 2)
            print(f"  Step {label:>8}: {count:5d} ({pct:5.1f}%) {bar}")

        print(f"\n轨迹长度统计:")
        print(f"  平均长度: {np.mean(trajectory_lengths):.1f}")
        print(f"  中位数长度: {np.median(trajectory_lengths):.1f}")
    else:
        print("没有找到夹爪关闭动作的数据！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze gripper closing timing in training data"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/vcar/LIBERO/modified_libero_rlds",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="libero_spatial_no_noops",
        choices=[
            "libero_spatial_no_noops",
            "libero_object_no_noops",
            "libero_goal_no_noops",
            "libero_10_no_noops",
        ],
    )
    args = parser.parse_args()

    analyze_training_gripper(dataset_name=args.dataset, data_root=args.data_root)
