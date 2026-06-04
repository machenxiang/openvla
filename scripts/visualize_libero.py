"""
visualize_libero.py

可视化 LIBERO 数据集中的训练样本。
显示图像、动作值、语言指令等信息。

Usage:
    python scripts/visualize_libero.py --dataset libero_spatial_no_noops --num_samples 5
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image


def load_tfrecord_dataset(data_root: Path, dataset_name: str, num_samples: int = 5):
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


def visualize_sample(sample, sample_idx=0):
    """可视化单个样本"""
    print(f"\n=== Sample {sample_idx} ===")

    # 获取基本数据
    actions = tf.sparse.to_dense(sample["steps/action"]).numpy()
    states = tf.sparse.to_dense(sample["steps/observation/state"]).numpy()
    images = tf.sparse.to_dense(sample["steps/observation/image"]).numpy()
    wrist_images = tf.sparse.to_dense(sample["steps/observation/wrist_image"]).numpy()
    instructions = tf.sparse.to_dense(sample["steps/language_instruction"])

    num_elements = actions.shape[0]
    action_dim = 7
    state_dim = 8

    # 扁平化数据，需要重新计算步数
    num_steps = num_elements // action_dim
    actions = actions[:num_steps * action_dim].reshape(num_steps, action_dim)
    states = states[:num_steps * state_dim].reshape(num_steps, state_dim)

    # images 和 wrist_images 每个step一张，不需要reshape
    num_images = images.shape[0]
    wrist_images = wrist_images[:num_steps]  # 截取到num_steps
    images = images[:num_steps]

    print(f"Number of steps: {num_steps}")
    print(f"Actions shape: {actions.shape}")
    print(f"States shape: {states.shape}")
    print(f"Number of images: {images.shape[0]}")

    # 解析 language instruction
    lang_instr = instructions[0].numpy().decode('utf-8') if len(instructions) > 0 else ""
    print(f"Language instruction: {lang_instr}")

    # 平均采样4个点
    num_plot_points = 4
    sample_indices = np.linspace(0, num_steps - 1, num_plot_points, dtype=int)
    print(f"Sampled indices: {sample_indices}")

    # 创建图像 + 动作图一起显示 (4行4列)
    fig = plt.figure(figsize=(16, 12))

    # 第一行：显示4张图像
    for i, idx in enumerate(sample_indices):
        ax_img = fig.add_subplot(4, 4, i + 1)
        img = tf.io.decode_image(images[idx]).numpy()
        ax_img.imshow(img)
        ax_img.set_title(f"Image Step {idx}")
        ax_img.axis('off')

        # 第二行：手腕图像
        ax_wrist = fig.add_subplot(4, 4, i + 5)
        wrist = tf.io.decode_image(wrist_images[idx]).numpy()
        ax_wrist.imshow(wrist)
        ax_wrist.set_title(f"Wrist Step {idx}")
        ax_wrist.axis('off')

    # 第三行和第四行：动作曲线
    action_names = ['base_x', 'base_y', 'base_z', 'base_rz', 'base_ry', 'base_rx', 'gripper']

    for i in range(7):
        ax = fig.add_subplot(4, 4, i + 9)
        ax.plot(sample_indices, actions[sample_indices, i], 'o-', markersize=8)
        ax.set_title(f"{action_names[i]}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.set_xticks(sample_indices)
        ax.set_xticklabels([str(idx) for idx in sample_indices])
        ax.grid(True)

    # 最后一个位置留空或显示额外信息
    ax_info = fig.add_subplot(4, 4, 16)
    ax_info.text(0.5, 0.5, f"Task: {lang_instr[:50]}...", ha='center', va='center', fontsize=10, wrap=True)
    ax_info.axis('off')

    plt.suptitle(f"Sample {sample_idx}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"sample_{sample_idx}_combined.png", dpi=100)
    print(f"Saved sample_{sample_idx}_combined.png")
    plt.close()

    return {
        "actions": actions,
        "states": states,
        "language_instruction": lang_instr,
    }


def main():
    parser = argparse.ArgumentParser(description="Visualize LIBERO dataset samples")
    parser.add_argument("--data_root", type=str, default="/home/vcar/LIBERO/modified_libero_rlds")
    parser.add_argument("--dataset", type=str, default="libero_spatial_no_noops",
                        choices=["libero_spatial_no_noops", "libero_object_no_noops",
                                 "libero_goal_no_noops", "libero_10_no_noops"])
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to visualize")
    parser.add_argument("--num_frames", type=int, default=5, help="Number of frames to show per sample")

    args = parser.parse_args()

    data_root = Path(args.data_root)
    dataset = load_tfrecord_dataset(data_root, args.dataset, args.num_samples)

    print(f"Loaded dataset: {args.dataset}")
    print(f"Data root: {data_root}")
    print(f"Number of samples to visualize: {args.num_samples}")

    for i, sample in enumerate(dataset.take(args.num_samples)):
        try:
            data = visualize_sample(sample, i)
            if data is not None:
                print(f"\nActions shape: {data['actions'].shape}")
                print(f"States shape: {data['states'].shape}")
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()

    print("\n=== Visualization complete! ===")
    print("Check current directory for saved PNG files.")


if __name__ == "__main__":
    main()