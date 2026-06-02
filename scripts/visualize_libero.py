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

    # 解析 example
    def parse_example(example):
        return tf.io.parse_single_example(
            example,
            {
                "features/steps": tf.io.VarLenFeature(tf.string),
            }
        )

    parsed_dataset = dataset.map(parse_example)
    return parsed_dataset


def decode_steps(steps_bytes):
    """解码 steps 序列"""
    steps = []
    for step_bytes in steps_bytes:
        step = tf.io.parse_tensor(step_bytes, out_type=tf.float32)
        steps.append(step)
    return steps


def visualize_sample(sample, sample_idx=0):
    """可视化单个样本"""
    raw_steps = sample["features/steps"]

    # 获取第一个 step 来查看结构
    first_step = tf.io.parse_tensor(raw_steps[0], out_type=tf.float32)
    print(f"\n=== Sample {sample_idx} ===")
    print(f"Number of steps: {len(raw_steps)}")
    print(f"Step shape: {first_step.shape}")
    print(f"Step dtype: {first_step.dtype}")

    # 解析所有 steps
    actions = []
    images = []
    wrist_images = []
    instructions = []
    states = []

    for i, step_bytes in enumerate(raw_steps):
        step = tf.io.parse_tensor(step_bytes, out_type=tf.float32).numpy()
        # step 格式: [action(7), image(256,256,3), wrist_image(256,256,3), state(8), joint_state(7), is_last, is_first, is_terminal, discount, reward, instruction_len, ...]
        # 需要根据实际情况解析

        # 假设 step 被扁平化了，需要根据 features.json 的结构来解析
        # action: 7, image: 256*256*3, wrist: 256*256*3, state: 8, joint: 7
        # 后面是标量: is_last, is_first, is_terminal, discount, reward, instruction_len
        # 然后是 language_instruction

        offset = 0
        action = step[offset:offset+7]; offset += 7
        img_flat = step[offset:offset+256*256*3]; offset += 256*256*3
        wrist_flat = step[offset:offset+256*256*3]; offset += 256*256*3
        state = step[offset:offset+8]; offset += 8
        joint = step[offset:offset+7]; offset += 7

        is_last = step[offset]; offset += 1
        is_first = step[offset]; offset += 1
        is_terminal = step[offset]; offset += 1
        discount = step[offset]; offset += 1
        reward = step[offset]; offset += 1

        actions.append(action)
        images.append(img_flat.reshape(256, 256, 3).astype(np.uint8))
        wrist_images.append(wrist_flat.reshape(256, 256, 3).astype(np.uint8))
        states.append(state)

    return {
        "actions": np.array(actions),
        "images": images,
        "wrist_images": wrist_images,
        "states": np.array(states),
    }


def plot_sample(data, sample_idx=0, num_frames=5):
    """绘制样本的关键帧"""
    n_frames = min(num_frames, len(data["images"]))
    indices = np.linspace(0, len(data["images"])-1, n_frames, dtype=int)

    fig, axes = plt.subplots(2, n_frames, figsize=(4*n_frames, 8))

    if n_frames == 1:
        axes = axes.reshape(2, 1)

    for i, idx in enumerate(indices):
        # 主相机图像
        axes[0, i].imshow(data["images"][idx])
        axes[0, i].set_title(f"Frame {idx}")
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("Main Camera", fontsize=12)

        # 腕部相机图像
        axes[1, i].imshow(data["wrist_images"][idx])
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("Wrist Camera", fontsize=12)

    plt.suptitle(f"Sample {sample_idx} - {n_frames} frames", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"sample_{sample_idx}_visualization.png", dpi=100)
    print(f"Saved sample_{sample_idx}_visualization.png")
    plt.close()


def plot_actions(data, sample_idx=0):
    """绘制动作曲线"""
    actions = data["actions"]  # shape: (num_steps, 7)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    action_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

    for i in range(7):
        axes[i].plot(actions[:, i])
        axes[i].set_title(f"Action {i}: {action_names[i]}")
        axes[i].set_xlabel("Step")
        axes[i].set_ylabel("Value")
        axes[i].grid(True)

    # 绘制 gripper 单独放大
    axes[7].plot(actions[:, 6])
    axes[7].set_title("Gripper (放大)")
    axes[7].set_xlabel("Step")
    axes[7].set_ylabel("Value")
    axes[7].grid(True)

    plt.suptitle(f"Sample {sample_idx} - Action Trajectory", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"sample_{sample_idx}_actions.png", dpi=100)
    print(f"Saved sample_{sample_idx}_actions.png")
    plt.close()


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
            print(f"\n--- Processing sample {i} ---")
            data = visualize_sample(sample, i)

            # 打印动作统计
            print(f"\nActions shape: {data['actions'].shape}")
            print(f"Action mean: {data['actions'].mean(axis=0)}")
            print(f"Action std: {data['actions'].std(axis=0)}")
            print(f"Action min: {data['actions'].min(axis=0)}")
            print(f"Action max: {data['actions'].max(axis=0)}")

            # 绘制图像
            plot_sample(data, i, args.num_frames)

            # 绘制动作曲线
            plot_actions(data, i)

            print(f"Sample {i} done!")

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()

    print("\n=== Visualization complete! ===")
    print("Check current directory for saved PNG files.")


if __name__ == "__main__":
    main()