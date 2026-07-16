"""
Debug script for evaluating OpenVLA on LIBERO
Usage: python my_libero_eval.py
Then use debugger to set breakpoints and step through.
"""

import os
import sys


# Set environment variables BEFORE importing anything else
os.environ["HF_TOKEN"] = ""  # Set your HF_TOKEN here
os.environ["PYTHONPATH"] = "/home/vcar/LIBERO:" + os.environ.get("PYTHONPATH", "")
# os.environ["MUJOCO_GL"] = "osmesa"

# 在你的 Python 脚本开头修改这两行
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

# Print debug info
print("Debug mode: Starting evaluation...")
print(f"Python: {sys.executable}")
print(f"Working directory: {os.getcwd()}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '')}")

if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [
            sys.executable,
            "experiments/robot/libero/run_libero_eval.py",
            "--model_family=openvla",
            "--pretrained_checkpoint=/home/vcar/openvla/merged_checkpoints/step_18000",
            "--task_suite_name=libero_spatial",
            "--center_crop=True",
            "--num_trials_per_task=10",  # 减少试验次数加快调试
            "--use_wandb=False",
            "--seed=7",
        ],
        cwd="/home/vcar/openvla",
    )
    sys.exit(result.returncode)
