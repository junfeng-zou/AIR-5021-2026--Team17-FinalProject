#!/usr/bin/env python3
"""
生成 Vision Projector 的 QNN 量化校准数据

从 HDF5 数据集中提取图像，预处理为模型输入格式，
保存为 raw float32 二进制文件 + filelist.txt

用法:
    python edge_optimization/scripts/generate_calib_data.py \
        --hdf5_dir data/vla_dataset \
        --output_dir edge_optimization/qnn_models/calib_data \
        --num_samples 100
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="生成 QNN 校准数据")
    p.add_argument("--hdf5_dir", type=str, default="data/vla_dataset")
    p.add_argument("--output_dir", type=str, default="edge_optimization/qnn_models/calib_data")
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--use_fused_backbone", action="store_true", default=True,
                   help="6 通道输入 (DINOv2+SigLIP fused)")
    return p.parse_args()


def _resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(repo_root, path)


def preprocess_image(img: np.ndarray, use_fused: bool) -> np.ndarray:
    """uint8 HWC → float32 NCHW (1, C, H, W)"""
    x = img.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # HWC → CHW
    if use_fused:
        x = np.concatenate([x, x], axis=0)  # (6, H, W)
    return np.expand_dims(x, 0)  # (1, C, H, W)


def main():
    args = parse_args()
    hdf5_dir = _resolve(args.hdf5_dir)
    output_dir = _resolve(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    import h5py

    hdf5_files = sorted(glob.glob(os.path.join(hdf5_dir, "episode_*.hdf5")))
    if not hdf5_files:
        print(f"[ERROR] 未找到 HDF5 文件: {hdf5_dir}")
        return 1

    print(f"找到 {len(hdf5_files)} 个 HDF5 文件")

    filelist_path = os.path.join(output_dir, "filelist.txt")
    count = 0

    with open(filelist_path, "w") as fl:
        for hf_path in hdf5_files:
            if count >= args.num_samples:
                break
            try:
                with h5py.File(hf_path, "r") as f:
                    images = f["observations/images/rgb"][:]
                step = max(1, len(images) // 5)
                for i in range(0, len(images), step):
                    if count >= args.num_samples:
                        break
                    img = images[i]
                    tensor = preprocess_image(img, args.use_fused_backbone)

                    # 保存为 raw float32 二进制
                    bin_name = f"sample_{count:04d}.raw"
                    bin_path = os.path.join(output_dir, bin_name)
                    tensor.tofile(bin_path)

                    fl.write(os.path.abspath(bin_path) + "\n")
                    count += 1
            except Exception as e:
                print(f"  [WARN] 跳过 {hf_path}: {e}")

    print(f"生成了 {count} 个校准样本 → {output_dir}")
    print(f"filelist: {filelist_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
