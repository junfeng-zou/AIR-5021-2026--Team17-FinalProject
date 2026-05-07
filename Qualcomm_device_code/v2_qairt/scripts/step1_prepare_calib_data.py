#!/usr/bin/env python3
"""
Step 1: 生成 qairt-quantizer 校准数据集
输入: HDF5 帧 或 图片目录
输出: (1,6,224,224) float32 .raw 文件 + input_list.txt

用法:
    python step1_prepare_calib_data.py --hdf5 /path/to/episode.hdf5 --num_samples 30
    python step1_prepare_calib_data.py --image_dir /path/to/images/  --num_samples 30
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np

_DINO_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_DINO_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_SIG_MEAN  = np.array([0.5, 0.5, 0.5],       dtype=np.float32)
_SIG_STD   = np.array([0.5, 0.5, 0.5],       dtype=np.float32)


def preprocess(rgb_uint8: np.ndarray) -> np.ndarray:
    """(H,W,3) uint8 → (1,6,224,224) float32，双骨干归一化"""
    import cv2
    if rgb_uint8.shape[:2] != (224, 224):
        rgb_uint8 = cv2.resize(rgb_uint8, (224, 224), interpolation=cv2.INTER_AREA)
    x = rgb_uint8.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # (3,224,224)
    dino = (x - _DINO_MEAN[:, None, None]) / _DINO_STD[:, None, None]
    sglp = (x - _SIG_MEAN[:, None, None])  / _SIG_STD[:, None, None]
    return np.concatenate([dino, sglp], axis=0)[np.newaxis].astype(np.float16)


def from_hdf5(hdf5_path: str, n: int, out_dir: str) -> list[str]:
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        if "observations/images/rgb" in f:
            imgs = f["observations/images/rgb"][:]
        elif "images" in f:
            imgs = f["images"][:]
        else:
            raise KeyError(f"未找到图像 key，可用: {list(f.keys())}")
    idxs = np.linspace(0, len(imgs) - 1, min(n, len(imgs)), dtype=int)
    print(f"  HDF5: {len(imgs)} 帧 → 采样 {len(idxs)} 张")
    paths = []
    for i, idx in enumerate(idxs):
        rgb = imgs[idx]
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        p = os.path.join(out_dir, f"calib_{i:04d}.raw")
        preprocess(rgb).tofile(p)
        paths.append(p)
    return paths


def from_image_dir(img_dir: str, n: int, out_dir: str) -> list[str]:
    import cv2
    files = sorted(p for p in Path(img_dir).iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not files:
        raise FileNotFoundError(f"目录 {img_dir} 中无图片")
    idxs = np.linspace(0, len(files) - 1, min(n, len(files)), dtype=int)
    print(f"  图片目录: {len(files)} 张 → 采样 {len(idxs)} 张")
    paths = []
    for i, idx in enumerate(idxs):
        bgr = cv2.imread(str(files[idx]))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        p = os.path.join(out_dir, f"calib_{i:04d}.raw")
        preprocess(rgb).tofile(p)
        paths.append(p)
    return paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5",       default="")
    p.add_argument("--image_dir",  default="")
    p.add_argument("--output_dir", default="edge_optimization_v2.0/data/calib")
    p.add_argument("--num_samples", type=int, default=30)
    args = p.parse_args()

    if not args.hdf5 and not args.image_dir:
        print("ERROR: 需要 --hdf5 或 --image_dir"); sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 50)
    print("Step 1: 生成量化校准数据集")
    print(f"  输出: {args.output_dir}")
    print(f"  每个 .raw: {1*6*224*224*4/1024:.0f} KB  (1,6,224,224) float32")

    if args.hdf5:
        raw_paths = from_hdf5(args.hdf5, args.num_samples, args.output_dir)
    else:
        raw_paths = from_image_dir(args.image_dir, args.num_samples, args.output_dir)

    # 写 input_list.txt（qairt-quantizer 需要绝对路径）
    list_path = os.path.join(args.output_dir, "input_list.txt")
    with open(list_path, "w") as f:
        for rp in raw_paths:
            f.write(os.path.abspath(rp) + "\n")

    print(f"\n✅ 完成: {len(raw_paths)} 个 .raw + {list_path}")

    # 快速验证文件大小
    expected = 1 * 6 * 224 * 224 * 4
    bad = [p for p in raw_paths if os.path.getsize(p) != expected]
    if bad:
        print(f"  ⚠️  {len(bad)} 个文件大小异常（预期 {expected} bytes）")
    else:
        print(f"  ✅ 所有文件大小正确 ({expected} bytes each)")


if __name__ == "__main__":
    main()
