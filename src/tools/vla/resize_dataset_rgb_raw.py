#!/usr/bin/env python3
"""
Resize HDF5 observations/images/rgb_raw for all episodes to target resolution.

默认安全模式：读取 --src，写入 --dst（不改原始数据）。
如需原地修改可加 --inplace（会先写临时文件再原子替换）。

Typical use:
  python tools/vla/resize_dataset_rgb_raw.py \
    --src "/media/zjf/新加卷/vla_dataset_real" \
    --dst "/media/zjf/新加卷/vla_dataset_real_rgbraw_640x360" \
    --width 640 --height 360

In-place:
  python tools/vla/resize_dataset_rgb_raw.py \
    --src "/media/zjf/新加卷/vla_dataset_real" \
    --width 640 --height 360 --inplace
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def _list_episode_files(data_dir: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(data_dir, "episode_*.hdf5")))
    paths.extend(sorted(glob.glob(os.path.join(data_dir, "*.hdf5"))))
    return sorted(set(paths))


def _copy_non_hdf5(src_dir: str, dst_dir: str) -> None:
    """Copy loose files (README, json, etc.); skip .hdf5."""
    for name in os.listdir(src_dir):
        if name.endswith(".hdf5"):
            continue
        s = os.path.join(src_dir, name)
        d = os.path.join(dst_dir, name)
        if os.path.isfile(s):
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(s, d)
        elif os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)


def _resize_rgb_raw(rgb_raw: np.ndarray, width: int, height: int) -> np.ndarray:
    """rgb_raw: (T, H, W, 3) uint8 -> (T, height, width, 3) uint8."""
    if rgb_raw.ndim != 4 or rgb_raw.shape[-1] != 3:
        raise ValueError(f"rgb_raw shape must be (T,H,W,3), got {rgb_raw.shape}")

    t = int(rgb_raw.shape[0])
    out = np.empty((t, height, width, 3), dtype=np.uint8)
    for i in range(t):
        img = Image.fromarray(rgb_raw[i])
        # 使用双线性插值，速度与质量比较均衡
        out[i] = np.asarray(img.resize((width, height), resample=Image.BILINEAR), dtype=np.uint8)
    return out


def _copy_group_with_resize(
    src_grp: h5py.Group,
    dst_grp: h5py.Group,
    width: int,
    height: int,
    stats: dict[str, int],
    group_path: str = "",
) -> None:
    for ak, av in src_grp.attrs.items():
        dst_grp.attrs[ak] = av

    for key in src_grp.keys():
        item = src_grp[key]
        cur_path = f"{group_path}/{key}" if group_path else key
        if isinstance(item, h5py.Group):
            sub = dst_grp.create_group(key)
            _copy_group_with_resize(item, sub, width, height, stats, cur_path)
            continue

        ds_kw: dict = {}
        if item.compression:
            ds_kw["compression"] = item.compression
        if item.compression_opts is not None:
            ds_kw["compression_opts"] = item.compression_opts

        if cur_path == "observations/images/rgb_raw":
            src_arr = np.asarray(item[:], dtype=np.uint8)
            src_shape = tuple(src_arr.shape)
            dst_arr = _resize_rgb_raw(src_arr, width, height)
            d = dst_grp.create_dataset(key, data=dst_arr, **ds_kw)
            stats["resized"] += 1
            stats["frames"] += int(dst_arr.shape[0])
            print(f"  - resized rgb_raw: {src_shape} -> {tuple(dst_arr.shape)}")
        else:
            d = dst_grp.create_dataset(key, data=item[()], **ds_kw)

        for ak, av in item.attrs.items():
            d.attrs[ak] = av


def process_episode(src_path: str, dst_path: str, width: int, height: int) -> tuple[bool, str]:
    stats = {"resized": 0, "frames": 0}

    with h5py.File(src_path, "r") as src:
        os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
        with h5py.File(dst_path, "w") as dst:
            for k, v in src.attrs.items():
                dst.attrs[k] = v

            for key in src.keys():
                item = src[key]
                if isinstance(item, h5py.Group):
                    g = dst.create_group(key)
                    _copy_group_with_resize(item, g, width, height, stats, key)
                else:
                    kw: dict = {}
                    if item.compression:
                        kw["compression"] = item.compression
                    if item.compression_opts is not None:
                        kw["compression_opts"] = item.compression_opts
                    d = dst.create_dataset(key, data=item[()], **kw)
                    for ak, av in item.attrs.items():
                        d.attrs[ak] = av

            if stats["resized"] > 0:
                dst.attrs["rgb_raw_height"] = int(height)
                dst.attrs["rgb_raw_width"] = int(width)
                dst.attrs["rgb_raw_resized"] = True
                dst.attrs["rgb_raw_resize_target"] = f"{width}x{height}"

    if stats["resized"] == 0:
        return False, "skip: no observations/images/rgb_raw"
    return True, f"ok: resized {stats['frames']} frames to {width}x{height}"


def _process_inplace(src_path: str, width: int, height: int) -> tuple[bool, str]:
    parent = str(Path(src_path).parent)
    name = Path(src_path).name
    tmp_path = os.path.join(parent, f".tmp_resize_{name}")
    ok, msg = process_episode(src_path, tmp_path, width, height)
    if ok:
        os.replace(tmp_path, src_path)
    else:
        # 未改动则删除临时文件
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return ok, msg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, help="Source folder with episode HDF5")
    p.add_argument(
        "--dst",
        default="",
        help="Output folder (required unless --inplace).",
    )
    p.add_argument("--width", type=int, default=640, help="Target rgb_raw width")
    p.add_argument("--height", type=int, default=360, help="Target rgb_raw height")
    p.add_argument("--inplace", action="store_true", help="Modify source files in place")
    p.add_argument("--dry_run", action="store_true", help="Only print planned operations")
    args = p.parse_args()

    src_dir = os.path.abspath(args.src)
    if not os.path.isdir(src_dir):
        print(f"[错误] 源目录不存在: {src_dir}", file=sys.stderr)
        sys.exit(1)
    if args.width <= 0 or args.height <= 0:
        print("[错误] width/height 必须为正整数", file=sys.stderr)
        sys.exit(1)

    if args.inplace and args.dst:
        print("[错误] --inplace 模式下不应设置 --dst", file=sys.stderr)
        sys.exit(1)
    if (not args.inplace) and (not args.dst):
        print("[错误] 非 --inplace 模式必须提供 --dst", file=sys.stderr)
        sys.exit(1)

    dst_dir = os.path.abspath(args.dst) if args.dst else ""
    if not args.inplace and os.path.normpath(dst_dir) == os.path.normpath(src_dir):
        print("[错误] 为避免覆盖，请使用不同的 --dst，或显式使用 --inplace", file=sys.stderr)
        sys.exit(1)

    files = _list_episode_files(src_dir)
    if not files:
        print(f"[错误] 未找到 episode_*.hdf5 或 *.hdf5: {src_dir}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        mode = "inplace" if args.inplace else f"{src_dir} -> {dst_dir}"
        print(f"[dry-run] 模式: {mode}")
        print(f"[dry-run] 将处理 {len(files)} 个 HDF5，目标尺寸 {args.width}x{args.height}")
        for fp in files[:5]:
            print(f"  {fp}")
        if len(files) > 5:
            print(f"  ... 共 {len(files)} 个")
        sys.exit(0)

    if not args.inplace:
        os.makedirs(dst_dir, exist_ok=True)
        _copy_non_hdf5(src_dir, dst_dir)

    ok_n = 0
    skip_n = 0
    for src_path in files:
        rel = os.path.relpath(src_path, src_dir)
        print(f"\n[{rel}]")
        if args.inplace:
            ok, msg = _process_inplace(src_path, args.width, args.height)
        else:
            dst_path = os.path.join(dst_dir, rel)
            ok, msg = process_episode(src_path, dst_path, args.width, args.height)
        print(msg)
        if ok:
            ok_n += 1
        else:
            skip_n += 1

    if args.inplace:
        print(f"\n完成: 原地修改 {ok_n} 个 episode，跳过 {skip_n} 个。目录: {src_dir}")
    else:
        print(f"\n完成: 写入 {ok_n} 个 episode，跳过 {skip_n} 个。输出目录: {dst_dir}")


if __name__ == "__main__":
    main()
