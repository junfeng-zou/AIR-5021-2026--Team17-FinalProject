#!/usr/bin/env python3
"""
Extract observations/images/rgb_raw from episode HDF5 files, and write new HDF5
without rgb_raw into another folder.

默认安全模式（不改原始数据）：
  - 读取 --src 下的 *.hdf5
  - 将 rgb_raw 缩放到 640x360 后保存到 --rgb_out_dir（每个 episode 一个 .npz）
  - 将删除 rgb_raw 后的 hdf5 写到 --dst

Typical use:
  python tools/vla/extract_rgb_raw_from_hdf5.py \
    --src "/media/zjf/新加卷/vla_dataset_real" \
    --dst "/home/zjf/pouring_VLA/data/vla_dataset_real_no_rgbraw" \
    --rgb_out_dir "/home/zjf/pouring_VLA/data/vla_dataset_real_rgbraw_only"
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


TARGET_RGB_RAW_WIDTH = 640
TARGET_RGB_RAW_HEIGHT = 360


def _list_hdf5_files(data_dir: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(data_dir, "episode_*.hdf5")))
    paths.extend(sorted(glob.glob(os.path.join(data_dir, "*.hdf5"))))
    return sorted(set(paths))


def _episode_index_from_name(path: str) -> int | None:
    """
    Parse episode index from file name like episode_0026.hdf5.
    Return None if name does not match this pattern.
    """
    name = os.path.basename(path)
    m = re.fullmatch(r"episode_(\d+)\.hdf5", name)
    if not m:
        return None
    return int(m.group(1))


def _copy_non_hdf5(src_dir: str, dst_dir: str) -> None:
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


def _resize_rgb_raw_to_target(rgb_raw: np.ndarray) -> np.ndarray:
    """Resize (T, H, W, 3) uint8 -> (T, 360, 640, 3) uint8."""
    if rgb_raw.ndim != 4 or rgb_raw.shape[-1] != 3:
        raise ValueError(f"rgb_raw shape must be (T,H,W,3), got {rgb_raw.shape}")
    t = int(rgb_raw.shape[0])
    out = np.empty((t, TARGET_RGB_RAW_HEIGHT, TARGET_RGB_RAW_WIDTH, 3), dtype=np.uint8)
    for i in range(t):
        img = Image.fromarray(rgb_raw[i])
        out[i] = np.asarray(
            img.resize((TARGET_RGB_RAW_WIDTH, TARGET_RGB_RAW_HEIGHT), resample=Image.BILINEAR),
            dtype=np.uint8,
        )
    return out


def _copy_group_drop_rgb_raw(src_grp: h5py.Group, dst_grp: h5py.Group, prefix: str = "") -> bool:
    """
    Copy src group recursively into dst group, while dropping observations/images/rgb_raw.
    Return True if dropped.
    """
    dropped = False
    for ak, av in src_grp.attrs.items():
        dst_grp.attrs[ak] = av

    for key in src_grp.keys():
        item = src_grp[key]
        cur_path = f"{prefix}/{key}" if prefix else key
        if isinstance(item, h5py.Group):
            sub = dst_grp.create_group(key)
            dropped = _copy_group_drop_rgb_raw(item, sub, cur_path) or dropped
            continue

        if cur_path == "observations/images/rgb_raw":
            dropped = True
            continue

        kw: dict = {}
        if item.compression:
            kw["compression"] = item.compression
        if item.compression_opts is not None:
            kw["compression_opts"] = item.compression_opts
        d = dst_grp.create_dataset(key, data=item[()], **kw)
        for ak, av in item.attrs.items():
            d.attrs[ak] = av

    return dropped


def _extract_one_rgb_raw(src_path: str, rgb_npz_path: str, meta_json_path: str) -> tuple[bool, str]:
    with h5py.File(src_path, "r") as f:
        ds_path = "observations/images/rgb_raw"
        if ds_path not in f:
            return False, "skip: no observations/images/rgb_raw"
        arr = np.asarray(f[ds_path][:], dtype=np.uint8)
        arr_resized = _resize_rgb_raw_to_target(arr)
        os.makedirs(os.path.dirname(rgb_npz_path) or ".", exist_ok=True)
        np.savez_compressed(rgb_npz_path, rgb_raw=arr_resized)
        meta = {
            "source_hdf5": os.path.abspath(src_path),
            "rgb_raw_npz": os.path.abspath(rgb_npz_path),
            "shape_before_resize": list(arr.shape),
            "shape_after_resize": list(arr_resized.shape),
            "target_resolution": f"{TARGET_RGB_RAW_WIDTH}x{TARGET_RGB_RAW_HEIGHT}",
            "dtype": str(arr_resized.dtype),
        }
        with open(meta_json_path, "w", encoding="utf-8") as fp:
            json.dump(meta, fp, indent=2, ensure_ascii=False)
    return (
        True,
        f"ok: saved rgb_raw {tuple(arr.shape)} -> {tuple(arr_resized.shape)} -> {rgb_npz_path}",
    )


def _write_hdf5_without_rgb_raw(src_path: str, dst_path: str, rgb_rel: str) -> tuple[bool, str]:
    with h5py.File(src_path, "r") as src:
        os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
        with h5py.File(dst_path, "w") as dst:
            for k, v in src.attrs.items():
                # 删除与内嵌 rgb_raw 强绑定的属性，避免误导
                if k in ("rgb_raw_height", "rgb_raw_width"):
                    continue
                dst.attrs[k] = v

            dropped = False
            for key in src.keys():
                item = src[key]
                if isinstance(item, h5py.Group):
                    g = dst.create_group(key)
                    dropped = _copy_group_drop_rgb_raw(item, g, key) or dropped
                else:
                    kw: dict = {}
                    if item.compression:
                        kw["compression"] = item.compression
                    if item.compression_opts is not None:
                        kw["compression_opts"] = item.compression_opts
                    d = dst.create_dataset(key, data=item[()], **kw)
                    for ak, av in item.attrs.items():
                        d.attrs[ak] = av

            if dropped:
                dst.attrs["rgb_raw_externalized"] = True
                dst.attrs["rgb_raw_external_relpath"] = rgb_rel

    if not dropped:
        return False, "skip: no observations/images/rgb_raw"
    return True, f"ok: wrote hdf5 without rgb_raw -> {dst_path}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, help="Source folder containing episode HDF5")
    p.add_argument("--dst", required=True, help="Output folder for HDF5 without rgb_raw")
    p.add_argument("--rgb_out_dir", required=True, help="Output folder for extracted rgb_raw .npz")
    p.add_argument(
        "--rgb_suffix",
        default="_rgb_raw.npz",
        help="Suffix for extracted rgb_raw files (default: _rgb_raw.npz)",
    )
    p.add_argument(
        "--start_episode_idx",
        type=int,
        default=0,
        help="Only process episode_XXXX.hdf5 with XXXX >= this value (default: 0)",
    )
    p.add_argument("--dry_run", action="store_true", help="Only print planned operations")
    args = p.parse_args()

    src_dir = os.path.abspath(args.src)
    dst_dir = os.path.abspath(args.dst)
    rgb_dir = os.path.abspath(args.rgb_out_dir)

    if not os.path.isdir(src_dir):
        print(f"[错误] 源目录不存在: {src_dir}", file=sys.stderr)
        sys.exit(1)
    if os.path.normpath(src_dir) == os.path.normpath(dst_dir):
        print("[错误] --dst 不能和 --src 相同", file=sys.stderr)
        sys.exit(1)

    files = _list_hdf5_files(src_dir)
    if not files:
        print(f"[错误] 未找到 episode_*.hdf5 或 *.hdf5: {src_dir}", file=sys.stderr)
        sys.exit(1)

    filtered: list[str] = []
    for fp in files:
        ep_idx = _episode_index_from_name(fp)
        if ep_idx is None:
            # 非 episode_XXXX 命名的 hdf5 保持处理（向后兼容）
            filtered.append(fp)
            continue
        if ep_idx >= args.start_episode_idx:
            filtered.append(fp)
    files = filtered
    if not files:
        print(
            f"[错误] 过滤后无可处理文件（--start_episode_idx={args.start_episode_idx}）",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.dry_run:
        print(f"[dry-run] src={src_dir}")
        print(f"[dry-run] dst(no rgb_raw)={dst_dir}")
        print(f"[dry-run] rgb_out_dir={rgb_dir}")
        print(f"[dry-run] start_episode_idx={args.start_episode_idx}")
        print(f"[dry-run] files={len(files)}")
        for fp in files[:5]:
            print(f"  {fp}")
        if len(files) > 5:
            print(f"  ... 共 {len(files)} 个")
        return

    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(rgb_dir, exist_ok=True)
    _copy_non_hdf5(src_dir, dst_dir)

    ok_n = 0
    skip_n = 0
    for src_path in files:
        rel = os.path.relpath(src_path, src_dir)
        stem = Path(rel).with_suffix("").as_posix()
        rgb_rel = f"{stem}{args.rgb_suffix}"
        rgb_npz = os.path.join(rgb_dir, rgb_rel)
        meta_json = os.path.join(rgb_dir, f"{stem}_rgb_raw_meta.json")
        dst_h5 = os.path.join(dst_dir, rel)

        print(f"\n[{rel}]")
        ok_ext, msg_ext = _extract_one_rgb_raw(src_path, rgb_npz, meta_json)
        print(msg_ext)
        ok_h5, msg_h5 = _write_hdf5_without_rgb_raw(src_path, dst_h5, rgb_rel=rgb_rel)
        print(msg_h5)

        if ok_ext and ok_h5:
            ok_n += 1
        else:
            skip_n += 1

    print(f"\n完成: 成功处理 {ok_n} 个 episode，跳过 {skip_n} 个。")
    print(f"  新 hdf5 目录: {dst_dir}")
    print(f"  rgb_raw 输出目录: {rgb_dir}")


if __name__ == "__main__":
    main()
