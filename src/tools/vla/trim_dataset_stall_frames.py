#!/usr/bin/env python3
"""
Copy VLA HDF5 episodes and drop stall frames globally based on action deltas.

判定使用根部的 actions（与 rgb 时间长度一致）：
- position 范数 = ||actions[:3]||
- rotation 范数 = ||actions[3:6]||
- gripper（actions[6]）不参与判定

仅当某帧的 position 与 rotation 范数都小于各自阈值时，视为 stall 帧。
本脚本执行“全局逐帧删除”：凡是 stall 帧都会被删除（不只是掐头去尾）。
会对**文件中所有与时间轴对齐的数据集**做同一索引筛选
（rgb、depth、state、actions、timestamps 等），保证每帧各类数据仍一一对齐。

Typical use (repo root):
  python tools/vla/trim_dataset_stall_frames.py \\
    --src data/vla_dataset --dst data/vla_dataset_processed \\
    --pos_threshold 0.01 --rot_threshold 0.01
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys

import h5py
import numpy as np


def _list_episode_files(data_dir: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(data_dir, "episode_*.hdf5")))
    paths.extend(sorted(glob.glob(os.path.join(data_dir, "*.hdf5"))))
    return sorted(set(paths))


def _time_length_rgb(src: h5py.File) -> int:
    rgb = src["observations/images/rgb"]
    return int(rgb.shape[0])


def _keep_indices_global(
    actions: np.ndarray, pos_threshold: float, rot_threshold: float
) -> np.ndarray | None:
    """Return kept frame indices after global stall filtering, or None if empty."""
    if actions.ndim != 2 or actions.shape[1] < 6:
        raise ValueError(f"actions bad shape {actions.shape}, need (T, >=6)")
    pos_norms = np.linalg.norm(actions[:, :3].astype(np.float64), axis=1)
    rot_norms = np.linalg.norm(actions[:, 3:6].astype(np.float64), axis=1)
    keep_mask = ~((pos_norms < pos_threshold) & (rot_norms < rot_threshold))
    keep_idx = np.flatnonzero(keep_mask).astype(np.int64)
    if keep_idx.size == 0:
        return None
    return keep_idx


def _time_slice_for_dataset(
    src_ds: h5py.Dataset,
    keep_idx: np.ndarray,
    t_old: int,
) -> tuple[np.ndarray, str]:
    """
    Return (array_to_store, mode) where mode is 'axis0', 'axis_last', or 'full'.

    - axis0: 第一维为时间 T，与 rgb 一致（当前 VLA 写入格式）
    - axis_last: 仅当第一维不是 T 且最后一维是 T 时，按最后一维切片（兼容时间维在末尾的布局）
    - full: 非按步展开的数组，原样拷贝
    """
    sh = src_ds.shape
    if not sh:
        return np.asarray(src_ds[()]), "full"
    if sh[0] == t_old:
        return np.asarray(src_ds[()])[keep_idx], "axis0"
    if len(sh) >= 1 and sh[-1] == t_old and sh[0] != t_old:
        full = np.asarray(src_ds[()])
        idx = (slice(None),) * (len(sh) - 1) + (keep_idx,)
        return np.asarray(full[idx]), "axis_last"
    return np.asarray(src_ds[()]), "full"


def _copy_dataset_sliced(
    src_ds: h5py.Dataset,
    dst_parent: h5py.Group,
    name: str,
    keep_idx: np.ndarray,
    t_old: int,
) -> None:
    data, _mode = _time_slice_for_dataset(src_ds, keep_idx, t_old)
    kw: dict = {}
    if src_ds.compression:
        kw["compression"] = src_ds.compression
    if src_ds.compression_opts is not None:
        kw["compression_opts"] = src_ds.compression_opts
    d = dst_parent.create_dataset(name, data=data, **kw)
    for ak, av in src_ds.attrs.items():
        d.attrs[ak] = av


def _copy_group_sliced(
    src_grp: h5py.Group,
    dst_grp: h5py.Group,
    keep_idx: np.ndarray,
    t_old: int,
) -> None:
    for ak, av in src_grp.attrs.items():
        dst_grp.attrs[ak] = av
    for key in src_grp.keys():
        item = src_grp[key]
        if isinstance(item, h5py.Dataset):
            _copy_dataset_sliced(item, dst_grp, key, keep_idx, t_old)
        else:
            sub = dst_grp.create_group(key)
            _copy_group_sliced(item, sub, keep_idx, t_old)


def _copy_root_attrs(src: h5py.File, dst: h5py.File, new_num_steps: int) -> None:
    for k, v in src.attrs.items():
        dst.attrs[k] = v
    dst.attrs["num_steps"] = new_num_steps


def process_episode(
    src_path: str, dst_path: str, pos_threshold: float, rot_threshold: float
) -> tuple[bool, str]:
    with h5py.File(src_path, "r") as src:
        if "actions" not in src:
            return False, "skip: no actions"
        t_old = _time_length_rgb(src)
        actions = np.asarray(src["actions"][:], dtype=np.float32)
        if actions.shape[0] != t_old:
            return False, f"skip: actions T={actions.shape[0]} != rgb T={t_old}"
        keep_idx = _keep_indices_global(actions, pos_threshold, rot_threshold)
        if keep_idx is None:
            return False, "skip: all frames below thresholds (empty)"
        new_t = int(keep_idx.size)
        drop_n = int(t_old - new_t)

        os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
        with h5py.File(dst_path, "w") as dst:
            for key in src.keys():
                item = src[key]
                if isinstance(item, h5py.Dataset):
                    _copy_dataset_sliced(item, dst, key, keep_idx, t_old)
                else:
                    g = dst.create_group(key)
                    _copy_group_sliced(item, g, keep_idx, t_old)
            _copy_root_attrs(src, dst, new_t)
            dst.attrs["trimmed_global_stall"] = True
            dst.attrs["trim_source_dataset"] = "actions"
            dst.attrs["trim_stall_pos_l2_threshold"] = pos_threshold
            dst.attrs["trim_stall_rot_l2_threshold"] = rot_threshold
            dst.attrs["trim_stall_ignore_gripper"] = True
            dst.attrs["trim_removed_total"] = drop_n

        with h5py.File(dst_path, "r") as verify:
            tr = int(verify["observations/images/rgb"].shape[0])
            if tr != new_t:
                return False, f"internal error: wrote rgb T={tr} expected {new_t}"
            for ds_path in ("actions", "actions_raw", "timestamps"):
                if ds_path in verify:
                    if int(verify[ds_path].shape[0]) != new_t:
                        return False, f"internal error: {ds_path} T mismatch after trim"

    return True, f"ok: {t_old} -> {new_t} frames (drop total {drop_n})"


def copy_non_hdf5(src_dir: str, dst_dir: str) -> None:
    """Copy loose files (README, json, etc.); skip .hdf5 (handled separately)."""
    if not os.path.isdir(src_dir):
        return
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default="data/vla_dataset", help="Source folder with episode HDF5")
    p.add_argument("--dst", default="data/vla_dataset_processed", help="Output folder")
    p.add_argument("--pos_threshold", type=float, default=0.01, help="L2 norm threshold for actions[:3]")
    p.add_argument("--rot_threshold", type=float, default=0.01, help="L2 norm threshold for actions[3:6]")
    p.add_argument("--dry_run", action="store_true", help="Only print planned operations")
    args = p.parse_args()

    src_dir = os.path.abspath(args.src)
    dst_dir = os.path.abspath(args.dst)

    if not os.path.isdir(src_dir):
        print(f"[错误] 源目录不存在: {src_dir}", file=sys.stderr)
        sys.exit(1)

    files = _list_episode_files(src_dir)
    if not files:
        print(f"[错误] 未找到 episode_*.hdf5 或 *.hdf5: {src_dir}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print(f"[dry-run] 将处理 {len(files)} 个 HDF5 -> {dst_dir}")
        for fp in files[:5]:
            print(f"  {fp}")
        if len(files) > 5:
            print(f"  ... 共 {len(files)}")
        sys.exit(0)

    os.makedirs(dst_dir, exist_ok=True)
    copy_non_hdf5(src_dir, dst_dir)

    ok_n = skip_n = 0
    for src_path in files:
        rel = os.path.relpath(src_path, src_dir)
        dst_path = os.path.join(dst_dir, rel)
        os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
        ok, msg = process_episode(
            src_path,
            dst_path,
            args.pos_threshold,
            args.rot_threshold,
        )
        print(f"{rel}: {msg}")
        if ok:
            ok_n += 1
        else:
            skip_n += 1

    print(f"\n完成: 写入 {ok_n} 个 episode，跳过 {skip_n} 个。输出目录: {dst_dir}")


if __name__ == "__main__":
    main()
