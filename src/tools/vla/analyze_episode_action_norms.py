#!/usr/bin/env python3
"""
Analyze action-delta norm distributions for one episode or a whole dataset.

This script reads root dataset "actions" with shape (T, >=7) or (T, >=6),
then computes:
  - position norm: ||actions[:, :3]||
  - rotation norm: ||actions[:, 3:6]||

The gripper dimension (actions[:, 6]) is ignored.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from typing import Iterable

import h5py
import numpy as np


def _fmt(x: float) -> str:
    return f"{x:.6g}"


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
    }


def _percentiles(values: np.ndarray, ps: Iterable[float]) -> list[tuple[float, float]]:
    ps_arr = np.asarray(list(ps), dtype=np.float64)
    vals = np.percentile(values, ps_arr)
    return [(float(p), float(v)) for p, v in zip(ps_arr, vals)]


def _print_stats(title: str, values: np.ndarray) -> None:
    stats = _summary_stats(values)
    print(f"\n[{title}]")
    print(
        "  min={min} max={max} mean={mean} std={std} median={median}".format(
            **{k: _fmt(v) for k, v in stats.items()}
        )
    )


def _print_percentiles(title: str, values: np.ndarray) -> None:
    ps = [0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100]
    rows = _percentiles(values, ps)
    print(f"\n[{title} percentiles]")
    for p, v in rows:
        print(f"  p{int(p):>3}: {_fmt(v)}")


def _ascii_hist(title: str, values: np.ndarray, bins: int = 24, width: int = 50) -> None:
    hist, edges = np.histogram(values, bins=bins)
    m = int(np.max(hist)) if hist.size else 0
    print(f"\n[{title} histogram, bins={bins}]")
    if m == 0:
        print("  (empty)")
        return
    for i, c in enumerate(hist):
        left = edges[i]
        right = edges[i + 1]
        bar_len = int(round((c / m) * width)) if m > 0 else 0
        bar = "#" * bar_len
        print(f"  [{_fmt(left):>10}, {_fmt(right):>10}) | {bar} {c}")


def _global_stall_filter_stats(
    pos_norm: np.ndarray,
    rot_norm: np.ndarray,
    pos_thr: float,
    rot_thr: float,
) -> tuple[int, int]:
    stall = (pos_norm < pos_thr) & (rot_norm < rot_thr)
    drop_total = int(np.count_nonzero(stall))
    keep_total = int(stall.shape[0] - drop_total)
    return drop_total, keep_total


def _list_hdf5_files(data_dir: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(data_dir, "episode_*.hdf5")))
    paths.extend(sorted(glob.glob(os.path.join(data_dir, "*.hdf5"))))
    return sorted(set(paths))


def _episode_index_from_name(path: str) -> int | None:
    name = os.path.basename(path)
    m = re.fullmatch(r"episode_(\d+)\.hdf5", name)
    if not m:
        return None
    return int(m.group(1))


def _as_percentile_dict(values: np.ndarray) -> dict[str, float]:
    ps = [0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100]
    rows = _percentiles(values, ps)
    return {f"p{int(p)}": float(v) for p, v in rows}


def _load_actions(ep: str) -> np.ndarray:
    with h5py.File(ep, "r") as f:
        if "actions" not in f:
            raise KeyError("dataset 'actions' not found in episode")
        actions = np.asarray(f["actions"][:], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] < 6:
        raise ValueError(f"invalid actions shape {actions.shape}, expected (T, >=6)")
    return actions


def _compute_norms(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos_norm = np.linalg.norm(actions[:, :3].astype(np.float64), axis=1)
    rot_norm = np.linalg.norm(actions[:, 3:6].astype(np.float64), axis=1)
    return pos_norm, rot_norm


def _analyze_single_episode(ep: str, args: argparse.Namespace) -> int:
    ep_abs = os.path.abspath(ep)
    if not os.path.isfile(ep_abs):
        print(f"[ERROR] episode file not found: {ep_abs}", file=sys.stderr)
        return 1

    try:
        actions = _load_actions(ep_abs)
    except (OSError, KeyError, ValueError) as exc:
        print(f"[ERROR] failed to load actions from {ep_abs}: {exc}", file=sys.stderr)
        return 2

    t = int(actions.shape[0])
    pos_norm, rot_norm = _compute_norms(actions)

    print(f"Episode: {ep_abs}")
    print(f"Frames: {t}")
    print("Norm definition:")
    print("  pos_norm = ||actions[:3]||")
    print("  rot_norm = ||actions[3:6]||")
    print("  gripper(actions[6]) is ignored")

    _print_stats("position norm", pos_norm)
    _print_percentiles("position norm", pos_norm)

    _print_stats("rotation norm", rot_norm)
    _print_percentiles("rotation norm", rot_norm)

    if not args.no_hist:
        _ascii_hist("position norm", pos_norm, bins=max(2, int(args.bins)))
        _ascii_hist("rotation norm", rot_norm, bins=max(2, int(args.bins)))

    if args.probe_pos_threshold is not None and args.probe_rot_threshold is not None:
        drop_total, kept = _global_stall_filter_stats(
            pos_norm,
            rot_norm,
            float(args.probe_pos_threshold),
            float(args.probe_rot_threshold),
        )
        print("\n[probe filter result]")
        print(f"  pos_threshold={_fmt(float(args.probe_pos_threshold))}")
        print(f"  rot_threshold={_fmt(float(args.probe_rot_threshold))}")
        print("  rule: drop frame if pos_norm<thr AND rot_norm<thr")
        print(f"  drop_total={drop_total} keep={kept}/{t}")
    elif args.probe_pos_threshold is not None or args.probe_rot_threshold is not None:
        print(
            "\n[WARN] probe requires both --probe_pos_threshold and --probe_rot_threshold."
        )

    return 0


def _analyze_dataset(data_dir: str, out_dir: str, args: argparse.Namespace) -> int:
    src = os.path.abspath(data_dir)
    if not os.path.isdir(src):
        print(f"[ERROR] data_dir not found: {src}", file=sys.stderr)
        return 1

    files = _list_hdf5_files(src)
    if not files:
        print(f"[ERROR] no hdf5 found under: {src}", file=sys.stderr)
        return 2

    os.makedirs(out_dir, exist_ok=True)
    frame_csv = os.path.join(out_dir, "frame_norms.csv")
    episode_csv = os.path.join(out_dir, "episode_norm_summary.csv")
    summary_json = os.path.join(out_dir, "episode_norm_summary.json")

    summaries: list[dict] = []
    ok_n = 0
    skip_n = 0

    frame_writer = None
    frame_f = None
    if args.export_frame_csv:
        frame_f = open(frame_csv, "w", newline="", encoding="utf-8")
        frame_writer = csv.writer(frame_f)
        frame_writer.writerow(
            [
                "episode_file",
                "episode_index",
                "frame_index",
                "pos_norm",
                "rot_norm",
                "is_stall",
            ]
        )

    try:
        for ep in files:
            ep_abs = os.path.abspath(ep)
            rel = os.path.relpath(ep_abs, src)
            ep_idx = _episode_index_from_name(ep_abs)
            try:
                actions = _load_actions(ep_abs)
            except (OSError, KeyError, ValueError) as exc:
                summaries.append(
                    {
                        "episode_file": rel,
                        "episode_index": ep_idx,
                        "status": "error",
                        "error": str(exc),
                    }
                )
                print(f"[skip] {rel}: {exc}")
                skip_n += 1
                continue

            pos_norm, rot_norm = _compute_norms(actions)
            t = int(actions.shape[0])
            stall_mask = None
            if args.probe_pos_threshold is not None and args.probe_rot_threshold is not None:
                stall_mask = (pos_norm < float(args.probe_pos_threshold)) & (
                    rot_norm < float(args.probe_rot_threshold)
                )

            if frame_writer is not None:
                for i in range(t):
                    is_stall_val = (
                        int(bool(stall_mask[i])) if stall_mask is not None else ""
                    )
                    frame_writer.writerow(
                        [
                            rel,
                            ep_idx if ep_idx is not None else "",
                            i,
                            float(pos_norm[i]),
                            float(rot_norm[i]),
                            is_stall_val,
                        ]
                    )

            rec: dict = {
                "episode_file": rel,
                "episode_index": ep_idx,
                "status": "ok",
                "frames": t,
                "position_norm": {
                    "stats": _summary_stats(pos_norm),
                    "percentiles": _as_percentile_dict(pos_norm),
                },
                "rotation_norm": {
                    "stats": _summary_stats(rot_norm),
                    "percentiles": _as_percentile_dict(rot_norm),
                },
            }

            if stall_mask is not None:
                drop_total = int(np.count_nonzero(stall_mask))
                rec["probe"] = {
                    "pos_threshold": float(args.probe_pos_threshold),
                    "rot_threshold": float(args.probe_rot_threshold),
                    "rule": "drop frame if pos_norm<thr AND rot_norm<thr",
                    "drop_total": drop_total,
                    "keep_total": int(t - drop_total),
                }

            summaries.append(rec)
            ok_n += 1
    finally:
        if frame_f is not None:
            frame_f.close()

    # 一行一个 episode，方便你在表格里快速筛选与排序
    percentile_keys = [f"p{x}" for x in (0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100)]
    with open(episode_csv, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        head = [
            "episode_file",
            "episode_index",
            "status",
            "frames",
            "pos_min",
            "pos_max",
            "pos_mean",
            "pos_std",
            "pos_median",
            "rot_min",
            "rot_max",
            "rot_mean",
            "rot_std",
            "rot_median",
        ]
        head.extend([f"pos_{k}" for k in percentile_keys])
        head.extend([f"rot_{k}" for k in percentile_keys])
        if args.probe_pos_threshold is not None and args.probe_rot_threshold is not None:
            head.extend(["probe_drop_total", "probe_keep_total"])
        writer.writerow(head)

        for rec in summaries:
            if rec.get("status") != "ok":
                row = [
                    rec.get("episode_file", ""),
                    rec.get("episode_index", ""),
                    rec.get("status", ""),
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
                row.extend([""] * len(percentile_keys))
                row.extend([""] * len(percentile_keys))
                if args.probe_pos_threshold is not None and args.probe_rot_threshold is not None:
                    row.extend(["", ""])
                writer.writerow(row)
                continue

            ps = rec["position_norm"]["stats"]
            rs = rec["rotation_norm"]["stats"]
            pp = rec["position_norm"]["percentiles"]
            rp = rec["rotation_norm"]["percentiles"]
            row = [
                rec.get("episode_file", ""),
                rec.get("episode_index", ""),
                "ok",
                rec.get("frames", ""),
                ps["min"],
                ps["max"],
                ps["mean"],
                ps["std"],
                ps["median"],
                rs["min"],
                rs["max"],
                rs["mean"],
                rs["std"],
                rs["median"],
            ]
            row.extend([pp.get(k, "") for k in percentile_keys])
            row.extend([rp.get(k, "") for k in percentile_keys])
            if args.probe_pos_threshold is not None and args.probe_rot_threshold is not None:
                probe = rec.get("probe", {})
                row.extend([probe.get("drop_total", ""), probe.get("keep_total", "")])
            writer.writerow(row)

    with open(summary_json, "w", encoding="utf-8") as fjs:
        json.dump(
            {
                "data_dir": src,
                "episodes_total": len(files),
                "episodes_ok": ok_n,
                "episodes_skipped": skip_n,
                "frame_csv": os.path.abspath(frame_csv),
                "episodes": summaries,
            },
            fjs,
            indent=2,
            ensure_ascii=False,
        )

    print("\n[dataset analysis done]")
    print(f"  data_dir: {src}")
    print(f"  episodes: total={len(files)} ok={ok_n} skipped={skip_n}")
    if args.export_frame_csv:
        print(f"  frame-level output: {os.path.abspath(frame_csv)}")
    else:
        print("  frame-level output: (disabled)")
    print(f"  episode-level csv: {os.path.abspath(episode_csv)}")
    print(f"  episode-level output: {os.path.abspath(summary_json)}")
    if args.probe_pos_threshold is None or args.probe_rot_threshold is None:
        print("  note: is_stall 列为空（未提供完整 probe 阈值）")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode", default="", help="Path to one episode_XXXX.hdf5")
    p.add_argument("--data_dir", default="", help="Folder with episode_*.hdf5 for batch analysis")
    p.add_argument(
        "--output_dir",
        default="",
        help="Batch output directory (default: <data_dir>/analysis_action_norms)",
    )
    p.add_argument(
        "--export_frame_csv",
        action="store_true",
        help="Also export frame_norms.csv (one row per frame). Default: off",
    )
    p.add_argument("--bins", type=int, default=24, help="Histogram bins")
    p.add_argument("--no_hist", action="store_true", help="Disable ASCII histograms")
    p.add_argument("--probe_pos_threshold", type=float, default=None, help="Optional trial threshold for ||actions[:3]||")
    p.add_argument("--probe_rot_threshold", type=float, default=None, help="Optional trial threshold for ||actions[3:6]||")
    args = p.parse_args()

    if bool(args.episode) == bool(args.data_dir):
        print(
            "[ERROR] exactly one of --episode or --data_dir must be provided.",
            file=sys.stderr,
        )
        return 1

    if args.episode:
        return _analyze_single_episode(args.episode, args)

    out_dir = (
        os.path.abspath(args.output_dir)
        if args.output_dir
        else os.path.abspath(os.path.join(args.data_dir, "analysis_action_norms"))
    )
    return _analyze_dataset(args.data_dir, out_dir, args)


if __name__ == "__main__":
    raise SystemExit(main())
