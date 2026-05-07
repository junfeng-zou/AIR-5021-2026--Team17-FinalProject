#!/usr/bin/env python3
"""
VLA Dataset Writer — OpenVLA-7B Compatible
============================================

Collects per-step data during teleoperation and saves each episode
as an HDF5 file for fine-tuning OpenVLA-7B.

HDF5 layout
------------
episode_XXXX.hdf5
├── observations/
│   ├── images/
│   │   ├── rgb          (T, H_lora, W_lora, 3)  uint8  — resized for OpenVLA / LoRA
│   │   ├── rgb_raw      (T, H, W, 3)  uint8    — optional, full-resolution RGB
│   │   ├── depth        (T, H, W, 1)  float32  — main camera depth (224² or full-res)
│   │   └── rgb_side     (T, H, W, 3)  uint8    — side camera (optional)
│   └── state            (T, D_state)  float32
├── actions               (T, 7)       float32   — [dx,dy,dz,droll,dpitch,dyaw,gripper]
├── actions_raw           (T, 7)       float32   — physical delta action (optional)
├── timestamps            (T,)         float64
└── attrs
    ├── num_steps         int
    ├── dt                float
    ├── task              str          — language instruction for VLA
    ├── action_names      list[str]
    ├── action_range      str
    └── model_target      str          — "OpenVLA-7B"
"""

from __future__ import annotations

import os
import time
from typing import Optional

import cv2
import h5py
import numpy as np


class EpisodeRecorder:
    """Buffer per-step data and flush to HDF5 when an episode ends."""

    def __init__(
        self,
        image_size: int = 224,
        task_description: str = "pour water from bottle to cup",
        action_range: str = "[-1, 1]",
    ):
        self.image_size = image_size
        self.task_description = task_description
        self.action_range = action_range
        self.reset()

    # ---- Public API ----------------------------------------------------------

    def reset(self):
        """Clear all buffers to start a new episode."""
        self._rgb: list[np.ndarray] = []
        self._rgb_raw: list[np.ndarray] = []
        self._depth: list[np.ndarray] = []
        self._rgb_side: list[np.ndarray] = []
        self._state: list[np.ndarray] = []
        self._action: list[np.ndarray] = []
        self._action_raw: list[np.ndarray] = []
        self._timestamps: list[float] = []
        self._phases: list[int] = []

    @property
    def num_steps(self) -> int:
        return len(self._timestamps)

    def add_step(
        self,
        rgb: np.ndarray,                        # (H, W, 3) uint8 — full-res when store_rgb_raw
        depth: np.ndarray,                      # (H, W) or (H, W, 1) float32
        state: np.ndarray,                      # (D_state,) float32
        action: np.ndarray,                     # (D_action,) float32
        action_raw: Optional[np.ndarray] = None,  # (D_action,) float32, optional physical action
        timestamp: Optional[float] = None,
        rgb_side: Optional[np.ndarray] = None,  # (H, W, 3) uint8, optional side camera
        *,
        store_rgb_raw: bool = False,
        store_full_depth: bool = False,
        phase: Optional[int] = None,
    ):
        """Add one time-step of data to the episode buffer."""
        rgb_u8 = np.asarray(rgb, dtype=np.uint8)
        if store_rgb_raw:
            self._rgb_raw.append(rgb_u8.copy())

        rgb_resized = cv2.resize(rgb_u8, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        self._rgb.append(rgb_resized)

        depth_arr = np.asarray(depth, dtype=np.float32)
        if depth_arr.ndim == 2:
            depth_arr = depth_arr[:, :, np.newaxis]

        if store_full_depth:
            self._depth.append(depth_arr.copy())
        else:
            depth_resized = cv2.resize(
                depth_arr.squeeze(-1), (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST
            )
            self._depth.append(depth_resized[:, :, np.newaxis])

        if rgb_side is not None:
            side_resized = cv2.resize(rgb_side, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
            self._rgb_side.append(side_resized)

        self._state.append(state.astype(np.float32))
        self._action.append(action.astype(np.float32))
        if action_raw is not None:
            self._action_raw.append(action_raw.astype(np.float32))
        self._timestamps.append(timestamp if timestamp is not None else time.time())
        if phase is not None:
            self._phases.append(phase)

    def save(self, filepath: str, dt: float = 0.1) -> str:
        """
        Write the buffered episode to an HDF5 file.

        Args:
            filepath: Full path to the output .hdf5 file.
            dt: Simulation time-step between recorded frames.

        Returns:
            The filepath that was written.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with h5py.File(filepath, "w") as f:
            T = self.num_steps

            obs_grp = f.create_group("observations")
            img_grp = obs_grp.create_group("images")
            img_grp.create_dataset(
                "rgb",
                data=np.stack(self._rgb, axis=0),
                compression="gzip",
                compression_opts=4,
            )
            if self._rgb_raw:
                if len(self._rgb_raw) != T:
                    raise ValueError(f"rgb_raw length {len(self._rgb_raw)} != num_steps {T}")
                img_grp.create_dataset(
                    "rgb_raw",
                    data=np.stack(self._rgb_raw, axis=0),
                    compression="gzip",
                    compression_opts=4,
                )
                r0 = self._rgb_raw[0]
                f.attrs["rgb_raw_height"] = int(r0.shape[0])
                f.attrs["rgb_raw_width"] = int(r0.shape[1])

            img_grp.create_dataset(
                "depth",
                data=np.stack(self._depth, axis=0).astype(np.float32),
                compression="gzip",
                compression_opts=4,
            )
            d0 = self._depth[0]
            f.attrs["depth_height"] = int(d0.shape[0])
            f.attrs["depth_width"] = int(d0.shape[1])

            if self._rgb_side and len(self._rgb_side) == T:
                img_grp.create_dataset(
                    "rgb_side",
                    data=np.stack(self._rgb_side, axis=0),
                    compression="gzip",
                    compression_opts=4,
                )

            obs_grp.create_dataset(
                "state",
                data=np.stack(self._state, axis=0),
                compression="gzip",
            )

            f.create_dataset(
                "actions",
                data=np.stack(self._action, axis=0),
                compression="gzip",
            )
            if self._action_raw and len(self._action_raw) == T:
                f.create_dataset(
                    "actions_raw",
                    data=np.stack(self._action_raw, axis=0),
                    compression="gzip",
                )

            f.create_dataset("timestamps", data=np.array(self._timestamps, dtype=np.float64))

            # -- phase labels (per-step) --
            if self._phases and len(self._phases) == T:
                f.create_dataset(
                    "phase_labels",
                    data=np.array(self._phases, dtype=np.int32),
                )
                f.attrs["num_phases"] = int(max(self._phases)) + 1

            # -- metadata (OpenVLA-7B compatible) --
            f.attrs["num_steps"] = T
            f.attrs["dt"] = dt
            f.attrs["task"] = self.task_description
            f.attrs["image_size"] = self.image_size
            f.attrs["lora_image_size"] = self.image_size
            f.attrs["model_target"] = "OpenVLA-7B"
            f.attrs["action_dim"] = 7
            f.attrs["action_names"] = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
            f.attrs["action_range"] = self.action_range
            f.attrs["robot"] = "DOBOT CR5"

        return filepath


class StreamingEpisodeRecorder:
    """
    边采边写入 HDF5（可扩展数据集），单条 episode 再长也不占满内存。
    布局与 EpisodeRecorder.save 一致；写入阶段不使用 gzip（追加快），文件体积略大。
    """

    def __init__(
        self,
        filepath: str,
        image_size: int = 224,
        task_description: str = "pour water from bottle to cup",
        action_range: str = "[-1, 1]",
    ):
        self._path = filepath
        self.image_size = image_size
        self.task_description = task_description
        self.action_range = action_range
        self._f: h5py.File | None = None
        self._img_grp = None
        self._obs_grp = None
        self._d_rgb = None
        self._d_rgb_raw = None
        self._d_depth = None
        self._d_state = None
        self._d_actions = None
        self._d_ts = None
        self._T = 0
        self._store_rgb_raw = False
        self._depth_hw: tuple[int, int] | None = None
        self._state_dim: int | None = None
        self._phases: list[int] = []

    @property
    def num_steps(self) -> int:
        return self._T

    @property
    def output_path(self) -> str:
        return self._path

    def _lazy_create(
        self,
        rgb_small: np.ndarray,
        rgb_full: np.ndarray | None,
        depth_hwc: np.ndarray,
        state_dim: int,
    ) -> None:
        if self._f is not None:
            return
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        self._f = h5py.File(self._path, "w")
        f = self._f
        self._obs_grp = f.create_group("observations")
        self._img_grp = self._obs_grp.create_group("images")

        self._d_rgb = self._img_grp.create_dataset(
            "rgb",
            shape=(0, self.image_size, self.image_size, 3),
            maxshape=(None, self.image_size, self.image_size, 3),
            dtype=np.uint8,
            chunks=(1, self.image_size, self.image_size, 3),
        )
        if rgb_full is not None:
            h, w = int(rgb_full.shape[0]), int(rgb_full.shape[1])
            self._store_rgb_raw = True
            self._d_rgb_raw = self._img_grp.create_dataset(
                "rgb_raw",
                shape=(0, h, w, 3),
                maxshape=(None, h, w, 3),
                dtype=np.uint8,
                chunks=(1, h, w, 3),
            )
            f.attrs["rgb_raw_height"] = h
            f.attrs["rgb_raw_width"] = w

        dh, dw, dc = depth_hwc.shape[0], depth_hwc.shape[1], depth_hwc.shape[2]
        self._depth_hw = (dh, dw)
        self._d_depth = self._img_grp.create_dataset(
            "depth",
            shape=(0, dh, dw, dc),
            maxshape=(None, dh, dw, dc),
            dtype=np.float32,
            chunks=(1, dh, dw, dc),
        )
        f.attrs["depth_height"] = dh
        f.attrs["depth_width"] = dw

        sd = int(state_dim)
        self._state_dim = sd
        self._d_state = self._obs_grp.create_dataset(
            "state",
            shape=(0, sd),
            maxshape=(None, sd),
            dtype=np.float32,
            chunks=(1, sd),
        )
        self._d_actions = f.create_dataset(
            "actions",
            shape=(0, 7),
            maxshape=(None, 7),
            dtype=np.float32,
            chunks=(1, 7),
        )
        self._d_ts = f.create_dataset(
            "timestamps",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64,
            chunks=(1024,),
        )

    def add_step(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        timestamp: Optional[float] = None,
        action_raw: Optional[np.ndarray] = None,
        rgb_side: Optional[np.ndarray] = None,
        *,
        store_rgb_raw: bool = False,
        store_full_depth: bool = False,
        phase: Optional[int] = None,
    ) -> None:
        if action_raw is not None or rgb_side is not None:
            raise NotImplementedError("StreamingEpisodeRecorder 暂不支持 action_raw / rgb_side")
        rgb_u8 = np.asarray(rgb, dtype=np.uint8)
        rgb_small = cv2.resize(rgb_u8, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        depth_arr = np.asarray(depth, dtype=np.float32)
        if depth_arr.ndim == 2:
            depth_arr = depth_arr[:, :, np.newaxis]
        if store_full_depth:
            depth_hwc = depth_arr.copy()
        else:
            rz = cv2.resize(depth_arr.squeeze(-1), (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            depth_hwc = rz[:, :, np.newaxis]

        st = state.astype(np.float32)
        ac = action.astype(np.float32)
        ts = float(timestamp if timestamp is not None else time.time())

        raw_for_init = rgb_u8.copy() if store_rgb_raw else None
        self._lazy_create(rgb_small, raw_for_init, depth_hwc, st.shape[0])

        if store_rgb_raw and self._d_rgb_raw is None:
            raise RuntimeError("首帧未开启 store_rgb_raw，后续不能开启")
        if store_rgb_raw:
            if rgb_u8.shape != (self._d_rgb_raw.shape[1], self._d_rgb_raw.shape[2], 3):
                raise ValueError(f"rgb_raw 形状不一致: got {rgb_u8.shape}, expect H,W,3 与首帧相同")

        new_t = self._T + 1
        self._d_rgb.resize((new_t, self.image_size, self.image_size, 3))
        self._d_rgb[new_t - 1] = rgb_small

        if self._d_rgb_raw is not None:
            self._d_rgb_raw.resize((new_t,) + self._d_rgb_raw.shape[1:])
            self._d_rgb_raw[new_t - 1] = rgb_u8

        self._d_depth.resize((new_t,) + self._d_depth.shape[1:])
        self._d_depth[new_t - 1] = depth_hwc.astype(np.float32, copy=False)

        self._d_state.resize((new_t, self._state_dim))
        self._d_state[new_t - 1] = st

        self._d_actions.resize((new_t, 7))
        self._d_actions[new_t - 1] = ac

        self._d_ts.resize((new_t,))
        self._d_ts[new_t - 1] = ts

        if phase is not None:
            self._phases.append(phase)

        self._T = new_t

    def finalize(self, dt: float = 0.1) -> str:
        """写入 attrs / language_instruction 并关闭文件。"""
        if self._f is None:
            return self._path
        f = self._f
        T = self._T
        str_dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("language_instruction", data=np.array(self.task_description, dtype=str_dt))

        # -- phase labels (per-step) --
        if self._phases and len(self._phases) == T:
            f.create_dataset(
                "phase_labels",
                data=np.array(self._phases, dtype=np.int32),
            )
            f.attrs["num_phases"] = int(max(self._phases)) + 1

        f.attrs["num_steps"] = T
        f.attrs["dt"] = dt
        f.attrs["task"] = self.task_description
        f.attrs["image_size"] = self.image_size
        f.attrs["lora_image_size"] = self.image_size
        f.attrs["model_target"] = "OpenVLA-7B"
        f.attrs["action_dim"] = 7
        f.attrs["action_names"] = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
        f.attrs["action_range"] = self.action_range
        f.attrs["robot"] = "DOBOT CR5"
        f.close()
        self._f = None
        self._d_rgb = None
        return self._path

    def abort(self) -> None:
        """丢弃本段录制：关闭并删除未完成的 HDF5。"""
        path = self._path
        if self._f is not None:
            try:
                self._f.close()
            except Exception:
                pass
            self._f = None
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass


def next_episode_path(save_dir: str) -> str:
    """Return the path for the next episode file, e.g. episode_0003.hdf5.

    Uses max existing index + 1 to avoid overwriting after deletions.
    """
    os.makedirs(save_dir, exist_ok=True)
    max_idx = -1
    for f in os.listdir(save_dir):
        if f.startswith("episode_") and f.endswith(".hdf5"):
            try:
                idx = int(f[len("episode_"):-len(".hdf5")])
                max_idx = max(max_idx, idx)
            except ValueError:
                continue
    return os.path.join(save_dir, f"episode_{max_idx + 1:04d}.hdf5")
