"""
Map-style PyTorch dataset: pouring_VLA HDF5 episodes -> OpenVLA training tensors.
"""

from __future__ import annotations

import bisect
import glob
import json
import os
from typing import Any, Callable, Mapping, Type

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from action_tokenizer import ActionTokenizer
from collator import IGNORE_INDEX
from prompting_vicuna import PromptBuilder, VicunaV15ChatPromptBuilder


def _decode_task(task_attr: Any) -> str:
    if isinstance(task_attr, bytes):
        return task_attr.decode("utf-8")
    if isinstance(task_attr, np.ndarray) and task_attr.dtype.type is np.str_:
        return str(task_attr.item())
    return str(task_attr)


def _list_hdf5_files(data_dir: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(data_dir, "episode_*.hdf5")))
    paths.extend(sorted(glob.glob(os.path.join(data_dir, "*.hdf5"))))
    # De-dup (episode_*.hdf5 matched twice)
    return sorted(set(paths))


class PouringHDF5VLADataset(Dataset):
    """
    One sample = one (image, language, 7-DoF action) transition from HDF5.

    Matches OpenVLA / RLDSBatchTransform label masking: only action (+ stop) tokens contribute loss.
    """

    def __init__(
        self,
        data_dir: str,
        action_tokenizer: ActionTokenizer,
        base_tokenizer,
        image_transform: Callable,
        prompt_builder_fn: Type[PromptBuilder] = VicunaV15ChatPromptBuilder,
        predict_stop_token: bool = True,
        action_dim: int = 7,
        action_source: str = "actions",
        episode_weights: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.data_dir = os.path.abspath(data_dir)
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.predict_stop_token = predict_stop_token
        self.action_dim = action_dim
        if action_source not in ("actions", "actions_raw"):
            raise ValueError(f"action_source must be 'actions' or 'actions_raw', got {action_source!r}")
        self.action_source = action_source

        files = _list_hdf5_files(self.data_dir)
        if not files:
            raise FileNotFoundError(f"No episode_*.hdf5 under {self.data_dir}")

        self._episodes: list[tuple[str, int, str]] = []
        self._episode_weights: list[float] = []
        for fp in files:
            with h5py.File(fp, "r") as f:
                if self.action_source not in f:
                    raise ValueError(f"{fp}: missing dataset '{self.action_source}'")
                actions = f[self.action_source]
                if actions.shape[-1] != action_dim:
                    raise ValueError(
                        f"{fp}: expected {self.action_source}[..., {action_dim}], got {actions.shape}"
                    )
                t = int(actions.shape[0])
                if t < 1:
                    continue
                lang = _decode_task(f.attrs["task"])
            self._episodes.append((fp, t, lang))
            ep_name = os.path.basename(fp)
            w = 1.0
            if episode_weights is not None:
                w = float(
                    episode_weights.get(ep_name, episode_weights.get(fp, episode_weights.get(os.path.abspath(fp), 1.0)))
                )
            self._episode_weights.append(max(1e-8, w))

        if not self._episodes:
            raise RuntimeError("No non-empty episodes found.")

        self._cum: list[int] = [0]
        for _, t, _ in self._episodes:
            self._cum.append(self._cum[-1] + t)

        self.dataset_statistics = self._compute_action_stats()

        # 预计算归一化数组：__getitem__ 中把 raw action 归一化到 [-1, 1]
        # 使得 ActionTokenizer 的 256 bin 能充分覆盖有效动态范围。
        # 实际使用 q05/q95（而非 q01/q99），改善 dz/dpitch 的中心集中问题。
        # 推理端 unnormalize_action_q01q99() 用同一份统计量做反操作。
        _stats = self.dataset_statistics["pouring_hdf5"]["action"]
        self._q01 = np.array(_stats["q01"], dtype=np.float64)
        self._q99 = np.array(_stats["q99"], dtype=np.float64)
        self._q_range = self._q99 - self._q01
        # 若某一维 q_low≈q_high（常量维度），保持原值不做归一化
        self._q_range = np.where(np.abs(self._q_range) < 1e-8, 1.0, self._q_range)

    def _compute_action_stats(self) -> dict:
        """OpenVLA-style stats blob for downstream denormalization (q05/q95 per dim).

        使用 q05/q95 而非 q01/q99：dz 和 dpitch 在 q01/q99 下仍有 43%/75% 的帧
        落在中心 2 bin，导致模型倾向输出零动作。q05/q95 将中心集中度降至 ~5%。
        JSON key 仍沿用 'q01'/'q99' 以兼容推理端 unnormalize 函数。
        """
        actions_all: list[np.ndarray] = []
        for fp, t, _ in self._episodes:
            with h5py.File(fp, "r") as f:
                actions_all.append(np.asarray(f[self.action_source][:], dtype=np.float32))
        stacked = np.concatenate(actions_all, axis=0)
        q_low = np.quantile(stacked, 0.05, axis=0).astype(np.float32)
        q_high = np.quantile(stacked, 0.95, axis=0).astype(np.float32)
        return {"pouring_hdf5": {"action": {"q01": q_low.tolist(), "q99": q_high.tolist()}}}

    def __len__(self) -> int:
        return self._cum[-1]

    def _locate(self, idx: int) -> tuple[str, int, str]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        ep = bisect.bisect_right(self._cum, idx) - 1
        t = idx - self._cum[ep]
        path, T, lang = self._episodes[ep]
        assert 0 <= t < T
        return path, t, lang

    def index_weight(self, idx: int) -> float:
        """Sample weight for one flattened frame index."""
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        ep = bisect.bisect_right(self._cum, idx) - 1
        return float(self._episode_weights[ep])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path, t, default_lang = self._locate(idx)

        with h5py.File(path, "r") as f:
            rgb = np.asarray(f["observations/images/rgb"][t], dtype=np.uint8)
            action_raw = np.asarray(f[self.action_source][t], dtype=np.float32)
            # 优先使用每帧的 phase_task（三阶段子任务），否则回退到 episode 级 task
            if "phase_task" in f:
                lang = _decode_task(f["phase_task"][t])
            else:
                lang = _decode_task(f.attrs.get("task", default_lang))

        if action_raw.shape != (self.action_dim,):
            raise RuntimeError(f"Bad action shape {action_raw.shape} in {path} step {t}")

        # q01/q99 归一化到 [-1, 1]，使得 256 bin 离散化能覆盖有效动态范围。
        # 推理端 unnormalize_action_q01q99() 用相同 q01/q99 做反操作，恢复物理量。
        action = (2.0 * (action_raw.astype(np.float64) - self._q01)
                  / self._q_range - 1.0).astype(np.float32)

        img = Image.fromarray(rgb)
        pixel_values = self.image_transform(img)

        lang_l = lang.lower()
        tok = self.base_tokenizer
        at = self.action_tokenizer

        # 1) Text through ASSISTANT: only (no decode→encode round-trip for actions).
        prompt_builder = self.prompt_builder_fn("openvla")
        human_msg = f"<image>\nWhat action should the robot take to {lang_l}?"
        prompt_builder.add_turn("human", human_msg)
        text_prompt = prompt_builder.get_prompt()

        input_ids = list(tok(text_prompt, add_special_tokens=True).input_ids)

        # 2) Discrete action token IDs (same binning as ActionTokenizer, ids at vocab tail).
        action_clip = np.clip(
            action.astype(np.float64),
            float(at.min_action),
            float(at.max_action),
        )
        discretized = np.digitize(action_clip, at.bins)
        action_token_ids = (tok.vocab_size - discretized).astype(np.int64).tolist()
        assert len(action_token_ids) == self.action_dim

        input_ids.extend(action_token_ids)
        eos_id = tok.eos_token_id
        if eos_id is None:
            eos_id = getattr(tok, "pad_token_id", None) or 2
        input_ids.append(int(eos_id))

        # 3) Labels: supervise action tokens + EOS only.
        labels = [IGNORE_INDEX] * len(input_ids)
        n_sup = len(action_token_ids) + 1
        labels[-n_sup:] = input_ids[-n_sup:]
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def save_dataset_statistics(self, run_dir: str) -> None:
        os.makedirs(run_dir, exist_ok=True)
        out = os.path.join(run_dir, "dataset_statistics.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self.dataset_statistics, f, indent=2)
