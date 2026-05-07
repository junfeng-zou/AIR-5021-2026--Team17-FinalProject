"""
RLDS Dataset Builder for DOBOT CR5 Pouring Task.

Converts HDF5 episodes to RLDS format for OpenVLA LoRA fine-tuning.
Still frames are removed before conversion.

Dataset: DOBOT CR5 water pouring demonstrations
- 224x224 RGB images
- 7-DoF EEF delta actions [dx, dy, dz, droll, dpitch, dyaw, gripper]
- Gripper: 1 = open, 0 = close (mapped from +1/-1 in HDF5)
"""

from typing import Any, Iterator, Tuple

import glob
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


# ── Still frame removal thresholds ──
POS_THR = 0.0015    # 1.5mm — position L2 norm threshold
ROT_THR = 0.008     # 0.008 rad — rotation L2 norm threshold
GRIPPER_PROTECT = 0  # Only the exact gripper-switch frame is protected; all adjacent still frames are filtered

# ── HDF5 data source ──
HDF5_DATA_DIR = "/home/zjf/pouring_VLA/data/vla_dataset_overfit"

# ── Unified language instruction ──
LANGUAGE_INSTRUCTION = "pour cola into cup"


def _compute_keep_mask(actions: np.ndarray) -> np.ndarray:
    """
    Compute a boolean mask for active (non-still) frames.

    A frame is considered still if ALL of the following are true:
      1. Position delta L2 norm < POS_THR (1.5mm)
      2. Rotation delta L2 norm < ROT_THR (0.008 rad)
      3. Not within ±GRIPPER_PROTECT frames of a gripper switch event

    Returns: bool array of shape (T,), True = keep (active), False = remove (still)
    """
    n = len(actions)
    gripper = actions[:, 6]
    pos_norm = np.linalg.norm(actions[:, 0:3], axis=1)
    rot_norm = np.linalg.norm(actions[:, 3:6], axis=1)

    # Basic still condition
    still = (pos_norm < POS_THR) & (rot_norm < ROT_THR)

    # Protect frames near gripper switch events
    protected = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if (gripper[i - 1] > 0 and gripper[i] < 0) or \
           (gripper[i - 1] < 0 and gripper[i] > 0):
            lo = max(0, i - GRIPPER_PROTECT)
            hi = min(n, i + GRIPPER_PROTECT + 1)
            protected[lo:hi] = True

    # Final: still AND not protected → remove
    still_final = still & ~protected

    return ~still_final  # True = keep


def _map_gripper(gripper_val: np.ndarray) -> np.ndarray:
    """Map gripper from HDF5 convention (+1=open, -1=close) to OXE convention, discretized to {0.0, 1.0}.

    Steps:
      1. Map [-1, 1] → [0, 1]:  continuous = (val + 1) / 2
      2. Discretize:             1.0 if continuous >= 0.5 else 0.0
    This eliminates the soft-transition interpolation values (~0.3s window) produced by
    GripperController.get_state() during actuation, keeping gripper action strictly binary.
    Accepts both scalar float and numpy array.
    """
    continuous = (np.asarray(gripper_val, dtype=np.float32) + 1.0) / 2.0
    return np.where(continuous >= 0.5, 1.0, 0.0).astype(np.float32)


class DobotPouring(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for DOBOT CR5 pouring dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "88 episodes with still frame removal, unified prompt.",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # No USE language embedding model needed — we fill zeros

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=(224, 224, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="Main camera RGB observation.",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(7,),
                                        dtype=np.float32,
                                        doc="Robot state: [EE_x, EE_y, EE_z, EE_rx, EE_ry, EE_rz, gripper_state]. "
                                        "Gripper: 1=open, 0=close.",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(7,),
                                dtype=np.float32,
                                doc="Robot action: [dx, dy, dz, droll, dpitch, dyaw, gripper]. "
                                "Gripper: 1=open, 0=close.",
                            ),
                            "discount": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Discount if provided, default to 1.",
                            ),
                            "reward": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Reward if provided, 1 on final step for demos.",
                            ),
                            "is_first": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on first step of the episode.",
                            ),
                            "is_last": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step of the episode.",
                            ),
                            "is_terminal": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step of the episode if it is a terminal step, True for demos.",
                            ),
                            "language_instruction": tfds.features.Text(
                                doc="Language instruction for this step.",
                            ),
                            "language_embedding": tfds.features.Tensor(
                                shape=(512,),
                                dtype=np.float32,
                                doc="Kona language embedding. Filled with zeros (not used by OpenVLA).",
                            ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(
                                doc="Path to the original HDF5 file.",
                            ),
                        }
                    ),
                }
            )
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits — train only, OpenVLA auto-splits 95%/5%."""
        return {
            "train": self._generate_examples(
                path=os.path.join(HDF5_DATA_DIR, "episode_*.hdf5")
            ),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""
        import h5py

        episode_paths = sorted(glob.glob(path))
        included_count = 0
        total_orig = 0
        total_kept = 0
        zero_embedding = np.zeros(512, dtype=np.float32)

        for episode_path in episode_paths:
            with h5py.File(episode_path, "r") as f:
                actions = np.array(f["actions"], dtype=np.float32)  # (T, 7)
                state_full = np.array(
                    f["observations/state"], dtype=np.float32
                )  # (T, 19)
                images = np.array(f["observations/images/rgb"])  # (T, 224, 224, 3)

            T_orig = actions.shape[0]

            # ── Remove still frames ──
            keep_mask = _compute_keep_mask(actions)
            actions = actions[keep_mask]
            state_full = state_full[keep_mask]
            images = images[keep_mask]
            T = actions.shape[0]

            total_orig += T_orig
            total_kept += T

            # ── Build state: EE pose (6) + gripper (1) ──
            ee_state = state_full[:, 13:19]  # (T, 6) = xyz + rpy
            gripper_obs = state_full[:, 12:13]  # (T, 1) = gripper
            # Map gripper observation: +1=open → 1, -1=close → 0
            gripper_obs_mapped = (gripper_obs + 1.0) / 2.0
            state = np.concatenate([ee_state, gripper_obs_mapped], axis=-1).astype(
                np.float32
            )  # (T, 7)

            # ── Map gripper in actions: +1=open → 1, -1=close → 0 ──
            actions_mapped = actions.copy()
            actions_mapped[:, 6] = _map_gripper(actions[:, 6])

            # ── Construct episode steps ──
            episode = []
            for i in range(T):
                episode.append(
                    {
                        "observation": {
                            "image": images[i],
                            "state": state[i],
                        },
                        "action": actions_mapped[i],
                        "discount": 1.0,
                        "reward": float(i == T - 1),
                        "is_first": i == 0,
                        "is_last": i == T - 1,
                        "is_terminal": i == T - 1,
                        "language_instruction": LANGUAGE_INSTRUCTION,
                        "language_embedding": zero_embedding,
                    }
                )

            sample = {
                "steps": episode,
                "episode_metadata": {
                    "file_path": episode_path,
                },
            }

            included_count += 1
            yield episode_path, sample

        removed = total_orig - total_kept
        pct = removed / total_orig * 100 if total_orig > 0 else 0
        print(f"\n[DobotPouring] Included {included_count} episodes.")
        print(f"[DobotPouring] Still frame removal: {total_orig} → {total_kept} "
              f"(-{removed}, {pct:.1f}% removed)")
