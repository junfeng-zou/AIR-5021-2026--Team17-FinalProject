#!/usr/bin/env python3
"""
OpenVLA-7B + LoRA — Isaac Lab PouringEnv 单进程闭环（仿真 + 推理同一 Python）。

若希望仿真与推理 **分进程 / 分环境**（推荐），请改用：
  • `scripts/inference/pouring_sim_policy_server.py` — 仅 Isaac
  • `scripts/inference/openvla_lora_policy_client.py` — 仅 OpenVLA+LoRA

共享推理逻辑见 `tools/vla/openvla_lora_runtime.py`。

依赖（isaaclab 环境）：
  pip install peft accelerate timm einops
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.vla.openvla_lora_runtime import (
    apply_cart_action_gain,
    build_vicuna_prompt,
    check_openvla_runtime_dependencies,
    ensure_rgb_uint8_hwc,
    infer_action_vector,
    load_openvla_with_lora,
    resolve_base_checkpoint_path,
    resolve_dataset_stats_path_from_args,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenVLA-7B + LoRA 在 PouringEnv 中推理控制（单进程）")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--base_checkpoint", type=str, default="openvla/openvla-7b")
    p.add_argument(
        "--lora_path",
        type=str,
        default=os.path.join(_REPO_ROOT, "LoRA_train", "runs", "pouring_lora_a10080", "final_lora"),
    )
    p.add_argument("--dataset_stats", type=str, default="")
    p.add_argument(
        "--processor_path",
        type=str,
        default="",
        help="Processor/tokenizer 目录；默认与 --base_checkpoint 相同",
    )
    p.add_argument("--task", type=str, default="pour cola from bottle into cup")
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--warmup_steps", type=int, default=5)
    p.add_argument("--unnorm_key", type=str, default="pouring_hdf5")
    p.add_argument("--merge_lora", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--ema_alpha", type=float, default=0.0)
    p.add_argument(
        "--cart_action_gain",
        type=float,
        default=8.0,
        help="仅放大前 6 维笛卡尔命令；与分进程客户端 --cart_action_gain 含义相同",
    )
    p.add_argument(
        "--show_ee_frame",
        action="store_true",
        help="在末端 link_6 显示坐标系：X 红、Y 绿、Z 蓝（USD 箭头）",
    )
    p.add_argument(
        "--physical_cart_actions",
        action="store_true",
        help="前 6 维为米/步与弧度/步（反归一化物理增量），PouringEnv 不再乘 pos/rot scale；夹爪仍 [-1,1]",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dataset_stats_path = resolve_dataset_stats_path_from_args(args)
    base_ckpt = resolve_base_checkpoint_path(args.base_checkpoint, _REPO_ROOT)
    processor_path = args.processor_path or base_ckpt

    device = torch.device(args.device)
    use_bf16 = device.type == "cuda"
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    check_openvla_runtime_dependencies()

    from isaacsim import SimulationApp

    sim_app = SimulationApp(
        {
            "headless": args.headless,
            "width": 1280,
            "height": 720,
            "enable_cameras": True,
        }
    )

    import carb

    carb.settings.get_settings().set_bool("/isaaclab/cameras_enabled", True)

    import cv2

    from envs.pouring_env import PouringEnv, PouringEnvCfg

    cfg = PouringEnvCfg()
    cfg.scene.num_envs = 1
    if args.show_ee_frame:
        cfg.debug_visualize_ee_frame = True
    if args.physical_cart_actions:
        cfg.cart_action_is_physical_delta = True
    env = PouringEnv(cfg)

    model, processor, action_dim = load_openvla_with_lora(
        base_checkpoint=base_ckpt,
        lora_path=os.path.abspath(args.lora_path),
        processor_path=processor_path,
        dataset_stats_path=dataset_stats_path,
        device=device,
        dtype=dtype,
        unnorm_key=args.unnorm_key,
        merge_lora=args.merge_lora,
    )

    prompt = build_vicuna_prompt(args.task)
    print("=" * 64)
    print("OpenVLA-7B + LoRA — PouringEnv 推理（单进程）")
    print("=" * 64)
    print(f"  dataset_stats : {dataset_stats_path}")
    print(f"  control_hz    : {args.hz}")
    print(f"  cart_action_gain: {args.cart_action_gain}")
    print(f"  dry_run       : {args.dry_run}")
    print("=" * 64)

    sim_dt = cfg.sim.dt * cfg.decimation
    control_interval = max(1, int(round(1.0 / (args.hz * sim_dt))))
    print(f"  sim_dt={sim_dt:.6f}s  → 每 {control_interval} 个 sim step 推理一次")

    obs, _ = env.reset()
    step = 0
    steps_since_reset = 0
    ema_prev: Optional[np.ndarray] = None

    win = "OpenVLA LoRA — main camera"
    if not args.headless:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        while True:
            if args.max_steps > 0 and step >= args.max_steps:
                print("达到 max_steps，退出。")
                break

            run_policy = steps_since_reset > args.warmup_steps and (step % control_interval == 0)

            rgb_np: Optional[np.ndarray] = None
            cam_data = env._camera.data
            if "rgb" in cam_data.output and cam_data.output["rgb"].numel() > 0:
                rgb = cam_data.output["rgb"][0].cpu().numpy()
                if rgb.shape[-1] == 4:
                    rgb = rgb[:, :, :3]
                rgb_np = rgb

            action_np = np.zeros(7, dtype=np.float32)

            if (not args.dry_run) and run_policy and rgb_np is not None:
                action_np, ema_out = infer_action_vector(
                    model,
                    processor,
                    rgb_np,
                    prompt,
                    device,
                    dtype,
                    action_dim,
                    args.unnorm_key,
                    args.ema_alpha,
                    ema_prev,
                )
                action_np = apply_cart_action_gain(action_np, args.cart_action_gain)
                if ema_out is not None:
                    ema_prev = ema_out

            if args.dry_run:
                action_np = np.zeros(7, dtype=np.float32)

            action_t = torch.tensor(action_np, device=env.device, dtype=torch.float32).unsqueeze(0)
            obs, _r, terminated, truncated, _info = env.step(action_t)
            step += 1
            steps_since_reset += 1
            if bool(terminated[0]) or bool(truncated[0]):
                obs, _ = env.reset()
                steps_since_reset = 0
                ema_prev = None

            if not args.headless and rgb_np is not None:
                bgr = cv2.cvtColor(ensure_rgb_uint8_hwc(rgb_np), cv2.COLOR_RGB2BGR)
                cv2.imshow(win, bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("按 q 退出。")
                    break

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("KeyboardInterrupt，退出。")
    finally:
        if not args.headless:
            cv2.destroyAllWindows()
        sim_app.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
