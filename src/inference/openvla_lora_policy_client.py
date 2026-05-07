#!/usr/bin/env python3
"""
OpenVLA-7B + LoRA 策略端 — 无 Isaac，仅 GPU 推理；通过 TCP 接收仿真端 JPEG 并回传 7 维动作。

配对 `pouring_sim_policy_server.py` 时：服务端默认仅在策略步发送观测，本客户端在两次发送之间阻塞于 recv（正常）。相对笛卡尔环境下 SKIP 或等价路径应回传前 6 维为 0、第 7 维保持上一拍夹爪。

依赖（可用独立 conda/venv，不必与 Isaac 同环境）：
  pip install peft accelerate timm einops torch transformers pillow numpy opencv-python

示例：
    python scripts/inference/openvla_lora_policy_client.py \\
        --host 127.0.0.1 --port 9875 \\
        --base_checkpoint /path/to/openvla-7b \\
        --lora_path LoRA_train/runs/.../final_lora

若仿真端使用 pouring_sim_policy_server.py 的 --physical_cart_actions（前 6 维已为米/步与弧度/步），
请设 --cart_action_gain 1，避免再被放大并 clip 到 [-1,1]。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import socket
import sys

import cv2
import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.vla.openvla_lora_runtime import (
    apply_cart_action_gain,
    build_vicuna_prompt,
    check_openvla_runtime_dependencies,
    infer_action_vector,
    load_openvla_with_lora,
    resolve_base_checkpoint_path,
    resolve_dataset_stats_path,
)
from tools.vla.pouring_policy_wire import handshake_client, recv_obs_jpeg_or_skip, send_action7


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenVLA+LoRA 远程策略客户端")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=9875)
    p.add_argument(
        "--base_checkpoint",
        type=str,
        default="openvla/openvla-7b",
    )
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
        help="Processor/tokenizer 目录；默认与 --base_checkpoint 相同（勿用 LoRA 目录，易触发 tokenizers 解析错误）",
    )
    p.add_argument("--task", type=str, default="move the bottle above the cup and pour coke")
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--unnorm_key", type=str, default="pouring_hdf5")
    p.add_argument("--merge_lora", action="store_true")
    p.add_argument("--ema_alpha", type=float, default=0.0)
    p.add_argument(
        "--cart_action_gain",
        type=float,
        default=8.0,
        help="仅放大前 6 维（位姿）再 clip 到 [-1,1]；遥操归一化时常用。反归一化已是物理增量且仿真用 --physical_cart_actions 时请用 1",
    )
    p.add_argument(
        "--recv_timeout_s",
        type=float,
        default=0.0,
        help="等待下一帧观测的 socket 超时；0 表示阻塞",
    )
    p.add_argument(
        "--debug_jpeg_md5",
        action="store_true",
        help="每 30 帧统计行附带上一帧 JPEG 的 md5 前 12 位；若始终相同则服务端画面未更新或发同一缓冲",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    check_openvla_runtime_dependencies()

    dataset_stats_path = resolve_dataset_stats_path(args.dataset_stats, args.lora_path)
    base_ckpt = resolve_base_checkpoint_path(args.base_checkpoint, _REPO_ROOT)
    processor_path = args.processor_path or base_ckpt

    device = torch.device(args.device)
    use_bf16 = device.type == "cuda"
    dtype = torch.bfloat16 if use_bf16 else torch.float32

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
    print(f"[客户端] cart_action_gain={args.cart_action_gain}（前 6 维；夹爪不放大）")
    print(f"[客户端] 连接 {args.host}:{args.port} …")
    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if args.recv_timeout_s > 0:
        conn.settimeout(args.recv_timeout_s)
    try:
        conn.connect((args.host, args.port))
    except ConnectionRefusedError:
        print(
            "[错误] 无法连接仿真端（Connection refused）。请先在本机启动 TCP 服务端并等待其打印「已连接」前的监听状态，例如：\n"
            "  isaaclab.sh -p scripts/inference/pouring_sim_policy_server.py -- --port "
            f"{args.port}\n"
            "（需在 Isaac Lab 环境中执行；端口与 --host/--port 须与客户端一致。）",
            file=sys.stderr,
        )
        return 1
    handshake_client(conn)
    print("[客户端] 握手完成，进入推理循环（Ctrl+C 退出）")

    ema_prev: np.ndarray | None = None
    n = 0
    n_skip_win = 0
    n_vla_win = 0
    last_vla_action7: np.ndarray | None = None
    # 相对 IK 增量：SKIP 时前 6 维必须为 0，否则同一 delta 会在多仿真步上重复施加。夹爪为绝对映射，保持上一拍第 7 维。
    held_gripper = 0.0
    last_jpeg_md5: str = ""
    try:
        while True:
            jpeg = recv_obs_jpeg_or_skip(conn)
            if jpeg is None:
                n_skip_win += 1
                action_np = np.zeros(7, dtype=np.float32)
                action_np[6] = np.float32(held_gripper)
            else:
                n_vla_win += 1
                arr = np.frombuffer(jpeg, dtype=np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError("JPEG 解码失败")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                # --- 新增的 Debug 代码 ---
                if n < 5:  # 只保存前 5 帧看看
                    cv2.imwrite(f"debug_vla_obs_{n}.jpg", bgr)
                # -------------------------
                action_np, ema_prev = infer_action_vector(
                    model,
                    processor,
                    rgb,
                    prompt,
                    device,
                    dtype,
                    action_dim,
                    args.unnorm_key,
                    args.ema_alpha,
                    ema_prev,
                )
                action_np = apply_cart_action_gain(action_np, args.cart_action_gain)
                held_gripper = float(action_np[6])
                last_vla_action7 = action_np.copy()
                if args.debug_jpeg_md5 and jpeg:
                    last_jpeg_md5 = hashlib.md5(jpeg).hexdigest()[:12]
            n += 1
            if n % 30 == 0:
                vla_s = (
                    np.round(last_vla_action7, 3).tolist()
                    if last_vla_action7 is not None
                    else "—"
                )
                tail = "末帧SKIP→Δ6=0 保持夹爪" if jpeg is None else "末帧=VLA"
                dbg = f"  jpeg_md5[:12]={last_jpeg_md5}" if args.debug_jpeg_md5 and last_jpeg_md5 else ""
                print(
                    f"[客户端] 往返 {n}  近30帧 SKIP={n_skip_win} VLA={n_vla_win}  "
                    f"窗口内最近VLA输出(7)={vla_s}  末帧回传(7)={np.round(action_np, 3).tolist()} ({tail}){dbg}"
                )
                n_skip_win = 0
                n_vla_win = 0
                last_vla_action7 = None

            send_action7(conn, action_np.astype(np.float32).tobytes())
    except KeyboardInterrupt:
        print("[客户端] KeyboardInterrupt")
    except (ConnectionError, TimeoutError, OSError, ValueError) as exc:
        print(f"[客户端] 结束: {exc}")
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
