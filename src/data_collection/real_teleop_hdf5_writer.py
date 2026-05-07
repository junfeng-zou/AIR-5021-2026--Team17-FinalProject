#!/usr/bin/env python3
"""
DOBOT CR5 真机遥操作 — HDF5 录制端（Linux PC）

功能：
  - 经 ZMQ 接收机械臂 PC 发来的观测 + 动作
  - 解析 RGB（PNG/JPEG）+ Depth + State + Action
  - 写入 HDF5 episodes

启动（Linux PC）：
    python scripts/collect/real_teleop_hdf5_writer.py --port 9875 --save_dir data/vla_dataset_real

协议（ZMQ multipart）：
  命令： [b"CMD", json_utf8]，如 {"cmd":"start"}；start 可带 "task":"英文" 写入 HDF5（OpenVLA LoRA）
  观测： [b"OBS", header_json, rgb_bytes, depth_f32, state_f32, action_f32, ts_f64_be]

训练时 task 会嵌入：What action should the robot take to {task}?（见 LoRA_train/code/dataset_hdf5.py）
"""

from __future__ import annotations

import argparse
import json
import os
import random
import struct
import sys
import time

import cv2
import numpy as np

try:
    import zmq  # type: ignore[import-not-found]
    HAS_ZMQ = True
except ImportError:
    HAS_ZMQ = False
    zmq = None

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from tools.vla.vla_data_writer import EpisodeRecorder, StreamingEpisodeRecorder, next_episode_path


def decode_rgb_from_payload(header: dict | None, rgb_bytes: bytes) -> np.ndarray | None:
    """按发送端 rgb_encoding 解码 PNG/JPEG，得到 RGB uint8 (H,W,3)。失败返回 None。"""
    if not rgb_bytes:
        return None
    hdr = header or {}
    enc = hdr.get("rgb_encoding", "jpeg")
    if enc in ("none", "", None):
        return None
    if enc in ("jpeg", "jpg", "png"):
        bgr = cv2.imdecode(np.frombuffer(rgb_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    bgr = cv2.imdecode(np.frombuffer(rgb_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def parse_args():
    p = argparse.ArgumentParser(description="DOBOT CR5 HDF5 录制端（Linux PC，ZMQ）")
    p.add_argument("--port", type=int, default=9875, help="ZMQ PULL 监听端口")
    p.add_argument("--bind", type=str, default="0.0.0.0", help="监听地址")
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument(
        "--rgb_raw_width",
        type=int,
        default=None,
        help="保存 observations/images/rgb_raw 的宽；不填则保持发送端原始宽度",
    )
    p.add_argument(
        "--rgb_raw_height",
        type=int,
        default=None,
        help="保存 observations/images/rgb_raw 的高；不填则保持发送端原始高度",
    )
    p.add_argument(
        "--task",
        type=str,
        default="pour water into cup",
        help="OpenVLA 用：填入 'What action should the robot take to {task}?' 的 task 子句（英文）",
    )
    p.add_argument(
        "--task_mode",
        type=str,
        default="fixed",
        choices=["fixed", "augment"],
        help="fixed=每段 episode 均用同一 task（或与 start 里带的 task）；augment=从同义改写池中随机（仅增强数据）",
    )
    p.add_argument("--min_steps", type=int, default=10)
    p.add_argument("--timeout", type=float, default=5.0, help="录制中无观测帧超时秒数")
    p.add_argument(
        "--idle_sleep_ms",
        type=float,
        default=10.0,
        help="本轮未收到任何 ZMQ 消息时的休眠毫秒数；过小会导致空转占满 CPU、IDE 卡顿（默认 10）",
    )
    p.add_argument(
        "--record_mode",
        type=str,
        default="stream",
        choices=["stream", "memory"],
        help="stream=边采边写 HDF5，单条 episode 内存几乎不随步数增长；memory=停录后一次性写入（旧行为，长 episode 易 OOM）",
    )
    args = p.parse_args()
    only_one_set = (args.rgb_raw_width is None) ^ (args.rgb_raw_height is None)
    if only_one_set:
        p.error("--rgb_raw_width 与 --rgb_raw_height 必须同时设置，或同时不设置")
    if args.rgb_raw_width is not None and args.rgb_raw_width <= 0:
        p.error("--rgb_raw_width 必须为正整数")
    if args.rgb_raw_height is not None and args.rgb_raw_height <= 0:
        p.error("--rgb_raw_height 必须为正整数")
    return args


def _pour_task_augment_pool(base_task: str) -> list[str]:
    """与默认「瓶→杯倒水」语义一致的英文改写，供 augment 模式；训练模板同上。"""
    base = base_task.strip()
    candidates = [
        base,
        "pour water from bottle into cup",
        "pour water from the bottle into the cup",
        "pour the water into the cup",
        "pick up the bottle and pour water into the cup",
        "fill the cup with water from the bottle",
        "grasp the bottle and pour water into the cup",
        "transfer water from the bottle to the cup",
        "pour water from the bottle into the target cup",
        "grab the bottle, move it over the cup, and pour water",
        "tilt the bottle to pour water into the cup",
        "carefully pour water from bottle to cup",
        "move the bottle above the cup and pour water",
        "complete the water pouring into the cup",
    ]
    seen: set[str] = set()
    out: list[str] = []
    for p in candidates:
        k = p.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(p.strip())
    return out


def _resolve_task_for_episode(cmd: dict, args: argparse.Namespace) -> str:
    """确定本段 episode 写入 HDF5 的 task 字符串。"""
    t = (cmd.get("task") or "").strip()
    if t:
        return t
    if args.task_mode == "fixed":
        return args.task.strip()
    pool = _pour_task_augment_pool(args.task)
    return random.choice(pool)


def main():
    args = parse_args()
    if not HAS_ZMQ:
        raise RuntimeError("pyzmq 未安装，请先 pip install pyzmq")
    save_dir = args.save_dir or os.path.join(_REPO_ROOT, "data", "vla_dataset_real")
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 64)
    print("DOBOT CR5 — HDF5 录制端（Linux PC）")
    print("=" * 64)
    print(f"  ZMQ 监听   : tcp://{args.bind}:{args.port}")
    print(f"  保存目录   : {save_dir}")
    print(f"  图像尺寸   : {args.image_size}")
    if args.rgb_raw_width is not None and args.rgb_raw_height is not None:
        print(f"  rgb_raw尺寸: {args.rgb_raw_width}x{args.rgb_raw_height}")
    else:
        print("  rgb_raw尺寸: 跟随发送端原始分辨率")
    print(f"  默认 task  : {args.task}")
    print(f"  task 模式  : {args.task_mode}")
    print(f"  最少步数   : {args.min_steps}")
    print(f"  空闲轮询   : {args.idle_sleep_ms:g} ms（无消息时休眠，减轻 CPU / IDE 卡顿）")
    print(f"  录制模式   : {args.record_mode}")
    print("=" * 64)

    zmq_ctx = zmq.Context()
    zmq_sock = zmq_ctx.socket(zmq.PULL)
    zmq_sock.setsockopt(zmq.RCVHWM, 10)
    zmq_sock.bind(f"tcp://{args.bind}:{args.port}")
    print(f"[INFO] ZMQ PULL bind tcp://{args.bind}:{args.port}")

    recorder: EpisodeRecorder | StreamingEpisodeRecorder | None = None
    episode_count = 0
    recording = False
    pause_recording = False
    print("\n等待机械臂PC连接...\n")
    last_recv_time = time.time()
    recv_stall_warned = False

    idle_sleep_s = max(0.0, float(args.idle_sleep_ms) / 1000.0)

    try:
        while True:
            got_zmq = False
            got_obs_this_round = False

            while True:
                try:
                    parts = zmq_sock.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
                got_zmq = True

                cmd = None
                frame = None
                try:
                    if len(parts) >= 2 and parts[0] == b"CMD":
                        cmd = json.loads(parts[1].decode("utf-8"))
                    elif len(parts) == 7 and parts[0] == b"OBS":
                        header = json.loads(parts[1].decode("utf-8"))
                        depth_shape = tuple(header.get("depth_shape", [480, 640]))
                        frame = {
                            "header": header,
                            "rgb_bytes": parts[2],
                            "depth": np.frombuffer(parts[3], dtype=np.float32).reshape(depth_shape),
                            "state": np.frombuffer(parts[4], dtype=np.float32).copy(),
                            "action": np.frombuffer(parts[5], dtype=np.float32).copy(),
                            "timestamp": struct.unpack(">d", parts[6])[0],
                        }
                except Exception as e:
                    print(f"[WARN] ZMQ 消息解析错误: {e}")
                    continue

                if cmd is not None:
                    c = cmd.get("cmd", "")
                    if c == "start":
                        if not recording:
                            recording = True
                            pause_recording = False
                            last_recv_time = time.time()
                            recv_stall_warned = False
                            task_desc = _resolve_task_for_episode(cmd, args)
                            arng = "feedback pose delta: dx,dy,dz (m), droll,dpitch,dyaw (rad), gripper [-1,1]"
                            if args.record_mode == "stream":
                                ep_path = next_episode_path(save_dir)
                                recorder = StreamingEpisodeRecorder(
                                    ep_path,
                                    image_size=args.image_size,
                                    task_description=task_desc,
                                    action_range=arng,
                                )
                                print(
                                    f"[●REC] 开始录制 — episode {episode_count}  stream→{ep_path}  "
                                    f"OpenVLA task={task_desc!r}"
                                )
                            else:
                                recorder = EpisodeRecorder(
                                    image_size=args.image_size,
                                    task_description=task_desc,
                                    action_range=arng,
                                )
                                print(f"[●REC] 开始录制 — episode {episode_count}  OpenVLA task={task_desc!r}")

                    elif c == "stop":
                        if recording and recorder is not None:
                            if recorder.num_steps >= args.min_steps:
                                if isinstance(recorder, StreamingEpisodeRecorder):
                                    recorder.finalize(dt=1.0 / 10.0)
                                    ep_path = recorder.output_path
                                else:
                                    ep_path = next_episode_path(save_dir)
                                    recorder.save(ep_path, dt=1.0 / 10.0)
                                print(f"[✓保存] Episode 已保存: {ep_path} ({recorder.num_steps} 步)")
                                episode_count += 1
                            else:
                                print(f"[✗跳过] Episode 太短 ({recorder.num_steps} < {args.min_steps})，已丢弃")
                                if isinstance(recorder, StreamingEpisodeRecorder):
                                    recorder.abort()
                            recording = False
                            pause_recording = False
                            recorder = None
                            recv_stall_warned = False

                    elif c == "discard":
                        if recording:
                            print(f"[✗丢弃] Episode 已丢弃")
                            if isinstance(recorder, StreamingEpisodeRecorder):
                                recorder.abort()
                            recording = False
                            pause_recording = False
                            recorder = None
                            recv_stall_warned = False

                    elif c == "pause":
                        if recording and not pause_recording:
                            pause_recording = True
                            print("[⏸PAUSE] 已暂停写入")

                    elif c == "resume":
                        if recording and pause_recording:
                            pause_recording = False
                            last_recv_time = time.time()
                            recv_stall_warned = False
                            print("[▶RESUME] 继续写入")

                if frame is not None:
                    got_obs_this_round = True
                    last_recv_time = time.time()
                    recv_stall_warned = False
                    rgb_bytes = frame["rgb_bytes"]
                    depth = frame["depth"]
                    state = frame["state"]
                    action = frame["action"]
                    ts = frame["timestamp"]
                    frame_header = frame.get("header")
                    frame_phase = frame_header.get("phase", 0) if isinstance(frame_header, dict) else 0

                    if recording and (not pause_recording) and recorder is not None:
                        rgb = decode_rgb_from_payload(
                            frame_header if isinstance(frame_header, dict) else None, rgb_bytes
                        )
                        if (
                            rgb is not None
                            and args.rgb_raw_width is not None
                            and args.rgb_raw_height is not None
                        ):
                            rgb = cv2.resize(
                                rgb,
                                (args.rgb_raw_width, args.rgb_raw_height),
                                interpolation=cv2.INTER_AREA,
                            )

                        if rgb is None:
                            fallback_h = args.rgb_raw_height if args.rgb_raw_height is not None else 224
                            fallback_w = args.rgb_raw_width if args.rgb_raw_width is not None else 224
                            rgb_for_record = np.zeros((fallback_h, fallback_w, 3), dtype=np.uint8)
                        else:
                            rgb_for_record = rgb

                        depth_for_record = depth if depth.size > 0 else np.zeros((480, 640), dtype=np.float32)
                        recorder.add_step(
                            rgb=rgb_for_record,
                            depth=depth_for_record,
                            state=state,
                            action=action,
                            timestamp=ts,
                            store_rgb_raw=False,
                            store_full_depth=True,
                            phase=frame_phase,
                        )

            if not got_obs_this_round:
                now = time.time()
                if recording and (not pause_recording) and (now - last_recv_time) > args.timeout:
                    if not recv_stall_warned:
                        print(
                            f"[WARN] 已 {args.timeout:.1f}s 未收到观测帧 (OBS)，可能断连或发送端未发图；"
                            "本提示同一录制段内只报一次。"
                        )
                        recv_stall_warned = True

            if not got_zmq and idle_sleep_s > 0:
                time.sleep(idle_sleep_s)

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C")
        if recording and recorder is not None:
            if recorder.num_steps >= args.min_steps:
                if isinstance(recorder, StreamingEpisodeRecorder):
                    recorder.finalize(dt=1.0 / 10.0)
                    ep_path = recorder.output_path
                else:
                    ep_path = next_episode_path(save_dir)
                    recorder.save(ep_path, dt=1.0 / 10.0)
                print(f"[✓保存] 保存: {ep_path} ({recorder.num_steps} 步)")
            elif isinstance(recorder, StreamingEpisodeRecorder):
                recorder.abort()
    finally:
        zmq_sock.close(linger=0)
        zmq_ctx.term()
        print(f"\n[INFO] 完成。共保存 {episode_count} 个 episodes")

    return 0


if __name__ == "__main__":
    sys.exit(main())
