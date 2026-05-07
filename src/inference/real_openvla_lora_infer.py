#!/usr/bin/env python3
"""
DOBOT CR5 — 真机 OpenVLA-7B + LoRA 推理闭环
============================================

基于 `scripts/collect/real_teleop_collect.py` 的真机接口（DobotCR5 / ServoP /
Orbbec Femto Bolt / Pico 2 W 夹爪），把手柄遥操替换为 OpenVLA+LoRA 策略。

动作语义（务必与训练一致）：
    infer_action_vector() 输出的 7 维动作已经完成 q01/q99 反归一化；因此
        action[0:3] —— 位置增量，**单位 m**（对应 HDF5 中 feedback 差分）
        action[3:6] —— 姿态增量，**单位 rad**
        action[6]   —— 夹爪，**[-1, 1]**（模型输出仅供参考，实际由手柄控制）
    所以默认 `--cart_action_gain 1.0`（与仿真侧的 8.0 截然不同）。
    real_teleop_collect.py 用 `--pos_scale 2 mm/step` 把手柄 [-1,1] 放大，
    本脚本不再经过那一层缩放，直接把 action 累加到 `robot.cartesian_pose`。

    ★ 夹爪控制和阶段切换由手柄负责（而非模型输出）：
        X (2)  — 切换夹爪开/关（同时 Grasp→Move 阶段自动推进）
        RB (5) — 手动推进到下一阶段

安全措施（强烈建议首次试运行）：
    1. `--dry_run`        —— 不发 ServoP，只打印命令，用来核对量级是否合理
    2. `--max_pos_step_m` —— 每一步位置增量绝对值上限（默认 5 mm）
    3. `--max_rot_step_rad` —— 每一步姿态增量绝对值上限（默认 3°）
    4. `--bbox_xyz_m`     —— 目标位姿工作空间 bbox，超出则夹住（默认关闭）
    5. 手柄：A 暂停/继续推理；Y 回 home；Start 退出；B 紧急急停（调用
       `robot.stop_move()` 并进入暂停状态）；X 夹爪；RB 切阶段。

典型用法：
    # 先空跑，确认每步命令量级合理（位置 ≤ 5 mm、角度 ≤ 3° 的同阶）
    python scripts/inference/real_openvla_lora_infer.py \\
        --lora_path LoRA_train/runs/pouring_lora_a10080/final_lora \\
        --dry_run

    # 实际上机（第一次建议 --speed_ratio 10，`--hz 5` 放慢确认）
    python scripts/inference/real_openvla_lora_infer.py \\
        --lora_path LoRA_train/runs/pouring_lora_a10080/final_lora \\
        --speed_ratio 20 --hz 10
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import cv2
import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# 注意：ROS2（/opt/ros/humble）自带一个同名 `scripts` 包，会屏蔽仓库里的 scripts/ 目录。
# 这里把 scripts/collect/ 直接加到 sys.path，按模块文件名 import，避开 `scripts.` 命名冲突。
_COLLECT_DIR = os.path.join(_REPO_ROOT, "scripts", "collect")
if _COLLECT_DIR not in sys.path:
    sys.path.insert(0, _COLLECT_DIR)

from dobot_cr5 import DobotCR5
from real_teleop_collect import (  # noqa: E402  — 复用真机采集里的硬件抽象
    CameraManager,
    GripperController,
    _auto_detect_gripper_port,
    _build_display,
)
from tools.teleop.gamepad_receiver import GamepadReceiver
from tools.vla.openvla_lora_runtime import (
    apply_cart_action_gain,
    build_vicuna_prompt,
    check_openvla_runtime_dependencies,
    debug_print_action_stats,
    ensure_rgb_uint8_hwc,
    infer_action_vector,
    inject_dataset_statistics,
    load_openvla_with_lora,
    resolve_base_checkpoint_path,
    resolve_dataset_stats_path,
)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DOBOT CR5 + OpenVLA-7B(LoRA) 真机推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # 机器人 / 相机 / 夹爪（与 real_teleop_collect 对齐）
    p.add_argument("--ip", type=str, default="192.168.50.104")
    p.add_argument("--cam_id", type=int, default=0)
    p.add_argument("--cam_width", type=int, default=1280)
    p.add_argument("--cam_height", type=int, default=720)
    p.add_argument("--gripper_port", type=str, default=None)
    p.add_argument("--speed_ratio", type=int, default=20)
    p.add_argument(
        "--home_joints",
        type=float,
        nargs=6,
        default=[34.4919, 13.2380, 120.5698, -43.8078, -90.0024, -0.0243],
    )
    p.add_argument("--no_robot", action="store_true", help="仅跑推理，不连接机器人（调试用）")

    # VLA
    p.add_argument("--base_checkpoint", type=str, default="checkpoints\openvla-7b")
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
    p.add_argument("--task", type=str, default="",
                   help="固定 task 字符串；留空则启用三阶段自动切换")
    p.add_argument("--tilt_threshold_deg", type=float, default=20.0,
                   help="Move→Pour 切换的 EE 倾斜角阈值（度）")
    p.add_argument("--unnorm_key", type=str, default="dobot_pouring")
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--merge_lora", action="store_true")
    p.add_argument("--no_lora", action="store_true",
                   help="直接加载已合并的完整模型（--base_checkpoint 指向合并后模型目录），跳过 LoRA 加载")
    p.add_argument("--ema_alpha", type=float, default=0.0)
    p.add_argument(
        "--cart_action_gain",
        type=float,
        default=1.0,
        help="前 6 维放大系数；真机推理时反归一化已是物理量，默认 1.0",
    )
    p.add_argument("--image_size", type=int, default=224,
                   help="喂给模型的 RGB 边长（与训练 EpisodeRecorder 一致）")
    p.add_argument("--center_crop", action="store_true",
                   help="推理前对图像做中心裁剪（面积 90%%），匹配 --image_aug True 训练时的 random_resized_crop")

    # 闭环
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument("--max_steps", type=int, default=0, help="0 表示不限")
    p.add_argument("--warmup_steps", type=int, default=5,
                   help="前 N 拍只采图不下发动作（等模型/相机稳定）")
    p.add_argument("--dry_run", action="store_true",
                   help="不发 ServoP 也不动夹爪；仅打印命令")

    # 安全 / 保护
    p.add_argument("--max_pos_step_m", type=float, default=0.1,
                   help="单步位置增量绝对值上限（m），默认 5 mm")
    p.add_argument("--max_rot_step_rad", type=float, default=np.deg2rad(10.0),
                   help="单步姿态增量绝对值上限（rad），默认 3°")
    p.add_argument("--bbox_xyz_m", type=float, nargs=6, default=None,
                   metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                   help="工作空间 bbox（基坐标 m），超出会把目标点夹进来；不设则不约束")
    p.add_argument("--gripper_hysteresis", type=float, default=0.2,
                   help="|action[6]| > 阈值 才切换夹爪，避免抖动")

    # 手柄（可选，纯辅助）
    p.add_argument("--gamepad_port", type=int, default=9876,
                   help="手柄 UDP 端口（可选辅助）；默认使用键盘控制")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════

def _resize_for_model(rgb: np.ndarray, size: int, center_crop: bool = False) -> np.ndarray:
    """相机原始分辨率 -> 训练时存的 224x224 方图。

    如果 center_crop=True，先裁剪中心 90% 面积（边长 ×√0.9 ≈ 94.9%），
    匹配训练时 random_resized_crop(scale=[0.9,0.9], ratio=[1.0,1.0]) 的效果。
    """
    u8 = ensure_rgb_uint8_hwc(rgb)
    # 先 resize 到 size×size
    if u8.shape[0] != size or u8.shape[1] != size:
        u8 = cv2.resize(u8, (size, size), interpolation=cv2.INTER_AREA)
    # 中心裁剪 90% 面积（边长 × sqrt(0.9)）
    if center_crop:
        import math
        crop_side = int(size * math.sqrt(0.9))  # 224 → 212
        offset = (size - crop_side) // 2
        u8 = u8[offset:offset + crop_side, offset:offset + crop_side]
        u8 = cv2.resize(u8, (size, size), interpolation=cv2.INTER_AREA)
    return u8


def _clip_step(action: np.ndarray, max_pos_m: float, max_rot_rad: float) -> np.ndarray:
    """单步增量安全 clip（逐轴）。仅裁剪前 6 维；夹爪保持。"""
    a = action.astype(np.float32).copy()
    a[0:3] = np.clip(a[0:3], -max_pos_m, max_pos_m)
    a[3:6] = np.clip(a[3:6], -max_rot_rad, max_rot_rad)
    return a


def _apply_workspace_bbox(
    target_xyz_mm: np.ndarray, bbox_xyz_m: Optional[list[float]]
) -> np.ndarray:
    """对目标点做基坐标 bbox clip。bbox_xyz_m 是 6 个 m，为 None 则不变。"""
    if bbox_xyz_m is None:
        return target_xyz_mm
    xmin, xmax, ymin, ymax, zmin, zmax = bbox_xyz_m
    out = target_xyz_mm.copy()
    out[0] = float(np.clip(out[0], xmin * 1000.0, xmax * 1000.0))
    out[1] = float(np.clip(out[1], ymin * 1000.0, ymax * 1000.0))
    out[2] = float(np.clip(out[2], zmin * 1000.0, zmax * 1000.0))
    return out


def _empty_events() -> dict:
    """返回一个空的事件字典。"""
    return {
        "toggle_pause": False,
        "estop": False,
        "toggle_gripper": False,
        "home": False,
        "next_phase": False,
        "quit": False,
    }


def _keyboard_events(key_code: int) -> dict:
    """从 cv2.waitKey 返回的键码构建事件字典。

    键盘映射（在 OpenCV 窗口获得焦点时生效）：
        Space  — 暂停/继续推理
        E      — 急停
        G      — 切换夹爪开/关
        H      — 回 home
        N      — 手动推进到下一阶段
        Q / ESC — 退出
    """
    events = _empty_events()
    if key_code < 0:
        return events
    k = key_code & 0xFF
    events["toggle_pause"]  = (k == ord(' '))
    events["estop"]         = (k == ord('e') or k == ord('E'))
    events["toggle_gripper"]= (k == ord('g') or k == ord('G'))
    events["home"]          = (k == ord('h') or k == ord('H'))
    events["next_phase"]    = (k == ord('n') or k == ord('N'))
    events["quit"]          = (k == 27 or k == ord('q') or k == ord('Q'))
    return events


def _gamepad_events(data: Optional[dict], prev_buttons: list[int]) -> tuple[dict, list[int]]:
    """手柄按键边沿检测（可选辅助）。"""
    events = _empty_events()
    if data is None:
        return events, []
    buttons = list(data.get("buttons", []))

    def rising(idx: int) -> bool:
        if idx >= len(buttons):
            return False
        curr = buttons[idx]
        prev = prev_buttons[idx] if idx < len(prev_buttons) else 0
        return curr == 1 and prev == 0

    events["toggle_pause"] = rising(0)    # A
    events["estop"] = rising(1)           # B
    # events["toggle_gripper"] = rising(2)  # X ← 已移除：夹爪改由模型输出驱动，手柄不再控制
    events["home"] = rising(3)            # Y
    events["next_phase"] = rising(5)      # RB
    events["quit"] = rising(7)            # Start
    return events, buttons


def _merge_events(a: dict, b: dict) -> dict:
    """合并两个事件字典（OR 逻辑）。"""
    return {k: a.get(k, False) or b.get(k, False) for k in a}


# ═══════════════════════════════════════════════════════════════════════════
# 三阶段自动切换状态机
# ═══════════════════════════════════════════════════════════════════════════

PHASE_TASKS = [
    # "pick up cola",        # phase 0: Grasp  (与训练数据一致)
    # "move to cup",         # phase 1: Move   (与训练数据一致)
    # "pour cola into cup",  # phase 2: Pour   (与训练数据一致)
    "pour cola into cup",  # phase 0: Grasp  (训练数据里虽然叫 Grasp，但实际是拿起并倾斜到 Move 的动作，和 Pour 一样都是倾斜状态)
    "pour cola into cup",  # phase 0: Grasp  (训练数据里虽然叫 Grasp，但实际是拿起并倾斜到 Move 的动作，和 Pour 一样都是倾斜状态)
    "pour cola into cup",  # phase 0: Grasp  (训练数据里虽然叫 Grasp，但实际是拿起并倾斜到 Move 的动作，和 Pour 一样都是倾斜状态)
]


def _compute_tilt_from_pose(pose_6: np.ndarray) -> float:
    """
    计算 EE 偏离竖直向下的倾斜角（度）。

    竖直向下时 rx ≈ ±180°, ry ≈ 0°。
    与 analyze_phases.compute_ee_tilt 逻辑一致。
    """
    rx = float(pose_6[3])  # degrees
    ry = float(pose_6[4])  # degrees
    rx_tilt = abs(((rx - 180.0 + 180.0) % 360.0) - 180.0)
    ry_tilt = abs(ry)
    return max(rx_tilt, ry_tilt)


class PhaseStateMachine:
    """
    三阶段任务切换状态机（手柄驱动）。

    状态转移完全由手柄控制：
        X 按钮  — 切换夹爪开/关；若当前 phase=0(Grasp) 且夹爪闭合，自动推进到 phase=1(Move)
        RB 按钮 — 手动推进到下一阶段

    与训练数据的 phase_task 标注方式完全一致。
    """

    def __init__(self, tilt_threshold_deg: float = 20.0):
        self.tilt_threshold = tilt_threshold_deg
        self.phase = 0  # 0=Grasp, 1=Move, 2=Pour

    @property
    def task(self) -> str:
        return PHASE_TASKS[self.phase]

    @property
    def phase_name(self) -> str:
        return ["Grasp", "Move", "Pour"][self.phase]

    def advance(self) -> bool:
        """手动推进到下一阶段。返回 True 表示发生了阶段切换。"""
        if self.phase < 2:
            old_name = self.phase_name
            self.phase += 1
            print(f"\n[PHASE] ★ {old_name} → {self.phase_name}  (手柄手动切换)")
            return True
        return False

    def on_gripper_close(self) -> bool:
        """夹爪闭合事件：若当前在 Grasp 阶段，自动推进到 Move。"""
        if self.phase == 0:
            self.phase = 1
            print(f"\n[PHASE] ★ Grasp → Move  (手柄夹爪闭合触发)")
            return True
        return False

    def reset(self):
        self.phase = 0


# ═══════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    args = parse_args()
    check_openvla_runtime_dependencies()

    if args.bbox_xyz_m is not None:
        xmin, xmax, ymin, ymax, zmin, zmax = args.bbox_xyz_m
        if not (xmin < xmax and ymin < ymax and zmin < zmax):
            print("[错误] --bbox_xyz_m 必须满足 min < max（顺序: xmin xmax ymin ymax zmin zmax）",
                  file=sys.stderr)
            return 2

    base_ckpt = resolve_base_checkpoint_path(args.base_checkpoint, _REPO_ROOT)
    processor_path = args.processor_path or base_ckpt

    # ── 解析 dataset_statistics 路径 ──
    if args.no_lora:
        # 合并模型模式：从模型目录或 --dataset_stats 找统计文件
        if args.dataset_stats and os.path.isfile(args.dataset_stats):
            dataset_stats_path = os.path.abspath(args.dataset_stats)
        else:
            candidate = os.path.join(base_ckpt, "dataset_statistics.json")
            if os.path.isfile(candidate):
                dataset_stats_path = candidate
            else:
                print(
                    f"[错误] 合并模型目录下未找到 dataset_statistics.json，"
                    f"请用 --dataset_stats 指定。\n  模型路径: {base_ckpt}",
                    file=sys.stderr,
                )
                return 2
    else:
        dataset_stats_path = resolve_dataset_stats_path(args.dataset_stats, args.lora_path)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print("\n" + "=" * 68)
    print("DOBOT CR5 × OpenVLA 真机推理" + ("" if args.no_lora else " (+LoRA)"))
    print("=" * 68)
    print(f"  机器人 IP         : {args.ip}    |  连接: {'否' if args.no_robot else '是'}")
    print(f"  相机              : {args.cam_width}x{args.cam_height} -> {args.image_size}² (模型)")
    if args.no_lora:
        print(f"  合并模型          : {base_ckpt}")
    else:
        print(f"  基座模型          : {base_ckpt}")
        print(f"  LoRA              : {args.lora_path}")
    print(f"  dataset_stats     : {dataset_stats_path}")
    print(f"  device / dtype    : {device} / {dtype}")
    print(f"  控制频率          : {args.hz} Hz   warmup={args.warmup_steps}")
    print(f"  每步增量上限     : Δp≤{args.max_pos_step_m*1000:.1f} mm  Δr≤{np.rad2deg(args.max_rot_step_rad):.1f}°")
    print(f"  bbox              : {args.bbox_xyz_m}")
    print(f"  cart_action_gain  : {args.cart_action_gain}   dry_run={args.dry_run}")
    use_phase_sm = not args.task  # task 为空则启用状态机
    if use_phase_sm:
        print(f"  任务模式          : 三阶段自动切换 (tilt>{args.tilt_threshold_deg}°)")
        print(f"  Phase 0 (Grasp)   : {PHASE_TASKS[0]}")
        print(f"  Phase 1 (Move)    : {PHASE_TASKS[1]}")
        print(f"  Phase 2 (Pour)    : {PHASE_TASKS[2]}")
    else:
        print(f"  任务              : {args.task}")
    print("=" * 68 + "\n")

    # ── 加载模型 ──────────────────────────────────────────
    if args.no_lora:
        from transformers import AutoModelForVision2Seq, AutoProcessor

        print(f"加载 Processor: {processor_path}")
        processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)

        print(f"加载合并模型: {base_ckpt}")
        model = AutoModelForVision2Seq.from_pretrained(
            base_ckpt,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        inject_dataset_statistics(model, dataset_stats_path)
        model = model.to(device)
        model.eval()

        action_dim = model.get_action_dim(args.unnorm_key)
        print(f"unnorm_key={args.unnorm_key}  action_dim={action_dim}")
        debug_print_action_stats(model, args.unnorm_key)
    else:
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
    phase_sm: PhaseStateMachine | None = None
    if use_phase_sm:
        phase_sm = PhaseStateMachine(tilt_threshold_deg=args.tilt_threshold_deg)
        prompt = build_vicuna_prompt(phase_sm.task)
        print(f"[VLA] action_dim={action_dim}  初始 phase={phase_sm.phase_name}  task='{phase_sm.task}'\n")
    else:
        prompt = build_vicuna_prompt(args.task)
        print(f"[VLA] action_dim={action_dim}  prompt='{args.task}'\n")

    # ── 连接机器人 ────────────────────────────────────────
    robot: DobotCR5 | None = None
    if not args.no_robot:
        robot = DobotCR5(ip_address=args.ip)
        try:
            robot.connect()
            time.sleep(1)
            print(f"[ROBOT] mode={robot.robot_mode}")
            robot.enable()
            time.sleep(2)
            robot.set_speed_ratio(args.speed_ratio)
            print(f"[ROBOT] enable ok, speed_ratio={args.speed_ratio}")
            print("[ROBOT] moving to home ...")
            robot.joint_mov_j(list(args.home_joints))
            time.sleep(5)
            print(f"[ROBOT] joints={[round(a,2) for a in robot.joint_angles]}")
            print(f"[ROBOT] pose  ={[round(p,4) for p in robot.cartesian_pose]}")
        except Exception as e:
            print(f"[ERROR] robot connect/enable failed: {e}", file=sys.stderr)
            return 3

    # ── 相机 ──────────────────────────────────────────────
    camera = CameraManager(cam_id=args.cam_id, width=args.cam_width, height=args.cam_height, use_depth=False)
    try:
        camera.open()
    except RuntimeError as e:
        print(f"[ERROR] camera open failed: {e}", file=sys.stderr)
        if robot is not None:
            robot.disable()
            robot.disconnect()
        return 4

    # ── 夹爪 ──────────────────────────────────────────────
    gripper_port = args.gripper_port
    if gripper_port is None:
        print("[GRIPPER] 自动检测串口 ...")
        gripper_port = _auto_detect_gripper_port()
        if gripper_port is None:
            print("[ERROR] 未检测到夹爪，使用 --gripper_port 指定。", file=sys.stderr)
            camera.close()
            if robot is not None:
                robot.disable(); robot.disconnect()
            return 5
    gripper = GripperController(port=gripper_port)
    # 程序启动时确保夹爪处于打开状态
    if not args.dry_run:
        gripper.open()
        print("[GRIPPER] 初始化 → OPEN")

    # ── 手柄（可选辅助） ───────────────────────────────────
    receiver: GamepadReceiver | None = None
    gripper_open = True  # 夹爪状态跟踪
    try:
        receiver = GamepadReceiver(port=args.gamepad_port)
        receiver.start()
        print(f"[GAMEPAD] listening UDP :{args.gamepad_port}  （可选辅助，键盘始终可用）")
    except Exception as e:
        print(f"[GAMEPAD] 初始化失败（仅用键盘控制）: {e}")
        receiver = None

    # ── 主循环 ────────────────────────────────────────────
    WIN = "真机推理 — OpenVLA+LoRA"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 900, 480)

    control_dt = 1.0 / args.hz
    ema_prev: np.ndarray | None = None
    paused = False
    prev_buttons: list[int] = []
    step = 0
    inference_steps = 0
    infer_time_total: float = 0.0  # 累计 VLA 推理耗时（秒），用于计算平均频率
    last_infer_ms: float = 0.0    # 最近一次推理耗时（ms），用于日志显示
    last_action: np.ndarray = np.zeros(7, dtype=np.float32)
    last_key: int = -1  # 上一帧 cv2.waitKey 的返回值
    _phase_switch_verbose = 0  # 阶段切换后的详细日志倒计时

    print("\n" + "-" * 50)
    print("键盘控制（OpenCV 窗口需获得焦点）：")
    print("  Space — 暂停/继续推理")
    print("  G     — 切换夹爪 开/关")
    print("  N     — 手动切换到下一阶段")
    print("  H     — 回 home")
    print("  E     — 急停")
    print("  Q/ESC — 退出")
    print("-" * 50 + "\n")
    try:
        while True:
            loop_t0 = time.time()

            # 1) 键盘 + 手柄事件
            kb_events = _keyboard_events(last_key)
            last_key = -1  # 消费掉
            if receiver is not None:
                gp_data = receiver.get_latest()
                gp_events, prev_buttons = _gamepad_events(gp_data, prev_buttons)
                events = _merge_events(kb_events, gp_events)
            else:
                events = kb_events

            if events["quit"]:
                print("[INFO] 收到 Start，退出。")
                break

            if events["estop"] and robot is not None:
                try:
                    robot.stop_move()
                except Exception as e:
                    print(f"[ERROR] stop_move 失败: {e}")
                paused = True
                print("[!!] 急停！已发送 StopScript；当前暂停推理。按 A 继续。")

            if events["home"]:
                paused = True
                if robot is not None:
                    print("[INFO] 回 home ...")
                    robot.joint_mov_j(list(args.home_joints))
                    time.sleep(4)
                ema_prev = None
                gripper_open = True
                if not args.dry_run:
                    gripper.open()
                if phase_sm is not None:
                    phase_sm.reset()
                    prompt = build_vicuna_prompt(phase_sm.task)
                    print(f"[PHASE] 重置 → {phase_sm.phase_name}: '{phase_sm.task}'")
                print("[INFO] home 完成；仍处于暂停状态，按 A 继续。")

            # ── 键盘夹爪手动接管 (G 键) ──
            # 键盘 G 可随时手动切换夹爪，优先级高于模型输出；模型在下一推理步后恢复控制权
            if events["toggle_gripper"]:
                gripper_open = not gripper_open
                if not args.dry_run:
                    if gripper_open:
                        gripper.open()
                    else:
                        gripper.close()
                state_str = "OPEN" if gripper_open else "CLOSED"
                print(f"[GRIPPER] 键盘手动接管 → {state_str}")
                # 夹爪关闭时：无论是否切阶段，都重置 EMA
                # 原因：训练数据中夹爪关闭前后存在大量静止帧，模型倾向于输出近零动作；
                # 重置 EMA 可避免这些近零输出继续污染后续推理步
                if not gripper_open:
                    ema_prev = None
                    print("[EMA] 夹爪关闭 → EMA 重置")
                # 若当前在 Grasp 阶段且夹爪闭合，自动推进到 Move
                if (not gripper_open) and phase_sm is not None:
                    switched = phase_sm.on_gripper_close()
                    if switched:
                        prompt = build_vicuna_prompt(phase_sm.task)
                        print(f"[PHASE] 切换 prompt → '{phase_sm.task}'")
                        print(f"[PHASE] full prompt = {prompt!r}")
                        ema_prev = None
                        _phase_switch_verbose = 10  # 切换后打印 10 步详细动作

            # ── 手柄手动切阶段 (RB 按钮) ──
            if events["next_phase"] and phase_sm is not None:
                switched = phase_sm.advance()
                if switched:
                    prompt = build_vicuna_prompt(phase_sm.task)
                    print(f"[PHASE] 切换 prompt → '{phase_sm.task}'")
                    print(f"[PHASE] full prompt = {prompt!r}")
                    ema_prev = None
                    _phase_switch_verbose = 10

            if events["toggle_pause"]:
                paused = not paused
                print(f"[INFO] {'暂停' if paused else '继续'} 推理")
                if not paused:
                    ema_prev = None  # 重置平滑

            # 2) 读图
            rgb, depth = camera.read()
            if rgb is None:
                time.sleep(0.005)
                continue

            # 3) 推理（暂停 / warmup 期间不推理，填零动作占位显示）
            do_infer = (not paused) and (step >= args.warmup_steps)
            action_np = np.zeros(7, dtype=np.float32)
            action_np[6] = 0.0  # 占位；推理后由模型 action[6] 驱动夹爪

            if do_infer:
                rgb_224 = _resize_for_model(rgb, args.image_size, center_crop=args.center_crop)
                _t_infer_start = time.time()
                action_np, ema_prev = infer_action_vector(
                    model,
                    processor,
                    rgb_224,
                    prompt,
                    device,
                    dtype,
                    action_dim,
                    args.unnorm_key,
                    args.ema_alpha,
                    ema_prev,
                )
                _t_infer_end = time.time()
                last_infer_ms = (_t_infer_end - _t_infer_start) * 1000.0
                infer_time_total += (_t_infer_end - _t_infer_start)
                action_np = apply_cart_action_gain(action_np, args.cart_action_gain)
                # 真机安全 clip（逐轴限幅）
                action_np = _clip_step(
                    action_np, args.max_pos_step_m, args.max_rot_step_rad
                )
                inference_steps += 1
                last_action = action_np.copy()

                # ── 模型驱动夹爪控制 ──
                # action[6] 经过 q01/q99 反归一化后处于物理空间 [0, 1]：
                #   0.0 = 关闭（训练数据 gripper=0）
                #   1.0 = 打开（训练数据 gripper=1）
                # 以 0.5 为分界线，加 hysteresis 防抖：
                #   > 0.5+hys → OPEN；  < 0.5-hys → CLOSE
                # 键盘 G 已在本循环前处理，若用户手动切换则 gripper_open 已更新，
                # 此处仍按模型输出继续执行，实现模型接管（键盘仅生效一帧）。
                if not args.dry_run:
                    hys = args.gripper_hysteresis  # default 0.2
                    open_thresh  = 0.5 + hys   # > 0.7 → open
                    close_thresh = 0.5 - hys   # < 0.3 → close
                    if action_np[6] > open_thresh and not gripper_open:
                        gripper_open = True
                        gripper.open()
                        print(f"[GRIPPER] 模型→ OPEN   (action[6]={action_np[6]:+.3f}, threshold>{open_thresh:.2f})")
                        # 开夹爪不触发阶段切换（阶段切换由键盘 N/RB 或闭夹爪触发）
                    elif action_np[6] < close_thresh and gripper_open:
                        gripper_open = False
                        gripper.close()
                        print(f"[GRIPPER] 模型→ CLOSE  (action[6]={action_np[6]:+.3f}, threshold<{close_thresh:.2f})")
                        ema_prev = None  # 重置EMA，避免静止帧污染后续输出
                        # 若在 Grasp 阶段闭合，自动推进到 Move
                        if phase_sm is not None:
                            switched = phase_sm.on_gripper_close()
                            if switched:
                                prompt = build_vicuna_prompt(phase_sm.task)
                                print(f"[PHASE] 切换 prompt → '{phase_sm.task}'")
                                ema_prev = None
                                _phase_switch_verbose = 10

            # 4) 下发（仅臂运动；夹爪由手柄控制，不在此处处理）
            if do_infer and (not args.dry_run) and robot is not None:
                curr = np.asarray(robot.cartesian_pose, dtype=np.float64)
                # 位置：action 单位 m，robot.cartesian_pose 前 3 维 m；ServoP 需要 mm
                target_xyz_mm = (curr[:3] + action_np[:3]) * 1000.0
                target_xyz_mm = _apply_workspace_bbox(target_xyz_mm, args.bbox_xyz_m)
                # 姿态：action 单位 rad，robot.cartesian_pose 后 3 维 度；ServoP 需要 度
                target_rpy_deg = curr[3:] + np.rad2deg(action_np[3:6])
                try:
                    robot.servo_p(
                        float(target_xyz_mm[0]),
                        float(target_xyz_mm[1]),
                        float(target_xyz_mm[2]),
                        float(target_rpy_deg[0]),
                        float(target_rpy_deg[1]),
                        float(target_rpy_deg[2]),
                    )
                except Exception as e:
                    print(f"[ERROR] servo_p 失败: {e}")
                    paused = True

            # 5) HUD（沿用采集脚本的显示函数，把 state 塞进去）
            curr_state = np.zeros(19, dtype=np.float32)
            if robot is not None:
                curr_state[0:6] = robot.joint_angles
                curr_state[6:12] = robot.actual_joint_speeds
                curr_state[13:19] = robot.cartesian_pose
            curr_state[12] = gripper.get_state()

            display = _build_display(
                rgb, depth, curr_state,
                recording=False,     # 推理模式没有 REC
                episode_count=inference_steps,
                rec_steps=step,
                gripper=gripper,
                mode="推理" + (f" [{phase_sm.phase_name}]" if phase_sm else "") + (" (PAUSE)" if paused else ""),
                gp_connected=(receiver.is_connected if receiver is not None else None),
                paused=paused,
            )
            # 叠加一行：最近一次动作（物理量）
            a = last_action
            line = (f"a: dxyz=[{a[0]*1000:+.1f},{a[1]*1000:+.1f},{a[2]*1000:+.1f}] mm  "
                    f"drpy=[{np.rad2deg(a[3]):+.1f},{np.rad2deg(a[4]):+.1f},{np.rad2deg(a[5]):+.1f}] "
                    f"deg  g={a[6]:+.2f}")
            cv2.putText(display, line, (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 220), 1, cv2.LINE_AA)
            cv2.imshow(WIN, display)
            last_key = cv2.waitKey(1)  # 键盘事件在下一轮迭代处理

            # 6) 日志
            if do_infer and (_phase_switch_verbose > 0 or inference_steps % 20 == 0):
                avg_hz = inference_steps / infer_time_total if infer_time_total > 0 else 0.0
                print(f"[{inference_steps:>5d}] {line}  phase={phase_sm.phase_name if phase_sm else 'N/A'}  "
                      f"infer={last_infer_ms:.0f}ms  avg_hz={avg_hz:.2f}")
                if _phase_switch_verbose > 0:
                    _phase_switch_verbose -= 1

            step += 1
            if args.max_steps > 0 and inference_steps >= args.max_steps:
                print(f"[INFO] 达到 max_steps={args.max_steps}，退出。")
                break

            # 7) 节流
            elapsed = time.time() - loop_t0
            sleep_time = control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C 中断")
    finally:
        cv2.destroyAllWindows()
        if receiver is not None:
            receiver.stop()
        try:
            gripper.disconnect()
        except Exception:
            pass
        camera.close()
        if robot is not None:
            try:
                robot.stop_move()
            except Exception:
                pass
            time.sleep(0.3)
            robot.disable()
            time.sleep(0.3)
            robot.disconnect()

    return 0


if __name__ == "__main__":
    sys.exit(main())
