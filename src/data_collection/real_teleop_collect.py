#!/usr/bin/env python3
"""
DOBOT CR5 — 真机 VLA 数据采集（手柄遥操作）
============================================

手柄映射与控制逻辑与 scripts/collect/teleop_collect.py 一致；末端运动由 ServoP 执行。
写入 HDF5 的 actions 为相邻两帧之间、由 feedback 口解析更新的笛卡尔位姿差分（米 / 弧度），
不归一化；夹爪通道为 [-1,1] 语义。

图像：observations/images/rgb 为缩放至 --image_size（默认 224）的 LoRA/OpenVLA 输入；
      observations/images/rgb_raw 为相机原始分辨率 RGB。
      observations/images/depth 为原始分辨率深度（与 rgb_raw 对齐）。

启动示例：
    python scripts/collect/real_teleop_collect.py --port 9876

手柄控制映射（Xbox / 通用双摇杆）— 与 teleop_collect.py 相同：
    左摇杆 X/Y      → EE Y / X
    右摇杆 X/Y      → EE yaw / Z
    LT / RT (轴 4/5) → pitch
    LB / RB (按钮 4/5) → roll
    A (0)  切换录制   |  B (1) 丢弃 episode  |  X (2) 夹爪
    Y (3)  复位（回初始关节）|  Back (6) 暂停/继续录制  |  Start (7) 退出

真机上 Back 仅暂停/继续写入数据，不恢复机器人位姿（与仿真 snapshot 不同）。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

# -- 让项目根目录可导入 --
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from dobot_cr5 import DobotCR5
from tools.teleop.gamepad_receiver import GamepadReceiver
from tools.vla.vla_data_writer import EpisodeRecorder, next_episode_path


# ═══════════════════════════════════════════════════════════════════════════
# 夹爪自动检测
# ═══════════════════════════════════════════════════════════════════════════

def _auto_detect_gripper_port() -> str | None:
    """遍历可用串口，发送查询指令，识别 Pico 2 W 夹爪。

    Returns:
        串口路径（如 /dev/ttyACM0），未找到返回 None。
    """
    import serial
    import serial.tools.list_ports

    for port_info in serial.tools.list_ports.comports():
        try:
            with serial.Serial(
                port=port_info.device,
                baudrate=115200,
                timeout=0.1,
            ) as conn:
                conn.flushInput()
                conn.write(b"Q\n")
                conn.flush()
                resp = conn.readline().decode("ascii", errors="ignore").strip()
                if resp.startswith("A") and resp[1:].replace(".", "").replace("-", "").isdigit():
                    return port_info.device
        except (serial.SerialException, OSError):
            continue
    return None


# ═══════════════════════════════════════════════════════════════════════════
# 夹爪控制器 — Pico 2 W 串口 PWM 控制（LDX-335MG 数字舵机，开环）
# ═══════════════════════════════════════════════════════════════════════════

class GripperController:
    """Pico 2 W 夹爪控制器，通过串口发送 PWM 指令（无编码器反馈，开环控制）。

    夹爪物理范围：打开 = 80°，关闭 = 120°
    串口协议（115200 8N1）：
        O\\n  — 全开（80°）
        C\\n  — 全关（120°）
        G{angle}\\n  — 设置角度（度）

    state 语义：0.0 = 关闭（120°），1.0 = 打开（80°）。
    无真实反馈，通过时间插值模拟夹爪过渡过程。
    """

    ANGLE_OPEN = 115.0    # 舵机角度（度）：夹爪完全张开时的舵机位置
    ANGLE_CLOSE = 160.0   # 舵机角度（度）：夹爪完全闭合时的舵机位置
    ACTUATION_TIME = 0.8  # 夹爪从开到关（或关到开）的物理运动时间（秒），用于模拟过渡状态

    def __init__(self, port: str = "/dev/ttyACM0", baudrate: int = 115200):
        import serial
        self._serial = serial
        self._port = port
        self._baudrate = baudrate
        self._conn = None
        self._target_state = 1.0
        self._start_state = 1.0
        self._last_toggle_time = 0.0
        self._connect()

    def _connect(self):
        try:
            self._conn = self._serial.Serial(
                port=self._port,
                baudrate=self._baudrate,
                timeout=0.1,
                bytesize=self._serial.EIGHTBITS,
                parity=self._serial.PARITY_NONE,
                stopbits=self._serial.STOPBITS_ONE,
            )
            time.sleep(0.1)
            self._conn.flushInput()
            print(f"[GRIPPER] Connected to {self._port}")
        except self._serial.SerialException as e:
            print(f"[GRIPPER] Connection failed: {e}")
            self._conn = None

    def is_connected(self) -> bool:
        return self._conn is not None and self._conn.is_open

    def _send_cmd(self, cmd: bytes) -> bool:
        if not self.is_connected():
            return False
        try:
            self._conn.write(cmd)
            self._conn.flush()
            return True
        except (self._serial.SerialException, OSError):
            self._conn = None
            return False

    def open(self):
        """打开夹爪 → 80°"""
        if self._target_state != 1.0:
            self._start_state = self.get_state()
            self._target_state = 1.0
            self._last_toggle_time = time.time()
        self._send_cmd(f"G{self.ANGLE_OPEN:.1f}\n".encode())
        print(f"[GRIPPER] Open → {self.ANGLE_OPEN}°")

    def close(self):
        """关闭夹爪 → 120°"""
        if self._target_state != 0.0:
            self._start_state = self.get_state()
            self._target_state = 0.0
            self._last_toggle_time = time.time()
        self._send_cmd(f"G{self.ANGLE_CLOSE:.1f}\n".encode())
        print(f"[GRIPPER] Close → {self.ANGLE_CLOSE}°")

    def toggle(self):
        """切换开/关状态"""
        if self._target_state > 0.5:
            self.close()
        else:
            self.open()

    def get_state(self) -> float:
        """返回夹爪开度 (0.0=关闭, 1.0=打开)，通过时间插值模拟过渡过程。"""
        elapsed = time.time() - self._last_toggle_time
        if elapsed >= self.ACTUATION_TIME:
            return self._target_state
        progress = elapsed / self.ACTUATION_TIME
        return self._start_state + (self._target_state - self._start_state) * progress

    def disconnect(self):
        if self._conn and self._conn.is_open:
            self._conn.close()
            print("[GRIPPER] Disconnected")


# ═══════════════════════════════════════════════════════════════════════════
# 手柄 → 动作映射（与 teleop_collect.py 中 GamepadMapper 一致）
# ═══════════════════════════════════════════════════════════════════════════

class GamepadMapper:
    """Convert raw gamepad data to a 7-DoF Cartesian EE action + button events.

    Action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
    Outputs are in [-1, 1] for the arm; gripper is ±1. Real script scales arm for ServoP.
    """

    def __init__(self, deadzone: float = 0.15):
        self.deadzone = deadzone
        self.gripper_open = True
        self._prev_buttons: list[int] = []
        self._prev_hats: list[list[int]] = []

    def _apply_deadzone(self, value: float) -> float:
        if abs(value) < self.deadzone:
            return 0.0
        sign = 1.0 if value > 0 else -1.0
        return sign * (abs(value) - self.deadzone) / (1.0 - self.deadzone)

    def _button_pressed(self, buttons: list[int], idx: int) -> bool:
        if idx >= len(buttons):
            return False
        curr = buttons[idx]
        prev = self._prev_buttons[idx] if idx < len(self._prev_buttons) else 0
        return curr == 1 and prev == 0

    def map(self, data: dict | None) -> tuple[np.ndarray, dict]:
        action = np.zeros(7, dtype=np.float32)
        events = {
            "toggle_record": False,
            "discard": False,
            "toggle_gripper": False,
            "toggle_pause_record": False,
            "reset_env": False,
            "quit": False,
            "next_phase": False,
            "prev_phase": False,
        }

        if data is None:
            self._prev_buttons = []
            self._prev_hats = []
            action[6] = 1.0 if self.gripper_open else -1.0
            return action, events

        axes = data.get("axes", [])
        buttons = data.get("buttons", [])
        hats = data.get("hats", [])

        def axis(idx: int) -> float:
            return self._apply_deadzone(axes[idx]) if idx < len(axes) else 0.0

        action[0] = -axis(1)
        action[1] = axis(0)
        action[2] = -axis(3)
        action[5] = axis(2)

        if len(axes) > 5:
            lt = (axes[4] + 1.0) / 2.0
            rt = (axes[5] + 1.0) / 2.0
            action[4] = rt - lt

        lb_held = buttons[4] if len(buttons) > 4 else 0
        rb_held = buttons[5] if len(buttons) > 5 else 0
        action[3] = float(rb_held - lb_held)

        action[:6] *= -1.0

        if self._button_pressed(buttons, 2):
            self.gripper_open = not self.gripper_open
            events["toggle_gripper"] = True

        action[6] = 1.0 if self.gripper_open else -1.0

        events["toggle_record"] = self._button_pressed(buttons, 0)
        events["discard"] = self._button_pressed(buttons, 1)
        events["reset_env"] = self._button_pressed(buttons, 3)
        events["toggle_pause_record"] = self._button_pressed(buttons, 6)
        events["quit"] = self._button_pressed(buttons, 7)

        # D-pad (hat) 阶段切换：右 = 下一阶段，左 = 上一阶段
        if hats:
            hat_x = hats[0][0] if isinstance(hats[0], (list, tuple)) else 0
            prev_hat_x = (
                self._prev_hats[0][0]
                if self._prev_hats and isinstance(self._prev_hats[0], (list, tuple))
                else 0
            )
            # 边沿检测：从 0 变到 1 才触发
            if hat_x == 1 and prev_hat_x != 1:
                events["next_phase"] = True
            if hat_x == -1 and prev_hat_x != -1:
                events["prev_phase"] = True

        self._prev_buttons = list(buttons)
        self._prev_hats = [list(h) if isinstance(h, (list, tuple)) else [0, 0] for h in hats]
        return action, events


# ═══════════════════════════════════════════════════════════════════════════
# 相机管理器
# ═══════════════════════════════════════════════════════════════════════════

try:
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode
except ImportError:
    print("[WARNING] 未找到 pyorbbecsdk, 请安装。例如: pip install pyorbbecsdk")
    Pipeline, Config, OBSensorType, OBFormat, OBAlignMode = None, None, None, None, None


class CameraManager:
    """管理 Orbbec Femto Bolt 相机（通过 pyorbbecsdk），支持 D2C 对齐"""

    def __init__(self, cam_id: int = 0, width: int = 640, height: int = 480, use_depth: bool = True):
        self.cam_id = cam_id  # 对于单设备可能不使用
        self.width = width
        self.height = height
        self.use_depth = use_depth
        self.pipeline = None

    def open(self):
        """打开相机；use_depth=False 时仅开启 Color 流，跳过深度和 D2C 对齐。"""
        if Pipeline is None:
            raise RuntimeError("无法导入 pyorbbecsdk，无法使用 Orbbec 相机。")
            
        self.pipeline = Pipeline()
        config = Config()
        
        # 配置 Color 流
        try:
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = profile_list.get_video_stream_profile(self.width, 0, OBFormat.RGB, 30)
            config.enable_stream(color_profile)
        except Exception as e:
            print(f"[相机] 颜色流配置失败: {e}")
            raise
            
        # 配置 Depth 流（仅 use_depth=True 时开启）
        if self.use_depth:
            try:
                profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
                depth_profile = profile_list.get_video_stream_profile(self.width, 0, OBFormat.Y16, 30)
                config.enable_stream(depth_profile)
            except Exception as e:
                print(f"[相机] 深度流配置失败: {e}")
                raise

            # 开启 D2C 对齐 (硬件对齐)
            config.set_align_mode(OBAlignMode.HW_MODE)
        
        self.pipeline.start(config)
        mode_str = "RGB-D + D2C 对齐" if self.use_depth else "仅 RGB（深度已关闭）"
        print(f"[相机] Orbbec Femto Bolt 已启动 — {mode_str} ({self.width}x{self.height})")

    def read(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """读取一帧图像。
        返回:
            rgb: (H, W, 3) uint8, RGB 格式, 或 None
            depth: (H, W) float32, 单位 mm, 或 None（use_depth=False 时始终 None）
        """
        if self.pipeline is None:
            return None, None

        # 等待一帧数据
        frames = self.pipeline.wait_for_frames(100)
        if frames is None:
            return None, None

        color_frame = frames.get_color_frame()
        if color_frame is None:
            return None, None

        # 提取 RGB 图像
        color_data = np.asanyarray(color_frame.get_data())
        rgb = color_data.reshape((color_frame.get_height(), color_frame.get_width(), 3))

        # 提取深度图像（仅 use_depth=True）
        if not self.use_depth:
            return rgb, None

        depth_frame = frames.get_depth_frame()
        if depth_frame is None:
            return rgb, None

        depth_data = np.asanyarray(depth_frame.get_data())
        # pyorbbecsdk 返回的 depth_data 是 1D array
        if depth_data.dtype == np.uint8:
            depth_data = depth_data.view(np.uint16)
        depth = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))

        return rgb, depth.astype(np.float32)

    def close(self):
        """释放相机"""
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
            print("[相机] 已释放")


# ═══════════════════════════════════════════════════════════════════════════
# CLI 参数
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="DOBOT CR5 真机 VLA 数据采集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--ip", type=str, default="192.168.50.104",
                    help="机器人 IP 地址 (默认: 192.168.50.102)")
    p.add_argument("--cam_id", type=int, default=0,
                    help="相机设备索引 (默认: 0)")
    p.add_argument("--cam_width", type=int, default=1280, help="相机宽度 (默认: 640)")
    p.add_argument("--cam_height", type=int, default=960, help="相机高度 (默认: 480)")
    p.add_argument("--port", type=int, default=9876, help="手柄 UDP 端口 (默认: 9876)")
    p.add_argument("--hz", type=int, default=5, help="采集/控制频率 Hz (默认: 5)")
    p.add_argument("--deadzone", type=float, default=0.15, help="摇杆死区 (默认: 0.15)")
    p.add_argument("--image_size", type=int, default=224, help="保存图像尺寸 (默认: 224)")
    p.add_argument("--save_dir", type=str, default=None,
                    help="数据保存目录 (默认: data/vla_dataset_real/)")
    p.add_argument("--task", type=str, default="pour cola into cup",
                    help="任务语言描述 (默认: pour water from bottle into cup)")
    p.add_argument("--min_steps", type=int, default=10,
                    help="有效 episode 最少步数 (默认: 10)")
    p.add_argument("--no_robot", action="store_true", default=False,
                    help="仅相机模式（不连接机器人）")
    p.add_argument("--gripper_port", type=str, default=None,
                    help="夹爪串口（默认: 自动检测）")
    p.add_argument("--speed_ratio", type=int, default=50,
                    help="机器人速度比例 1-100 (默认: 30)")

    p.add_argument("--pos_scale", type=float, default=6.0,
                    help="ServoP: 手柄 [-1,1] → 位置增量 mm/step (默认: 2.0)")
    p.add_argument("--rot_scale", type=float, default=3.0,
                    help="ServoP: 手柄 [-1,1] → 姿态增量 °/step (默认: 1.0)")

    # -- 初始关节角度（度），与 log/experiment_log.md 机械臂初始位置一致 --
    p.add_argument("--home_joints", type=float, nargs=6,
                    default=[34.4919, 13.2380, 120.5698, -43.8078, -90.0024, -0.0243],
                    help="初始关节角度（度）(默认: experiment_log 记录位姿)")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# 状态向量构建
# ═══════════════════════════════════════════════════════════════════════════

def build_state_vector(robot: DobotCR5, gripper: GripperController) -> np.ndarray:
    """构建 19 维状态向量

    布局: [关节角(6) + 关节速度(6) + 夹爪状态(1) + 末端位姿(6)] = 19 维

    关节角: 度
    关节速度: 度/秒
    夹爪状态: 1.0=打开, 0.0=关闭
    末端位姿: [x(m), y(m), z(m), rx(°), ry(°), rz(°)]
    """
    state = np.zeros(19, dtype=np.float32)

    # 关节角度 (6)
    state[0:6] = robot.joint_angles

    # 关节速度 (6)
    state[6:12] = robot.actual_joint_speeds

    # 夹爪状态 (1)
    state[12] = gripper.get_state()

    # 末端位姿 (6): cartesian_pose 前3维是位置(m)，后3维是姿态(°)
    state[13:19] = robot.cartesian_pose

    return state


def build_dummy_state(gripper: GripperController) -> np.ndarray:
    """无机器人连接时的虚拟状态向量"""
    state = np.zeros(19, dtype=np.float32)
    state[12] = gripper.get_state()
    return state


# ═══════════════════════════════════════════════════════════════════════════
# 动作向量：由 feedback 口更新的位姿在相邻采样间的差分（非手柄指令）
# ═══════════════════════════════════════════════════════════════════════════

def _normalize_angle_degrees(delta_deg: np.ndarray) -> np.ndarray:
    """逐分量限制到 [-180, 180]"""
    return (delta_deg + 180.0) % 360.0 - 180.0


def build_action_from_feedback_delta(
    prev_state: np.ndarray, curr_state: np.ndarray, gripper: GripperController
) -> np.ndarray:
    """7-DoF 动作：由 Dobot feedback 解析得到的笛卡尔位姿相邻帧差分。

    state 布局与 build_state_vector 一致；关节/速度来自同一 feedback 包更新。
    位置: m；姿态差: rad；夹爪: [-1, 1]。
    """
    action = np.zeros(7, dtype=np.float32)
    action[0:3] = curr_state[13:16] - prev_state[13:16]
    action[3:6] = _normalize_angle_degrees(curr_state[16:19] - prev_state[16:19]) * (
        np.pi / 180.0
    )
    action[6] = gripper.get_state() * 2.0 - 1.0
    return action


# ═══════════════════════════════════════════════════════════════════════════
# 手柄遥操作主循环（与 teleop_collect 按键/映射一致）
# ═══════════════════════════════════════════════════════════════════════════

PHASE_NAMES = ["phase_0_grasp", "phase_1_move", "phase_2_pour"]
NUM_PHASES = len(PHASE_NAMES)


def run_gamepad_mode(
    args,
    robot: DobotCR5 | None,
    camera: CameraManager,
    gripper: GripperController,
    recorder: EpisodeRecorder,
    receiver: GamepadReceiver,
    save_dir: str,
):
    """ServoP 手柄控制；HDF5 actions 为 feedback 位姿差分。"""

    mapper = GamepadMapper(deadzone=args.deadzone)

    recording = False
    pause_recording = False
    episode_count = 0
    control_dt = 1.0 / args.hz
    prev_rec_state: np.ndarray | None = None
    current_phase: int = 0  # 0=grasp, 1=move, 2=pour

    WIN_NAME = "真机采集 — 手柄 (teleop_collect 一致)"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 800, 500)

    print("\n操作说明：")
    print("  A = 切换录制 | B = 丢弃 | X = 夹爪 | Y = 复位 | Back = 暂停/继续录制 | Start = 退出")
    print("  D-pad → = 下一阶段 | D-pad ← = 上一阶段")
    print(f"  阶段列表: {' → '.join(PHASE_NAMES)}")
    print("  真机 Back 仅暂停写入，不恢复机器人状态。")
    print("  等待手柄连接...\n")

    try:
        while True:
            loop_t0 = time.time()

            gp_data = receiver.get_latest()
            action_np, events = mapper.map(gp_data)

            if events["quit"]:
                print("\n[INFO] 退出请求")
                break

            if events["reset_env"]:
                if recording:
                    print("[WARN] 录制停止并丢弃（复位）")
                    recorder.reset()
                    recording = False
                    pause_recording = False
                    prev_rec_state = None
                mapper.gripper_open = True
                if robot is not None:
                    print("[INFO] 回到初始位姿...")
                    robot.joint_mov_j(list(args.home_joints))
                    time.sleep(3)
                    print("[INFO] 复位完成")
                continue

            if events["toggle_gripper"]:
                gripper.toggle()
                state = "OPEN" if gripper.get_state() > 0.5 else "CLOSED"
                print(f"[GRIPPER] {state}")

            if events["toggle_record"]:
                if not recording:
                    recording = True
                    pause_recording = False
                    prev_rec_state = None
                    current_phase = 0
                    recorder.reset()
                    recorder.task_description = args.task
                    print(f"[●REC] 开始录制 — episode {episode_count}")
                    print(f"       任务: \"{args.task}\"")
                    print(f"       阶段: {current_phase}")
                else:
                    if recorder.num_steps >= args.min_steps:
                        ep_path = next_episode_path(save_dir)
                        recorder.save(ep_path, dt=1.0 / args.hz)
                        print(f"[✓保存] Episode 已保存: {ep_path}  ({recorder.num_steps} 步)")
                        episode_count += 1
                    else:
                        print(
                            f"[✗跳过] Episode 太短 ({recorder.num_steps} < {args.min_steps})，已丢弃"
                        )
                    recording = False
                    pause_recording = False
                    prev_rec_state = None
                    recorder.reset()

            if events["discard"]:
                if recording:
                    print(f"[✗丢弃] Episode 已丢弃 ({recorder.num_steps} 步)")
                    recorder.reset()
                    recording = False
                    pause_recording = False
                    prev_rec_state = None

            if events["toggle_pause_record"]:
                if not recording:
                    print("[WARN] 未在录制，忽略暂停。")
                elif not pause_recording:
                    pause_recording = True
                    print("[⏸PAUSE] 已暂停写入（机器人仍可动手柄控制）。")
                else:
                    pause_recording = False
                    if robot is not None:
                        curr_state = build_state_vector(robot, gripper)
                    else:
                        curr_state = build_dummy_state(gripper)
                    prev_rec_state = curr_state.copy()
                    print("[▶RESUME] 继续录制（已对齐 feedback 差分基准，未做位姿恢复）。")

            # ── 阶段切换 ──
            if events["next_phase"]:
                if not recording:
                    print("[WARN] 未在录制，忽略阶段切换。")
                elif current_phase < NUM_PHASES - 1:
                    current_phase += 1
                    print(f"[▶PHASE] 切换到阶段 {current_phase}")
                else:
                    print(f"[WARN] 已在最后阶段 ({current_phase})，无法前进。")

            if events["prev_phase"]:
                if not recording:
                    print("[WARN] 未在录制，忽略阶段切换。")
                elif current_phase > 0:
                    current_phase -= 1
                    print(f"[◀PHASE] 切换到阶段 {current_phase}")
                else:
                    print(f"[WARN] 已在第一阶段 ({current_phase})，无法后退。")

            has_motion = np.any(np.abs(action_np[:6]) > 1e-6)
            if has_motion and robot is not None:
                curr_pose = list(robot.cartesian_pose)
                target_x = curr_pose[0] * 1000.0 + action_np[0] * args.pos_scale
                target_y = curr_pose[1] * 1000.0 + action_np[1] * args.pos_scale
                target_z = curr_pose[2] * 1000.0 + action_np[2] * args.pos_scale
                target_rx = curr_pose[3] + action_np[3] * args.rot_scale
                target_ry = curr_pose[4] + action_np[4] * args.rot_scale
                target_rz = curr_pose[5] + action_np[5] * args.rot_scale
                robot.servo_p(target_x, target_y, target_z, target_rx, target_ry, target_rz)

            rgb, depth = camera.read()

            if robot is not None:
                curr_state = build_state_vector(robot, gripper)
            else:
                curr_state = build_dummy_state(gripper)

            if recording and (not pause_recording) and rgb is not None:
                if prev_rec_state is None:
                    prev_rec_state = curr_state.copy()
                depth_for_record = (
                    depth
                    if depth is not None
                    else np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
                )
                action_fb = build_action_from_feedback_delta(
                    prev_rec_state, curr_state, gripper
                )
                prev_rec_state = curr_state.copy()

                recorder.add_step(
                    rgb=rgb,
                    depth=depth_for_record,
                    state=curr_state,
                    action=action_fb,
                    timestamp=time.time(),
                    store_rgb_raw=False,
                    store_full_depth=True,
                    phase=current_phase,
                )

            display = _build_display(
                rgb,
                depth,
                curr_state,
                recording,
                episode_count,
                recorder.num_steps,
                gripper,
                mode="手柄",
                gp_connected=receiver.is_connected,
                paused=pause_recording,
                phase=current_phase,
            )
            cv2.imshow(WIN_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

            elapsed = time.time() - loop_t0
            sleep_time = control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C 中断")
        if recording and recorder.num_steps >= args.min_steps:
            ep_path = next_episode_path(save_dir)
            recorder.save(ep_path, dt=1.0 / args.hz)
            print(f"[✓保存] Ctrl+C 保存: {ep_path}")
            episode_count += 1

    cv2.destroyAllWindows()
    return episode_count


# ═══════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════


def _build_display(
    rgb: np.ndarray | None,
    depth: np.ndarray | None,
    state: np.ndarray,
    recording: bool,
    episode_count: int,
    rec_steps: int,
    gripper: GripperController,
    mode: str = "手柄",
    gp_connected: bool | None = None,
    paused: bool = False,
    phase: int = 0,
) -> np.ndarray:
    """构建 OpenCV 可视化画面"""

    panel_h = 400

    # RGB 画面
    if rgb is not None:
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = rgb_bgr.shape[:2]
        panel_w = int(w * panel_h / h)
        rgb_panel = cv2.resize(rgb_bgr, (panel_w, panel_h))
    else:
        panel_w = int(panel_h * 4 / 3)
        rgb_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        cv2.putText(rgb_panel, "No Camera", (10, panel_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)

    panels = [rgb_panel]

    # 深度画面
    if depth is not None:
        valid = np.isfinite(depth) & (depth > 0)
        if valid.any():
            d_min, d_max = depth[valid].min(), depth[valid].max()
            depth_norm = np.zeros_like(depth, dtype=np.uint8)
            if d_max > d_min:
                depth_norm[valid] = ((depth[valid] - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            depth_norm = np.zeros_like(depth, dtype=np.uint8)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
        h, w = depth_color.shape[:2]
        dp_w = int(w * panel_h / h)
        depth_panel = cv2.resize(depth_color, (dp_w, panel_h))
        cv2.putText(depth_panel, "Depth", (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        panels.append(depth_panel)

    display = np.hstack(panels)

    # HUD 叠加信息
    if recording and paused:
        rec_text = f"[PAUSED {rec_steps}]"
    elif recording:
        rec_text = f"[REC {rec_steps}]"
    else:
        rec_text = "[IDLE]"
    grip_text = "OPEN" if gripper.get_state() > 0.5 else "CLOSED"
    ee_pose = state[13:19]

    phase_name = f"Phase {phase}"
    hud_line1 = f"Mode:{mode}  {rec_text}  Ep:{episode_count}  {phase_name}  Grip:{grip_text}"
    if gp_connected is not None:
        hud_line1 += f"  GP:{'OK' if gp_connected else '--'}"

    hud_line2 = (
        f"EE: [{ee_pose[0]:.3f}, {ee_pose[1]:.3f}, {ee_pose[2]:.3f}] m  "
        f"[{ee_pose[3]:.1f}, {ee_pose[4]:.1f}, {ee_pose[5]:.1f}] deg"
    )

    color = (0, 255, 0) if recording else (200, 200, 200)
    cv2.putText(display, hud_line1, (10, display.shape[0] - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    cv2.putText(display, hud_line2, (10, display.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    if recording and paused:
        cv2.putText(
            display,
            "PAUSED  (Back to Resume)",
            (20, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )

    # 录制红点
    if recording and (not paused):
        cv2.circle(display, (display.shape[1] - 25, 25), 12, (0, 0, 255), -1)

    return display


# ═══════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # 数据保存目录
    save_dir = args.save_dir or os.path.join(_REPO_ROOT, "data", "vla_dataset_real")

    # ── 打印配置 ──
    print("\n" + "=" * 64)
    print("DOBOT CR5 — 真机 VLA 数据采集（手柄 / feedback 动作）")
    print("=" * 64)
    print(f"  机器人 IP      : {args.ip}")
    print(f"  相机分辨率     : {args.cam_width}x{args.cam_height}")
    print(f"  LoRA RGB 边长 : {args.image_size}（observations/images/rgb）")
    print(f"  采集频率       : {args.hz} Hz")
    print(f"  保存目录       : {save_dir}")
    print(f"  任务描述       : {args.task}（语言池与 teleop_collect 一致）")
    print(f"  最少步数       : {args.min_steps}")
    print(f"  连接机器人     : {'否' if args.no_robot else '是'}")
    print(f"  ServoP 位置增量: {args.pos_scale} mm/step（手柄 [-1,1]）")
    print(f"  ServoP 姿态增量: {args.rot_scale} °/step")
    print(f"  手柄 UDP 端口  : {args.port}")
    print(f"  初始关节角度   : {args.home_joints}")
    print("=" * 64)

    # ── 连接机器人 ──
    robot = None
    if not args.no_robot:
        robot = DobotCR5(ip_address=args.ip)
        try:
            robot.connect()
            time.sleep(1)
            print(f"[INFO] 机器人模式: {robot.robot_mode}")

            robot.enable()
            time.sleep(2)
            print(f"[INFO] 使能后模式: {robot.robot_mode}")

            robot.set_speed_ratio(args.speed_ratio)
            print(f"[INFO] 速度比例: {args.speed_ratio}%")

            print("[INFO] 运动到初始位姿...")
            robot.joint_mov_j(list(args.home_joints))
            time.sleep(5)
            print(f"[INFO] 当前关节角: {[round(a, 2) for a in robot.joint_angles]}")
            print(f"[INFO] 当前末端位姿: {[round(p, 4) for p in robot.cartesian_pose]}")

        except Exception as e:
            print(f"[ERROR] 机器人连接失败: {e}")
            robot = None
    else:
        print("[INFO] --no_robot 模式，不连接机器人")

    # ── 打开相机 ──
    camera = CameraManager(cam_id=args.cam_id, width=args.cam_width, height=args.cam_height)
    try:
        camera.open()
    except RuntimeError as e:
        print(f"[WARN] {e}")
        print("[WARN] 将继续运行但无相机画面")

    # ── 夹爪 ──
    gripper_port = args.gripper_port
    if gripper_port is None:
        print("[INFO] 未指定夹爪串口，开始自动检测...")
        gripper_port = _auto_detect_gripper_port()
        if gripper_port is None:
            print("[ERROR] 未检测到 Pico 2 W 夹爪，请检查连接或用 --gripper_port 指定端口")
            return
        print(f"[INFO] 自动检测到夹爪: {gripper_port}")
    gripper = GripperController(port=gripper_port)

    action_range_meta = (
        "feedback pose delta: dx,dy,dz (m), droll,dpitch,dyaw (rad), gripper [-1,1]"
    )
    recorder = EpisodeRecorder(
        image_size=args.image_size,
        task_description=args.task,
        action_range=action_range_meta,
    )

    receiver = GamepadReceiver(port=args.port)
    receiver.start()

    try:
        episode_count = run_gamepad_mode(
            args, robot, camera, gripper, recorder, receiver, save_dir
        )
    finally:
        # ── 清理 ──
        if receiver is not None:
            receiver.stop()
        gripper.disconnect()
        camera.close()
        if robot is not None:
            robot.disable()
            time.sleep(0.5)
            robot.disconnect()

    print(f"\n[INFO] 完成。共保存 {episode_count} 个 episodes")


if __name__ == "__main__":
    main()
