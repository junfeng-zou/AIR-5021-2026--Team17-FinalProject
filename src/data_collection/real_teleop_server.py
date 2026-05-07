#!/usr/bin/env python3
"""
DOBOT CR5 真机遥操作 — 数据发送端（Windows 机械臂PC）

功能：
  - 本地直接读取游戏手柄（pygame）
  - Dobot SDK 执行 ServoP 运动
  - Orbbec 相机采集 RGB-D
  - 通过 ZMQ (TCP) 将观测+动作数据发送到 Linux PC 进行 HDF5 录制

启动（Windows 机械臂PC）：
    python scripts/collect/real_teleop_server.py --robot_ip 192.168.50.102 --save_host 192.168.50.x --save_port 9875
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
import cv2
import numpy as np
import pygame

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from dobot_cr5 import DobotCR5

# -- Orbbec 相机 --
try:
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode
    HAS_ORBBEC = True
except ImportError:
    print("[WARN] pyorbbecsdk 未安装，相机功能不可用")
    HAS_ORBBEC = False
    Pipeline, Config, OBSensorType, OBFormat, OBAlignMode = None, None, None, None, None

# -- 夹爪 --
try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False
    print("[WARN] pyserial 未安装，夹爪功能不可用")

try:
    import zmq
    HAS_ZMQ = True
except ImportError:
    HAS_ZMQ = False
    print("[WARN] pyzmq 未安装，ZMQ 传输不可用")


# ═══════════════════════════════════════════════════════════════════════════
# 协议：ZMQ multipart — CMD / OBS
# ═══════════════════════════════════════════════════════════════════════════


class DataSender:
    """ZMQ PUSH → Linux 端 PULL（同一 tcp://host:port）。"""

    def __init__(self, host: str, port: int):
        if not HAS_ZMQ:
            raise RuntimeError("pyzmq 未安装，无法使用 ZMQ 传输")
        self.zmq_ctx = zmq.Context()
        self.zmq_sock = self.zmq_ctx.socket(zmq.PUSH)
        self.zmq_sock.setsockopt(zmq.SNDHWM, 10)
        self.zmq_sock.connect(f"tcp://{host}:{port}")

    def send_cmd(self, cmd: dict) -> None:
        payload = json.dumps(cmd, separators=(",", ":")).encode("utf-8")
        self.zmq_sock.send_multipart([b"CMD", payload])

    def send_obs(self, seq: int, rgb_bytes: bytes, depth: np.ndarray, state: np.ndarray,
                 action: np.ndarray, gripper_state: float, timestamp: float,
                 rgb_encoding: str = "jpeg", phase: int = 0) -> None:
        header = json.dumps(
            {
                "seq": seq,
                "rgb_len": len(rgb_bytes),
                "rgb_encoding": rgb_encoding,
                "depth_shape": list(depth.shape),
                "gripper": float(gripper_state),
                "timestamp": float(timestamp),
                "phase": int(phase),
            },
            separators=(",", ":"),
        ).encode("utf-8")
        self.zmq_sock.send_multipart(
            [
                b"OBS",
                header,
                rgb_bytes,
                depth.astype(np.float32).tobytes(),
                state.astype(np.float32).tobytes(),
                action.astype(np.float32).tobytes(),
                struct.pack(">d", timestamp),
            ]
        )

    def close(self):
        self.zmq_sock.close(linger=0)
        self.zmq_ctx.term()


# ═══════════════════════════════════════════════════════════════════════════
# 手柄映射（与 real_teleop_collect.py 相同，通过 pygame 本地读取）
# ═══════════════════════════════════════════════════════════════════════════

class GamepadMapper:
    """通过 pygame 本地读取游戏手柄（Xbox/通用双摇杆）"""

    def __init__(self, deadzone: float = 0.15):
        pygame.init()
        pygame.joystick.init()
        self.deadzone = deadzone
        self.gripper_open = True
        self._prev_buttons = []
        self._prev_hats = []
        self._last_axes = []
        self._last_buttons = []
        self._last_probe_ts = 0.0
        self._probe_interval_s = 1.0

        self._js = None
        self._try_connect_joystick(force_log=True)

    def _try_connect_joystick(self, force_log: bool = False) -> None:
        """尝试连接第一个可用手柄（支持运行时热插拔）"""
        try:
            pygame.joystick.quit()
            pygame.joystick.init()
            n = pygame.joystick.get_count()
        except pygame.error:
            n = 0

        if n == 0:
            if force_log:
                print("[WARN] 未检测到手柄")
            self._js = None
            self._prev_buttons = []
            return

        try:
            js = pygame.joystick.Joystick(0)
            js.init()
            self._js = js
            self._prev_buttons = []
            print(f"[INFO] 手柄已连接: {self._js.get_name()}")
        except pygame.error:
            self._js = None
            self._prev_buttons = []
            if force_log:
                print("[WARN] 手柄初始化失败")

    def _apply_deadzone(self, value: float) -> float:
        if abs(value) < self.deadzone:
            return 0.0
        sign = 1.0 if value > 0 else -1.0
        return sign * (abs(value) - self.deadzone) / (1.0 - self.deadzone)

    def _button_pressed(self, buttons: list, idx: int) -> bool:
        if idx >= len(buttons):
            return False
        curr = buttons[idx]
        prev = self._prev_buttons[idx] if idx < len(self._prev_buttons) else 0
        return curr == 1 and prev == 0

    def map(self) -> tuple[np.ndarray, dict]:
        """读取当前帧手柄状态，返回动作和事件"""
        action = np.zeros(7, dtype=np.float32)
        events = {
            "toggle_record": False,
            "discard": False,
            "toggle_gripper": False,
            "toggle_pause": False,
            "reset_env": False,
            "quit": False,
            "next_phase": False,
            "prev_phase": False,
        }

        if self._js is None:
            now = time.time()
            if now - self._last_probe_ts >= self._probe_interval_s:
                self._last_probe_ts = now
                self._try_connect_joystick()
            self._last_axes = []
            self._last_buttons = []
            action[6] = 1.0 if self.gripper_open else -1.0
            return action, events

        # 轮询 SDL 事件（比 event.get() 更稳，避免个别驱动下 KeyError/SystemError）
        try:
            pygame.event.pump()
            axes = [self._js.get_axis(i) for i in range(self._js.get_numaxes())]
            buttons = [self._js.get_button(i) for i in range(self._js.get_numbuttons())]
            hats = [list(self._js.get_hat(i)) for i in range(self._js.get_numhats())]
            self._last_axes = list(axes)
            self._last_buttons = list(buttons)
        except Exception:
            # 覆盖 pygame.error / SystemError 等异常，统一进入重连流程
            self._js = None
            self._prev_buttons = []
            self._prev_hats = []
            self._last_axes = []
            self._last_buttons = []
            action[6] = 1.0 if self.gripper_open else -1.0
            return action, events

        def axis(idx: int) -> float:
            return self._apply_deadzone(axes[idx]) if idx < len(axes) else 0.0

        # 左摇杆 X/Y → EE Y/X
        action[0] = -axis(1)
        action[1] = axis(0)
        # 右摇杆 X/Y → EE yaw / Z
        action[2] = -axis(3)
        action[5] = axis(2)
        # LT/RT → pitch
        if len(axes) > 5:
            lt = (axes[4] + 1.0) / 2.0
            rt = (axes[5] + 1.0) / 2.0
            action[4] = rt - lt
        # LB/RB → roll
        lb_held = buttons[4] if len(buttons) > 4 else 0
        rb_held = buttons[5] if len(buttons) > 5 else 0
        action[3] = float(rb_held - lb_held)

        action[:6] *= -1.0

        # A(0) 切换录制
        if self._button_pressed(buttons, 0):
            events["toggle_record"] = True
        # B(1) 丢弃
        if self._button_pressed(buttons, 1):
            events["discard"] = True
        # X(2) 夹爪
        if self._button_pressed(buttons, 2):
            self.gripper_open = not self.gripper_open
            events["toggle_gripper"] = True
        # Y(3) 复位
        if self._button_pressed(buttons, 3):
            events["reset_env"] = True
        # Back(6) 暂停/继续
        if self._button_pressed(buttons, 6):
            events["toggle_pause"] = True
        # Start(7) 退出
        if self._button_pressed(buttons, 7):
            events["quit"] = True

        # D-pad (hat) 阶段切换：右 = 下一阶段，左 = 上一阶段
        if hats:
            hat_x = hats[0][0] if isinstance(hats[0], (list, tuple)) else 0
            prev_hat_x = (
                self._prev_hats[0][0]
                if self._prev_hats and isinstance(self._prev_hats[0], (list, tuple))
                else 0
            )
            if hat_x == 1 and prev_hat_x != 1:
                events["next_phase"] = True
            if hat_x == -1 and prev_hat_x != -1:
                events["prev_phase"] = True

        action[6] = 1.0 if self.gripper_open else -1.0

        self._prev_buttons = list(buttons)
        self._prev_hats = [list(h) if isinstance(h, (list, tuple)) else [0, 0] for h in hats]
        return action, events

    def debug_snapshot(self) -> tuple[list, list]:
        return list(self._last_axes), list(self._last_buttons)


# ═══════════════════════════════════════════════════════════════════════════
# 相机管理器（与 real_teleop_collect.py 相同）
# ═══════════════════════════════════════════════════════════════════════════

class CameraManager:
    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.pipeline = None

    def open(self):
        if not HAS_ORBBEC:
            raise RuntimeError("pyorbbecsdk 未安装")
        # 固定配置（Femto Bolt）
        # RGB: 1280x720@30 MJPEG
        # Depth: 640x576@30 Y16 (NFoV unbinned)
        # 对齐模式按 HW -> SW -> 关闭 逐级回退，避免 "not support hardware d2c process"
        align_candidates = [
            ("HW_MODE", getattr(OBAlignMode, "HW_MODE", None)),
            ("SW_MODE", getattr(OBAlignMode, "SW_MODE", None)),
            ("DISABLE", getattr(OBAlignMode, "DISABLE", None)),
            ("ALIGN_DISABLE", getattr(OBAlignMode, "ALIGN_DISABLE", None)),
        ]

        last_err = None
        for align_name, align_mode in align_candidates:
            if align_mode is None:
                continue
            try:
                self.pipeline = Pipeline()
                config = Config()
                color_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
                color_profile = color_list.get_video_stream_profile(1280, 720, OBFormat.MJPG, 30)
                config.enable_stream(color_profile)
                depth_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
                depth_profile = depth_list.get_video_stream_profile(640, 576, OBFormat.Y16, 30)
                config.enable_stream(depth_profile)
                config.set_align_mode(align_mode)
                self.pipeline.start(config)
                print(
                    "[Camera] Orbbec 已启动: "
                    f"RGB 1280x720@30 MJPG, Depth 640x576@30 Y16, align={align_name}"
                )
                return
            except Exception as e:
                last_err = e
                try:
                    if self.pipeline is not None:
                        self.pipeline.stop()
                except Exception:
                    pass
                self.pipeline = None

        raise RuntimeError(f"相机配置失败（固定分辨率）: {last_err}")

    def read(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self.pipeline is None:
            return None, None
        frames = self.pipeline.wait_for_frames(100)
        if frames is None:
            return None, None
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            return None, None
        color_data = np.asanyarray(color_frame.get_data())
        h = color_frame.get_height()
        w = color_frame.get_width()

        rgb = None
        # 固定配置当前是 MJPG，先按 JPEG 解码；若失败再尝试原始重排
        try:
            encoded = color_data.reshape(-1).astype(np.uint8, copy=False)
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is not None:
                # OpenCV 输出 BGR，转换为 RGB
                rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        except Exception:
            rgb = None

        if rgb is None:
            # 回退：按未压缩 3 通道格式解析
            if color_data.size == h * w * 3:
                rgb = color_data.reshape((h, w, 3))
            else:
                return None, None

        depth_data = np.asanyarray(depth_frame.get_data())
        if depth_data.dtype == np.uint8:
            depth_data = depth_data.view(np.uint16)
        depth = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))
        return rgb, depth.astype(np.float32)

    def close(self):
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None


# ═══════════════════════════════════════════════════════════════════════════
# 夹爪控制器（与 real_teleop_collect.py 相同）
# ═══════════════════════════════════════════════════════════════════════════

class GripperController:
    ANGLE_OPEN = 80.0
    ANGLE_CLOSE = 120.0
    ACTUATION_TIME = 0.3

    def __init__(self, port: str = "COM3", baudrate: int = 115200):
        if not HAS_SERIAL:
            raise RuntimeError("pyserial 未安装")
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
            print(f"[Gripper] Connected to {self._port}")
        except self._serial.SerialException as e:
            print(f"[Gripper] Connection failed: {e}")
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
        self._start_state = self.get_state()
        self._send_cmd(f"G{self.ANGLE_OPEN:.1f}\n".encode())
        self._target_state = 1.0
        self._last_toggle_time = time.time()

    def close(self):
        self._start_state = self.get_state()
        self._send_cmd(f"G{self.ANGLE_CLOSE:.1f}\n".encode())
        self._target_state = 0.0
        self._last_toggle_time = time.time()

    def toggle(self):
        if self._target_state > 0.5:
            self.close()
        else:
            self.open()

    def get_state(self) -> float:
        elapsed = time.time() - self._last_toggle_time
        if elapsed >= self.ACTUATION_TIME:
            return self._target_state
        progress = elapsed / self.ACTUATION_TIME
        return self._start_state + (self._target_state - self._start_state) * progress

    def get_target_state(self) -> float:
        return self._target_state

    def disconnect(self):
        if self._conn and self._conn.is_open:
            self._conn.close()


# ═══════════════════════════════════════════════════════════════════════════
# 命令行参数
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="DOBOT CR5 数据发送端（机械臂PC）")
    p.add_argument("--robot_ip", type=str, default="192.168.50.102")
    p.add_argument("--save_host", type=str, required=True, help="Linux PC IP")
    p.add_argument("--save_port", type=int, default=9875, help="ZMQ 目标端口（Linux 端 PULL bind）")
    p.add_argument("--cam_width", type=int, default=640)
    p.add_argument("--cam_height", type=int, default=480)
    p.add_argument("--gripper_port", type=str, default="COM3", help="夹爪串口")
    p.add_argument("--speed_ratio", type=int, default=30)
    p.add_argument("--jpeg_quality", type=int, default=85)
    p.add_argument(
        "--rgb_codec",
        type=str,
        default="png",
        choices=["png", "jpeg"],
        help="RGB 网络编码：png 无损；jpeg 有损（体积更小）",
    )
    p.add_argument("--png_compression", type=int, default=3, help="PNG 压缩等级 0~9（无损，仅影响体积/速度）")
    p.add_argument("--control_hz", type=int, default=30, help="机器人控制频率(ServoP)，建议 30~33Hz")
    p.add_argument("--record_hz", type=int, default=5, help="数据采集/发送频率")
    # 兼容旧参数：--hz 等价于 --record_hz
    p.add_argument("--hz", dest="record_hz", type=int, help="兼容参数，等价于 --record_hz")
    p.add_argument("--pos_scale", type=float, default=6.0, help="手柄→mm/step")
    p.add_argument("--rot_scale", type=float, default=2.0, help="手柄→°/step")
    p.add_argument("--deadzone", type=float, default=0.20, help="手柄死区，增大可抑制抖动")
    p.add_argument("--motion_alpha", type=float, default=0.35, help="动作低通滤波系数(0~1)，越小越稳")
    p.add_argument("--max_pos_step", type=float, default=24, help="单步最大平移(mm)")
    p.add_argument("--max_rot_step", type=float, default=9, help="单步最大转角(°)")
    p.add_argument("--home_joints", type=float, nargs=6,
                    default=[34.4919, 13.2380, 120.5698, -43.8078, -90.0024, -0.0243])
    p.add_argument("--debug_gamepad", action="store_true", help="打印手柄原始 axes/buttons")
    p.add_argument("--debug_motion", action="store_true", help="打印运动目标与逆解失败信息")
    p.add_argument("--debug_interval", type=float, default=1.0, help="调试日志最小间隔秒")
    p.add_argument(
        "--task",
        type=str,
        default="pour water into cup",
        help="写入 HDF5 的语言指令（英文），随 start 发给 Linux；OpenVLA 模板：What action should the robot take to {task}?",
    )
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# 状态构建
# ═══════════════════════════════════════════════════════════════════════════

def build_state_vector(robot: DobotCR5, gripper: GripperController) -> np.ndarray:
    state = np.zeros(19, dtype=np.float32)
    state[0:6] = robot.get_joint_angles()
    state[6:12] = robot.get_actual_joint_speeds()
    state[12] = gripper.get_state()
    state[13:19] = robot.get_cartesian_pose()
    return state


# ═══════════════════════════════════════════════════════════════════════════
# 主循环
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    print("=" * 64)
    print("DOBOT CR5 — 数据发送端（机械臂PC）")
    print("=" * 64)
    print(f"  机器人 IP    : {args.robot_ip}")
    print(f"  本地手柄     : pygame 直接读取")
    print(f"  数据发送至   : tcp://{args.save_host}:{args.save_port} (ZMQ)")
    print(f"  RGB 编码     : {args.rgb_codec}" + (" (PNG 无损)" if args.rgb_codec == "png" else ""))
    print(f"  控制频率     : {args.control_hz} Hz")
    print(f"  采集频率     : {args.record_hz} Hz")
    print(f"  OpenVLA task : {args.task}")
    print("=" * 64)

    # -- 机器人 --
    print(f"[INFO] 连接机器人 {args.robot_ip}...")
    robot = DobotCR5(ip_address=args.robot_ip)
    robot.connect()
    time.sleep(1)
    robot.enable(2.0)
    time.sleep(2)
    robot.set_speed_ratio(args.speed_ratio)
    print(f"[INFO] 机器人已使能，速度比例 {args.speed_ratio}%")
    print(f"[INFO] 运动到初始位姿...")
    robot.joint_mov_j(list(args.home_joints))
    time.sleep(5)
    print(f"[INFO] 当前末端位姿: {[round(p, 4) for p in robot.get_cartesian_pose()]}")

    # -- 相机 --
    camera = None
    if HAS_ORBBEC:
        camera = CameraManager(width=args.cam_width, height=args.cam_height)
        try:
            camera.open()
        except Exception as e:
            print(f"[WARN] 相机启动失败: {e}")
            camera = None
    else:
        print("[WARN] pyorbbecsdk 未安装，无法使用相机")

    # -- 夹爪 --
    gripper = None
    if HAS_SERIAL:
        gripper = GripperController(port=args.gripper_port)
    else:
        print("[INFO] pyserial 未安装，夹爪不可用")

    # -- 本地手柄 --
    mapper = GamepadMapper(deadzone=args.deadzone)

    # -- 数据发送 --
    sender = DataSender(args.save_host, args.save_port)
    print(f"[INFO] ZMQ PUSH → tcp://{args.save_host}:{args.save_port}")

    # -- 状态机 --
    recording = False
    pause_recording = False
    seq = 0
    control_dt = 1.0 / max(1, args.control_hz)
    record_dt = 1.0 / max(1, args.record_hz)
    last_record_ts = 0.0
    prev_state = None
    show_preview = True
    filtered_action = np.zeros(7, dtype=np.float32)
    last_debug_ts = 0.0

    print("\n操作说明（手柄直连本机）：")
    print("  A = 切换录制 | B = 丢弃 | X = 夹爪 | Y = 复位 | Back = 暂停/继续 | Start = 退出")
    print("  D-pad → = 下一阶段 | D-pad ← = 上一阶段  (3 阶段: 0=Grasp / 1=Move / 2=Pour)")
    print("  键盘 q = 退出预览并结束程序")
    print("  等待手柄...\n")

    NUM_PHASES = 3
    current_phase = 0

    try:
        while True:
            loop_t0 = time.time()

            # -- 本地读取手柄 --
            action_np, events = mapper.map()
            alpha = float(np.clip(args.motion_alpha, 0.0, 1.0))
            filtered_action[:6] = (1.0 - alpha) * filtered_action[:6] + alpha * action_np[:6]
            filtered_action[6] = action_np[6]
            action_cmd = filtered_action.copy()
            if args.debug_gamepad:
                now = time.time()
                if now - last_debug_ts >= args.debug_interval:
                    axes, buttons = mapper.debug_snapshot()
                    print(f"[DBG][GAMEPAD] axes={np.round(np.array(axes), 3).tolist()} buttons={buttons} action_raw={np.round(action_np, 3).tolist()} action_filt={np.round(action_cmd, 3).tolist()}")
                    last_debug_ts = now

            if events["quit"]:
                print("\n[INFO] 退出请求")
                break

            if events["reset_env"]:
                if recording:
                    recording = False
                    pause_recording = False
                    sender.send_cmd({"cmd": "stop"})
                mapper.gripper_open = True
                print("[INFO] 运动到初始位姿...")
                robot.joint_mov_j(list(args.home_joints))
                time.sleep(5)
                prev_state = None
                continue

            if events["toggle_gripper"]:
                if gripper:
                    gripper.toggle()
                    print(f"[GRIPPER] {'OPEN' if gripper.get_target_state() > 0.5 else 'CLOSED'}")

            # -- 录制命令 --
            if events["toggle_record"]:
                if not recording:
                    recording = True
                    pause_recording = False
                    current_phase = 0
                    sender.send_cmd({"cmd": "start", "task": args.task})
                    print(f"[●REC] 开始录制  task={args.task!r}  phase={current_phase}")
                else:
                    recording = False
                    pause_recording = False
                    sender.send_cmd({"cmd": "stop"})
                    print("[■REC] 停止录制")
                    prev_state = None
                    continue

            if events["discard"]:
                if recording:
                    recording = False
                    pause_recording = False
                    sender.send_cmd({"cmd": "discard"})
                    print("[✗丢弃] Episode 已丢弃")
                    prev_state = None

            if events["toggle_pause"]:
                if not recording:
                    print("[WARN] 未在录制")
                elif not pause_recording:
                    pause_recording = True
                    sender.send_cmd({"cmd": "pause"})
                    print("[⏸PAUSE] 已暂停")
                else:
                    pause_recording = False
                    sender.send_cmd({"cmd": "resume"})
                    print("[▶RESUME] 继续录制")

            # -- 阶段切换 --
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

            # -- 执行末端运动 --
            has_motion = np.any(np.abs(action_cmd[:6]) > 1e-4)
            if has_motion:
                curr_pose = robot.get_cartesian_pose()
                dx = float(np.clip(action_cmd[0] * args.pos_scale, -args.max_pos_step, args.max_pos_step))
                dy = float(np.clip(action_cmd[1] * args.pos_scale, -args.max_pos_step, args.max_pos_step))
                dz = float(np.clip(action_cmd[2] * args.pos_scale, -args.max_pos_step, args.max_pos_step))
                drx = float(np.clip(action_cmd[3] * args.rot_scale, -args.max_rot_step, args.max_rot_step))
                dry = float(np.clip(action_cmd[4] * args.rot_scale, -args.max_rot_step, args.max_rot_step))
                drz = float(np.clip(action_cmd[5] * args.rot_scale, -args.max_rot_step, args.max_rot_step))
                target_x = curr_pose[0] * 1000.0 + dx
                target_y = curr_pose[1] * 1000.0 + dy
                target_z = curr_pose[2] * 1000.0 + dz
                target_rx = curr_pose[3] + drx
                target_ry = curr_pose[4] + dry
                target_rz = curr_pose[5] + drz
                if args.debug_motion:
                    print(
                        "[DBG][MOTION] "
                        f"curr_mm_deg={[round(curr_pose[0]*1000,2), round(curr_pose[1]*1000,2), round(curr_pose[2]*1000,2), round(curr_pose[3],2), round(curr_pose[4],2), round(curr_pose[5],2)]} "
                        f"delta_raw={[round(float(v),3) for v in action_np[:6]]} "
                        f"delta_filt={[round(float(v),3) for v in action_cmd[:6]]} "
                        f"delta_cmd_mm_deg={[round(dx,3), round(dy,3), round(dz,3), round(drx,3), round(dry,3), round(drz,3)]} "
                        f"target_mm_deg={[round(target_x,2), round(target_y,2), round(target_z,2), round(target_rx,2), round(target_ry,2), round(target_rz,2)]}"
                    )
                try:
                    robot.servo_p(target_x, target_y, target_z, target_rx, target_ry, target_rz)
                except Exception as e:
                    print(
                        "[ERR][IK] servo_p failed: "
                        f"{e}; target_mm_deg={[round(target_x,2), round(target_y,2), round(target_z,2), round(target_rx,2), round(target_ry,2), round(target_rz,2)]}"
                    )

            # -- 采集相机 --
            rgb, depth = (camera.read() if camera else (None, None))
            if show_preview and rgb is not None and depth is not None:
                try:
                    # 缩小一半后拼接显示（左 RGB，右 Depth）
                    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    rgb_show = cv2.resize(rgb_bgr, (rgb_bgr.shape[1] // 2, rgb_bgr.shape[0] // 2))

                    # 深度图拉伸到 8bit 便于可视化
                    dmin = float(np.min(depth))
                    dmax = float(np.max(depth))
                    if dmax > dmin:
                        depth_u8 = np.clip((depth - dmin) / (dmax - dmin) * 255.0, 0, 255).astype(np.uint8)
                    else:
                        depth_u8 = np.zeros_like(depth, dtype=np.uint8)
                    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
                    depth_show = cv2.resize(depth_color, (depth_color.shape[1] // 2, depth_color.shape[0] // 2))

                    if rgb_show.shape[:2] != depth_show.shape[:2]:
                        depth_show = cv2.resize(depth_show, (rgb_show.shape[1], rgb_show.shape[0]))
                    preview = np.hstack([rgb_show, depth_show])
                    cv2.imshow("RGB | Depth Preview (1/2)", preview)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("[INFO] 收到键盘退出请求(q)")
                        break
                except Exception as e:
                    print(f"[WARN] 预览显示失败: {e}")
                    show_preview = False

            # -- 构建状态 --
            if gripper:
                curr_state = build_state_vector(robot, gripper)
                gripper_state = gripper.get_state()
            else:
                curr_state = np.zeros(19, dtype=np.float32)
                curr_state[0:6] = robot.get_joint_angles()
                curr_state[6:12] = robot.get_actual_joint_speeds()
                curr_state[13:19] = robot.get_cartesian_pose()
                gripper_state = 1.0

            # -- 计算 feedback delta 动作 --
            if prev_state is None:
                action_to_send = action_np.copy()
            else:
                action_to_send = np.zeros(7, dtype=np.float32)
                action_to_send[0:3] = curr_state[13:16] - prev_state[13:16]
                delta_angle = curr_state[16:19] - prev_state[16:19]
                delta_angle = (delta_angle + 180.0) % 360.0 - 180.0
                action_to_send[3:6] = delta_angle * (np.pi / 180.0)
                action_to_send[6] = gripper_state * 2.0 - 1.0

            prev_state = curr_state.copy()

            # -- 发送到 Linux PC --
            if recording and (not pause_recording):
                now_ts = time.time()
                if now_ts - last_record_ts < record_dt:
                    elapsed = time.time() - loop_t0
                    sleep_time = control_dt - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    continue
                last_record_ts = now_ts
                timestamp = now_ts
                if rgb is not None:
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    rgb_buf = None
                    enc_tag = "none"
                    if args.rgb_codec == "png":
                        pc = int(np.clip(args.png_compression, 0, 9))
                        ok, enc = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, pc])
                        if ok:
                            rgb_buf = enc
                            enc_tag = "png"
                    else:
                        ok, enc = cv2.imencode(
                            ".jpg",
                            bgr,
                            [cv2.IMWRITE_JPEG_QUALITY, int(np.clip(args.jpeg_quality, 1, 100))],
                        )
                        if ok:
                            rgb_buf = enc
                            enc_tag = "jpeg"
                    if rgb_buf is None:
                        continue
                    payload = rgb_buf.tobytes()
                    sender.send_obs(
                        seq,
                        payload,
                        depth,
                        curr_state,
                        action_to_send,
                        gripper_state,
                        timestamp,
                        rgb_encoding=enc_tag,
                        phase=current_phase,
                    )
                else:
                    sender.send_obs(
                        seq,
                        b"",
                        np.zeros((480, 640), dtype=np.float32),
                        curr_state,
                        action_to_send,
                        gripper_state,
                        timestamp,
                        rgb_encoding="none",
                        phase=current_phase,
                    )
                seq += 1

            elapsed = time.time() - loop_t0
            sleep_time = control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C")
    finally:
        if camera:
            camera.close()
        if gripper:
            gripper.disconnect()
        if robot:
            robot.disable()
            time.sleep(0.5)
            robot.disconnect()
        sender.close()
        cv2.destroyAllWindows()
        pygame.quit()
        print("[INFO] 已关闭")

    return 0


if __name__ == "__main__":
    sys.exit(main())