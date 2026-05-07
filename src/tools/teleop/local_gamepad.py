#!/usr/bin/env python3
"""
Local Gamepad Reader — 本机直连手柄
====================================

与 GamepadReceiver 接口兼容的本地手柄读取器。
手柄直接连在运行推理的同一台电脑上时使用此类，无需 UDP sender。

用法：
    from tools.teleop.local_gamepad import LocalGamepadReader

    reader = LocalGamepadReader()
    reader.start()

    data = reader.get_latest()   # 与 GamepadReceiver.get_latest() 格式完全一致
    # data["axes"], data["buttons"], data["hats"]

    reader.stop()

独立测试：
    python tools/teleop/local_gamepad.py
"""

import threading
import time
from typing import Optional


class LocalGamepadReader:
    """通过 pygame 直接读取本机手柄，后台线程轮询，线程安全。

    API 与 GamepadReceiver 完全一致，可作为 drop-in 替换。
    """

    def __init__(self, joystick_id: int = 0, poll_hz: float = 50.0):
        self._js_id = joystick_id
        self._poll_interval = 1.0 / poll_hz
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._latest: Optional[dict] = None

    # ---- Public API (与 GamepadReceiver 一致) --------------------------------

    def start(self):
        """启动后台读取线程。"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """停止读取并清理。"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        print("[LocalGamepad] Stopped.")

    def get_latest(self) -> Optional[dict]:
        """返回最新一帧手柄数据，格式与 GamepadReceiver 完全一致。

        Returns dict with keys:
            - "axes":    List[float]
            - "buttons": List[int]
            - "hats":    List[List[int]]
            - "seq":     int
            - "timestamp": float
        """
        with self._lock:
            return self._latest

    def get_axes(self, default=None):
        data = self.get_latest()
        return data["axes"] if data else default

    def get_buttons(self, default=None):
        data = self.get_latest()
        return data["buttons"] if data else default

    @property
    def is_connected(self) -> bool:
        return self._latest is not None

    # ---- Internal ------------------------------------------------------------

    def _read_loop(self):
        try:
            import pygame
        except ImportError:
            print("[LocalGamepad] ERROR: pygame 未安装。请运行: pip install pygame")
            self._running = False
            return

        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            print("[LocalGamepad] ERROR: 未检测到手柄，请确认连接。")
            self._running = False
            pygame.quit()
            return

        js = pygame.joystick.Joystick(self._js_id)
        js.init()
        print(f"[LocalGamepad] 手柄已连接: {js.get_name()}")
        print(f"[LocalGamepad]   Axes={js.get_numaxes()}  Buttons={js.get_numbuttons()}  Hats={js.get_numhats()}")

        seq = 0
        try:
            while self._running:
                pygame.event.pump()

                axes = [round(js.get_axis(i), 4) for i in range(js.get_numaxes())]
                buttons = [js.get_button(i) for i in range(js.get_numbuttons())]
                hats = [list(js.get_hat(i)) for i in range(js.get_numhats())]

                data = {
                    "axes": axes,
                    "buttons": buttons,
                    "hats": hats,
                    "seq": seq,
                    "timestamp": time.time(),
                }
                with self._lock:
                    self._latest = data

                seq += 1
                time.sleep(self._poll_interval)
        finally:
            pygame.quit()


# ---------------------------------------------------------------------------
# Stand-alone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    reader = LocalGamepadReader()
    reader.start()
    print("[INFO] 等待手柄数据... (Ctrl+C 退出)\n")
    try:
        while True:
            data = reader.get_latest()
            if data is not None:
                print(
                    f"  [seq={data['seq']}]"
                    f"  axes={data['axes']}"
                    f"  buttons={data['buttons']}"
                    f"  hats={data.get('hats', [])}"
                )
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()
    finally:
        reader.stop()
