#!/usr/bin/env python3
"""
Gamepad UDP Receiver — 远程 3090 端
====================================

线程安全的 UDP 手柄数据接收器，可嵌入 Isaac Lab 仿真循环。

独立测试：
    python tools/teleop/gamepad_receiver.py --port 9876

在 Isaac Lab 中使用：
    from tools.teleop.gamepad_receiver import GamepadReceiver

    receiver = GamepadReceiver(port=9876)
    receiver.start()

    # 在仿真循环中
    data = receiver.get_latest()
    if data is not None:
        axes    = data["axes"]      # List[float]
        buttons = data["buttons"]   # List[int]
        hats    = data["hats"]      # List[Tuple[int,int]]
        # ... 映射到机器人动作 ...

    receiver.stop()
"""

import argparse
import json
import socket
import threading
import time
from typing import Optional


class GamepadReceiver:
    """Thread-safe UDP receiver for gamepad data."""

    def __init__(self, port: int = 9876, bind_ip: str = "0.0.0.0"):
        self.port = port
        self.bind_ip = bind_ip

        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._latest: Optional[dict] = None

    # ---- Public API ----------------------------------------------------------

    def start(self):
        """Start background receiving thread."""
        if self._running:
            return
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.bind_ip, self.port))
        self._sock.settimeout(0.5)  # allow periodic check for stop signal
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        print(f"[GamepadReceiver] Listening on {self.bind_ip}:{self.port}")

    def stop(self):
        """Stop the receiver and clean up."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        print("[GamepadReceiver] Stopped.")

    def get_latest(self) -> Optional[dict]:
        """
        Return the latest gamepad state dict, or None if no data received yet.

        Returns dict with keys:
            - "axes":    List[float]        joystick axis values
            - "buttons": List[int]          button states (0/1)
            - "hats":    List[List[int]]    D-pad hat values
            - "seq":     int                sequence number
            - "timestamp": float            sender timestamp
        """
        with self._lock:
            return self._latest

    def get_axes(self, default: Optional[list] = None) -> Optional[list]:
        """Convenience: return just the axes list."""
        data = self.get_latest()
        return data["axes"] if data else default

    def get_buttons(self, default: Optional[list] = None) -> Optional[list]:
        """Convenience: return just the buttons list."""
        data = self.get_latest()
        return data["buttons"] if data else default

    @property
    def is_connected(self) -> bool:
        """True if at least one packet has been received."""
        return self._latest is not None

    # ---- Internal ------------------------------------------------------------

    def _recv_loop(self):
        while self._running:
            try:
                raw, addr = self._sock.recvfrom(4096)
                data = json.loads(raw.decode("utf-8"))
                with self._lock:
                    self._latest = data
            except socket.timeout:
                continue
            except (json.JSONDecodeError, OSError) as e:
                if self._running:
                    print(f"[GamepadReceiver] Error: {e}")
                continue


# ---------------------------------------------------------------------------
# Stand-alone test mode
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Gamepad UDP Receiver (standalone test)")
    parser.add_argument("--port", type=int, default=9876, help="UDP port (default: 9876)")
    parser.add_argument("--bind", type=str, default="0.0.0.0", help="Bind IP (default: 0.0.0.0)")
    args = parser.parse_args()

    receiver = GamepadReceiver(port=args.port, bind_ip=args.bind)
    receiver.start()

    print("[INFO] Waiting for gamepad data... (Ctrl+C to stop)\n")
    try:
        while True:
            data = receiver.get_latest()
            if data is not None:
                print(
                    f"  [seq={data.get('seq', '?')}]"
                    f"  axes={data['axes']}"
                    f"  buttons={data['buttons']}"
                    f"  hats={data.get('hats', [])}"
                )
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()
    finally:
        receiver.stop()


if __name__ == "__main__":
    main()
