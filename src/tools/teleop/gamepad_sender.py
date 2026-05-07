#!/usr/bin/env python3
"""
Gamepad UDP Sender — 本地 PC 端
================================

读取本地手柄（joystick）的摇杆轴 + 按键，
打包为 JSON 通过 UDP 发送到远程 3090。

用法：
    python tools/teleop/gamepad_sender.py --ip <3090_TAILSCALE_IP> --port 9876

依赖：
    pip install pygame
"""

import argparse
import json
import socket
import sys
import time

import pygame


def main():
    parser = argparse.ArgumentParser(description="Gamepad UDP Sender")
    parser.add_argument("--ip", type=str, required=True,
                        help="Remote receiver IP (e.g. 3090 Tailscale IP)")
    parser.add_argument("--port", type=int, default=9876,
                        help="Remote receiver UDP port (default: 9876)")
    parser.add_argument("--hz", type=float, default=50.0,
                        help="Send frequency in Hz (default: 50)")
    args = parser.parse_args()

    # ---- Init pygame & joystick ----
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("[ERROR] No joystick/gamepad detected. Please connect one and retry.")
        sys.exit(1)

    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"[INFO] Joystick detected: {js.get_name()}")
    print(f"[INFO]   Axes   : {js.get_numaxes()}")
    print(f"[INFO]   Buttons: {js.get_numbuttons()}")
    print(f"[INFO]   Hats   : {js.get_numhats()}")
    print(f"[INFO] Sending to {args.ip}:{args.port} @ {args.hz} Hz")
    print("[INFO] Press Ctrl+C to stop.\n")

    # ---- UDP socket ----
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (args.ip, args.port)
    interval = 1.0 / args.hz

    try:
        seq = 0
        while True:
            # Pump pygame events (required to update joystick state)
            pygame.event.pump()

            # Read axes
            axes = [round(js.get_axis(i), 4) for i in range(js.get_numaxes())]

            # Read buttons
            buttons = [js.get_button(i) for i in range(js.get_numbuttons())]

            # Read hats (D-pad)
            hats = [js.get_hat(i) for i in range(js.get_numhats())]

            # Build packet
            packet = {
                "axes": axes,
                "buttons": buttons,
                "hats": hats,
                "seq": seq,
                "timestamp": time.time(),
            }
            raw = json.dumps(packet).encode("utf-8")
            sock.sendto(raw, addr)

            # Print every ~1 second
            if seq % int(args.hz) == 0:
                print(f"[seq={seq}] axes={axes}  buttons={buttons}")

            seq += 1
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n[INFO] Sender stopped.")
    finally:
        sock.close()
        pygame.quit()


if __name__ == "__main__":
    main()
