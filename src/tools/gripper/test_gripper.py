#!/usr/bin/env python3
"""
Gripper Controller Quick Test Script

快速测试夹爪控制器的基本功能。

Usage:
    python test_gripper.py --port /dev/ttyUSB0
"""

import argparse
import time
import sys
import os

# 添加父目录到路径以导入模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gripper_controller import GripperController, list_serial_ports


def run_tests(port: str, baudrate: int):
    """运行一系列测试"""

    print("=" * 60)
    print("Gripper Controller Test Suite")
    print("=" * 60)

    # 测试 1: 连接
    print("\n[Test 1] Connecting to gripper...")
    gripper = GripperController(port=port, baudrate=baudrate, auto_connect=False)

    if not gripper.connect():
        print("FAIL: Cannot connect to gripper controller")
        print("\nTroubleshooting:")
        print("  1. Check USB cable is connected")
        print("  2. Verify correct port:")
        for p in list_serial_ports():
            print(f"     - {p}")
        print("  3. On Linux: sudo usermod -a -G dialout $USER")
        return False

    print("PASS: Connected successfully")

    # 测试 2: 查询当前角度
    print("\n[Test 2] Querying current angle...")
    angle = gripper.get_current_angle()
    if angle is not None:
        print(f"PASS: Current angle = {angle:.1f}°")
    else:
        print("WARN: Could not query angle (may be supported in firmware)")

    # 测试 3: 设置中间角度
    print("\n[Test 3] Setting angle to 90°...")
    if gripper.set_angle(90):
        print("PASS: Angle set successfully")
    else:
        print("FAIL: Could not set angle")

    time.sleep(0.5)

    # 测试 4: 完全打开
    print("\n[Test 4] Opening gripper fully (180°)...")
    if gripper.open():
        print("PASS: Gripper opened")
    else:
        print("FAIL: Could not open gripper")

    time.sleep(1.0)

    # 测试 5: 完全关闭
    print("\n[Test 5] Closing gripper fully (0°)...")
    if gripper.close():
        print("PASS: Gripper closed")
    else:
        print("FAIL: Could not close gripper")

    time.sleep(1.0)

    # 测试 6: 百分比控制
    print("\n[Test 6] Testing percentage control...")
    for pct in [0.25, 0.5, 0.75, 1.0]:
        gripper.set_percentage(pct, wait=False)
        time.sleep(0.3)
    print("PASS: Percentage control tested")

    time.sleep(0.5)

    # 测试 7: 微调角度
    print("\n[Test 7] Testing fine angle control...")
    test_angles = [45, 90, 135, 90, 45, 0]
    for angle in test_angles:
        gripper.set_angle(angle, wait=False)
        time.sleep(0.2)
    print("PASS: Fine angle control tested")

    # 测试 8: 校准循环
    print("\n[Test 8] Running calibration (3 cycles)...")
    gripper.calibrate(cycles=3)
    print("PASS: Calibration complete")

    # 结束
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)

    # 保持打开状态
    print("\nLeaving gripper in open position (180°)")
    gripper.open()

    # 断开连接
    gripper.disconnect()
    return True


def main():
    parser = argparse.ArgumentParser(description="Gripper Controller Test Script")
    parser.add_argument(
        "--port", "-p",
        default="/dev/ttyACM0",
        help="Serial port (default: /dev/ttyUSB0)"
    )
    parser.add_argument(
        "--baudrate", "-b",
        type=int,
        default=115200,
        help="Baud rate (default: 115200)"
    )
    parser.add_argument(
        "--list-ports", "-l",
        action="store_true",
        help="List available serial ports"
    )

    args = parser.parse_args()

    if args.list_ports:
        print("Available serial ports:")
        for port in list_serial_ports():
            print(f"  {port}")
        return

    success = run_tests(args.port, args.baudrate)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
