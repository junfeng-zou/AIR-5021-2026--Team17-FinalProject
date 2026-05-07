"""
Test script for Waveshare Raspberry Pi Pico 2 W Gripper Controller

Usage:
    python test_pico2_w.py --port /dev/ttyACM0
"""

import sys
import time
import argparse

# Add project root to path
sys.path.insert(0, "/home/zjf/pouring_VLA")

from tools.gripper.gripper_controller import GripperController


def test_sequence(controller: GripperController):
    """Run a test sequence on the gripper."""
    print("\n" + "=" * 50)
    print("Running Pico 2 W gripper test sequence...")
    print("=" * 50)

    # Test 1: Query current angle
    print("\n[Test 1] Query current angle...")
    angle = controller.get_current_angle()
    if angle is not None:
        print(f"  Current angle: {angle:.1f}°")
    else:
        print("  Failed to query angle (normal if gripper just started)")

    # Test 2: Sweep through range
    print("\n[Test 2] Sweep through 0° -> 90° -> 180° -> 90° -> 0°...")
    for target in [0, 90, 180, 90, 0]:
        print(f"  -> {target}°...", end=" ", flush=True)
        if controller.set_angle(target):
            print("OK")
        time.sleep(0.3)

    # Test 3: Open/Close
    print("\n[Test 3] Open/Close test...")
    print("  Opening...", end=" ", flush=True)
    controller.open()
    print("OK")
    time.sleep(0.5)

    print("  Closing...", end=" ", flush=True)
    controller.close()
    print("OK")
    time.sleep(0.5)

    # Test 4: Percentage control
    print("\n[Test 4] Percentage control (0%, 50%, 100%)...")
    for pct in [0.0, 0.5, 1.0]:
        print(f"  -> {pct * 100:.0f}%...", end=" ", flush=True)
        controller.set_percentage(pct)
        print("OK")
        time.sleep(0.3)

    print("\n" + "=" * 50)
    print("Test sequence complete!")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Test Pico 2 W Gripper Controller")
    parser.add_argument(
        "--port", "-p",
        default="/dev/ttyACM0",
        help="Serial port (default: /dev/ttyACM0)"
    )
    parser.add_argument(
        "--baudrate", "-b",
        type=int,
        default=115200,
        help="Baud rate (default: 115200)"
    )
    args = parser.parse_args()

    print(f"Connecting to {args.port} at {args.baudrate} baud...")
    controller = GripperController(
        port=args.port,
        baudrate=args.baudrate,
        auto_connect=True
    )

    if not controller.is_connected():
        print("ERROR: Failed to connect. Check:")
        print("  1. Pico 2 W is connected via USB")
        print("  2. Correct port is specified")
        print("  3. On Linux: check /dev/ttyACM0, /dev/ttyUSB0")
        print("  4. On Linux: 'sudo usermod -a -G dialout $USER' then re-login")
        return

    try:
        test_sequence(controller)
    finally:
        controller.disconnect()


if __name__ == "__main__":
    main()
