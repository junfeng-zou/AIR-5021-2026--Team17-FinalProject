"""
Gripper Controller for Dobot CR5/Nova 5

Controls a PWM-based gripper via CH340 USB-to-Serial adapter.
The gripper uses 500-2500us PWM (50Hz) corresponding to 0-180 degrees.

Hardware:
- STC8G1K08A-36I-SOP8 MCU generates 50Hz PWM
- CH340 handles USB-to-UART communication
- PWM range: 500us (0°) to 2500us (180°)
"""

import serial
import time
from typing import Optional


class GripperController:
    """
    Controller for PWM-based robotic gripper via serial communication.

    Protocol:
    - Command format: b'G{angle}\n' where angle is 0-180
    - Example: b'G90\n' sets gripper to 90 degrees
    """

    # PWM timing constants (microseconds)
    PWM_MIN_US = 500    # 0 degrees
    PWM_MAX_US = 2500   # 180 degrees
    PWM_FREQ_HZ = 50    # 50Hz PWM frequency

    # Angle range
    ANGLE_MIN = 0
    ANGLE_MAX = 180

    # Serial communication
    DEFAULT_BAUDRATE = 115200
    DEFAULT_TIMEOUT = 0.1

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = DEFAULT_BAUDRATE,
        timeout: float = DEFAULT_TIMEOUT,
        auto_connect: bool = True
    ):
        """
        Initialize the gripper controller.

        Args:
            port: Serial port path (e.g., '/dev/ttyUSB0' or 'COM3' on Windows)
            baudrate: Serial baudrate (default 115200)
            timeout: Serial read timeout in seconds
            auto_connect: Whether to automatically connect on init
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn: Optional[serial.Serial] = None
        self._current_angle: Optional[float] = None

        if auto_connect:
            self.connect()

    def connect(self) -> bool:
        """
        Establish serial connection to the gripper controller.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            # Wait for CH340/STC8G to initialize
            time.sleep(0.1)
            # Flush any pending data
            self.serial_conn.flushInput()
            self.serial_conn.flushOutput()
            print(f"[Gripper] Connected to {self.port}")
            return True
        except serial.SerialException as e:
            print(f"[Gripper] Failed to connect: {e}")
            return False

    def disconnect(self) -> None:
        """Close the serial connection."""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("[Gripper] Disconnected")

    def is_connected(self) -> bool:
        """Check if serial connection is open."""
        return self.serial_conn is not None and self.serial_conn.is_open

    def angle_to_pwm_us(self, angle: float) -> int:
        """
        Convert angle (degrees) to PWM pulse width (microseconds).

        Args:
            angle: Angle in degrees (0-180)

        Returns:
            PWM pulse width in microseconds (500-2500)
        """
        # Clamp angle to valid range
        angle = max(self.ANGLE_MIN, min(self.ANGLE_MAX, angle))
        # Linear mapping: 0° -> 500us, 180° -> 2500us
        pwm_us = self.PWM_MIN_US + (angle / self.ANGLE_MAX) * (self.PWM_MAX_US - self.PWM_MIN_US)
        return int(round(pwm_us))

    def pwm_us_to_angle(self, pwm_us: int) -> float:
        """
        Convert PWM pulse width (microseconds) to angle (degrees).

        Args:
            pwm_us: PWM pulse width in microseconds (500-2500)

        Returns:
            Angle in degrees (0-180)
        """
        pwm_us = max(self.PWM_MIN_US, min(self.PWM_MAX_US, pwm_us))
        angle = ((pwm_us - self.PWM_MIN_US) / (self.PWM_MAX_US - self.PWM_MIN_US)) * self.ANGLE_MAX
        return angle

    def set_angle(self, angle: float, wait: bool = True) -> bool:
        """
        Set the gripper to a specific angle.

        Args:
            angle: Target angle in degrees (0-180)
            wait: Whether to wait for acknowledgment

        Returns:
            True if command sent successfully
        """
        if not self.is_connected():
            print("[Gripper] Not connected, cannot set angle")
            return False

        # Clamp and validate angle
        angle = max(self.ANGLE_MIN, min(self.ANGLE_MAX, angle))
        pwm_us = self.angle_to_pwm_us(angle)

        # Format command: G{angle}\n (e.g., G90\n)
        command = f"G{angle:.1f}\n".encode('ascii')

        try:
            self.serial_conn.write(command)
            self.serial_conn.flush()

            if wait:
                # Wait for acknowledgment
                response = self.serial_conn.readline().decode('ascii').strip()
                if response.startswith("OK"):
                    self._current_angle = angle
                    print(f"[Gripper] Set to {angle:.1f}° (PWM: {pwm_us}us)")
                    return True
                else:
                    print(f"[Gripper] Unexpected response: {response}")
                    return False
            else:
                self._current_angle = angle
                print(f"[Gripper] Command sent: {angle:.1f}° (PWM: {pwm_us}us)")
                return True

        except serial.SerialException as e:
            print(f"[Gripper] Serial error: {e}")
            return False

    def get_current_angle(self) -> Optional[float]:
        """
        Query the current gripper angle.

        Returns:
            Current angle in degrees, or None if failed
        """
        if not self.is_connected():
            print("[Gripper] Not connected")
            return None

        try:
            self.serial_conn.write(b"Q\n")  # Query command
            self.serial_conn.flush()

            response = self.serial_conn.readline().decode('ascii').strip()
            if response.startswith("A"):  # Angle response: A{angle}
                angle = float(response[1:])
                self._current_angle = angle
                return angle
            else:
                print(f"[Gripper] Unexpected response: {response}")
                return None

        except (serial.SerialException, ValueError) as e:
            print(f"[Gripper] Error querying angle: {e}")
            return None

    def open(self, wait: bool = True) -> bool:
        """Open the gripper fully (180 degrees)."""
        return self.set_angle(180, wait=wait)

    def close(self, wait: bool = True) -> bool:
        """Close the gripper fully (0 degrees)."""
        return self.set_angle(0, wait=wait)

    def set_percentage(self, percentage: float, wait: bool = True) -> bool:
        """
        Set gripper position by percentage.

        Args:
            percentage: 0.0 (fully closed) to 1.0 (fully open)
        """
        percentage = max(0.0, min(1.0, percentage))
        angle = percentage * self.ANGLE_MAX
        return self.set_angle(angle, wait=wait)

    def calibrate(self, cycles: int = 3) -> bool:
        """
        Run a calibration cycle (open/close several times).

        Args:
            cycles: Number of open/close cycles
        """
        print(f"[Gripper] Running calibration ({cycles} cycles)...")

        for i in range(cycles):
            print(f"[Gripper] Cycle {i+1}/{cycles}")
            self.open()
            time.sleep(0.5)
            self.close()
            time.sleep(0.5)

        # End in open position
        self.open()
        print("[Gripper] Calibration complete")
        return True

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


def list_serial_ports() -> list:
    """List available serial ports."""
    import serial.tools.list_ports
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]


def main():
    """Interactive CLI for testing the gripper controller."""
    import argparse

    parser = argparse.ArgumentParser(description="Gripper Controller CLI")
    parser.add_argument(
        "--port", "-p",
        default="/dev/ttyUSB0",
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

    print("=" * 50)
    print("Gripper Controller CLI")
    print("=" * 50)
    print(f"Port: {args.port}")
    print(f"Baudrate: {args.baudrate}")
    print()
    print("Commands:")
    print("  o / open     - Open gripper fully (180°)")
    print("  c / close    - Close gripper fully (0°)")
    print("  <angle>      - Set specific angle (0-180)")
    print("  p <0-1>      - Set percentage (0.0-1.0)")
    print("  q / query    - Query current angle")
    print("  cal          - Run calibration cycle")
    print("  exit / quit  - Exit")
    print()

    controller = GripperController(port=args.port, baudrate=args.baudrate)

    if not controller.is_connected():
        print("Failed to connect. Check port and permissions.")
        print("On Linux, you may need: sudo usermod -a -G dialout $USER")
        return

    try:
        while True:
            try:
                cmd = input("> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if cmd in ("exit", "quit", "q"):
                break
            elif cmd in ("o", "open"):
                controller.open()
            elif cmd in ("c", "close"):
                controller.close()
            elif cmd in ("q", "query"):
                angle = controller.get_current_angle()
                if angle is not None:
                    print(f"Current angle: {angle:.1f}°")
            elif cmd == "cal":
                controller.calibrate()
            elif cmd.startswith("p "):
                try:
                    pct = float(cmd.split()[1])
                    controller.set_percentage(pct)
                except (IndexError, ValueError):
                    print("Usage: p <0-1>")
            else:
                try:
                    angle = float(cmd)
                    if 0 <= angle <= 180:
                        controller.set_angle(angle)
                    else:
                        print("Angle must be 0-180")
                except ValueError:
                    print("Unknown command")
    finally:
        controller.disconnect()


if __name__ == "__main__":
    main()
