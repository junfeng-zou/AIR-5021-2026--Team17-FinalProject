# Gripper Controller - 夹爪控制模块

通过 USB 串口控制 PWM 舵机夹爪。

## 硬件架构

```
┌─────────────┐      USB       ┌─────────────┐     PWM     ┌─────────────┐
│   Host PC   │ ◄──────────►   │ Pico 2 W    │ ◄────────► │   夹爪舵机  │
│  (Python)   │   Virtual COM  │  (RP2350)   │   GP2     │  500-2500us│
└─────────────┘                └─────────────┘            │   50Hz PWM  │
                                                           └─────────────┘
```

## 技术规格

| 参数 | 值 |
|------|-----|
| PWM 频率 | 50Hz (20ms 周期) |
| 脉宽范围 | 500μs (0°) ~ 2500μs (180°) |
| 角度分辨率 | 约 0.1° |
| 通讯波特率 | 115200 |
| MCU 型号 | Raspberry Pi Pico 2 W (RP2350) |
| PWM 引脚 | GP2 (Pin 4) |

## 文件说明

- `gripper_controller.py` - Python 控制库（主机端）
- `pico2_w_gripper/pico2_w_gripper.ino` - Pico 2 W Arduino 代码
- `test_pico2_w.py` - Pico 2 W 版本测试脚本
- `stc8g_gripper_firmware.c` - STC8G 单片机固件（旧方案）
- `test_gripper.py` - STC8G 版本测试脚本（旧方案）
- `README.md` - 本文档

## 硬件版本

### 方案 A: Raspberry Pi Pico 2 W (推荐)

- MCU: RP2350 (Dual-core Cortex-M33, 520KB SRAM, 4MB Flash)
- 特点: 内置 WiFi/BT，USB-C 接口，16-bit PWM
- 代码: `pico2_w_gripper/pico2_w_gripper.ino`
- PWM 引脚: GP2 (Pin 4)

### 方案 B: STC8G 单片机 (旧方案)

- MCU: STC8G1K08A-36I-SOP8
- 特点: 成本极低，外围简单
- 固件: `stc8g_gripper_firmware.c`

## 使用方法

### 1. 烧录固件 (Pico 2 W 版本)

1. 安装 [Arduino-Pico core](https://github.com/earlephilhower/arduino-pico)
   或使用 [Thonny](https://thonny.org/) + MicroPython
2. 打开 `pico2_w_gripper/pico2_w_gripper.ino`
3. 选择 Board: "Raspberry Pi Pico"
4. Upload sketch

**Arduino IDE 配置:**
- 工具 → 开发板 → Raspberry Pi RP2040 → "Raspberry Pi Pico"
- 工具 → Upload Speed → "115200"
- 工具 → CPU Speed → "133 MHz"

**PlatformIO (推荐):**
```ini
[env:pico]
platform = raspberrypi
board = pico
framework = arduino
```

### 2. Python 控制

```bash
# 安装依赖
pip install pyserial

# 交互式控制
python tools/gripper/gripper_controller.py --port /dev/ttyACM0

# 测试脚本
python tools/gripper/test_pico2_w.py --port /dev/ttyACM0
```

### 3. 代码集成

```python
from tools.gripper.gripper_controller import GripperController

# 创建控制器
gripper = GripperController(port="/dev/ttyACM0")

# 设置角度
gripper.set_angle(90)      # 90 度位置
gripper.open()             # 完全打开 (180 度)
gripper.close()            # 完全关闭 (0 度)

# 百分比控制
gripper.set_percentage(0.5)  # 50% 开合

# 查询状态
angle = gripper.get_current_angle()

# 使用上下文管理器
with GripperController(port="/dev/ttyACM0") as g:
    g.open()
    # ... 自动断开
```

## 通讯协议

### 命令格式

| 命令 | 描述 | 示例 |
|------|------|------|
| `G<angle>\n` | 设置角度 | `G90\n` → 90 度 |
| `Q\n` | 查询角度 | `Q\n` → `A90\n` |
| `O\n` | 打开夹爪 | `O\n` → 180 度 |
| `C\n` | 关闭夹爪 | `C\n` → 0 度 |

### 响应格式

| 响应 | 描述 |
|------|------|
| `OK<angle>\n` | 设置成功确认 |
| `A<angle>\n` | 当前角度 |
| `ERR...\n` | 错误信息 |

## 引脚连接

### Pico 2 W

| Pico 引脚 | 功能 | 连接 |
|----------|------|------|
| GP2 (Pin 4) | PWM 输出 | 舵机 PWM 输入 (白/黄线) |
| GND (Pin 3) | GND | 舵机 GND (黑线) |
| VBUS (Pin 40) | 5V | 舵机 VCC (红线, 如需供电) |

**注意:** 大型舵机建议外部 5V/2A 供电，不要从 Pico 取电。

## 常见问题

### 1. 无法连接设备

**Linux:**
```bash
# 检查设备
ls -l /dev/ttyACM* /dev/ttyUSB*

# 添加用户到 dialout 组
sudo usermod -a -G dialout $USER
# 重新登录生效
```

**Windows:**
- 安装驱动后查看设备管理器 COM 端口号

### 2. PWM 舵机抖动

- 检查电源是否稳定（建议单独 5V 供电）
- 确认 GND 共地
- 检查 PWM 线是否过长
- 降低 PWM 更新频率

### 3. 角度不准确

可以在 `gripper_controller.py` 中调整映射：

```python
# 校准参数 (根据实际舵机调整)
PWM_MIN_US = 500    # 0 度实际脉宽
PWM_MAX_US = 2500   # 180 度实际脉宽
```

## License

MIT License
