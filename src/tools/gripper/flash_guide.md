# 夹爪固件烧录指南

## 编译好的文件

`gripper.hex` - 已编译好的固件文件，可直接烧录

## 重新编译

如需修改代码后重新编译：

```bash
cd tools/gripper
sdcc --model-small --out-fmt-ihx stc8g_gripper_firmware.c
cp stc8g_gripper_firmware.ihx gripper.hex
```

## 烧录方法

### 方法 1：STC-ISP (Windows)

1. 下载 STC-ISP 软件：http://www.stcmcudata.com/
2. 打开 STC-ISP
3. 选择型号：**STC8G1K08A**
4. 选择串口：选择 CH340 对应的 COM 口
5. 波特率：首次建议 **2400**，成功后可尝试 **115200**
6. 加载 `gripper.hex` 文件
7. 断开 PCB 电源（冷启动）
8. 点击"下载/编程"按钮
9. 给 PCB 上电
10. 等待下载完成

### 方法 2：stcgal (Linux)

```bash
# 安装
pip install stcgal

# 烧录
stcgal -P stc89 -p /dev/ttyUSB0 -b 115200 gripper.hex
```

如果权限不足：
```bash
sudo stcgal -P stc89 -p /dev/ttyUSB0 -b 115200 gripper.hex
```

### 方法 3：stcgal 自动波特率

```bash
# 先断开设备电源
stcgal -P stc89 -p /dev/ttyUSB0 --low-BAUD=2400 gripper.hex
# 然后给设备上电
```

## 烧录后验证

烧录完成后，运行测试脚本验证：

```bash
# 安装 Python 串口库
pip install pyserial

# 运行测试
python test_gripper.py --port /dev/ttyUSB0
```

或者使用交互式控制：

```bash
python gripper_controller.py --port /dev/ttyUSB0
```

## 常见问题

### 1. 无法识别设备

参考根目录的 CH340 问题排查：
```bash
# 禁用 brltty 服务
sudo systemctl stop brltty-udev.service
sudo systemctl disable brltty-udev.service
```

### 2. 烧录失败

- 确保冷启动（先断电，点下载后再上电）
- 降低波特率到 2400 或 4800
- 检查接线（TX/RX 是否接反）
- 确保共地（GND 连接可靠）

### 3. 舵机不动作

- 检查 PWM 引脚（P3.7）是否连接到舵机信号线
- 确保舵机供电正常（5V）
- 检查舵机 GND 是否与 MCU GND 共地

## 引脚定义

| STC8G 引脚 | 功能 | 连接 |
|-----------|------|------|
| P3.0 | RXD | CH340 TXD |
| P3.1 | TXD | CH340 RXD |
| P3.7 | PWM | 舵机信号线（白色/黄色）|
| VCC | 电源 | 5V/3.3V |
| GND | 地 | GND |
