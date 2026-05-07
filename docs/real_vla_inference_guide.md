# 真机 OpenVLA 推理说明文档

用训练好的 OpenVLA-7B + LoRA 直接驱动 DOBOT CR5 完成倒水任务的完整流程。

相关脚本：

- [`scripts/inference/real_openvla_lora_infer.py`](../scripts/inference/real_openvla_lora_infer.py) — 真机推理入口
- [`scripts/robot/measure_workspace_bbox.py`](../scripts/robot/measure_workspace_bbox.py) — 工作空间 bbox 自动测量（拖拽模式）
- [`scripts/collect/real_teleop_collect.py`](../scripts/collect/real_teleop_collect.py) — 真机数据采集（本推理脚本复用其硬件抽象）

---

## 1. 架构与数据流

```
┌───────────────────────────────────────────────────────────────────┐
│                      同一台 GPU 主机（24/80 GB 显存）                │
│                                                                   │
│  ┌──────────────┐     RGB(1280x960→224)   ┌──────────────────┐    │
│  │ Orbbec Femto │ ───────────────────────►│  OpenVLA-7B +    │    │
│  │   Bolt RGBD  │                         │  LoRA adapter    │    │
│  └──────────────┘                         │  (bf16 on CUDA)  │    │
│                                           └────────┬─────────┘    │
│                                       action[7]    │              │
│                                  (米/弧度/[-1,1])   ▼              │
│                                       ┌────────────────────────┐  │
│                                       │ 安全 clip + bbox + 夹爪 │  │
│                                       │ 累加到 cartesian_pose   │  │
│                                       └────────────┬────────────┘  │
│                                  ServoP(mm, °) over TCP           │
│                                                    ▼              │
│                                           ┌────────────────┐      │
│                                           │   DOBOT CR5    │      │
│                                           │ (192.168.50.x) │      │
│                                           └────────────────┘      │
└───────────────────────────────────────────────────────────────────┘
                              ▲
                              │ UDP :9876  (可选)
                              │
                    ┌──────────────────┐
                    │ 手柄（急停 / 暂停 │
                    │   / 回 home）    │
                    └──────────────────┘
```

闭环频率默认 **10 Hz**，与训练 `dt=0.1s` 对齐；每一拍流程：

1. **采图**：Orbbec Femto Bolt 硬件 D2C 对齐后取 RGB；
2. **缩放**：`1280×960 → 224×224` 方图（与训练时 `EpisodeRecorder(image_size=224)` 一致）；
3. **推理**：`infer_action_vector(...)` 返回 7 维动作（物理量：米/弧度/[-1,1]）；
4. **安全**：逐轴 clip（默认位置 5 mm/步、姿态 3°/步）→ 累加到当前末端位姿 → bbox clip；
5. **下发**：`robot.servo_p(x_mm, y_mm, z_mm, rx°, ry°, rz°)` + 夹爪 `open()/close()`。

---

## 2. 硬件与软件准备

### 2.1 硬件

| 项 | 说明 |
|---|---|
| 机械臂 | DOBOT CR5 / Nova5（TCP/IP 远程控制，默认 IP `192.168.50.102`） |
| 相机 | Orbbec Femto Bolt（通过 `pyorbbecsdk` 启用 D2C 对齐） |
| 夹爪 | Pico 2 W 串口 PWM 控制 LDX-335MG（与采集脚本一致） |
| GPU | ≥ 24 GB 显存（bf16 OpenVLA-7B），RTX 3090/4090 或 A100 均可 |
| 手柄（可选） | Xbox / 通用双摇杆；仅用作急停 / 暂停 / 回 home / 退出 |

### 2.2 网络

- 主机和机械臂同网段，能 ping 通 `192.168.50.102`（Dobot 默认 IP，可用 `--ip` 改）。
- 若使用手柄：手柄发送端（本地 PC）能通过 UDP :9876 连到本主机（Tailscale 亦可）。

### 2.3 Python 依赖

真机推理**不需要 Isaac Lab**，建议用独立 venv / conda：

```bash
conda create -n vla_real python=3.10 -y
conda activate vla_real

# OpenVLA 推理栈（与 LoRA 训练对齐）
pip install torch torchvision transformers>=4.40 "peft>=0.11,<0.14" \
            accelerate timm einops pillow numpy opencv-python

# 真机 I/O
pip install pyserial pyorbbecsdk

# 如使用手柄
pip install pygame
```

### 2.4 检查清单

- [ ] 机械臂上电、急停按钮已解除
- [ ] 夹爪 Pico 2 W USB 已接 (`ls /dev/ttyACM*` 能看到)
- [ ] 相机 USB3 接好，`lsusb | grep Orbbec` 能看到
- [ ] 桌面 bbox 已测（见 §3）或你对极限位姿有信心
- [ ] 训练产物 `LoRA_train/runs/<run>/final_lora/` 和 `dataset_statistics.json` 已准备好

---

## 3. 首次上机流程（**严格按顺序**）

### Step 1 ─ 测工作空间 bbox（10 分钟）

```bash
python scripts/robot/measure_workspace_bbox.py \
    --ip 192.168.50.102 \
    --margin_m 0.03 \
    --out runs/workspace_bbox.json
```

- 脚本连完机器人后会自动进入 **StartDrag 协作拖拽模式**，你可以手推末端。
- 把末端推到任务可能到达的**每个极端位姿**：取瓶最低点、倒水最高点、左右两侧、前后远近。
- 每到一个极端点按 **A（手柄）/ 空格（键盘）** 暂停 → 挪动过渡 → 再按一次继续，避免把"过渡路径"污染 bbox。
- 按 **Start / ESC** 退出，终端会打印两份 bbox：裸的和带 margin 的。**推理用带 margin 那份**（详见 [bbox 作用](#bbox-作用机制)）。

### Step 2 ─ Dry-run 核对量级（2 分钟）

```bash
python scripts/inference/real_openvla_lora_infer.py \
    --lora_path LoRA_train/runs/pouring_lora_a10080/final_lora \
    --dry_run --hz 5
```

脚本**不会动机器人和夹爪**，只打印每一步的 `Δxyz / Δrpy / g`。预期量级：

| 维度 | 正常范围 | 含义 |
|---|---|---|
| `Δxyz` | 每轴 ~`±0.5 ~ ±5 mm` | 物理位置增量 |
| `Δrpy` | 每轴 ~`±0.2° ~ ±2°` | 物理姿态增量（弧度在屏幕上转成度显示）|
| `g` | `−1.0 / +1.0`（偶尔中间值）| 夹爪命令，带 ±0.2 滞回 |

**如果 `Δxyz` 整体看起来是 ±0.01 以下并且动作几乎恒定**，说明 `dataset_statistics.json` 没加载正确或 `unnorm_key` 错了 → 检查 §6 排障。

### Step 3 ─ 降速安全试运行（5-10 分钟）

```bash
python scripts/inference/real_openvla_lora_infer.py \
    --lora_path LoRA_train/runs/pouring_lora_a10080/final_lora \
    --speed_ratio 10 \
    --hz 5 \
    --max_pos_step_m 0.003 \
    --max_rot_step_rad 0.035 \
    --bbox_xyz_m 0.2800 0.5400 -0.2200 0.2200 0.0300 0.4200
```

- `--speed_ratio 10`：ServoP 内部执行速度打到 10%，万一飞车也能救得回。
- `--hz 5`：控制频率减半，留更多反应时间。
- `--max_pos_step_m 0.003`：硬限制单步位置增量 ≤ 3 mm。
- `--bbox_xyz_m ...`：替换为 Step 1 测出来的带 margin bbox。
- 手柄按 **B = 急停**（`robot.stop_move()` 并暂停），**A = 暂停/继续**，**Y = 回 home**，**Start = 退出**。

确认：
- 机器人大致向目标（瓶子/杯子）移动；
- 没有反复卡在 bbox 边界；
- 夹爪开合发生在预期时刻（不是乱跳）。

### Step 4 ─ 正常速度运行

```bash
python scripts/inference/real_openvla_lora_infer.py \
    --lora_path LoRA_train/runs/pouring_lora_a10080/final_lora \
    --speed_ratio 20 \
    --hz 10 \
    --bbox_xyz_m 0.2800 0.5400 -0.2200 0.2200 0.0300 0.4200 \
    --task "pour cola from bottle into cup"
```

`--task` 支持训练时语言池里的任何一句（见 `real_teleop_collect._make_task_prompts`）。

---

## 4. 参数详解

### 4.1 动作语义（**最容易踩错**）

训练 HDF5 里的 `actions` 是 `build_action_from_feedback_delta` 写的 feedback 位姿差分：

| 维度 | 单位 | 语义 |
|---|---|---|
| `action[0:3]` | **m** | 相邻两帧末端位置增量 |
| `action[3:6]` | **rad** | 相邻两帧末端姿态增量（限到 [-π, π]）|
| `action[6]` | `[-1, 1]` | 夹爪（+1=开，−1=关）|

OpenVLA 输出经过 `q01/q99` 反归一化后**就是上面这些物理量**，所以：

```
--cart_action_gain 1.0          # 真机默认（✓ 物理量直接累加）
--cart_action_gain 8.0          # 仿真默认（⚠ 真机上会飞）
```

### 4.2 相机分辨率

- Orbbec Femto Bolt 实际输出 **1280×960 RGB**；采集、推理都在这一分辨率上捕获、随后 resize 到 224。
- `--cam_width 1280 --cam_height 960`（脚本默认已改为这一组）。
- **务必和训练数据捕获分辨率一致**，否则 FOV 不同，视觉分布偏移。

### 4.3 安全参数

| 参数 | 默认 | 作用 |
|---|---|---|
| `--max_pos_step_m` | `0.005` (5 mm) | 单步位置增量逐轴上限；VLA 偶发大值时兜底 |
| `--max_rot_step_rad` | `np.deg2rad(3)` | 单步姿态增量逐轴上限 |
| `--bbox_xyz_m` | 未设 | 目标点基坐标 bbox clip；强烈建议设 |
| `--gripper_hysteresis` | `0.2` | `|action[6]|<0.2` 不切换，防夹爪抖动 |
| `--warmup_steps` | `5` | 前 N 拍只采图不动，让相机/模型稳定 |
| `--dry_run` | `False` | 只打印，不下发 ServoP 和夹爪 |

#### bbox 作用机制

bbox 只夹**目标点**，不夹动作本身：

```python
target_xyz_mm = (curr_pose_m + action_m) * 1000
target_xyz_mm = clip(target_xyz_mm, bbox)   # ← 仅在这一步 clip
robot.servo_p(*target_xyz_mm, *target_rpy_deg)
```

所以 bbox 定得稍大（margin ≥ 2 cm）不会让策略失效，只是让"想走出桌面"的那一瞬被拉回到桌面边界上；margin 定得太小，策略会在边界反复卡。

### 4.4 模型与任务

| 参数 | 说明 |
|---|---|
| `--lora_path` | 训练产物 `final_lora/` 目录，必填 |
| `--base_checkpoint` | 默认 `openvla/openvla-7b`（HF Hub），也可给本地绝对路径 |
| `--dataset_stats` | 留空会自动在 `final_lora/` 同级/上层递归找 `dataset_statistics.json` |
| `--unnorm_key` | 默认 `pouring_hdf5`，必须与训练时写进 stats 的 key 一致 |
| `--task` | 语言指令；选训练语言池里的任意一句 |
| `--ema_alpha` | 动作 EMA 平滑，`0` 关闭；抖动大可以试 `0.3~0.7` |
| `--merge_lora` | 启动时把 LoRA 合并进基座（启动慢但每拍快一点）|

---

## 5. 控制面板

### 手柄（`--gamepad_port 9876`）

| 按键 | 动作 |
|---|---|
| **A**（button 0）| 暂停 / 继续推理 |
| **B**（button 1）| 紧急停止（`robot.stop_move()` + 暂停）|
| **Y**（button 3）| 回 home 位（会先进入暂停）|
| **Start**（button 7）| 退出程序（正常关停流程）|

### 键盘（OpenCV 窗口聚焦）

| 键 | 动作 |
|---|---|
| `ESC` | 退出 |

### HUD 信息

- 顶部彩色行：`a: dxyz=[±mm] drpy=[±°] g=±` —— 最近一次推理动作的物理量
- 采集脚本沿用的中部面板：EE 位姿、夹爪状态、episode 步数、手柄连接状态

---

## 6. 常见问题排障

### Q1. 动作量级接近零，机械臂几乎不动

- 检查 `dataset_statistics.json` 是否被加载（启动日志会打印路径和 `q01/q99`）；`q01/q99` 如果全是 ±1 左右而不是 ±0.01，说明 stats 没注入。
- 确认 `--unnorm_key pouring_hdf5` 和训练时写入的 key 一致（可以 `python -c "import json; print(list(json.load(open('runs/.../dataset_statistics.json')).keys()))"` 确认）。
- 确认用的是 `real` 训练的 LoRA，不是 sim 训练的（尺度差一两个数量级）。

### Q2. `ServoP` 报错 / 机器人不动但日志有输出

- 检查 `robot.robot_mode`：启动日志里应该是 `ENABLE`；若是 `ERROR`，先手动 `robot.clear_error()` 或面板复位。
- `--speed_ratio` 不要设为 0。
- Dobot ServoP 命令缓冲有时被塞满；降低 `--hz`（比如 5）或加大 `speed_ratio` 都能缓解。

### Q3. 相机报 `pyorbbecsdk` 无法导入

```bash
pip install pyorbbecsdk
# 若 pypi 没有适配：从官方源码编译（Orbbec 仓库 README）
```

如果相机打不开但推理流程仍想测试，可以先把相机换成固定帧图片测试流程（改造一下 `CameraManager.read` 临时 `return rgb, None`）。

### Q4. `from scripts.collect... import` 的 `ModuleNotFoundError`

如果是因为 `source /opt/ros/humble/setup.bash` 把 ROS 的 `scripts` 包引入了：推理脚本里已经用 `sys.path.insert(0, scripts/collect)` + `from real_teleop_collect import ...` 绕开。若你新写脚本，别用 `from scripts.xxx import`，直接 import 顶层模块。

### Q5. 机械臂在 bbox 边界反复卡

- bbox margin 太小。重跑 `measure_workspace_bbox.py`，margin 给到 3–5 cm。
- 或者 VLA 输出偏离训练分布较远；降 `--hz`、调 `--ema_alpha 0.5` 平滑一下。

### Q6. 夹爪反复开合

- 增大 `--gripper_hysteresis`（比如 0.4）。
- 若还是抖，说明 `action[6]` 在 0 附近跳，大概率是视觉输入和训练分布偏差大：检查光照、相机位姿、目标物是否和训练时一致。

---

## 7. 与仿真推理的关键区别

| 项 | 仿真（`openvla_lora_pouring_infer.py`）| 真机（`real_openvla_lora_infer.py`） |
|---|---|---|
| 运行环境 | Isaac Lab（`isaaclab.sh -p`）| 普通 Python venv（无 Isaac Lab）|
| 动作语义 | [-1, 1] 归一化，走 IK 乘 `pos_action_scale=5e-3` | **已是物理量**（米/弧度/[-1,1]）|
| `cart_action_gain` 默认 | `8.0`（需放大让 IK 看得见）| **`1.0`**（直接累加）|
| 执行器 | Isaac IK + ArticulationController | DOBOT TCP `ServoP` + 串口夹爪 |
| 观察源 | Isaac `CameraCfg` | Orbbec Femto Bolt (`pyorbbecsdk`) |
| 初始位姿 | env reset 随机化 | `JointMovJ(home_joints)` |
| 安全措施 | 仿真天然无害 | bbox clip + 单步 clip + 急停按钮 + dry_run |
| 退出失败影响 | 最多重启仿真 | **可能砸东西**；始终留急停通路 |

---

## 8. 快速命令速查

```bash
# 测 bbox
python scripts/robot/measure_workspace_bbox.py --out runs/ws.json

# dry-run
python scripts/inference/real_openvla_lora_infer.py \
    --lora_path LoRA_train/runs/pouring_lora_a10080/final_lora --dry_run

# 降速试跑
python scripts/inference/real_openvla_lora_infer.py \
    --lora_path LoRA_train/runs/pouring_lora_a10080/final_lora \
    --speed_ratio 10 --hz 5 --max_pos_step_m 0.003 \
    --bbox_xyz_m 0.28 0.54 -0.22 0.22 0.03 0.42

# 正常运行
python scripts/inference/real_openvla_lora_infer.py \
    --lora_path LoRA_train/runs/pouring_lora_a10080/final_lora \
    --speed_ratio 20 --hz 10 \
    --bbox_xyz_m 0.28 0.54 -0.22 0.22 0.03 0.42 \
    --task "pour cola from bottle into cup"
```

---

## 附录 A ─ 训练 ↔ 推理数据一致性检查清单

推理前建议跑一次：

```bash
python - << 'EOF'
import h5py
with h5py.File('data/vla_dataset_real/episode_0000.hdf5', 'r') as f:
    print("rgb shape :", f['observations/images/rgb'].shape)         # (T,224,224,3)
    print("state shape:", f['observations/state'].shape)             # (T,19)
    print("action shape:", f['actions'].shape)                       # (T,7)
    print("action_range:", f.attrs.get('action_range'))
    print("dt         :", f.attrs.get('dt'))
EOF
```

对照真机推理：

| 项 | 训练数据 | 真机推理 |
|---|---|---|
| 图像输入分辨率 | 224×224（downscale from 1280×960）| 224×224（同路径）✓ |
| 控制频率 | `1/dt = 10 Hz` | `--hz 10` ✓ |
| 动作单位 | 米/弧度/[-1,1] | `infer_action_vector` 反归一化后同单位 ✓ |
| `cart_action_gain` | N/A（采集时不乘）| **1.0** ✓ |

只要这张表每一行都打勾，模型训练时"看到了什么、输出了什么"和真机推理时一致，才不会因为 preprocess 差异走神。
