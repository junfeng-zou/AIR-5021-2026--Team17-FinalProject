# 数据采集说明文档

面向 OpenVLA-7B 微调的遥操作演示数据采集指南。

## 1. 概述

本项目通过 Isaac Lab 仿真环境 + 手柄遥操作的方式，采集 DOBOT CR5 机械臂执行倒水任务的演示数据，用于后续 OpenVLA-7B 的微调训练。

### 系统架构

```
┌──────────────────┐   UDP (50Hz)   ┌─────────────────────────────────┐
│  本地 PC          │  ──────────►  │  3090 服务器                     │
│  gamepad_sender   │               │  ┌─────────────────────────┐    │
│  (pygame + 手柄)  │               │  │ Isaac Lab 仿真环境       │    │
└──────────────────┘               │  │  - DOBOT CR5 机械臂      │    │
                                   │  │  - 桌面 + 杯子 + 瓶子    │    │
                                   │  │  - 80个水球粒子          │    │
                                   │  │  - 双相机（主 + 侧）     │    │
                                   │  └────────────┬────────────┘    │
                                   │               │                  │
                                   │  ┌────────────▼────────────┐    │
                                   │  │ HDF5 Episode 保存        │    │
                                   │  │  data/vla_dataset/       │    │
                                   │  └─────────────────────────┘    │
                                   └─────────────────────────────────┘
```

### 任务描述

默认语言指令：`"pour water from bottle into cup"`

机械臂需要：抓取桌上的瓶子 → 移动到杯子上方 → 倾倒水球进入杯中 → 放回瓶子。

## 2. 环境准备

### 依赖

- **服务器端**（3090）：Isaac Sim + Isaac Lab 已安装
- **本地端**（有手柄的 PC）：`pip install pygame`

### 关键文件

| 文件 | 用途 |
|------|------|
| `scripts/collect/teleop_collect.py` | 主采集脚本（服务器端运行） |
| `tools/teleop/gamepad_sender.py` | 手柄数据发送（本地端运行） |
| `tools/teleop/gamepad_receiver.py` | UDP 手柄数据接收 |
| `tools/vla/vla_data_writer.py` | HDF5 数据写入 |
| `envs/pouring_env.py` | 仿真环境定义 |

## 3. 启动流程

### 第一步：启动仿真环境（3090 服务器）

```bash
# 正常模式（有可视化窗口）
/home/zjf/IsaacLab/isaaclab.sh -p scripts/collect/teleop_collect.py

# 无头模式（无 GUI，节省资源）
/home/zjf/IsaacLab/isaaclab.sh -p scripts/collect/teleop_collect.py --headless
```

### 第二步：启动手柄发送（本地 PC）

```bash
python tools/teleop/gamepad_sender.py --ip <3090的Tailscale_IP>
```

### 第三步：开始采集

连接成功后，控制台会显示 `GP:OK`。使用手柄遥操作机械臂，按 **A 键**开始/停止录制。

## 4. 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--headless` | `False` | 无头模式运行 |
| `--port` | `9876` | UDP 手柄接收端口 |
| `--hz` | `10` | 数据录制频率（Hz） |
| `--deadzone` | `0.15` | 摇杆死区阈值 |
| `--save_dir` | `data/vla_dataset/` | 数据保存目录 |
| `--image_size` | `224` | 保存图像尺寸（OpenVLA 要求 224×224） |
| `--task` | `"pour water from bottle into cup"` | 语言任务指令 |
| `--min_steps` | `10` | 最小 episode 步数（不足则丢弃） |
| `--debug` | `False` | 启用调试输出 |

**示例：自定义采集**

```bash
/home/zjf/IsaacLab/isaaclab.sh -p scripts/collect/teleop_collect.py \
    --hz 15 \
    --task "pick up the bottle and pour water into the cup" \
    --save_dir data/custom_dataset/ \
    --min_steps 20
```

## 5. 手柄按键映射

详细映射参见 [gamepad_controls.md](gamepad_controls.md)。

### 运动控制

| 输入 | 控制 | 说明 |
|------|------|------|
| 左摇杆 ↑↓ | EE 前后 (X) | 推杆向前 → +X |
| 左摇杆 ←→ | EE 左右 (Y) | 推杆向右 → +Y |
| 右摇杆 ↑↓ | EE 上下 (Z) | 推杆向上 → +Z |
| 右摇杆 ←→ | EE 偏航 (Yaw) | 推杆向右 → 顺时针 |
| LT / RT | EE 俯仰 (Pitch) | LT 俯 / RT 仰 |
| LB / RB | EE 翻滚 (Roll) | LB 逆时针 / RB 顺时针 |

### 功能按键

| 按键 | 功能 | 说明 |
|------|------|------|
| **A** | 录制开关 | 按一次开始录制，再按一次保存并停止 |
| **B** | 丢弃 episode | 删除当前未完成的录制数据 |
| **X** | 夹爪开合 | 每按一次切换开/关 |
| **Y** | 重置环境 | 回到初始状态（正在录制会被丢弃） |
| **Start** | 退出 | 安全退出程序 |
| **ESC**（键盘） | 退出 | 通过可视化窗口退出 |

## 6. 数据格式

每个 episode 保存为一个独立的 HDF5 文件。

### 文件命名

```
data/vla_dataset/
├── episode_0000.hdf5
├── episode_0001.hdf5
├── episode_0002.hdf5
└── ...
```

### HDF5 内部结构

```
episode_XXXX.hdf5
├── observations/
│   ├── images/
│   │   ├── rgb          (T, 224, 224, 3)   uint8     — 主相机 RGB
│   │   ├── depth        (T, 224, 224, 1)   float32   — 主相机深度图
│   │   └── rgb_side     (T, 224, 224, 3)   uint8     — 侧视相机 RGB（可选）
│   └── state            (T, 20)            float32   — 本体感受状态
├── actions               (T, 7)            float32   — 7-DoF 动作
├── timestamps            (T,)              float64   — 时间戳
└── attrs
    ├── num_steps         int               — 总步数
    ├── dt                float             — 采样间隔（秒）
    ├── task              str               — 语言指令
    ├── image_size        int               — 图像尺寸
    ├── model_target      str               — "OpenVLA-7B"
    ├── action_dim        int               — 7
    ├── action_names      list[str]         — 各维名称
    ├── action_range      str               — "[-1, 1]"
    └── robot             str               — "DOBOT CR5"
```

### 动作空间（7 维）

| 索引 | 名称 | 含义 | 范围 |
|------|------|------|------|
| 0 | `dx` | 末端 X 方向平移 | [-1, 1] |
| 1 | `dy` | 末端 Y 方向平移 | [-1, 1] |
| 2 | `dz` | 末端 Z 方向平移 | [-1, 1] |
| 3 | `droll` | 末端绕 X 轴旋转 | [-1, 1] |
| 4 | `dpitch` | 末端绕 Y 轴旋转 | [-1, 1] |
| 5 | `dyaw` | 末端绕 Z 轴旋转 | [-1, 1] |
| 6 | `gripper` | 夹爪状态 | +1=打开, -1=关闭 |

动作值为归一化的 [-1, 1]，环境内部通过 `pos_action_scale`（0.005 m/step）和 `rot_action_scale`（0.01 rad/step）缩放为实际位移。

### 状态向量（20 维）

| 索引 | 内容 | 维度 |
|------|------|------|
| 0–5 | 手臂关节位置 | 6 |
| 6–11 | 手臂关节速度（×0.1） | 6 |
| 12–13 | 夹爪手指位置 | 2 |
| 14–16 | 末端执行器位置 (xyz) | 3 |
| 17–19 | 杯子位置 (xyz) | 3 |

### 相机配置

| 参数 | 主相机 | 侧视相机 |
|------|--------|----------|
| 原始分辨率 | 960×540 | 640×480 |
| 保存尺寸 | 224×224 | 224×224 |
| 数据类型 | RGB + 深度 | RGB |
| 更新频率 | 20 Hz | 10 Hz |
| 视角 | 斜前方俯视工作区 | 侧面俯瞰 |

## 7. 采集流程建议

### 单次 Episode 采集步骤

1. 按 **Y** 重置环境，确认机械臂回到初始位置
2. 按 **A** 开始录制（HUD 显示 `[REC]`，绿色）
3. 使用左摇杆控制末端移动到瓶子位置
4. 按 **X** 关闭夹爪，夹住瓶子
5. 移动瓶子到杯子上方，倾倒水球
6. 完成后按 **A** 停止录制并保存

### 数据多样性机制

系统内置两套自动多样化机制，无需手动调整：

**位置随机化**：每次环境 reset（按 Y 或 episode 自然结束）时，杯子和瓶子的位置会在以下范围内随机采样：

| 物体 | X 范围 (m) | Y 范围 (m) | 备注 |
|------|-----------|-----------|------|
| 杯子 | [0.30, 0.55] | [-0.15, 0.15] | 桌面中央区域 |
| 瓶子 | [0.30, 0.55] | [-0.15, 0.25] | 与杯子至少 0.10m 距离 |

这意味着每次 reset 后杯子和瓶子的相对位置都不同，操作者需要根据当前布局调整策略。

如需关闭随机化（调试时），在 `envs/pouring_env.py` 中设置：
```python
randomize_positions = False
```

**语言指令多样化**：每次按 A 开始录制时，系统从预定义的 8 条语义等价指令中随机选一条。示例：
- "pour water from bottle into cup"
- "pick up the bottle and pour water into the cup"
- "fill the cup with water from the bottle"
- "grasp the bottle and pour its contents into the cup"

控制台会打印当前 episode 选中的指令。也可通过 `--task` 自定义基础指令。

### 数据质量控制

- **最小步数保护**：不足 `--min_steps`（默认 10）步的 episode 自动丢弃
- **相机预热**：每次 reset 后前 5 步不录制，避免空白帧
- **频率一致性**：数据严格以 `--hz` 频率采样，HDF5 中的 `dt` 与实际采样间隔一致
- **实时同步**：仿真与挂钟时间同步，遥操作手感接近真实

### 建议采集量

OpenVLA-7B 微调一般建议：
- **最少 50 个 episode**（能跑通训练流程）
- **推荐 200–500 个 episode**（获得较好泛化效果）
- **理想 500–1000 个 episode**（位置变化大时需要更多数据）
- 每个 episode 通常 50–200 步（5–20 秒 @10Hz）

### 采集策略建议

1. **不要追求每次都完美**——偶尔的失败轨迹也有价值，模型可以学到纠错
2. **按 Y 重置后观察新位置**——确认杯子和瓶子的位置已变化再开始录制
3. **均匀覆盖不同布局**——如果连续几次随机到很近的布局，按 Y 多重置几次
4. **分批采集**——每次采集 50–100 条后休息，避免疲劳导致操作质量下降

## 8. 数据验证

采集完成后可以用 Python 快速验证：

```python
import h5py
import numpy as np

with h5py.File("data/vla_dataset/episode_0000.hdf5", "r") as f:
    print("=== Episode Info ===")
    print(f"  Task:       {f.attrs['task']}")
    print(f"  Steps:      {f.attrs['num_steps']}")
    print(f"  dt:         {f.attrs['dt']} s")
    print(f"  Robot:      {f.attrs['robot']}")
    print(f"  Model:      {f.attrs['model_target']}")

    rgb = f["observations/images/rgb"][:]
    actions = f["actions"][:]
    state = f["observations/state"][:]
    print(f"\n=== Data Shapes ===")
    print(f"  RGB:        {rgb.shape}  dtype={rgb.dtype}")
    print(f"  Actions:    {actions.shape}  dtype={actions.dtype}")
    print(f"  State:      {state.shape}  dtype={state.dtype}")
    print(f"  Action range: [{actions.min():.3f}, {actions.max():.3f}]")
```

## 9. 后续步骤：OpenVLA-7B 微调

当前采集的 HDF5 格式是中间存储格式。OpenVLA-7B 微调需要将数据转换为 RLDS（Reinforcement Learning Datasets）格式。转换流程：

```
HDF5 episodes  ──►  RLDS (TFRecord)  ──►  OpenVLA fine-tuning
```

转换时需要确保：
1. **图像**：224×224 RGB，uint8（已满足）
2. **动作**：7 维连续 action，归一化到 [-1, 1]（已满足）
3. **语言指令**：每个 episode 对应一条自然语言指令（已通过 `task` 属性记录）
4. **Episode 边界**：每个 HDF5 文件对应一个完整 episode（已满足）

## 10. 故障排查

| 问题 | 可能原因 | 解决方法 |
|------|----------|----------|
| HUD 显示 `GP:--` | 手柄数据未到达 | 检查网络连通性和端口（默认 9876） |
| 操作没有反应 | 摇杆值在死区内 | 减小 `--deadzone`（如 0.08） |
| 移动太快/太慢 | 缩放参数不合适 | 调整 `pouring_env.py` 中的 `pos_action_scale` / `rot_action_scale` |
| Episode 自动丢弃 | 步数不足 `--min_steps` | 录制更长的演示，或减小 `--min_steps` |
| 第一帧图像异常 | 正常现象 | 系统已自动跳过 reset 后的前 5 步 |
| `[✗SKIP]` 提示 | Episode 太短 | 确保操作时间足够长再停止录制 |
