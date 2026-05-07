# OpenVLA 边缘部署优化方案 (V2.0)

> **目标设备**: Fogwise® AIRbox Q900 (Qualcomm QCS9100, 36GB LPDDR5, Hexagon HTP NPU)
> **模型**: OpenVLA-7B + LoRA (checkpoint-14000, `dobot_pouring`)
> **任务**: Dobot CR5 机械臂三阶段倒水 (Grasp → Move → Pour)

---

## 架构总览

```
【工作站离线处理】
原始 7B 模型 (BF16, ~14GB)
    ↓ LoRA 合并
合并模型 (~14GB)
    ├─ [SigLIP + DINOv2 + Projector] → ONNX → vision_projector.onnx
    ├─ [LLM Backbone (Llama-2-7B)] → GGUF Q4_K_M → openvla-llm-Q4_K_M.gguf (~4GB)
    └─ [动作解码参数] → action_head_params.json

【Q900 实时推理】
摄像头帧 (1280×720)
    ↓ resize + 双骨干归一化
(1,6,224,224) float32
    ↓ ONNX Runtime (Vulkan/CPU)
vision_emb (256, 4096) float32
    ↓ 注入 LLM KV-cache
llama-cpp-python (QNN HTP/CPU)
    ↓ greedy decode, n=7
action_token_ids [7]
    ↓ detokenize + 反归一化
7-DoF 物理动作 [m, m, m, rad, rad, rad, [0,1]]
    ↓
Dobot CR5 ServoP
```

---

## 关键设计决策

### 1. 为什么不用 mmproj.gguf？

OpenVLA 使用**融合双骨干**（DINOv2 + SigLIP，6通道），而 llama.cpp 的 `clip.cpp` 仅支持标准单骨干 CLIP（3通道）。直接生成 mmproj.gguf 需要大量修改 llama.cpp 源码，工程成本极高。

**选用方案**：ONNX Runtime 负责视觉，llama-cpp-python 的底层 C API 负责 embedding 注入。

### 2. 为什么 Action Head 不是独立模块？

OpenVLA 使用**动作离散化（Action Tokenization）**，不存在单独的连续输出线性层。
动作通过标准 LM Head（已包含在 GGUF 中）输出离散 Token ID，通过 `action_head_params.json` 中的映射表还原为物理量。

**解码公式**：
```
bin_index = vocab_size - token_id - 1    (= 32000 - token_id - 1)
normalized = bin_centers[bin_index]       (值域 ≈ [-1, 1])
physical   = 0.5 * (normalized + 1) * (q99 - q01) + q01   (对 mask=True 的维度)
```

### 3. 为什么必须 Greedy Decoding？

机器人控制要求动作确定性。任何温度 > 0 都会引入随机性导致动作漂移。
```bash
--temp 0.0 --n-predict 7
```

---

## 目录结构

```
edge_optimization/
├── README.md                          ← 本文档
├── log.md                             ← 执行日志
├── merged_model/                      ← LoRA 合并后的完整模型
├── components/
│   ├── llm_llama2_7b/                 ← HF 格式 LLM（转 GGUF 的中间产物）
│   ├── vision_projector/
│   │   ├── vision_projector.onnx      ← 主要推理文件（带外部数据）
│   │   ├── vision_projector.onnx.data ← ONNX 权重数据（~2.83 GB）
│   │   ├── vision_projector.pt        ← PyTorch state_dict 备份
│   │   └── vision_meta.json           ← 视觉模块元信息
│   └── action_head/
│       └── action_head_params.json    ← 动作解码参数（bin_centers + 归一化统计量）
├── gguf_models/
│   ├── openvla-llm-Q4_K_M.gguf       ← 主要部署文件（~4 GB）
│   └── openvla-llm-Q8_0.gguf         ← 高精度备用（~7 GB）
├── llama.cpp/                         ← llama.cpp 源码（含 QNN 后端）
├── scripts/
│   ├── step0_merge_lora.py            ← Phase 0: LoRA 合并
│   ├── step1_extract_components.py    ← Phase 0: 组件提取 + ONNX 导出
│   ├── step2_convert_llm_gguf.py      ← Phase 1: LLM → GGUF
│   ├── step3_quantize_vision.py       ← Phase 1: Vision INT8 量化（备用）
│   ├── step4_verify_pipeline.py       ← Phase 1: 端到端精度验证
│   ├── step5_verify_onnx_alignment.py ← Phase 1: ONNX 对齐验证 ⭐新增
│   ├── q900_inference.py              ← 旧版推理（已废弃，ctypes 不稳定）
│   ├── q900_inference_v2.py           ← V2.0 推理 ⭐新增，正式版本
│   └── q900_npu_inference.py          ← NPU 极限性能版（实验性）
└── qairt_v2.42.0.251225/              ← 高通 AIMET 工具链（备用路线）
```

---

## 工作站执行步骤（已完成）

| 步骤 | 脚本 | 状态 |
|------|------|------|
| LoRA 合并 | `step0_merge_lora.py` | ✅ 完成 |
| 组件提取 + ONNX 导出 | `step1_extract_components.py` | ✅ 完成 |
| LLM → GGUF (Q4_K_M, Q8_0) | `step2_convert_llm_gguf.py` | ✅ 完成 |
| ONNX 精度验证 | `step5_verify_onnx_alignment.py` | ⬜ 待执行 |

### 执行 ONNX 精度验证（必须通过后再部署）

```bash
cd /home/zjf/pouring_VLA
conda run -n openVLA python edge_optimization/scripts/step5_verify_onnx_alignment.py \
    --merged_model edge_optimization/merged_model \
    --onnx_path edge_optimization/components/vision_projector/vision_projector.onnx \
    --num_tests 5
# 要求: 余弦相似度 > 0.999 全部通过
```

> **重要**: 如果验证失败，最可能的原因是图像归一化不一致。
> 请检查 `preprocess_image()` 中 DINOv2/SigLIP 的 mean/std 是否与训练时一致。

---

## Q900 部署步骤

### 1. 拷贝文件到 Q900

> Q900 实际部署目录：`/home/radxa/pouring_vla/`（radxa 用户家目录下）

```bash
# 在工作站上执行，目标 IP: 192.168.50.122
Q900=radxa@192.168.50.122
DST=/home/radxa/pouring_vla

# GGUF 约 4GB，耗时较长，用 rsync 断点续传
rsync -avz --progress \
    edge_optimization/gguf_models/openvla-llm-Q4_K_M.gguf \
    ${Q900}:${DST}/

# ONNX 约 3GB（含外部数据文件）
rsync -avz --progress \
    edge_optimization/components/vision_projector/vision_projector.onnx \
    edge_optimization/components/vision_projector/vision_projector.onnx.data \
    ${Q900}:${DST}/

# 其他文件（小）
rsync -avz \
    edge_optimization/components/action_head/action_head_params.json \
    edge_optimization/scripts/q900_inference_v2.py \
    ${Q900}:${DST}/
```

传输完成后在 Q900 验证文件完整性：
```bash
# 在 Q900 上
ls -lh ~/pouring_vla/
# 应看到: openvla-llm-Q4_K_M.gguf (~4.0GB)
#         vision_projector.onnx (~387KB)
#         vision_projector.onnx.data (~2.8GB)
#         action_head_params.json (~9KB)
#         q900_inference_v2.py
```

### 2. 在 Q900 安装依赖

> **注意**：以下步骤（2、3）全部在 **Q900 上执行**，不是在工作站上。

```bash
pip install onnxruntime opencv-python numpy h5py
# llama-cpp-python 单独处理，见步骤3
```

### 3. 在 Q900 上编译并安装 llama-cpp-python（带 QNN HTP 支持）

> ⚠️ **必须在 Q900 上编译，不能在工作站上交叉编译。** 原因：
> - QNN SDK 的 `libQnnHtp.so` 等库只存在于 Q900 设备上
> - 编译产物是 ARM64 二进制，工作站（x86_64）无法使用
> - pip 安装 llama-cpp-python 时会在本机编译 C++ 扩展

**前提条件**：Q900 上已确认 QNN SDK 路径为 `~/qairt/2.37.1.250807`

```bash
# 在 Q900 上执行（用户: radxa，conda env: pouring_vla）

# ① 清除 pip 缓存（防止使用预编译 wheel）
pip cache purge

# ② 用 export 方式传递 CMAKE_ARGS（inline 赋值方式可能被 pip 忽略）
export QNN_SDK_ROOT=~/qairt/2.37.1.250807
export CMAKE_ARGS="-DGGML_QNN=on -DQNN_SDK_PATH=${QNN_SDK_ROOT}"

# ③ 强制从源码编译（耗时约 10-20 分钟，会看到大量 gcc 编译输出）
pip install llama-cpp-python \
    --force-reinstall \
    --no-binary llama-cpp-python \
    --no-cache-dir \
    -v 2>&1 | tee /tmp/llama_build.log

# ④ 验证 QNN 是否真正编译进去（非预编译 wheel）
python -c "
import ctypes, sys
try:
    lib = ctypes.CDLL(None)
    lib.ggml_backend_qnn_reg_devices
    print('✅ QNN backend 已编译')
except AttributeError:
    print('❌ QNN backend 未编译，仍是预编译 wheel')
    sys.exit(1)
"
```

> ⚠️ **如果 `pip install` 只用了几秒就完成**，说明 pip 使用了预编译 wheel，
> QNN 没有生效。必须看到大量 C++ 编译日志才算正确。

**如果暂时不用 QNN（纯 CPU 验证）**，直接：
```bash
pip install llama-cpp-python
```

### 4. 验证基础推理

```bash
# 在 Q900 上，先用 CPU + HDF5 离线验证功能正确性
# 确认文件存在
ls -lh ~/pouring_vla/openvla-llm-Q4_K_M.gguf  # 必须存在且 ~4GB

conda activate pouring_vla
python ~/pouring_vla/q900_inference_v2.py \
    --gguf_path ~/pouring_vla/openvla-llm-Q4_K_M.gguf \
    --vision_onnx ~/pouring_vla/vision_projector.onnx \
    --action_params ~/pouring_vla/action_head_params.json \
    --hdf5_path ~/pouring_vla/episode_0006.hdf5 \
    --task "pour cola into cup" \
    --n_gpu_layers 0 \
    --num_steps 5
```

> ⚠️ **如果出现 `Failed to load model from file`**，先确认 GGUF 文件大小：
> ```bash
> ls -lh ~/pouring_vla/*.gguf
> # 正常大小应为 ~3.8-4.2 GB，若远小于此说明传输不完整
> # 用 rsync 重传（会自动续传）：
> # rsync -avz --progress radxa@<工作站IP>:/path/to/openvla-llm-Q4_K_M.gguf ~/pouring_vla/
> ```

### 5. 逐步开启 QNN HTP 加速（可选）

```bash
# 确认 CPU 推理正确后，逐步增加 GPU 层数
python ~/pouring_vla/q900_inference_v2.py \
    --gguf_path ~/pouring_vla/openvla-llm-Q4_K_M.gguf \
    --vision_onnx ~/pouring_vla/vision_projector.onnx \
    --action_params ~/pouring_vla/action_head_params.json \
    --n_gpu_layers 16 \
    --camera_id 0 \
    --task "pour cola into cup" \
    --hz 2.0
```

---

## 动作解码参数速查

```
vocab_size          = 32000
n_action_bins       = 256
action_token_range  = [31744, 31999]  (= vocab_size - n_bins ~ vocab_size - 1)

维度含义（与训练数据一致）:
  action[0:3]  — 位置增量 [m]  (q01/q99 归一化, mask=True)
  action[3:6]  — 姿态增量 [rad] (q01/q99 归一化, mask=True)
  action[6]    — 夹爪 [0,1]   (不归一化, mask=False)

归一化统计量 (dobot_pouring):
  q01: [-0.00321, -0.00289, -0.00364, -0.01391, -9.8e-5, -0.00088, 0.0]
  q99: [ 0.00356,  0.00227,  0.00380,  0.01396,  1.3e-4,  0.01217, 1.0]
```

---

## 延迟预估（Q900 QCS9100）

| 阶段 | 运行位置 | 预估延迟 |
|------|---------|---------|
| 图像预处理 | CPU | < 5ms |
| Vision ONNX 编码 | Vulkan (Adreno) | 80–150ms |
| LLM Prefill (~350 tokens + 256 vis) | QNN HTP 或 CPU | 200–400ms |
| LLM Decode (7 action tokens) | QNN HTP 或 CPU | 30–60ms |
| Detokenize | CPU | < 1ms |
| **总计** | | **310–616ms → ~2 Hz** |

> 倒水任务需要 2~5 Hz 控制频率，**Route A-1 方案完全满足需求**。

---

## 已知问题与注意事项

### ⚠️ 旧版 q900_inference.py 已废弃
旧版通过 `ctypes.memmove` 修改 LLM embedding 表内存，在量化模型（Q4_K_M）上不可靠且极易崩溃。**请使用 `q900_inference_v2.py`**。

### ⚠️ 图像归一化是关键
OpenVLA 的 fused backbone 要求：
- 通道 0-2（DINOv2）：ImageNet normalization (`mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]`)
- 通道 3-5（SigLIP）：SigLIP normalization (`mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]`)

错误的归一化会导致视觉特征完全错误，表现为机械臂动作混乱（但程序不报错）。

### ⚠️ 视觉 embedding 注入速度
V2.0 的 `_inject_vision_embeddings()` 采用逐 token 注入（256次 llama_decode 调用），可能成为性能瓶颈。如需加速，可改为批量注入（一次 batch size=256 的 decode 调用）。

### ℹ️ 为什么 ONNX 比 mmproj.gguf 更合适
OpenVLA 的 DINOv2+SigLIP 融合骨干（6通道输入）与 llama.cpp 期望的标准 CLIP（3通道）结构不兼容，生成合法的 mmproj.gguf 需修改 `clip.cpp` 源码。现阶段用 ONNX Runtime 代替，功能等效，工程代价更低。

---

## 后续优化路线图

```
阶段 2（当前）:
  ✅ ONNX 视觉 + llama-cpp-python LLM (CPU/QNN)
  ✅ 正确的双骨干图像预处理
  ✅ 稳健的 embedding 注入

阶段 3（可选，性能优化）:
  □ 修改 clip.cpp 支持 6 通道 fused backbone → 生成 mmproj.gguf
  □ 将视觉 embedding 注入改为 batch 模式（256 tokens 一次 decode）
  □ 在 Q900 上测量实际延迟，决定是否需要 Route B（纯 QNN .so）

阶段 4（可选，精度优化）:
  □ 对比 FP32 ONNX vs INT8 ONNX 的动作精度（用 HDF5 验证集）
  □ 必要时对 SigLIP 做 W8A8 量化（仅在 FP32 精度已验证的前提下）
```
