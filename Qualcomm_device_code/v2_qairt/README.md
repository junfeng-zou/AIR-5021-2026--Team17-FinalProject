# OpenVLA Edge V2.0 — Vision NPU 加速方案

> **目标设备**: Radxa Q900 (QCS9100, Hexagon v73 HTP NPU)  
> **核心变化**: Vision Encoder 从 ONNX CPU/Vulkan → **Qualcomm NPU (INT8)**  
> **LLM**: 不变，继续用 GGUF + llama-cpp-python  
> **服务器 QAIRT SDK**: `edge_optimization_v2.0/qairt/2.42.0.251225`  
> **服务器 conda 环境**: `qairt` (Python 3.10)

---

## 当前起点（V1.0 已完成项）

```
edge_optimization/components/vision_projector/
├── vision_projector.onnx        (~387KB 主文件)
├── vision_projector.onnx.data   (~2.8GB 权重数据)
└── vision_meta.json

edge_optimization/gguf_models/
└── openvla-llm-Q4_K_M.gguf     (~4GB)

edge_optimization/components/action_head/
└── action_head_params.json
```

V2.0 **从现有的 ONNX 文件出发**，不需要重新导出。

---

## 整体流水线

```
【服务器 x86】
vision_projector.onnx (2.8GB)
    │
    ▼ Step 1: 生成量化校准数据集
edge_optimization_v2.0/data/calib/
    calib_0000.raw ... calib_0029.raw  (每张 1.2MB, float32)
    input_list.txt
    │
    ▼ Step 2: qairt-converter
vision_projector_fp32.dlc  (~2.8GB)
    │
    ▼ Step 3: qairt-quantizer
vision_projector_int8.dlc  (~700MB)
    │
    ✈ 传输到 Q900
    │
【Q900 aarch64】
    ▼ Step 4: qnn-context-binary-generator (必须在设备上运行)
vision_projector_v73.bin   (~700MB, NPU 直接执行格式)
    │
    ▼ Step 5: 验证 + 推理
q900_inference_v3.py
```

---

## 分步说明

### Step 1 — 校准数据生成（服务器）

```bash
cd /home/zjf/pouring_VLA
python edge_optimization_v2.0/scripts/step1_prepare_calib_data.py \
    --hdf5 /path/to/episode_0006.hdf5 \
    --output_dir edge_optimization_v2.0/data/calib \
    --num_samples 30
```

- 输入: HDF5 或图片目录
- 输出: 30 个 `.raw` 文件 + `input_list.txt`
- 每个 `.raw` = `(1, 6, 224, 224) float32`，双骨干归一化完全一致

---

### Step 2 — ONNX → FP32 DLC（服务器）

```bash
conda activate qairt
bash edge_optimization_v2.0/scripts/step2_convert_onnx_to_dlc.sh
```

关键参数（已内置）：
- `--target_backend HTP --target_soc_model QCS9100` —— 直接针对 QCS9100 HTP 做图优化，运行时比通用 DLC 更快
- **必须在 `.onnx` 和 `.onnx.data` 同目录下运行**（脚本已自动切换目录）

预期耗时：30-60 分钟（2.8GB 图处理）

---

### Step 3 — FP32 DLC → INT8 DLC（服务器）

```bash
conda activate qairt
bash edge_optimization_v2.0/scripts/step3_quantize_dlc.sh
```

使用 `qairt-quantizer`，量化方案 `tf_enhanced` W8A8。

预期耗时：10-30 分钟（依校准集大小）

---

### Step 4 — 生成 Context Binary（在 Q900 上运行）

> ⚠️ **此步骤必须在 Q900 上运行**，因为需要加载 ARM HTP 运行时库

```bash
# 先把 INT8 DLC 传到 Q900
rsync -avz --progress \
    edge_optimization_v2.0/output/vision_projector_int8.dlc \
    edge_optimization_v2.0/soc_config_qcs9100.json \
    radxa@192.168.50.122:/home/radxa/pouring_vla/

# 在 Q900 上执行
bash /home/radxa/pouring_vla/step4_gen_context_binary.sh
```

---

### Step 5 — Q900 推理

```bash
# 在 Q900 上
python q900_inference_v3.py \
    --context_bin /home/radxa/pouring_vla/vision_projector_v73.bin \
    --gguf_path /home/radxa/pouring_vla/openvla-llm-Q4_K_M.gguf \
    --action_params /home/radxa/pouring_vla/action_head_params.json \
    --camera_id 0 \
    --task "pour cola into cup"
```

Vision 后端自动选择优先级：
1. QNN Context Binary（NPU，最快）
2. ONNX Runtime QNN EP（NPU，中等）
3. ONNX Runtime CPU（兜底）

---

## 性能预期

| 阶段 | V1.0 (ONNX CPU) | V2.0 (NPU INT8) | 提升 |
|------|----------------|-----------------|------|
| Vision 编码 | ~150ms | **~30-60ms** | **2.5-5×** |
| LLM (不变) | ~350ms | ~350ms | — |
| **总计** | **~500ms (2Hz)** | **~380-410ms (~2.5Hz)** | ✅ |

---

## 目录结构

```
edge_optimization_v2.0/
├── README.md                          ← 本文档
├── qairt/
│   └── 2.42.0.251225/                 ← QAIRT SDK (qairt conda 环境)
├── soc_config_qcs9100.json            ← QCS9100 HTP 配置
├── data/
│   └── calib/                         ← Step 1 生成
│       ├── calib_0000.raw
│       └── input_list.txt
├── output/                            ← Steps 2-3 生成
│   ├── vision_projector_fp32.dlc
│   └── vision_projector_int8.dlc
└── scripts/
    ├── step1_prepare_calib_data.py    ← 校准数据生成
    ├── step2_convert_onnx_to_dlc.sh   ← ONNX → DLC (服务器, 需 qairt 环境)
    ├── step3_quantize_dlc.sh          ← 量化 (服务器, 需 qairt 环境)
    ├── step4_gen_context_binary.sh    ← .bin 生成 (Q900)
    ├── step5_verify_accuracy.py       ← 精度对比验证
    ├── run_server_pipeline.sh         ← 一键执行 Steps 1-3 (需 qairt 环境)
    └── q900_inference_v3.py           ← Q900 推理脚本
```

---

## soc_id 说明

QCS9100 的 `soc_id` 在不同文档中有歧义：
- Radxa 官方文档: `77`
- 高通内部规格: `43`

`step4_gen_context_binary.sh` 会先尝试 `43`，失败后自动切换 `77`。

可在 Q900 上通过以下命令确认：
```bash
cat /sys/devices/soc0/soc_id
```

---

## 常见问题

**Q: qairt-converter 报错 `Cannot find external data file`？**  
A: 必须先 `cd` 到包含 `.onnx` 和 `.onnx.data` 的目录再运行，或在脚本中指定 `--working_dir`。

**Q: qairt-quantizer 报错 `Input dimensions mismatch`？**  
A: 检查 `input_list.txt` 中的 `.raw` 文件大小，应为 `6*224*224*4 = 1,204,224 bytes`。

**Q: Q900 上 context binary 加载失败？**  
A: 确认 Q900 上的 QAIRT SDK 版本，建议与服务器版本一致或更新。可尝试在 Q900 上直接重新执行 Steps 2-4（如果内存够用）。

**Q: 量化后动作精度下降明显？**  
A: 增加校准样本数（50→100），或改用 `W8A8` 而非 `W4A8`（默认 W8A8 已足够）。
