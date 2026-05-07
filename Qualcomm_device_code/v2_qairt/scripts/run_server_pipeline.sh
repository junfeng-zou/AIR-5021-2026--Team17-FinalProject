#!/bin/bash
# ============================================================
# 服务器端一键流水线: Steps 1-3
# 在服务器 (x86 Linux) 上运行
#
# ⚠️  需要在 qairt conda 环境中运行:
#     conda activate qairt
#     bash run_server_pipeline.sh [hdf5_path] [image_dir]
# ============================================================
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/edge_optimization_v2.0/scripts"

# 检查 conda 环境
if [ "${CONDA_DEFAULT_ENV}" != "qairt" ]; then
    echo "WARNING: 当前 conda 环境为 '${CONDA_DEFAULT_ENV:-未激活}'"
    echo "         建议先执行: conda activate qairt"
fi

echo "========================================================"
echo " OpenVLA Edge V2.0 — 服务器端流水线 (Steps 1-3)"
echo "========================================================"
echo "  Repo: $REPO_ROOT"

# ---------- 参数配置 ----------
HDF5_PATH="${1:-}"          # 第一个参数: HDF5 文件路径（可选）
IMAGE_DIR="${2:-}"          # 第二个参数: 图片目录（可选）
NUM_SAMPLES=30

# Step 1: 校准数据
echo ""
echo "▶ Step 1: 生成校准数据集"
echo "──────────────────────────"
if [ -n "$HDF5_PATH" ] && [ -f "$HDF5_PATH" ]; then
    python "$SCRIPT_DIR/step1_prepare_calib_data.py" \
        --hdf5 "$HDF5_PATH" \
        --output_dir "${REPO_ROOT}/edge_optimization_v2.0/data/calib" \
        --num_samples $NUM_SAMPLES
elif [ -n "$IMAGE_DIR" ] && [ -d "$IMAGE_DIR" ]; then
    python "$SCRIPT_DIR/step1_prepare_calib_data.py" \
        --image_dir "$IMAGE_DIR" \
        --output_dir "${REPO_ROOT}/edge_optimization_v2.0/data/calib" \
        --num_samples $NUM_SAMPLES
else
    echo "[WARNING] 未提供 HDF5 或图片目录，将从 merged_model 目录寻找图片..."
    # 尝试从 data 目录找现有校准数据
    CALIB_DIR="${REPO_ROOT}/edge_optimization_v2.0/data/calib"
    if [ -f "${CALIB_DIR}/input_list.txt" ]; then
        EXISTING=$(wc -l < "${CALIB_DIR}/input_list.txt")
        echo "  发现已有校准数据: ${EXISTING} 张，跳过 Step 1"
    else
        echo "ERROR: 需要提供 HDF5 文件或图片目录"
        echo "用法: bash run_server_pipeline.sh [hdf5_path] [image_dir]"
        exit 1
    fi
fi

# Step 2: ONNX → FP32 DLC
echo ""
echo "▶ Step 2: ONNX → FP32 DLC"
echo "──────────────────────────"
bash "$SCRIPT_DIR/step2_convert_onnx_to_dlc.sh"

# Step 3: FP32 DLC → INT8 DLC
echo ""
echo "▶ Step 3: FP32 DLC → INT8 DLC"
echo "─────────────────────────────"
bash "$SCRIPT_DIR/step3_quantize_dlc.sh"

# Step 5: 精度验证（可选但推荐）
echo ""
echo "▶ Step 5: 精度验证 (INT8 DLC vs FP32 ONNX)"
echo "────────────────────────────────────────────"
python "$SCRIPT_DIR/step5_verify_accuracy.py" \
    --onnx "${REPO_ROOT}/edge_optimization/components/vision_projector/vision_projector.onnx" \
    --int8_dlc "${REPO_ROOT}/edge_optimization_v2.0/output/vision_projector_int8.dlc" \
    --calib_dir "${REPO_ROOT}/edge_optimization_v2.0/data/calib" \
    --num_tests 5 || echo "[WARNING] 精度验证失败（snpe-net-run 不可用），可跳过"

# 汇总
echo ""
echo "========================================================"
echo " ✅ 服务器端完成！产出文件:"
echo "========================================================"
ls -lh "${REPO_ROOT}/edge_optimization_v2.0/output/" 2>/dev/null || true
echo ""
echo "下一步: 传输文件到 Q900"
echo ""
echo "  Q900_IP=radxa@192.168.50.122"
echo "  DST=/home/radxa/pouring_vla"
echo ""
echo "  rsync -avz --progress \\"
echo "    edge_optimization_v2.0/output/vision_projector_int8.dlc \\"
echo "    edge_optimization_v2.0/soc_config_qcs9100.json \\"
echo "    edge_optimization_v2.0/scripts/step4_gen_context_binary.sh \\"
echo "    edge_optimization_v2.0/scripts/q900_inference_v3.py \\"
echo "    \${Q900_IP}:\${DST}/"
echo ""
echo "  在 Q900 上: bash \${DST}/step4_gen_context_binary.sh"
echo "========================================================"
