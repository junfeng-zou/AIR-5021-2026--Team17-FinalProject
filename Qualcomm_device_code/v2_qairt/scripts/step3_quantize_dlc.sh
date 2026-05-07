#!/bin/bash
# ============================================================
# Step 3: FP32 DLC → INT8 DLC (量化)
# 在服务器 (x86 Linux) 上运行
# 依赖: Step 1 的校准数据, Step 2 的 FP32 DLC
#
# ⚠️  需要在 qairt conda 环境中运行:
#     conda activate qairt
# ============================================================
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# SDK 路径: edge_optimization_v2.0/qairt/2.42.0.251225 (qairt conda 环境)
QAIRT_SDK="${REPO_ROOT}/edge_optimization_v2.0/qairt/2.42.0.251225"
OUTPUT_DIR="${REPO_ROOT}/edge_optimization_v2.0/output"
CALIB_DIR="${REPO_ROOT}/edge_optimization_v2.0/data/calib"
QAIRT_QUANTIZER="${QAIRT_SDK}/bin/x86_64-linux-clang/qairt-quantizer"

echo "========================================================"
echo " Step 3: FP32 DLC → INT8 DLC (qairt-quantizer)"
echo "========================================================"
echo "  输入 DLC   : $OUTPUT_DIR/vision_projector_fp32.dlc"
echo "  校准数据   : $CALIB_DIR/input_list.txt"
echo "  输出 DLC   : $OUTPUT_DIR/vision_projector_int8.dlc"

# 检查
if [ ! -f "${OUTPUT_DIR}/vision_projector_fp32.dlc" ]; then
    echo "ERROR: fp32 DLC 不存在，请先运行 step2"
    exit 1
fi
if [ ! -f "${CALIB_DIR}/input_list.txt" ]; then
    echo "ERROR: input_list.txt 不存在，请先运行 step1"
    exit 1
fi

CALIB_COUNT=$(wc -l < "${CALIB_DIR}/input_list.txt")
echo "  校准样本数 : $CALIB_COUNT 张"
if [ "$CALIB_COUNT" -lt 10 ]; then
    echo "WARNING: 校准样本不足 10 张，量化精度可能下降，建议 20-50 张"
fi

source "${QAIRT_SDK}/bin/envsetup.sh"

echo ""
echo "▶ 开始量化（预计 10-30 分钟）..."
"$QAIRT_QUANTIZER" \
    --input_dlc "${OUTPUT_DIR}/vision_projector_fp32.dlc" \
    --input_list "${CALIB_DIR}/input_list.txt" \
    --output_dlc "${OUTPUT_DIR}/vision_projector_int8.dlc" \
    --quant_schemes tf_enhanced \
    --act_bitwidth 8 \
    --weights_bitwidth 8 \
    --bias_bitwidth 32 \
    --algorithms cle \
    2>&1 | tee "${OUTPUT_DIR}/step3_quantize.log"

INT8_SIZE=$(du -sh "${OUTPUT_DIR}/vision_projector_int8.dlc" 2>/dev/null | cut -f1)
FP32_SIZE=$(du -sh "${OUTPUT_DIR}/vision_projector_fp32.dlc" 2>/dev/null | cut -f1)
echo ""
echo "✅ 量化完成!"
echo "  FP32 DLC: $FP32_SIZE  →  INT8 DLC: $INT8_SIZE"
echo ""
echo "下一步: 将以下文件传输到 Q900，然后运行 step4_gen_context_binary.sh"
echo "  rsync -avz --progress \\"
echo "    ${OUTPUT_DIR}/vision_projector_int8.dlc \\"
echo "    ${REPO_ROOT}/edge_optimization_v2.0/soc_config_qcs9100.json \\"
echo "    ${REPO_ROOT}/edge_optimization_v2.0/scripts/step4_gen_context_binary.sh \\"
echo "    ${REPO_ROOT}/edge_optimization_v2.0/scripts/q900_inference_v3.py \\"
echo "    radxa@192.168.50.122:/home/radxa/pouring_vla/"
