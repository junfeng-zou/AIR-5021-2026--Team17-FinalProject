#!/bin/bash
# ============================================================
# Step 4: INT8 DLC → Context Binary (.bin)
# ⚠️  必须在 Q900 (aarch64) 上运行！
# ============================================================
set -e

# Q900 上的 QAIRT SDK 路径（根据实际安装调整）
QNN_SDK="${HOME}/qairt/2.37.1.250807"
if [ ! -d "$QNN_SDK" ]; then
    # 尝试其他常见路径
    for try_path in \
        "${HOME}/qairt/2.42.0.251225" \
        "/opt/qcom/aistack/qnn/2.37.1.250807" \
        "/opt/qairt/2.37.1.250807"; do
        if [ -d "$try_path" ]; then
            QNN_SDK="$try_path"
            break
        fi
    done
fi

if [ ! -d "$QNN_SDK" ]; then
    echo "ERROR: 未找到 QAIRT SDK，请设置 QNN_SDK 环境变量"
    echo "  例: export QNN_SDK=/home/radxa/qairt/2.37.1.250807"
    exit 1
fi

WORK_DIR="${HOME}/pouring_vla"
INPUT_DLC="${WORK_DIR}/vision_projector_int8.dlc"
SOC_CONFIG="${WORK_DIR}/soc_config_qcs9100.json"
OUTPUT_DIR="${WORK_DIR}/context_bin"
QNN_BIN="${QNN_SDK}/bin/aarch64-ubuntu-gcc9.4"
QNN_LIB="${QNN_SDK}/lib/aarch64-ubuntu-gcc9.4"

echo "========================================================"
echo " Step 4: INT8 DLC → Context Binary (Q900 NPU)"
echo "========================================================"
echo "  QNN SDK  : $QNN_SDK"
echo "  输入 DLC : $INPUT_DLC"
echo "  输出目录 : $OUTPUT_DIR"

source "${QNN_SDK}/bin/envsetup.sh" 2>/dev/null || true
export PATH="${QNN_BIN}:${PATH}"
export LD_LIBRARY_PATH="${QNN_LIB}:${LD_LIBRARY_PATH}"

if [ ! -f "$INPUT_DLC" ]; then
    echo "ERROR: $INPUT_DLC 不存在"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ── 按 Radxa 官方文档格式生成 Context Binary ─────────────────────
# 参考: docs.radxa.com/fogwise/airbox-q900/ai-dev/qairt-usage
#
# config_backend.json: dsp_arch=v73, soc_id=77 (QCS9075/QCS9100 = Hexagon v73)
# config_file.json: 引用 config_backend.json

# 1. 生成 config_backend.json（图名必须与 DLC 文件名不带后缀一致）
DLC_BASENAME=$(basename "$INPUT_DLC" .dlc)
cat > "${WORK_DIR}/config_backend.json" << EOF
{
  "graphs": [
    {
      "graph_names": ["${DLC_BASENAME}"],
      "vtcm_mb": 0
    }
  ],
  "devices": [
    {
      "dsp_arch": "v73",
      "soc_id": 77
    }
  ]
}
EOF

# 2. 生成 config_file.json
cat > "${WORK_DIR}/config_file.json" << EOF
{
  "backend_extensions": {
    "shared_library_path": "libQnnHtpNetRunExtensions.so",
    "config_file_path": "${WORK_DIR}/config_backend.json"
  }
}
EOF

echo "  config_backend.json: dsp_arch=v73, soc_id=77"
echo "  config_file.json: 引用 config_backend.json"
echo ""

# 3. 生成 Context Binary（官方文档命令格式）
echo "▶ 生成 Context Binary ..."
qnn-context-binary-generator \
    --model "${QNN_LIB}/libQnnModelDlc.so" \
    --backend "${QNN_LIB}/libQnnHtp.so" \
    --dlc_path "$INPUT_DLC" \
    --output_dir "$OUTPUT_DIR" \
    --binary_file "${DLC_BASENAME}" \
    --config_file "${WORK_DIR}/config_file.json" \
    2>&1 | tee "${OUTPUT_DIR}/step4_gen.log"

BIN_FILE="${OUTPUT_DIR}/vision_projector_v73.bin"
if [ -f "$BIN_FILE" ]; then
    BIN_SIZE=$(du -sh "$BIN_FILE" | cut -f1)
    echo ""
    echo "  输出文件: $BIN_FILE  ($BIN_SIZE)"
    echo ""
    echo "▶ 快速验证 (qnn-net-run)..."
    # 生成一个全零的测试输入
    python3 -c "
import numpy as np
x = np.zeros((1,6,224,224), dtype=np.float32)
x.tofile('/tmp/test_input.raw')
with open('/tmp/test_input_list.txt','w') as f:
    f.write('/tmp/test_input.raw\n')
"
    if qnn-net-run \
        --retrieve_context "$BIN_FILE" \
        --backend "${QNN_LIB}/libQnnHtp.so" \
        --input_list /tmp/test_input_list.txt \
        --output_dir /tmp/qnn_test_output/ \
        2>&1 | tail -5; then
        echo "✅ NPU 推理验证通过！"
    else
        echo "⚠️  验证失败，但 .bin 文件可能仍可用，请检查日志"
    fi
else
    echo "ERROR: 未生成 .bin 文件，请检查日志"
    exit 1
fi

echo ""
echo "========================================================"
echo " 所有准备完成，运行推理:"
echo "   python ${WORK_DIR}/q900_inference_v3.py \\"
echo "     --context_bin ${BIN_FILE} \\"
echo "     --gguf_path ${WORK_DIR}/openvla-llm-Q4_K_M.gguf \\"
echo "     --action_params ${WORK_DIR}/action_head_params.json \\"
echo "     --camera_id 0 --task 'pour cola into cup'"
echo "========================================================"
