#!/bin/bash
# ============================================================
# Step 2: ONNX → FP32 DLC (针对 QCS9100 HTP 优化)
# 在服务器 (x86 Linux) 上运行
#
# ⚠️  需要在 qairt conda 环境中运行:
#     conda activate qairt
#     bash step2_convert_onnx_to_dlc.sh
# ============================================================
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# 新 SDK 路径: edge_optimization_v2.0/qairt/2.42.0.251225
QAIRT_SDK="${REPO_ROOT}/edge_optimization_v2.0/qairt/2.42.0.251225"
ONNX_DIR="${REPO_ROOT}/edge_optimization/components/vision_projector"
OUTPUT_DIR="${REPO_ROOT}/edge_optimization_v2.0/output"
QAIRT_CONVERTER="${QAIRT_SDK}/bin/x86_64-linux-clang/qairt-converter"

echo "========================================================"
echo " Step 2: ONNX → FP32 DLC (qairt-converter)"
echo " 目标 SoC: QCS9100 / Hexagon v73 HTP"
echo "========================================================"
echo "  QAIRT SDK : $QAIRT_SDK"
echo "  ONNX 文件 : $ONNX_DIR/vision_projector.onnx"
echo "  输出目录  : $OUTPUT_DIR"

# 检查 conda 环境
if [ "${CONDA_DEFAULT_ENV}" != "qairt" ]; then
    echo "WARNING: 当前 conda 环境为 '${CONDA_DEFAULT_ENV:-未激活}'"
    echo "         建议先执行: conda activate qairt"
fi

# 检查文件
if [ ! -f "$ONNX_DIR/vision_projector.onnx" ]; then
    echo "ERROR: vision_projector.onnx 不存在: $ONNX_DIR"
    exit 1
fi
if [ ! -f "$ONNX_DIR/vision_projector.onnx.data" ]; then
    echo "ERROR: vision_projector.onnx.data 不存在（2.8GB 权重文件）"
    exit 1
fi
if [ ! -f "$QAIRT_CONVERTER" ]; then
    echo "ERROR: qairt-converter 不存在: $QAIRT_CONVERTER"
    echo "       请确认 SDK 已安装到: $QAIRT_SDK"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# 激活 QAIRT 环境变量（PATH / LD_LIBRARY_PATH）
source "${QAIRT_SDK}/bin/envsetup.sh"

# 关键: 必须在 ONNX 所在目录执行，否则找不到 .onnx.data 外部权重文件
echo ""
echo "▶ 检查依赖 (onnx <= 1.16.2, onnxsim) ..."
ONNX_VER=$(python3 -c "import onnx; print(onnx.__version__)" 2>/dev/null || echo "未安装")
if ! python3 -c "import onnx; v=tuple(int(x) for x in onnx.__version__.split('.')); exit(0 if v <= (1,16,2) else 1)" 2>/dev/null; then
    echo "  ⚠️  onnx $ONNX_VER 太新 → 降级到 1.16.2 ..."
    pip install "onnx==1.16.2" --force-reinstall -q
    echo "  ✅ onnx 已降级"
else
    echo "  ✅ onnx $ONNX_VER"
fi

python3 -c "import onnxsim" 2>/dev/null || { echo "  安装 onnxsim ..."; pip install onnxsim -q; }
python3 -c "import onnxsim; print('  ✅ onnxsim', onnxsim.__version__)"

# ── 阶段 1: 用 onnxsim 固化 shape，解决动态 Reshape 问题 ──────────
# 错误根源: qairt-converter 遇到含 -1 的动态 Reshape 时形状计算溢出
# 解法: 先用 onnxsim.simplify(overwrite_input_shapes=...) 常量折叠
STATIC_ONNX="${OUTPUT_DIR}/vision_projector_static.onnx"
echo ""
echo "▶ 阶段 1: 固化 ONNX 动态 shape (onnxsim)..."
echo "  输入: ${ONNX_DIR}/vision_projector.onnx"
echo "  输出: ${STATIC_ONNX}"
echo "  注意: 2.8GB 模型首次运行约需 5-15 分钟"

python3 "${SCRIPT_DIR}/step2_fix_onnx_shapes.py" \
    --onnx "${ONNX_DIR}/vision_projector.onnx" \
    --out  "${STATIC_ONNX}"

EC=$?
if [ $EC -ne 0 ]; then
    echo "❌ shape 固化失败，请检查上方错误"
    exit $EC
fi

# ── 阶段 2: ONNX → FP32 DLC ────────────────────────────────────────
# 注意: static ONNX 没有内嵌权重，external data (vision_projector.onnx.data)
# 仍在原目录，所以 converter 必须在 ONNX_DIR 下运行以找到 .data 文件
# 但 static ONNX 已在 OUTPUT_DIR，需要把它链接或复制过去
echo ""
echo "▶ 阶段 2: ONNX → FP32 DLC (qairt-converter) ..."

# 处理 external data: static ONNX 本身没有权重，但它的 external_data 字段
# 仍然指向原始 vision_projector.onnx.data。
# 方案A: 把 static ONNX 复制到 ONNX_DIR，在那运行 (data 就在同目录)
STATIC_IN_ONNXDIR="${ONNX_DIR}/vision_projector_static.onnx"
cp "${STATIC_ONNX}" "${STATIC_IN_ONNXDIR}"
cd "$ONNX_DIR"

# 官方文档命令格式 (Radxa docs + v2.42 新 API):
#   --input_network:  模型路径 (自动识别 ONNX)
#   -d / --desired_input_shape: 输入维度 (deprecated but still works)
"$QAIRT_CONVERTER" \
    --input_network vision_projector_static.onnx \
    --output_path "${OUTPUT_DIR}/vision_projector_fp32.dlc" \
    -d 'pixel_values' '1,6,224,224' \
    2>&1 | tee "${OUTPUT_DIR}/step2_convert.log"

EC=${PIPESTATUS[0]}

# 清理临时文件
rm -f "${STATIC_IN_ONNXDIR}"

if [ $EC -ne 0 ]; then
    echo ""
    echo "❌ 转换失败 (exit code=$EC)"
    echo "   日志: ${OUTPUT_DIR}/step2_convert.log"
    echo ""
    echo "   排查建议:"
    echo "   1. grep -i 'unsupported\|error' ${OUTPUT_DIR}/step2_convert.log"
    echo "   2. 若仍有 Reshape 错误，尝试: --onnx_skip_simplification"
    exit $EC
fi

DLC_SIZE=$(du -sh "${OUTPUT_DIR}/vision_projector_fp32.dlc" 2>/dev/null | cut -f1)
echo ""
echo "✅ 转换完成: ${OUTPUT_DIR}/vision_projector_fp32.dlc  ($DLC_SIZE)"
echo ""
echo "▶ 查看 DLC 层信息 (前 40 行):"
qairt-dlc-info -i "${OUTPUT_DIR}/vision_projector_fp32.dlc" 2>/dev/null | head -40 || true
