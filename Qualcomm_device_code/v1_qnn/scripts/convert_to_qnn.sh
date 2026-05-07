#!/bin/bash
# ==============================================================================
# OpenVLA → QNN 转换脚本
# ==============================================================================
# 在服务器 (x86_64) 上运行，将 OpenVLA 的 LLM 和 Vision 转换为 QNN 格式
# 产出文件拷贝到 Q900 (aarch64) 上部署
#
# 前置依赖：
#   1. 系统安装 LLVM libc++：
#        sudo apt-get install -y libc++1 libc++abi1 libunwind-14
#   2. 创建 QNN 专用 conda 环境（onnx 1.13.x，与 openVLA 隔离）：
#        conda create -n qnn_convert python=3.10 -y
#        conda run -n qnn_convert pip install \
#            "onnx==1.13.1" "numpy<2" pyyaml scipy colorlog pandas pytz packaging
#
# 用法:
#   cd pouring_VLA
#   bash edge_optimization/scripts/convert_to_qnn.sh
# ==============================================================================

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# --- QAIRT SDK ---
QAIRT_DIR="edge_optimization/qairt_v2.42.0.251225/2.42.0.251225"
if [ ! -d "$QAIRT_DIR" ]; then
    echo "[ERROR] QAIRT SDK 不存在: $QAIRT_DIR"
    exit 1
fi

export QAIRT_SDK_ROOT="$(readlink -f "$QAIRT_DIR")"
export QNN_SDK_ROOT="$QAIRT_SDK_ROOT"
export PATH="${QNN_SDK_ROOT}/bin/x86_64-linux-clang:${PATH}"

# QNN Clang 工具链需要 LLVM libc++ 和 QNN 自身的 x86 共享库
export LD_LIBRARY_PATH="${QNN_SDK_ROOT}/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}"

# QNN Python 工具需要专用 conda 环境（onnx==1.13.1），PYTHONPATH 指向 QNN SDK
QNN_PYTHON="conda run -n qnn_convert python"
QNN_PYTHONPATH="${QNN_SDK_ROOT}/lib/python/"

# 启动前检查 libc++ 是否已安装（用 find 直接查文件，避免 ldconfig 缓存未更新问题）
if ! ldconfig -p 2>/dev/null | grep -q "libc++\.so\.1" && \
   ! find /usr/lib /usr/local/lib /lib -name "libc++.so.1*" 2>/dev/null | grep -q .; then
    echo ""
    echo "❌ [ERROR] 缺少 LLVM libc++ 运行时库 (libc++.so.1)"
    echo "   请先安装："
    echo "     sudo apt-get install -y libc++1 libc++abi1 libunwind-14"
    echo ""
    exit 1
fi

# 检查 qnn_convert conda 环境
if ! conda env list 2>/dev/null | grep -q "^qnn_convert "; then
    echo ""
    echo "❌ [ERROR] 缺少 qnn_convert conda 环境"
    echo "   请先创建："
    echo "     conda create -n qnn_convert python=3.10 -y"
    echo "     conda run -n qnn_convert pip install \\"
    echo "         \"onnx==1.13.1\" \"numpy<2\" pyyaml scipy colorlog pandas pytz packaging"
    echo ""
    exit 1
fi

# 验证 qnn_convert 环境的 onnx 版本兼容（onnx.mapping 需要 <1.14）
if ! PYTHONPATH="$QNN_PYTHONPATH" $QNN_PYTHON -c "from onnx import mapping" 2>/dev/null; then
    echo "❌ [ERROR] qnn_convert 环境的 onnx 版本不兼容（需要 onnx < 1.14）"
    echo "   请运行：conda run -n qnn_convert pip install 'onnx==1.13.1'"
    exit 1
fi

echo "=================================================="
echo " OpenVLA → QNN 转换"
echo "=================================================="
echo "  QNN SDK: $QNN_SDK_ROOT"
echo "  Python:  qnn_convert conda env (onnx 1.13.1)"
echo ""

# --- 输出目录 ---
OUT_DIR="edge_optimization/qnn_models"
LLM_OUT="$OUT_DIR/llm"
VISION_OUT="$OUT_DIR/vision"
mkdir -p "$LLM_OUT" "$VISION_OUT"

# --- 工具路径 ---
QNN_CONVERTER="${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-onnx-converter"
QNN_GENAI="${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-genai-transformer-composer"
QNN_MODEL_LIB="${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-model-lib-generator"
QNN_CTX_GEN="${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-context-binary-generator"
HTP_BACKEND="${QNN_SDK_ROOT}/lib/x86_64-linux-clang/libQnnHtp.so"

# ==============================================================================
# Part 1: LLM → QNN (Genie Transformer Composer)
# ==============================================================================
echo ""
echo "▶ [1/4] LLM: GGUF → QNN (Genie)"
echo "─────────────────────────────────"

GGUF_PATH="edge_optimization/gguf_models/openvla-llm-Q4_K_M.gguf"
LLM_CONFIG="edge_optimization/components/openvla_llm_qnn.json"
# qnn-genai-transformer-composer 需要 HuggingFace 格式目录（.safetensors），不接受 GGUF
# GGUF 是 llama.cpp 格式；Genie composer 使用 HF 格式直接读取原始权重再做量化
LLM_HF_DIR="edge_optimization/components/llm_llama2_7b"
# --outfile 必须以 .bin 结尾（工具要求）
LLM_BIN_OUT="$LLM_OUT/openvla_llm.bin"

if [ ! -d "$LLM_HF_DIR" ]; then
    echo "[ERROR] LLM HuggingFace 目录不存在: $LLM_HF_DIR"
    echo "  请先运行 step1_extract_components.py 提取 LLM 权重"
    exit 1
fi

echo "  LLM HF dir: $LLM_HF_DIR"
echo "  Config:     $LLM_CONFIG"
echo "  Output:     $LLM_BIN_OUT"

# Genie composer: HF safetensors → QNN weight binaries
# --quantize Z4 = 4-bit Z-order 量化 (适合 Hexagon HTP)
PYTHONPATH="$QNN_PYTHONPATH" $QNN_PYTHON "$QNN_GENAI" \
    --model "$LLM_HF_DIR" \
    --config_file "$LLM_CONFIG" \
    --quantize Z4 \
    --outfile "$LLM_BIN_OUT" \
    --export_tokenizer_json \
    2>&1 | tee "$LLM_OUT/composer.log"

echo "  ✅ LLM QNN 转换完成"

# ==============================================================================
# Part 2: LLM → Hexagon context binary
# ==============================================================================
echo ""
echo "▶ [2/4] LLM: 编译 Hexagon context binary"
echo "──────────────────────────────────────────"

# Genie composer 的输出结构：
#   openvla_llm.bin           - 主权重二进制（可直接被 Genie 加载）
#   openvla_llm.cpp / .bin    - 如果 composer 同时输出 QNN model lib 格式
LLM_CPP=$(find "$LLM_OUT" -name "*.cpp" | head -1)
LLM_MODEL_BIN=$(find "$LLM_OUT" -name "*.bin" ! -name "openvla_llm.bin" | head -1)

if [ -z "$LLM_CPP" ]; then
    echo "  [INFO] Genie composer 没有输出 .cpp，跳过 qnn-model-lib-generator 步骤"
    echo "         (Genie 直接使用 .bin 文件，无需额外编译)"
else
    echo "  Model lib: $LLM_CPP"

    # 构建 model library (aarch64-oe-linux for Q900)
    $QNN_MODEL_LIB \
        -c "$LLM_CPP" \
        -b "$LLM_MODEL_BIN" \
        -t aarch64-oe-linux-gcc11.2 \
        -o "$LLM_OUT/model_libs" \
        2>&1 | tee "$LLM_OUT/model_lib.log"

    # 生成 Hexagon context binary
    LLM_SO=$(find "$LLM_OUT/model_libs" -name "*.so" | head -1)
    if [ -n "$LLM_SO" ]; then
        $QNN_CTX_GEN \
            --backend "$HTP_BACKEND" \
            --model "$LLM_SO" \
            --binary_file "$LLM_OUT/openvla_llm_htp" \
            2>&1 | tee "$LLM_OUT/ctx_gen.log"
        echo "  ✅ LLM Hexagon context binary 生成完成"
    else
        echo "  [WARN] 未找到 model library .so 文件"
    fi
fi

# ==============================================================================
# Part 3: Vision Projector → QNN
# ==============================================================================
echo ""
echo "▶ [3/4] Vision: ONNX → QNN (INT8 量化)"
echo "─────────────────────────────────────────"

VISION_ONNX="edge_optimization/components/vision_projector/vision_projector.onnx"
CALIB_FILELIST="edge_optimization/qnn_models/calib_data/filelist.txt"

# 注意：必须使用原始未量化的 ONNX（包含 external data）
# 动态量化后的 INT8 ONNX 含有 DynamicQuantizeLinear 算子，QNN converter 不支持
# QNN converter 自己通过 --param_quantizer/--act_quantizer 完成量化
if [ ! -f "$VISION_ONNX" ]; then
    echo "[ERROR] Vision ONNX 不存在: $VISION_ONNX"
    exit 1
fi
VISION_INPUT="$VISION_ONNX"
echo "  使用原始 ONNX: $VISION_ONNX"

# 生成校准数据（如果不存在）
if [ ! -f "$CALIB_FILELIST" ]; then
    echo "  生成校准数据..."
    conda run -n openVLA python edge_optimization/scripts/generate_calib_data.py \
        --hdf5_dir edge_optimization/data \
        --output_dir edge_optimization/qnn_models/calib_data \
        --num_samples 50
fi

# ONNX → QNN cpp+bin（带 INT8 量化）
# 注意：--input_dim 参数格式是 "name dim1,dim2,..."（空格分隔 name 和 dims）
PYTHONPATH="$QNN_PYTHONPATH" $QNN_PYTHON "$QNN_CONVERTER" \
    --input_network "$VISION_INPUT" \
    -d pixel_values 1,6,224,224 \
    --input_list "$CALIB_FILELIST" \
    --param_quantizer tf \
    --act_quantizer tf \
    --weights_bitwidth 8 \
    --act_bitwidth 8 \
    -o "$VISION_OUT/vision_encoder" \
    2>&1 | tee "$VISION_OUT/converter.log"

echo "  ✅ Vision QNN 转换完成"

# ==============================================================================
# Part 4: Vision → Hexagon context binary
# ==============================================================================
echo ""
echo "▶ [4/4] Vision: 编译 Hexagon context binary"
echo "─────────────────────────────────────────────"

VISION_CPP=$(find "$VISION_OUT" -name "*.cpp" | head -1)
VISION_BIN=$(find "$VISION_OUT" -name "*.bin" | head -1)

if [ -f "$VISION_CPP" ]; then
    echo "  Model lib: $VISION_CPP"

    $QNN_MODEL_LIB \
        -c "$VISION_CPP" \
        -b "$VISION_BIN" \
        -t aarch64-oe-linux-gcc11.2 \
        -o "$VISION_OUT/model_libs" \
        2>&1 | tee "$VISION_OUT/model_lib.log"

    VISION_SO=$(find "$VISION_OUT/model_libs" -name "*.so" | head -1)
    if [ -n "$VISION_SO" ]; then
        $QNN_CTX_GEN \
            --backend "$HTP_BACKEND" \
            --model "$VISION_SO" \
            --binary_file "$VISION_OUT/vision_encoder_htp" \
            2>&1 | tee "$VISION_OUT/ctx_gen.log"
        echo "  ✅ Vision Hexagon context binary 生成完成"
    else
        echo "  [WARN] 未找到 model library .so 文件"
    fi
else
    echo "  [WARN] 未找到 Vision .cpp 文件，跳过 Hexagon 编译"
fi

# ==============================================================================
# 汇总
# ==============================================================================
echo ""
echo "=================================================="
echo " ✅ QNN 转换完成！"
echo "=================================================="
echo ""
echo "产出文件:"
echo "  $LLM_OUT/            — LLM QNN 权重二进制"
echo "  $VISION_OUT/         — Vision QNN context binary"
echo ""
echo "需要拷贝到 Q900 的文件:"
echo "  1. $LLM_OUT/openvla_llm.bin  (Genie LLM 权重)"
echo "  2. $VISION_OUT/*.serialized.bin 或 $VISION_OUT/vision_encoder_htp.serialized.bin"
echo "  3. edge_optimization/components/action_head/action_head_params.json"
echo "  4. $QNN_SDK_ROOT/lib/aarch64-oe-linux-gcc11.2/  (QNN runtime)"
echo "  5. $QNN_SDK_ROOT/lib/hexagon-v*/               (Hexagon skel libs)"
echo "  6. edge_optimization/scripts/q900_npu_inference.py"
