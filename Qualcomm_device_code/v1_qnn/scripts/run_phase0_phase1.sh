#!/bin/bash
# ==============================================================================
# Phase 0 + Phase 1: 一键执行脚本
# ==============================================================================
# 在服务器（3090/A100）上运行，完成：
#   Phase 0: LoRA 合并 → 模型拆分
#   Phase 1: LLM 量化 (GGUF) → Vision 量化 (INT8)
#
# 用法:
#   cd pouring_VLA
#   bash edge_optimization/scripts/run_phase0_phase1.sh
# ==============================================================================

set -e  # 任何命令失败即退出

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

echo "=================================================="
echo " OpenVLA Edge Optimization Pipeline"
echo " Phase 0 + Phase 1"
echo "=================================================="
echo "  Repo root: $REPO_ROOT"
echo ""

# ---------- Phase 0 Step 1: 合并 LoRA ----------
echo ""
echo "▶ Phase 0 Step 1: 合并 LoRA 权重"
echo "──────────────────────────────────"
python edge_optimization/scripts/step0_merge_lora.py \
    --base_checkpoint checkpoints/openvla-7b \
    --lora_path edge_optimization/runs_overfit2/openvla-7b+dobot_pouring+b16+lr-0.0004+lora-r32+dropout-0.0--image_aug/checkpoint-14000 \
    --output_dir edge_optimization/merged_model \
    --dataset_stats edge_optimization/runs_overfit2/openvla-7b+dobot_pouring+b16+lr-0.0004+lora-r32+dropout-0.0--image_aug/checkpoint-14000/dataset_statistics.json

# ---------- Phase 0 Step 2: 拆分子模块 ----------
echo ""
echo "▶ Phase 0 Step 2: 拆分子模块 (LLM / Vision / Action Head)"
echo "──────────────────────────────────────────────────────────"
python edge_optimization/scripts/step1_extract_components.py \
    --merged_model edge_optimization/merged_model \
    --output_dir edge_optimization/components \
    --export_vision_onnx

# ---------- Phase 1 Step 1: LLM → GGUF + 量化 ----------
echo ""
echo "▶ Phase 1 Step 1: LLM → GGUF 转换 + Q4_K_M/Q8_0 量化"
echo "──────────────────────────────────────────────────────"
python edge_optimization/scripts/step2_convert_llm_gguf.py \
    --llm_dir edge_optimization/components/llm_llama2_7b \
    --output_dir edge_optimization/gguf_models \
    --skip_clone --skip_build \
    --quantize Q4_K_M Q8_0

# ---------- Phase 1 Step 2: Vision INT8 量化 ----------
echo ""
echo "▶ Phase 1 Step 2: Vision 编码器 INT8 量化"
echo "─────────────────────────────────────────"
python edge_optimization/scripts/step3_quantize_vision.py \
    --vision_onnx edge_optimization/components/vision_projector/vision_projector.onnx \
    --mode dynamic

# ---------- 验证 ----------
echo ""
echo "▶ 端到端精度验证"
echo "────────────────"
python edge_optimization/scripts/step4_verify_pipeline.py \
    --base_checkpoint checkpoints/openvla-7b \
    --lora_path edge_optimization/runs_overfit2/openvla-7b+dobot_pouring+b16+lr-0.0004+lora-r32+dropout-0.0--image_aug/checkpoint-14000 \
    --merged_model edge_optimization/merged_model \
    --unnorm_key dobot_pouring \
    --num_tests 3

# ---------- 汇总 ----------
echo ""
echo "=================================================="
echo " ✅ Phase 0 + Phase 1 完成！"
echo "=================================================="
echo ""
echo "产出文件:"
echo "  edge_optimization/merged_model/          — 合并后的完整模型"
echo "  edge_optimization/components/"
echo "    llm_llama2_7b/                          — HF 格式 LLM"
echo "    vision_projector/                       — Vision ONNX + state_dict"
echo "    action_head/                            — 解码参数 JSON"
echo "  edge_optimization/gguf_models/            — 量化后的 GGUF 模型"
echo ""
echo "需要拷贝到 Q900 的文件:"
echo "  1. edge_optimization/gguf_models/openvla-llm-Q4_K_M.gguf   (~4 GB)"
echo "  2. edge_optimization/components/vision_projector/vision_projector_int8_dynamic.onnx"
echo "  3. edge_optimization/components/vision_projector/vision_projector.pt  (备用)"
echo "  4. edge_optimization/components/action_head/action_head_params.json"
echo ""
echo "下一步: Phase 2 — 在 Q900 上部署推理 runtime"
