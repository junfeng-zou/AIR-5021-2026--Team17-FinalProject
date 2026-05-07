#!/usr/bin/env python3
"""
Step 5: ONNX 精度对齐验证 (在工作站运行)
==========================================

验证 vision_projector.onnx 与原始 PyTorch 模型的特征对齐程度。
余弦相似度必须 > 0.999 才可以继续部署。

用法:
    cd /home/zjf/pouring_VLA
    conda run -n openVLA python edge_optimization/scripts/step5_verify_onnx_alignment.py \\
        --merged_model edge_optimization/merged_model \\
        --onnx_path edge_optimization/components/vision_projector/vision_projector.onnx
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--merged_model", default="edge_optimization/merged_model")
    p.add_argument(
        "--onnx_path",
        default="edge_optimization/components/vision_projector/vision_projector.onnx",
    )
    p.add_argument("--num_tests", type=int, default=3, help="随机测试次数")
    p.add_argument("--cos_threshold", type=float, default=0.999, help="余弦相似度最低要求")
    return p.parse_args()


def get_image_normalization_params():
    """
    返回 OpenVLA 双骨干的归一化参数。

    OpenVLA 使用 fused backbone:
      通道 0-2 → DINOv2 (ImageNet normalization)
      通道 3-5 → SigLIP (mean=0.5, std=0.5)

    归一化公式: (x / 255.0 - mean) / std
    """
    dinov2_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    dinov2_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    siglip_mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    siglip_std  = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    return dinov2_mean, dinov2_std, siglip_mean, siglip_std


def preprocess_image(rgb_uint8: np.ndarray) -> np.ndarray:
    """
    将 224x224 RGB uint8 图像预处理为 ONNX 输入 (1, 6, 224, 224) float32。

    注意: ONNX 模型期望的输入是已归一化的像素值，
    而非原始 [0, 255] 范围。
    """
    dinov2_mean, dinov2_std, siglip_mean, siglip_std = get_image_normalization_params()
    x = rgb_uint8.astype(np.float32) / 255.0  # (224, 224, 3), [0, 1]
    x = np.transpose(x, (2, 0, 1))            # (3, 224, 224)

    # DINOv2 normalized channels
    dino_ch = (x - dinov2_mean[:, None, None]) / dinov2_std[:, None, None]
    # SigLIP normalized channels
    siglip_ch = (x - siglip_mean[:, None, None]) / siglip_std[:, None, None]

    fused = np.concatenate([dino_ch, siglip_ch], axis=0)  # (6, 224, 224)
    return fused[np.newaxis].astype(np.float32)            # (1, 6, 224, 224)


def preprocess_image_torch(rgb_uint8: np.ndarray) -> torch.Tensor:
    """同 preprocess_image，但返回 torch.Tensor。"""
    arr = preprocess_image(rgb_uint8)
    return torch.from_numpy(arr)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    args = parse_args()
    merged_dir = os.path.join(_REPO_ROOT, args.merged_model) if not os.path.isabs(args.merged_model) else args.merged_model
    onnx_path  = os.path.join(_REPO_ROOT, args.onnx_path)  if not os.path.isabs(args.onnx_path)  else args.onnx_path

    print("=" * 62)
    print("Step 5: ONNX 精度对齐验证")
    print("=" * 62)
    print(f"  合并模型: {merged_dir}")
    print(f"  ONNX:     {onnx_path}")
    print()

    # ── 加载 ONNX ────────────────────────────────────────────
    print("[1/3] 加载 ONNX Runtime session ...")
    try:
        import onnxruntime as ort
    except ImportError:
        print("  ❌ onnxruntime 未安装: pip install onnxruntime")
        return 1

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, sess_opts, providers=["CPUExecutionProvider"])
    print(f"  ✅ ONNX session 创建成功")

    # ── 加载 PyTorch 参考模型 ─────────────────────────────────
    print("\n[2/3] 加载 PyTorch 参考模型 (仅 CPU, 用于对比) ...")
    from transformers import AutoModelForVision2Seq

    model = AutoModelForVision2Seq.from_pretrained(
        merged_dir,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    ).eval()
    print(f"  ✅ 模型加载完毕")

    # ── 对齐测试 ─────────────────────────────────────────────
    print(f"\n[3/3] 运行 {args.num_tests} 次随机图像对齐测试 ...")
    print(f"  阈值: 余弦相似度 > {args.cos_threshold}")
    print()

    all_pass = True
    cos_sims  = []
    max_errs  = []

    for i in range(args.num_tests):
        # 生成随机 224x224 RGB 图像
        rng = np.random.default_rng(seed=i)
        rgb_uint8 = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)

        # ONNX 推理
        x_np = preprocess_image(rgb_uint8)
        onnx_out = session.run(None, {"pixel_values": x_np})[0]  # (1, 256, 4096)

        # PyTorch 参考
        x_torch = preprocess_image_torch(rgb_uint8)
        with torch.no_grad():
            patch_feat = model.vision_backbone(x_torch)
            pt_out = model.projector(patch_feat).numpy()           # (1, 256, 4096)

        # 指标
        cos = cosine_similarity(pt_out, onnx_out)
        max_err = float(np.max(np.abs(pt_out - onnx_out)))
        cos_sims.append(cos)
        max_errs.append(max_err)

        status = "✅" if cos > args.cos_threshold else "❌"
        print(f"  测试 {i+1}: cos_sim={cos:.6f}  max_abs_err={max_err:.6f}  {status}")
        if cos <= args.cos_threshold:
            all_pass = False

    print()
    print("=" * 62)
    print(f"  平均余弦相似度: {np.mean(cos_sims):.6f}")
    print(f"  平均最大绝对误差: {np.mean(max_errs):.6f}")
    print()
    if all_pass:
        print("✅ 所有测试通过！ONNX 精度验证成功，可以继续部署。")
        print()
        print("下一步: 将以下文件拷贝到 Q900")
        print("  1. edge_optimization/gguf_models/openvla-llm-Q4_K_M.gguf")
        print("  2. edge_optimization/components/vision_projector/vision_projector.onnx*")
        print("  3. edge_optimization/components/action_head/action_head_params.json")
    else:
        print("❌ 精度验证失败！请检查以下可能原因：")
        print("  1. 图像归一化方式不一致（DINOv2 vs SigLIP 的 mean/std）")
        print("  2. ONNX 导出时 opset 不兼容")
        print("  3. vision_backbone 内部是否包含了额外的预处理层")
        print()
        print("  调试建议: 在 preprocess_image() 中不做归一化，")
        print("  看是否对齐——若对齐则说明 backbone 内部做了归一化。")
    print("=" * 62)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
