#!/usr/bin/env python3
"""
step2_reexport_onnx_static.py
==============================
修复 "node_view ReshapeOp shape mismatch" 的根本方法：
重新导出 vision_projector.onnx，去掉 dynamic_axes。

根本原因: 原始导出用了 dynamic_axes={"pixel_values":{0:"batch_size"}}
         → 内部所有依赖 batch_size 的 Reshape 变成动态 → qairt-converter 溢出

修复:    dynamic_axes={} + do_constant_folding=True
         → 所有 shape 在导出时固化为常量

用法 (需要在含 OpenVLA 权重的环境中运行，约需 5-10 分钟加载模型):
    cd /home/zjf/pouring_VLA
    conda activate openvla   # 或你加载 OpenVLA 的环境
    python edge_optimization_v2.0/scripts/step2_reexport_onnx_static.py \
        --model_path /path/to/openvla-7b-finetuned
"""
from __future__ import annotations
import argparse, json, os, sys, time
from collections import OrderedDict
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True,
                   help="OpenVLA/LoRA 合并后模型路径（本地目录）")
    p.add_argument("--output_dir",
                   default="edge_optimization/components/vision_projector",
                   help="输出目录，会覆盖 vision_projector.onnx/.onnx.data")
    args = p.parse_args()

    import torch
    import torch.nn as nn

    # ── 加载模型 ───────────────────────────────────────────────────
    print(f"[1/3] 加载模型: {args.model_path}")
    print("  (约 5-10 分钟，需要 ~14GB RAM)")
    t0 = time.time()

    # 尝试用 step1 里已有的 PrismaticVLM 加载方式
    repo_root = Path(__file__).parent.parent.parent  # pouring_VLA/
    sys.path.insert(0, str(repo_root))

    try:
        from transformers import AutoModelForVision2Seq, AutoProcessor
        from peft import PeftModel
        print("  使用 transformers + peft 加载 ...")
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_path, 
            torch_dtype=torch.float32, 
            low_cpu_mem_usage=True,
            attn_implementation="eager"  # 🟢 灵魂改动：强行关闭 FlashAttention，回退到原生数学实现
        )
    except Exception:
        # 尝试 prismatic 直接加载
        try:
            import prismatic
            model = prismatic.load(args.model_path)
        except Exception as e:
            print(f"  ❌ 模型加载失败: {e}")
            print("  请确认 --model_path 正确，且在含 OpenVLA 依赖的 conda 环境中运行")
            sys.exit(1)

    print(f"  加载完成 ({time.time()-t0:.0f}s)")

    # ── 提取 vision_backbone + projector ──────────────────────────
    print("\n[2/3] 导出 ONNX (static shapes, 无 dynamic_axes) ...")
    vision_backbone = model.vision_backbone
    projector = model.projector

    # 读 meta
    meta_path = os.path.join(args.output_dir, "vision_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    channels = 6 if meta["use_fused_vision_backbone"] else 3
    h, w = meta["image_sizes"][0], meta["image_sizes"][-1]

    class VisionProjectorWrapper(nn.Module):
        def __init__(self, vb, proj):
            super().__init__()
            self.vb = vb
            self.proj = proj
        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            return self.proj(self.vb(pixel_values))

    # 🟢 彻底去掉了 .half()，退回标准的 FP32
    wrapper = VisionProjectorWrapper(vision_backbone, projector)
    wrapper = wrapper.cpu().eval()

    # 🟢 占位符也退回 float32
    dummy = torch.zeros(1, channels, h, w, dtype=torch.float32)

    # 先跑一次验证 wrapper 正常
    with torch.no_grad():
        out_pt = wrapper(dummy)
    print(f"  PyTorch 输出 shape: {out_pt.shape}")

    out_path = os.path.join(args.output_dir, "vision_projector.onnx")
    print(f"  输出: {out_path}")

    t1 = time.time()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            out_path,
            input_names=["pixel_values"],
            output_names=["projected_embeddings"],
            # 1. 绝对不要加 dynamic_axes！甚至明确传一个空字典进去
            dynamic_axes={}, 
            opset_version=17,
            # 2. 强制要求导出时就进行常量折叠
            do_constant_folding=True, 
            # 3. 非常关键：让 PyTorch 强制保留静态 shape 信息
            keep_initializers_as_inputs=False,
            export_params=True
        )
    print(f"  导出完成 ({time.time()-t1:.0f}s)")

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  文件大小: {size_mb:.0f} MB")

# ── 验证 ──────────────────────────────────────────────────────
    print("\n[3/3] 验证 ...")
    import onnx, onnxruntime as ort
    
    # 直接传文件路径进行验证，完美绕过 2GB 内存加载限制
    onnx.checker.check_model(out_path)
    print("  ✅ onnx.checker 通过")

    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    result = sess.run(None, {"pixel_values": dummy.numpy()})
    print(f"  ✅ onnxruntime 输出 shape: {result[0].shape}")
    print(f"\n✅ 导出成功！现在可以运行:")
    print(f"   bash edge_optimization_v2.0/scripts/step2_convert_onnx_to_dlc.sh")


if __name__ == "__main__":
    main()
