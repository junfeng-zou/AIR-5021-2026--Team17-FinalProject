#!/usr/bin/env python3
"""
Phase 0 Step 2: 拆分 OpenVLA 为独立子模块
==========================================

从合并后的完整模型中提取三个独立部分：
  1. Vision Backbone (DINOv2 + SigLIP) + Projector → PyTorch state_dict
  2. Language Model (Llama-2-7B) → 标准 HuggingFace Llama 格式
  3. Action Head 参数 (bin_centers, vocab_size, dataset_statistics) → JSON

Vision 模块导出为 ONNX（Phase 1 中做 INT8 量化）。
LLM 导出为标准 HF LlamaForCausalLM 格式（后续转 GGUF）。

用法:
    python edge_optimization/scripts/step1_extract_components.py \
        --merged_model edge_optimization/merged_model \
        --output_dir edge_optimization/components
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="拆分 OpenVLA 为独立子模块")
    p.add_argument(
        "--merged_model",
        type=str,
        default="edge_optimization/merged_model",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="edge_optimization/components",
    )
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    p.add_argument(
        "--export_vision_onnx",
        action="store_true",
        default=True,
        help="是否导出视觉编码器为 ONNX",
    )
    p.add_argument(
        "--no_export_vision_onnx",
        action="store_false",
        dest="export_vision_onnx",
    )
    return p.parse_args()


def _resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(repo_root, path)


def extract_llm_as_hf_llama(model, output_dir: str, dtype: torch.dtype) -> None:
    """
    从 OpenVLA 的 language_model 提取一个标准 HuggingFace LlamaForCausalLM，
    可直接被 llama.cpp 的 convert_hf_to_gguf.py 识别。
    """
    from transformers import LlamaForCausalLM, LlamaConfig

    llm = model.language_model
    print(f"  LLM type: {type(llm).__name__}")

    # 获取 LLM config
    llm_config = llm.config
    print(f"  LLM config: vocab_size={llm_config.vocab_size}, "
          f"hidden_size={llm_config.hidden_size}, "
          f"num_layers={llm_config.num_hidden_layers}, "
          f"num_heads={llm_config.num_attention_heads}")

    # 直接保存 LLM 子模块 — 它已经是 LlamaForCausalLM
    os.makedirs(output_dir, exist_ok=True)
    llm.save_pretrained(output_dir, safe_serialization=True)

    # 验证 config
    saved_config = LlamaConfig.from_pretrained(output_dir)
    print(f"  已保存 LLM: vocab_size={saved_config.vocab_size}")


def extract_vision_and_projector(
    model, output_dir: str, export_onnx: bool, dtype: torch.dtype
) -> dict:
    """
    提取 vision_backbone + projector 的 state_dict，并可选导出 ONNX。
    返回视觉模块的元数据。
    """
    os.makedirs(output_dir, exist_ok=True)

    vision_backbone = model.vision_backbone
    projector = model.projector

    # 保存 state_dict
    vision_sd = OrderedDict()
    for k, v in vision_backbone.state_dict().items():
        vision_sd[f"vision_backbone.{k}"] = v
    for k, v in projector.state_dict().items():
        vision_sd[f"projector.{k}"] = v

    sd_path = os.path.join(output_dir, "vision_projector.pt")
    torch.save(vision_sd, sd_path)
    sd_size_mb = os.path.getsize(sd_path) / 1024 / 1024
    print(f"  vision+projector state_dict: {sd_size_mb:.1f} MB ({len(vision_sd)} tensors)")

    # 保存元数据
    meta = {
        "use_fused_vision_backbone": vision_backbone.use_fused_vision_backbone,
        "embed_dim": vision_backbone.embed_dim,
        "image_sizes": list(model.config.image_sizes),
        "timm_model_ids": list(model.config.timm_model_ids),
        "timm_override_act_layers": list(model.config.timm_override_act_layers),
        "projector_vision_dim": projector.vision_dim,
        "projector_llm_dim": projector.llm_dim,
        "num_vision_params": sum(p.numel() for p in vision_backbone.parameters()),
        "num_projector_params": sum(p.numel() for p in projector.parameters()),
    }
    with open(os.path.join(output_dir, "vision_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  vision params: {meta['num_vision_params']:,}")
    print(f"  projector params: {meta['num_projector_params']:,}")
    print(f"  fused backbone: {meta['use_fused_vision_backbone']}")
    print(f"  embed_dim: {meta['embed_dim']}")

    # ONNX 导出
    if export_onnx:
        print("\n  导出 Vision+Projector 为 ONNX ...")
        _export_vision_onnx(vision_backbone, projector, output_dir, dtype, meta)

    return meta


class VisionProjectorWrapper(nn.Module):
    """封装 vision_backbone + projector 用于 ONNX 导出"""
    def __init__(self, vision_backbone, projector):
        super().__init__()
        self.vision_backbone = vision_backbone
        self.projector = projector

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        patch_features = self.vision_backbone(pixel_values)
        projected = self.projector(patch_features)
        return projected


def _export_vision_onnx(
    vision_backbone, projector, output_dir: str, dtype: torch.dtype, meta: dict
) -> None:
    """将 vision_backbone + projector 导出为单个 ONNX 模型"""
    wrapper = VisionProjectorWrapper(vision_backbone, projector)
    wrapper.eval()

    device = torch.device("cpu")
    # 对于 ONNX 导出，用 float32
    wrapper = wrapper.float().to(device)

    # fused backbone 时 pixel_values 是 [B, 6, H, W]（两个 3 通道图拼在一起）
    if meta["use_fused_vision_backbone"]:
        channels = 6
    else:
        channels = 3
    h, w = meta["image_sizes"][0], meta["image_sizes"][-1]

    dummy_input = torch.randn(1, channels, h, w, dtype=torch.float32, device=device)

    onnx_path = os.path.join(output_dir, "vision_projector.onnx")
    try:
        torch.onnx.export(
            wrapper,
            dummy_input,
            onnx_path,
            input_names=["pixel_values"],
            output_names=["projected_embeddings"],
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "projected_embeddings": {0: "batch_size"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        onnx_size_mb = os.path.getsize(onnx_path) / 1024 / 1024
        print(f"  ✅ ONNX 导出成功: {onnx_path} ({onnx_size_mb:.1f} MB)")

        # 验证 ONNX 模型
        try:
            import onnx
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print(f"  ✅ ONNX 模型验证通过")
        except ImportError:
            print("  [INFO] onnx 库未安装，跳过验证")

    except Exception as e:
        print(f"  ❌ ONNX 导出失败: {e}")
        print("  提示: 如果 timm 的 ViT 有不支持的算子，可以在边缘端使用 PyTorch 直接加载 state_dict")
        import traceback
        traceback.print_exc()


def extract_action_head_params(model, output_dir: str, stats_path: str) -> None:
    """提取 action head 解码所需的全部参数"""
    os.makedirs(output_dir, exist_ok=True)

    # bin_centers
    bin_centers = model.bin_centers
    if isinstance(bin_centers, torch.Tensor):
        bin_centers = bin_centers.detach().cpu().numpy()
    else:
        bin_centers = np.asarray(bin_centers)

    # vocab_size（用于解码 token → bin index）
    # OpenVLA: self.vocab_size = config.text_config.vocab_size - config.pad_to_multiple_of
    vocab_size = model.vocab_size
    n_action_bins = model.config.n_action_bins
    pad_to_multiple_of = model.config.pad_to_multiple_of

    action_params = {
        "vocab_size": int(vocab_size),
        "n_action_bins": int(n_action_bins),
        "pad_to_multiple_of": int(pad_to_multiple_of),
        "text_config_vocab_size": int(model.config.text_config.vocab_size),
        "bin_centers": bin_centers.tolist(),
    }

    # dataset_statistics
    if os.path.isfile(stats_path):
        with open(stats_path, "r") as f:
            ds_stats = json.load(f)
        action_params["dataset_statistics"] = ds_stats
    elif hasattr(model, "norm_stats"):
        # 从模型中提取（仅保留我们需要的 key）
        action_params["dataset_statistics"] = dict(model.norm_stats)

    params_path = os.path.join(output_dir, "action_head_params.json")
    with open(params_path, "w") as f:
        json.dump(action_params, f, indent=2)

    print(f"  vocab_size (for decode): {vocab_size}")
    print(f"  n_action_bins: {n_action_bins}")
    print(f"  bin_centers shape: {bin_centers.shape}")
    print(f"  pad_to_multiple_of: {pad_to_multiple_of}")
    if "dataset_statistics" in action_params:
        ds_keys = list(action_params["dataset_statistics"].keys())
        print(f"  dataset_statistics keys: {ds_keys}")
    print(f"  已保存到: {params_path}")


def main() -> int:
    args = parse_args()
    merged_dir = _resolve(args.merged_model)
    output_dir = _resolve(args.output_dir)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    if not os.path.isdir(merged_dir):
        print(f"[ERROR] 合并模型目录不存在: {merged_dir}", file=sys.stderr)
        print("  请先运行 step0_merge_lora.py", file=sys.stderr)
        return 1

    print("=" * 60)
    print("Phase 0 Step 2: 拆分 OpenVLA → 独立子模块")
    print("=" * 60)
    print(f"  输入: {merged_dir}")
    print(f"  输出: {output_dir}")
    print("=" * 60)

    # 加载合并后的模型
    from transformers import AutoModelForVision2Seq

    print("\n[1/4] 加载合并后模型 ...")
    model = AutoModelForVision2Seq.from_pretrained(
        merged_dir,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    print(f"  总参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 提取 LLM
    llm_dir = os.path.join(output_dir, "llm_llama2_7b")
    print(f"\n[2/4] 提取 LLM (Llama-2-7B) → {llm_dir}")
    extract_llm_as_hf_llama(model, llm_dir, dtype)

    # 保存 tokenizer 到 LLM 目录（llama.cpp 转换需要）
    print("  复制 tokenizer 到 LLM 目录 ...")
    for fname in [
        "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
        "added_tokens.json", "special_tokens_map.json",
        "generation_config.json",
    ]:
        src = os.path.join(merged_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(llm_dir, fname))
            print(f"    ✓ {fname}")

    # 提取 Vision + Projector
    vision_dir = os.path.join(output_dir, "vision_projector")
    print(f"\n[3/4] 提取 Vision+Projector → {vision_dir}")
    vision_meta = extract_vision_and_projector(
        model, vision_dir, args.export_vision_onnx, dtype
    )

    # 提取 Action Head 参数
    action_dir = os.path.join(output_dir, "action_head")
    stats_path = os.path.join(merged_dir, "dataset_statistics.json")
    print(f"\n[4/4] 提取 Action Head 参数 → {action_dir}")
    extract_action_head_params(model, action_dir, stats_path)

    # 总结
    print("\n" + "=" * 60)
    print("✅ 拆分完成！输出结构:")
    print("=" * 60)
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(output_dir, "").count(os.sep)
        indent = "  " * level
        basename = os.path.basename(root)
        print(f"{indent}{basename}/")
        sub_indent = "  " * (level + 1)
        for f in sorted(files):
            fp = os.path.join(root, f)
            sz = os.path.getsize(fp)
            if sz > 1024 * 1024:
                print(f"{sub_indent}{f}  ({sz / 1024 / 1024:.1f} MB)")
            else:
                print(f"{sub_indent}{f}  ({sz / 1024:.1f} KB)")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
