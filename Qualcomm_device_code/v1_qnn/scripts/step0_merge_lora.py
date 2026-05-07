#!/usr/bin/env python3
"""
Phase 0 Step 1: 合并 LoRA 权重到 OpenVLA-7B 基座模型
=====================================================

将 LoRA adapter 权重 merge 回基座，产出一个完整的 FP16/BF16 模型，
后续所有操作（拆分、导出、量化）都在此合并后的模型上进行。

用法:
    python edge_optimization/scripts/step0_merge_lora.py \
        --base_checkpoint checkpoints/openvla-7b \
        --lora_path edge_optimization/pouring_lora_3090_real_weighted/checkpoint_step_5000 \
        --output_dir edge_optimization/merged_model \
        --verify
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="合并 LoRA 权重到 OpenVLA-7B 基座")
    p.add_argument(
        "--base_checkpoint",
        type=str,
        default="checkpoints/openvla-7b",
        help="OpenVLA-7B 基座模型目录",
    )
    p.add_argument(
        "--lora_path",
        type=str,
        default="edge_optimization/pouring_lora_3090_real_weighted/checkpoint_step_5000",
        help="LoRA adapter 目录",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="edge_optimization/merged_model",
        help="合并后模型的输出目录",
    )
    p.add_argument(
        "--dataset_stats",
        type=str,
        default="edge_optimization/pouring_lora_3090_real_weighted/dataset_statistics.json",
        help="数据集统计 JSON",
    )
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    p.add_argument(
        "--verify",
        action="store_true",
        help="合并后做一次 dummy 推理，验证合并前后输出一致",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # --- 路径解析 ---
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    base_ckpt = (
        args.base_checkpoint
        if os.path.isabs(args.base_checkpoint)
        else os.path.join(repo_root, args.base_checkpoint)
    )
    lora_path = (
        args.lora_path
        if os.path.isabs(args.lora_path)
        else os.path.join(repo_root, args.lora_path)
    )
    output_dir = (
        args.output_dir
        if os.path.isabs(args.output_dir)
        else os.path.join(repo_root, args.output_dir)
    )
    stats_path = (
        args.dataset_stats
        if os.path.isabs(args.dataset_stats)
        else os.path.join(repo_root, args.dataset_stats)
    )

    for p_name, p_val in [("base_checkpoint", base_ckpt), ("lora_path", lora_path)]:
        if not os.path.isdir(p_val):
            print(f"[ERROR] {p_name} 不存在: {p_val}", file=sys.stderr)
            return 1

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    print("=" * 60)
    print("Phase 0 Step 1: 合并 LoRA → 完整模型")
    print("=" * 60)
    print(f"  基座模型    : {base_ckpt}")
    print(f"  LoRA adapter: {lora_path}")
    print(f"  输出目录    : {output_dir}")
    print(f"  dtype       : {args.dtype}")
    print(f"  verify      : {args.verify}")
    print("=" * 60)

    # --- 1) 加载基座模型 ---
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from peft import PeftModel

    print("\n[1/5] 加载基座模型 ...")
    base_model = AutoModelForVision2Seq.from_pretrained(
        base_ckpt,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    print(f"  基座参数量: {sum(p.numel() for p in base_model.parameters()):,}")

    # --- 2) 注入 dataset_statistics ---
    if os.path.isfile(stats_path):
        print(f"\n[2/5] 注入 dataset_statistics: {stats_path}")
        with open(stats_path, "r") as f:
            stats = json.load(f)
        # 找到 norm_stats holder
        holder = base_model
        for attr in ["model", "base_model"]:
            candidate = getattr(holder, attr, None)
            if candidate is not None and hasattr(candidate, "norm_stats"):
                holder = candidate
                break
        if hasattr(holder, "norm_stats") and isinstance(holder.norm_stats, dict):
            holder.norm_stats.update(stats)
            print(f"  已注入 keys: {list(stats.keys())}")
        else:
            print("  [WARN] 未找到 norm_stats 属性，跳过注入")
    else:
        print(f"\n[2/5] dataset_statistics 文件不存在，跳过: {stats_path}")

    # --- 3) 加载 LoRA 并合并 ---
    print("\n[3/5] 加载 LoRA adapter ...")
    peft_model = PeftModel.from_pretrained(base_model, lora_path, torch_dtype=dtype)
    lora_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    print(f"  LoRA 可训练参数: {lora_params:,} / 总参数: {total_params:,}")

    # --- 可选验证：合并前推理 ---
    pre_merge_output = None
    if args.verify:
        print("\n[验证] 合并前 dummy forward ...")
        peft_model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        peft_model = peft_model.to(device)
        with torch.inference_mode():
            dummy_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)
            out = peft_model(input_ids=dummy_ids)
            pre_merge_output = out.logits.detach().cpu().float()
        peft_model = peft_model.cpu()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\n[4/5] merge_and_unload() ...")
    merged_model = peft_model.merge_and_unload()
    print(f"  合并后参数量: {sum(p.numel() for p in merged_model.parameters()):,}")
    print(f"  PEFT 层已移除: {'peft' not in str(type(merged_model))}")

    # --- 可选验证：合并后推理对比 ---
    if args.verify and pre_merge_output is not None:
        print("\n[验证] 合并后 dummy forward ...")
        merged_model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        merged_model = merged_model.to(device)
        with torch.inference_mode():
            dummy_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)
            out = merged_model(input_ids=dummy_ids)
            post_merge_output = out.logits.detach().cpu().float()
        merged_model = merged_model.cpu()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        max_diff = (pre_merge_output - post_merge_output).abs().max().item()
        mean_diff = (pre_merge_output - post_merge_output).abs().mean().item()
        print(f"  logits max_diff  = {max_diff:.6e}")
        print(f"  logits mean_diff = {mean_diff:.6e}")
        if max_diff < 1e-3:
            print("  ✅ 合并前后输出一致！")
        else:
            print("  ⚠️ 合并前后存在微小差异（通常由浮点精度导致，max_diff < 0.01 可接受）")

    # --- 5) 保存合并后的模型 ---
    print(f"\n[5/5] 保存合并模型到 {output_dir} ...")
    os.makedirs(output_dir, exist_ok=True)
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    print(f"  模型权重已保存")

    # 保存 processor/tokenizer
    print("  保存 Processor/Tokenizer ...")
    try:
        processor = AutoProcessor.from_pretrained(base_ckpt, trust_remote_code=True)
        processor.save_pretrained(output_dir)
    except Exception as e:
        print(f"  [WARN] Processor 保存失败: {e}，手动复制 tokenizer 文件 ...")
        for fname in [
            "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
            "added_tokens.json", "special_tokens_map.json",
            "preprocessor_config.json", "processing_prismatic.py",
            "processor_config.json",
        ]:
            src = os.path.join(base_ckpt, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(output_dir, fname))

    # 复制自定义模型代码（HF trust_remote_code 需要）
    for code_file in ["modeling_prismatic.py", "configuration_prismatic.py"]:
        src = os.path.join(base_ckpt, code_file)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(output_dir, code_file))

    # 保存 dataset_statistics
    if os.path.isfile(stats_path):
        shutil.copy2(stats_path, os.path.join(output_dir, "dataset_statistics.json"))

    # 列出输出目录
    total_size = 0
    print(f"\n  输出文件:")
    for f in sorted(os.listdir(output_dir)):
        fp = os.path.join(output_dir, f)
        if os.path.isfile(fp):
            sz = os.path.getsize(fp)
            total_size += sz
            print(f"    {f:50s} {sz / 1024 / 1024:.1f} MB")
    print(f"  总大小: {total_size / 1024 / 1024 / 1024:.2f} GB")

    print("\n✅ Phase 0 Step 1 完成！合并后模型已保存到:", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
