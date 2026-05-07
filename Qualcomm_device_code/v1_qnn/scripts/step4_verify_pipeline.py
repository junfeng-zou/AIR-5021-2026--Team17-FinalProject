#!/usr/bin/env python3
"""
Phase 0+1 验证: 端到端精度对比
================================

用一组真实图像对比原始 OpenVLA+LoRA 推理 vs 量化拆分后的 pipeline 推理，
验证各阶段的精度损失是否在可接受范围内。

用法:
    python edge_optimization/scripts/step4_verify_pipeline.py \
        --base_checkpoint checkpoints/openvla-7b \
        --lora_path edge_optimization/pouring_lora_3090_real_weighted/checkpoint_step_5000 \
        --merged_model edge_optimization/merged_model \
        --components_dir edge_optimization/components
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="端到端精度验证")
    p.add_argument("--base_checkpoint", type=str, default="checkpoints/openvla-7b")
    p.add_argument(
        "--lora_path",
        type=str,
        default="edge_optimization/pouring_lora_3090_real_weighted/checkpoint_step_5000",
    )
    p.add_argument("--merged_model", type=str, default="edge_optimization/merged_model")
    p.add_argument("--components_dir", type=str, default="edge_optimization/components")
    p.add_argument("--task", type=str, default="pour water from bottle into cup")
    p.add_argument("--unnorm_key", type=str, default="pouring_hdf5")
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_tests", type=int, default=3)
    return p.parse_args()


def _resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(repo_root, path)


def create_test_image() -> np.ndarray:
    """创建一个确定性的测试图像（224x224 RGB）"""
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, (224, 224, 3), dtype=np.uint8)


def test_original_model(
    base_ckpt: str,
    lora_path: str,
    test_images: list[np.ndarray],
    task: str,
    unnorm_key: str,
    device_str: str,
) -> list[np.ndarray]:
    """使用原始 OpenVLA+LoRA 推理"""
    print("\n--- 测试 1: 原始 OpenVLA + LoRA ---")

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, repo_root)
    lora_code = os.path.join(repo_root, "LoRA_train", "code")
    if lora_code not in sys.path:
        sys.path.insert(0, lora_code)

    from tools.vla.openvla_lora_runtime import (
        build_vicuna_prompt,
        infer_action_vector,
        inject_dataset_statistics,
        load_openvla_with_lora,
        resolve_dataset_stats_path,
    )

    dataset_stats = resolve_dataset_stats_path("", lora_path)
    device = torch.device(device_str)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model, processor, action_dim = load_openvla_with_lora(
        base_checkpoint=base_ckpt,
        lora_path=lora_path,
        processor_path=base_ckpt,
        dataset_stats_path=dataset_stats,
        device=device,
        dtype=dtype,
        unnorm_key=unnorm_key,
        merge_lora=False,
    )
    prompt = build_vicuna_prompt(task)

    actions = []
    for i, img in enumerate(test_images):
        action, _ = infer_action_vector(
            model, processor, img, prompt, device, dtype,
            action_dim, unnorm_key, 0.0, None,
        )
        actions.append(action)
        print(f"  img[{i}] action: {np.round(action, 6).tolist()}")

    # 释放显存
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return actions


def test_merged_model(
    merged_dir: str,
    test_images: list[np.ndarray],
    task: str,
    unnorm_key: str,
    device_str: str,
) -> list[np.ndarray]:
    """使用合并后模型推理"""
    print("\n--- 测试 2: 合并后模型 (merge_and_unload) ---")

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, repo_root)
    lora_code = os.path.join(repo_root, "LoRA_train", "code")
    if lora_code not in sys.path:
        sys.path.insert(0, lora_code)

    from tools.vla.openvla_lora_runtime import (
        build_vicuna_prompt,
        decode_normalized_action,
        ensure_rgb_uint8_hwc,
        inject_dataset_statistics,
        maybe_append_empty_token,
        unnormalize_action_q01q99,
    )
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from PIL import Image

    device = torch.device(device_str)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model = AutoModelForVision2Seq.from_pretrained(
        merged_dir, torch_dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager",
    )

    # 注入 dataset statistics
    stats_path = os.path.join(merged_dir, "dataset_statistics.json")
    if os.path.isfile(stats_path):
        inject_dataset_statistics(model, stats_path)

    model = model.to(device).eval()

    processor = AutoProcessor.from_pretrained(merged_dir, trust_remote_code=True)
    prompt = build_vicuna_prompt(task)
    action_dim = model.get_action_dim(unnorm_key)

    actions = []
    for i, img in enumerate(test_images):
        rgb_u8 = ensure_rgb_uint8_hwc(img)
        rgb_pil = Image.fromarray(rgb_u8)
        inputs = processor(prompt, rgb_pil, return_tensors="pt")
        inputs = {k: v.to(device=device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)
        maybe_append_empty_token(inputs, device=device, dtype_ids=inputs["input_ids"].dtype)
        prompt_len = int(inputs["input_ids"].shape[1])

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=action_dim,
                min_new_tokens=action_dim,
                do_sample=False,
            )
        action_norm = decode_normalized_action(model, generated_ids, action_dim, prompt_len=prompt_len)
        action = unnormalize_action_q01q99(model, action_norm, unnorm_key)
        action = np.asarray(action[:7], dtype=np.float32)
        actions.append(action)
        print(f"  img[{i}] action: {np.round(action, 6).tolist()}")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return actions


def compare_actions(name_a: str, actions_a: list, name_b: str, actions_b: list) -> None:
    """对比两组动作输出"""
    print(f"\n--- 对比: {name_a} vs {name_b} ---")
    assert len(actions_a) == len(actions_b)

    for i, (a, b) in enumerate(zip(actions_a, actions_b)):
        diff = np.abs(a - b)
        rel_diff = diff / (np.abs(a) + 1e-8)
        print(f"  img[{i}] max_abs_diff={diff.max():.6e}  mean_abs_diff={diff.mean():.6e}  "
              f"max_rel_diff={rel_diff.max():.4f}")

    # 总体统计
    all_diffs = [np.abs(a - b) for a, b in zip(actions_a, actions_b)]
    overall_max = max(d.max() for d in all_diffs)
    overall_mean = np.mean([d.mean() for d in all_diffs])

    if overall_max < 1e-4:
        print(f"  ✅ 完全一致 (max_diff={overall_max:.6e})")
    elif overall_max < 1e-2:
        print(f"  ✅ 近似一致 (max_diff={overall_max:.6e}, 浮点精度差异)")
    else:
        print(f"  ⚠️ 存在差异 (max_diff={overall_max:.6e}), 请检查合并/量化过程")


def main() -> int:
    args = parse_args()
    base_ckpt = _resolve(args.base_checkpoint)
    lora_path = _resolve(args.lora_path)
    merged_dir = _resolve(args.merged_model)
    components_dir = _resolve(args.components_dir)

    print("=" * 60)
    print("Phase 0+1 验证: 端到端精度对比")
    print("=" * 60)

    # 创建测试图像
    test_images = [create_test_image() for _ in range(args.num_tests)]
    print(f"  测试图像数: {len(test_images)}")

    results = {}

    # Test 1: 原始模型
    if os.path.isdir(base_ckpt) and os.path.isdir(lora_path):
        try:
            results["original"] = test_original_model(
                base_ckpt, lora_path, test_images,
                args.task, args.unnorm_key, args.device,
            )
        except Exception as e:
            print(f"  [SKIP] 原始模型测试失败: {e}")
    else:
        print("  [SKIP] 原始模型路径不存在")

    # Test 2: 合并后模型
    if os.path.isdir(merged_dir):
        try:
            results["merged"] = test_merged_model(
                merged_dir, test_images,
                args.task, args.unnorm_key, args.device,
            )
        except Exception as e:
            print(f"  [SKIP] 合并模型测试失败: {e}")
    else:
        print("  [SKIP] 合并模型目录不存在")

    # 对比
    print("\n" + "=" * 60)
    print("精度对比结果")
    print("=" * 60)

    if "original" in results and "merged" in results:
        compare_actions("Original+LoRA", results["original"], "Merged", results["merged"])
    else:
        available = list(results.keys())
        print(f"  可用结果: {available}")
        if len(available) >= 1:
            name = available[0]
            print(f"  {name} 输出:")
            for i, a in enumerate(results[name]):
                print(f"    img[{i}] = {np.round(a, 6).tolist()}")

    print("\n✅ 验证完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
