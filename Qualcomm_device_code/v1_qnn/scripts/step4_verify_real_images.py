#!/usr/bin/env python3
"""用真实 HDF5 图像对比 Original+LoRA vs Merged 模型输出"""

import os, sys, json, argparse
import numpy as np
import torch
import h5py
from PIL import Image


def _resolve(path):
    if os.path.isabs(path): return path
    return os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), path)


def load_images_from_hdf5(hdf5_path, num_images=5):
    with h5py.File(hdf5_path, "r") as f:
        rgb = f["observations/images/rgb"][:]
    indices = np.linspace(0, len(rgb) - 1, num_images, dtype=int)
    return [rgb[i] for i in indices]


def test_original(base_ckpt, lora_path, images, task, unnorm_key, device_str):
    print("\n--- Original OpenVLA + LoRA ---")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, repo_root)
    from tools.vla.openvla_lora_runtime import (
        build_vicuna_prompt, infer_action_vector,
        resolve_dataset_stats_path, load_openvla_with_lora,
    )

    dataset_stats = resolve_dataset_stats_path("", lora_path)
    device = torch.device(device_str)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model, processor, action_dim = load_openvla_with_lora(
        base_checkpoint=base_ckpt, lora_path=lora_path,
        processor_path=base_ckpt, dataset_stats_path=dataset_stats,
        device=device, dtype=dtype, unnorm_key=unnorm_key, merge_lora=False,
    )
    prompt = build_vicuna_prompt(task)
    actions = []
    for i, img in enumerate(images):
        action, _ = infer_action_vector(model, processor, img, prompt, device, dtype, action_dim, unnorm_key, 0.0, None)
        actions.append(action)
        print(f"  img[{i}] action: {np.round(action, 6).tolist()}")
    del model; torch.cuda.empty_cache()
    return actions


def test_merged(merged_dir, images, task, unnorm_key, device_str):
    print("\n--- Merged Model ---")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, repo_root)
    from tools.vla.openvla_lora_runtime import (
        build_vicuna_prompt, decode_normalized_action,
        ensure_rgb_uint8_hwc, inject_dataset_statistics,
        maybe_append_empty_token, unnormalize_action_q01q99,
    )
    from transformers import AutoModelForVision2Seq, AutoProcessor

    device = torch.device(device_str)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model = AutoModelForVision2Seq.from_pretrained(
        merged_dir, torch_dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager",
    )
    stats_path = os.path.join(merged_dir, "dataset_statistics.json")
    if os.path.isfile(stats_path):
        inject_dataset_statistics(model, stats_path)
    model = model.to(device).eval()

    processor = AutoProcessor.from_pretrained(merged_dir, trust_remote_code=True)
    prompt = build_vicuna_prompt(task)
    action_dim = model.get_action_dim(unnorm_key)

    actions = []
    for i, img in enumerate(images):
        rgb_u8 = ensure_rgb_uint8_hwc(img)
        rgb_pil = Image.fromarray(rgb_u8)
        inputs = processor(prompt, rgb_pil, return_tensors="pt")
        inputs = {k: v.to(device=device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)
        maybe_append_empty_token(inputs, device=device, dtype_ids=inputs["input_ids"].dtype)
        prompt_len = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=action_dim, min_new_tokens=action_dim, do_sample=False)
        action_norm = decode_normalized_action(model, generated_ids, action_dim, prompt_len=prompt_len)
        action = unnormalize_action_q01q99(model, action_norm, unnorm_key)
        action = np.asarray(action[:7], dtype=np.float32)
        actions.append(action)
        print(f"  img[{i}] action: {np.round(action, 6).tolist()}")
    del model; torch.cuda.empty_cache()
    return actions


def compare(name_a, actions_a, name_b, actions_b):
    print(f"\n--- 对比: {name_a} vs {name_b} ---")
    for i, (a, b) in enumerate(zip(actions_a, actions_b)):
        diff = np.abs(a - b)
        rel = diff / (np.abs(a) + 1e-8)
        print(f"  img[{i}] max_abs={diff.max():.6e}  mean_abs={diff.mean():.6e}  max_rel={rel.max():.4f}")
    all_d = [np.abs(a - b) for a, b in zip(actions_a, actions_b)]
    mx = max(d.max() for d in all_d)
    mn = np.mean([d.mean() for d in all_d])
    tag = "完全一致" if mx < 1e-4 else ("近似一致" if mx < 1e-2 else "存在差异")
    print(f"  {tag}  overall max_diff={mx:.6e}  mean_diff={mn:.6e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5_path", type=str, required=True)
    p.add_argument("--base_checkpoint", type=str, default="checkpoints/openvla-7b")
    p.add_argument("--lora_path", type=str, required=True)
    p.add_argument("--merged_model", type=str, default="edge_optimization/merged_model")
    p.add_argument("--task", type=str, default="pour water from bottle into cup")
    p.add_argument("--unnorm_key", type=str, default="dobot_pouring")
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_images", type=int, default=5)
    args = p.parse_args()

    hdf5_path = _resolve(args.hdf5_path)
    base_ckpt = _resolve(args.base_checkpoint)
    lora_path = _resolve(args.lora_path)
    merged_dir = _resolve(args.merged_model)

    print("=" * 60)
    print("真实图像验证: Original+LoRA vs Merged")
    print("=" * 60)
    print(f"  HDF5: {hdf5_path}")
    print(f"  图像数: {args.num_images}")

    images = load_images_from_hdf5(hdf5_path, args.num_images)
    print(f"  加载了 {len(images)} 张图像, shape={images[0].shape}")

    results = {}
    try:
        results["original"] = test_original(base_ckpt, lora_path, images, args.task, args.unnorm_key, args.device)
    except Exception as e:
        print(f"  [SKIP] 原始模型: {e}")

    try:
        results["merged"] = test_merged(merged_dir, images, args.task, args.unnorm_key, args.device)
    except Exception as e:
        print(f"  [SKIP] 合并模型: {e}")

    if "original" in results and "merged" in results:
        compare("Original+LoRA", results["original"], "Merged", results["merged"])

    print("\nDone.")


if __name__ == "__main__":
    main()
