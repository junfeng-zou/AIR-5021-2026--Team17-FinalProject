"""
OpenVLA-7B + LoRA 推理共用逻辑（供单进程脚本与分进程 client 复用）。
"""

from __future__ import annotations

import json
import os
import sys
from argparse import Namespace
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_LORA_CODE = os.path.join(_REPO_ROOT, "LoRA_train", "code")
if _LORA_CODE not in sys.path:
    sys.path.insert(0, _LORA_CODE)


def check_openvla_runtime_dependencies() -> None:
    failed: list[tuple[str, ImportError]] = []
    for mod in ("timm", "peft", "einops", "transformers"):
        try:
            __import__(mod)
        except ImportError as exc:
            failed.append((mod, exc))
    if failed:
        lines = [
            "[错误] OpenVLA 推理依赖导入失败（可能未安装，或与 transformers/peft 版本不兼容）:",
        ]
        for name, exc in failed:
            lines.append(f"  - {name}: {exc}")
        lines.append(
            "  建议: pip install peft accelerate timm einops；或与 LoRA 训练对齐 "
            '`pip install "peft>=0.11,<0.14"` + transformers>=4.40'
        )
        lines.append(f"当前解释器: {sys.executable}")
        print("\n".join(lines), file=sys.stderr)
        raise SystemExit(1)


def build_vicuna_prompt(instruction: str) -> str:
    from prompting_vicuna import VicunaV15ChatPromptBuilder

    text = instruction.lower().strip()
    builder = VicunaV15ChatPromptBuilder("openvla")
    human_msg = f"<image>\nWhat action should the robot take to {text}?"
    builder.add_turn("human", human_msg)
    return builder.get_prompt()


def _find_norm_stats_holder(model: torch.nn.Module) -> torch.nn.Module:
    cur: Any = model
    for _ in range(8):
        ns = getattr(cur, "norm_stats", None)
        if isinstance(ns, dict) and len(ns) > 0:
            return cur
        nxt = getattr(cur, "model", None)
        if nxt is not None and nxt is not cur:
            cur = nxt
            continue
        nxt = getattr(cur, "base_model", None)
        if nxt is not None and nxt is not cur:
            cur = nxt
            continue
        break
    return model


def inject_dataset_statistics(model: torch.nn.Module, json_path: str) -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    holder = _find_norm_stats_holder(model)
    ns = getattr(holder, "norm_stats", None)
    if not isinstance(ns, dict):
        raise RuntimeError("无法在模型上找到 dict 类型的 norm_stats，请确认基座为 OpenVLA。")
    ns.update(blob)


def resolve_dataset_stats_path(
    dataset_stats: str,
    lora_path: str,
) -> str:
    if dataset_stats and os.path.isfile(dataset_stats):
        return os.path.abspath(dataset_stats)
    root = os.path.abspath(os.path.expanduser(os.path.expandvars(lora_path.strip())))
    candidates: list[str] = [
        # 训练 output_dir 与 final_lora 同级（最常见）
        os.path.join(os.path.dirname(root), "dataset_statistics.json"),
        # 有人把 json 放在 LoRA 适配器目录内
        os.path.join(root, "dataset_statistics.json"),
    ]
    cur = os.path.dirname(root)
    for _ in range(8):
        if not cur or cur == os.path.dirname(cur):
            break
        candidates.append(os.path.join(cur, "dataset_statistics.json"))
        cur = os.path.dirname(cur)

    seen: set[str] = set()
    for c in candidates:
        c = os.path.abspath(c)
        if c in seen:
            continue
        seen.add(c)
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        "未找到 dataset_statistics.json。请用 --dataset_stats 指定文件路径，"
        "或将 dataset_statistics.json 放在 LoRA 的上一级（训练 output_dir）、LoRA 目录内，或上层 run 目录中。\n"
        f"  当前 lora_path={root!r}"
    )


def resolve_dataset_stats_path_from_args(args: Namespace) -> str:
    return resolve_dataset_stats_path(getattr(args, "dataset_stats", "") or "", args.lora_path)


def resolve_base_checkpoint_path(name_or_path: str, project_root: str) -> str:
    s = os.path.expandvars(os.path.expanduser((name_or_path or "").strip()))
    if not s:
        raise ValueError("base_checkpoint 不能为空")

    candidates: list[str] = []
    if os.path.isabs(s):
        candidates.append(s)
    else:
        candidates.append(os.path.abspath(s))
        candidates.append(os.path.join(project_root, s))

    for c in candidates:
        if c and os.path.isdir(c):
            return os.path.abspath(c)

    norm = s.replace("\\", "/")
    if "/" not in norm or norm.count("/") == 1:
        return s

    raise FileNotFoundError(
        f"未找到基座模型目录: {name_or_path!r}（已尝试 cwd 与项目根 {project_root}）。\n"
        "请使用绝对路径或 HF id（如 openvla/openvla-7b）。"
    )


def _unwrap_for_action_decode(model: torch.nn.Module) -> torch.nn.Module:
    """PeftModel / 包装层上可能没有 vocab_size；显式解包到含动作头的基座模块。"""
    m: Any = model
    if hasattr(m, "vocab_size") and hasattr(m, "bin_centers"):
        return m
    if hasattr(m, "base_model"):
        bm = m.base_model
        inner = getattr(bm, "model", bm)
        if hasattr(inner, "vocab_size") and hasattr(inner, "bin_centers"):
            return inner
    return model


def decode_normalized_action(
    model,
    generated_ids: torch.Tensor,
    action_dim: int,
    *,
    prompt_len: int,
) -> np.ndarray:
    """只解码「新生成」的 action_dim 个 token；用 [-action_dim:] 会在提前 EOS 时混入 prompt token，动作会异常且常重复。"""
    core = _unwrap_for_action_decode(model)
    row = generated_ids[0]
    end = int(prompt_len) + int(action_dim)
    if row.shape[0] < end:
        raise RuntimeError(
            f"生成长度不足：seq_len={row.shape[0]} 需要 ≥ prompt_len+action_dim={end}。"
            "请确认 generate 传入 min_new_tokens=action_dim，且未截断提示。"
        )
    predicted_token_ids = row[prompt_len:end].detach().cpu().numpy()
    print(
        "[DEBUG] raw_action_token_ids="
        f"{predicted_token_ids.tolist()} (prompt_len={prompt_len}, action_dim={action_dim})"
    )
    discretized = core.vocab_size - predicted_token_ids
    discretized = np.clip(discretized - 1, 0, core.bin_centers.shape[0] - 1)
    bin_centers = core.bin_centers
    if isinstance(bin_centers, torch.Tensor):
        bin_centers_np = bin_centers.detach().cpu().numpy()
    else:
        bin_centers_np = np.asarray(bin_centers)
    return bin_centers_np[discretized]


def unnormalize_action_q01q99(model, normalized: np.ndarray, unnorm_key: str) -> np.ndarray:
    core = _unwrap_for_action_decode(model)
    stats = core.get_action_stats(unnorm_key)
    mask = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
    action_high = np.array(stats["q99"], dtype=np.float64)
    action_low = np.array(stats["q01"], dtype=np.float64)
    normalized_np = np.asarray(normalized, dtype=np.float64)
    return np.where(
        mask,
        0.5 * (normalized_np + 1.0) * (action_high - action_low) + action_low,
        normalized_np,
    )


def debug_print_action_stats(model, unnorm_key: str) -> None:
    stats = model.get_action_stats(unnorm_key)
    q01 = np.array(stats["q01"], dtype=np.float64)
    q99 = np.array(stats["q99"], dtype=np.float64)
    print(f"[DEBUG] action_stats key={unnorm_key}")
    print(f"[DEBUG] q01={q01.tolist()}")
    print(f"[DEBUG] q99={q99.tolist()}")


def apply_cart_action_gain(action_7: np.ndarray, gain: float) -> np.ndarray:
    """放大前 6 维笛卡尔命令并 clip 到 [-1,1]；第 7 维夹爪不变。

    VLA 反归一化后常见量级约 ±0.01，而手柄遥操多为 ±1；PouringEnv 再乘 pos_action_scale≈5e-3，
    过小会导致 IK 步长肉眼不可见。与训练分布严格一致时将 gain 设为 1。"""
    g = float(gain)
    if g == 1.0:
        return action_7
    a = np.asarray(action_7, dtype=np.float32).copy()
    a[:6] = np.clip(a[:6] * g, -1.0, 1.0)
    return a


def _normalize_pretrained_path(name_or_path: str) -> str:
    """本地目录转绝对路径；HF Hub id 或非目录路径原样返回。"""
    s = (name_or_path or "").strip()
    if not s:
        return s
    exp = os.path.expandvars(os.path.expanduser(s))
    return os.path.abspath(exp) if os.path.isdir(exp) else s


def load_openvla_with_lora(
    base_checkpoint: str,
    lora_path: str,
    processor_path: str,
    dataset_stats_path: str,
    device: torch.device,
    dtype: torch.dtype,
    unnorm_key: str,
    merge_lora: bool,
):
    from peft import PeftModel
    from transformers import AutoModelForVision2Seq, AutoProcessor

    proc_candidates: list[str] = []
    for raw in (processor_path, base_checkpoint, lora_path):
        q = _normalize_pretrained_path(raw)
        if q and q not in proc_candidates:
            proc_candidates.append(q)

    processor = None
    last_exc: BaseException | None = None
    for proc_dir in proc_candidates:
        print(f"加载 Processor: {proc_dir}")
        try:
            processor = AutoProcessor.from_pretrained(proc_dir, trust_remote_code=True)
            break
        except Exception as exc:
            last_exc = exc
            print(f"[警告] Processor 加载失败 ({proc_dir}): {exc}", file=sys.stderr)
    if processor is None:
        raise RuntimeError(
            "无法加载 Processor，已尝试: "
            + ", ".join(proc_candidates)
            + "。LoRA 导出目录里的 tokenizer.json 常与当前 tokenizers 版本不兼容，"
            "请让 --processor_path 指向基座 OpenVLA 目录，或执行 pip install -U tokenizers transformers。"
        ) from last_exc

    print(f"加载基座: {base_checkpoint}")
    base_kw: dict[str, Any] = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
        "attn_implementation": "eager",
    }
    try:
        base_kw["dtype"] = dtype
        base = AutoModelForVision2Seq.from_pretrained(base_checkpoint, **base_kw)
    except TypeError:
        base_kw.pop("dtype", None)
        base_kw["torch_dtype"] = dtype
        base = AutoModelForVision2Seq.from_pretrained(base_checkpoint, **base_kw)

    print(f"加载 LoRA: {lora_path}")
    try:
        model = PeftModel.from_pretrained(base, lora_path, dtype=dtype)
    except TypeError:
        model = PeftModel.from_pretrained(base, lora_path, torch_dtype=dtype)
    if merge_lora:
        print("merge_and_unload() …")
        model = model.merge_and_unload()

    inject_dataset_statistics(model, dataset_stats_path)
    model = model.to(device)
    model.eval()

    if not hasattr(model, "get_action_dim") or not hasattr(model, "bin_centers"):
        raise RuntimeError("模型缺少 OpenVLA 动作头（get_action_dim / bin_centers）。")

    action_dim = model.get_action_dim(unnorm_key)
    print(f"unnorm_key={unnorm_key}  action_dim={action_dim}")
    debug_print_action_stats(model, unnorm_key)

    return model, processor, action_dim


def ensure_rgb_uint8_hwc(rgb: np.ndarray) -> np.ndarray:
    """Isaac / 部分管线给出 float32 [0,1] 或 [0,255] RGB；JPEG/OpenVLA 需要 uint8。错误类型会导致 imencode/视觉特征近似恒定。"""
    x = np.asarray(rgb)
    if x.ndim != 3 or x.shape[-1] < 3:
        raise ValueError(f"期望 HWC 且通道≥3，得到 shape={x.shape}")
    x = x[..., :3]
    if x.dtype == np.uint8:
        return np.ascontiguousarray(x)
    if np.issubdtype(x.dtype, np.floating):
        m = float(np.nanmax(x)) if x.size else 0.0
        if m <= 1.5:
            y = np.clip(x, 0.0, 1.0) * 255.0
        else:
            y = np.clip(x, 0.0, 255.0)
        return np.ascontiguousarray(y.round().astype(np.uint8))
    return np.ascontiguousarray(np.clip(x, 0, 255).astype(np.uint8))


def maybe_append_empty_token(inputs: dict, device: torch.device, dtype_ids: torch.dtype) -> None:
    """与 OpenVLA `predict_action` 一致：在提示末尾补 Llama 空 token 29871；须同步加长 attention_mask。"""
    if "input_ids" not in inputs:
        return
    ids = inputs["input_ids"]
    if torch.all(ids[:, -1] == 29871):
        return
    extra = torch.tensor([[29871]], dtype=dtype_ids, device=device)
    inputs["input_ids"] = torch.cat([ids, extra], dim=1)
    if "attention_mask" in inputs and inputs["attention_mask"] is not None:
        am = inputs["attention_mask"]
        pad = torch.ones((am.shape[0], 1), dtype=am.dtype, device=am.device)
        inputs["attention_mask"] = torch.cat([am, pad], dim=1)


def infer_action_vector(
    model,
    processor,
    rgb_uint8_hwc: np.ndarray,
    prompt: str,
    device: torch.device,
    dtype: torch.dtype,
    action_dim: int,
    unnorm_key: str,
    ema_alpha: float,
    ema_prev: Optional[np.ndarray],
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """返回 (7 维 float32 动作，反归一化后多为训练统计的物理量级；夹爪等未必在 [-1,1])，更新后的 EMA 状态。"""
    rgb_u8 = ensure_rgb_uint8_hwc(rgb_uint8_hwc)
    rgb_pil = Image.fromarray(rgb_u8)
    inputs = processor(prompt, rgb_pil, return_tensors="pt")
    inputs = {k: v.to(device=device) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)
    maybe_append_empty_token(inputs, device=device, dtype_ids=inputs["input_ids"].dtype)
    prompt_len = int(inputs["input_ids"].shape[1])

    with torch.inference_mode():
        # 未 merge 的 PeftModel：必须走外层 .generate，否则会调到基座 predict_action 绕过 LoRA。
        use_generate_only = getattr(model, "peft_config", None) is not None
        if not use_generate_only and callable(getattr(model, "predict_action", None)):
            action_cmd = model.predict_action(
                unnorm_key=unnorm_key,
                do_sample=False,
                min_new_tokens=action_dim,
                **inputs,
            )
            action_cmd = np.asarray(action_cmd[:7], dtype=np.float32)
        else:
            gen_kw = dict(inputs)
            if getattr(model, "generation_config", None) is not None:
                gc = model.generation_config
                if getattr(gc, "pad_token_id", None) is None and getattr(model.config, "pad_token_id", None) is not None:
                    gen_kw.setdefault("pad_token_id", model.config.pad_token_id)
            generated_ids = model.generate(
                **gen_kw,
                max_new_tokens=action_dim,
                min_new_tokens=action_dim,
                do_sample=False,
            )
            action_norm = decode_normalized_action(
                model, generated_ids, action_dim, prompt_len=prompt_len
            )
            action_cmd = unnormalize_action_q01q99(model, action_norm, unnorm_key)
            action_cmd = np.asarray(action_cmd[:7], dtype=np.float32)
    # 不在此处 clip：反归一化后为物理空间，强行压到 [-1,1] 会破坏分布；限制交给 apply_cart_action_gain / 环境

    alpha = float(ema_alpha)
    if alpha > 0.0:
        if ema_prev is None:
            ema_next = action_cmd.copy()
        else:
            ema_next = alpha * action_cmd + (1.0 - alpha) * ema_prev
        return ema_next.copy(), ema_next
    return action_cmd, None
