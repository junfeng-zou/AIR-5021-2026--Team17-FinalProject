#!/usr/bin/env python3
"""
LoRA fine-tune OpenVLA-7B on pouring_VLA HDF5 episodes (image + language + 7-DoF actions).

Run from repo root (recommended):
  pip install -r LoRA_train/requirements.txt
  accelerate launch LoRA_train/train_lora.py --data_dir data/vla_dataset --output_dir runs/openvla_pouring_lora

Single GPU:
  python LoRA_train/train_lora.py --data_dir data/vla_dataset --output_dir runs/openvla_pouring_lora

`--vla_path` 可为已下载的本地目录（含 config.json），此时默认不向 Hub 发请求；
也可为 Hub 模型 id（需能联网或使用缓存）。远程代码需 `trust_remote_code`。
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

# Local imports (LoRA_train as script directory)
_LORA_DIR = Path(__file__).resolve().parent
if str(_LORA_DIR) not in sys.path:
    sys.path.insert(0, str(_LORA_DIR))

import torch

# 开启 A100/H100 专属的 TF32 硬件加速
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import tqdm
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from transformers import AutoProcessor, get_scheduler

from action_tokenizer import ActionTokenizer
from collator import PaddedCollatorForActionPrediction
from collator import IGNORE_INDEX
from dataset_hdf5 import PouringHDF5VLADataset
from prompting_vicuna import VicunaV15ChatPromptBuilder

# 传给 Hub / 缓存解析的参数（勿混入 torch_dtype、low_cpu_mem_usage 等到 AutoConfig）
_HUB_FROM_PRETRAINED_KEYS = frozenset(
    {
        "local_files_only",
        "cache_dir",
        "force_download",
        "revision",
        "proxies",
        "subfolder",
        "token",
        "resume_download",
    }
)


def _align_labels_to_logits_seq(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Prismatic/OpenVLA 多模态 forward 在 BOS 后插入视觉 patch，LM logits 序列比 input_ids/labels 更长；
    与 modeling_prismatic 一致：multimodal_labels = [labels[:,0], -100*patches, labels[:,1:]]。
    """
    lm_len, lab_len = logits.size(1), labels.size(1)
    if lm_len == lab_len:
        return labels
    if lm_len < lab_len:
        return labels  # 调用方会因长度不一致跳过
    n_patch = lm_len - lab_len
    patch = torch.full(
        (labels.size(0), n_patch),
        IGNORE_INDEX,
        dtype=labels.dtype,
        device=labels.device,
    )
    return torch.cat([labels[:, :1], patch, labels[:, 1:]], dim=1)


def supervised_token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """与因果 LM 的 CE 对齐：在 labels!=IGNORE_INDEX 的位置上算 argmax 准确率。"""
    labels_aligned = _align_labels_to_logits_seq(logits, labels)
    if labels_aligned.size(1) != logits.size(1):
        return float("nan")
    shift_logits = logits[:, :-1, :]
    shift_labels = labels_aligned[:, 1:]
    mask = shift_labels.ne(IGNORE_INDEX)
    if mask.sum().item() == 0:
        return float("nan")
    preds = shift_logits.argmax(dim=-1)
    correct = (preds.eq(shift_labels) & mask).sum().float()
    return (correct / mask.sum().float()).item()


def run_validation_pass(
    accelerator: Accelerator,
    vla: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    bf16: bool,
) -> float:
    """返回验证集样本平均 loss（跨进程加权平均）。"""
    vla.eval()
    local_ls, local_n = 0.0, 0
    with torch.no_grad():
        for batch in tqdm.tqdm(
            val_loader,
            desc="val",
            leave=False,
            disable=not accelerator.is_local_main_process,
        ):
            pv = batch["pixel_values"]
            if isinstance(pv, torch.Tensor):
                pv = pv.to(device=device, dtype=torch.bfloat16 if bf16 else torch.float32)
            elif isinstance(pv, dict):
                pv = {
                    k: v.to(device=device, dtype=torch.bfloat16 if bf16 else torch.float32)
                    for k, v in pv.items()
                }
            out = vla(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                pixel_values=pv,
                labels=batch["labels"].to(device),
            )
            bs = batch["input_ids"].size(0)
            local_ls += out.loss.detach().float().item() * bs
            local_n += bs
    t_sum = torch.tensor([local_ls], device=device, dtype=torch.float32)
    t_cnt = torch.tensor([float(local_n)], device=device, dtype=torch.float32)
    reduced_sum = accelerator.reduce(t_sum, reduction="sum")
    reduced_cnt = accelerator.reduce(t_cnt, reduction="sum")
    return (reduced_sum / reduced_cnt.clamp(min=1.0)).item()


def _is_local_model_directory(path_str: str) -> bool:
    p = Path(path_str).expanduser()
    p = p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()
    return p.is_dir() and (p / "config.json").is_file()


def _load_episode_weights_csv(path_str: str) -> dict[str, float]:
    """
    CSV format:
      episode,weight
      episode_0001.hdf5,1.0
      episode_0103.hdf5,0.2
    """
    path = Path(path_str).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"episode weight csv not found: {path}")

    weights: dict[str, float] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"empty csv: {path}")
        for row in reader:
            ep = (
                row.get("episode")
                or row.get("episode_file")
                or row.get("file")
                or row.get("name")
                or ""
            ).strip()
            if not ep:
                continue
            w_raw = row.get("weight", "").strip()
            if not w_raw:
                continue
            w = max(1e-8, float(w_raw))
            weights[ep] = w
    return weights


def load_openvla_model(pretrained_model_name_or_path: str, *, trust_remote_code: bool = True, **kwargs):
    """Load OpenVLA weights. transformers 5.x removed `AutoModelForVision2Seq`; hub `auto_map` still uses that key."""
    hub_kw = {k: v for k, v in kwargs.items() if k in _HUB_FROM_PRETRAINED_KEYS}
    try:
        from transformers import AutoModelForVision2Seq

        return AutoModelForVision2Seq.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
        )
    except ImportError:
        from transformers import AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        from transformers.models.auto.auto_factory import add_generation_mixin_to_remote_model

        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            **hub_kw,
        )
        auto_map = getattr(config, "auto_map", None) or {}
        class_ref = auto_map.get("AutoModelForVision2Seq")
        if not class_ref:
            raise ImportError(
                "当前 transformers 版本没有 AutoModelForVision2Seq，且 checkpoint 的 config.auto_map 中缺少对应项。"
            ) from None

        model_cls = get_class_from_dynamic_module(
            class_ref, pretrained_model_name_or_path, **hub_kw
        )
        model_cls = add_generation_mixin_to_remote_model(model_cls)
        return model_cls.from_pretrained(
            pretrained_model_name_or_path,
            config=config,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )


# LoRA on Llama blocks only (typical VLA finetune: freeze vision, adapt LLM).
LLM_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenVLA LoRA fine-tuning on HDF5 pouring demos")
    p.add_argument("--data_dir", type=str, default="data/vla_dataset", help="Folder with episode_*.hdf5")
    p.add_argument(
        "--action_source",
        type=str,
        default="actions",
        choices=("actions", "actions_raw"),
        help="训练使用的动作字段：actions 或 actions_raw",
    )
    p.add_argument(
        "--vla_path",
        type=str,
        default="openvla/openvla-7b",
        help="本地权重目录（含 config.json）或 Hub 模型 id；本地目录会自动 local_files_only",
    )
    p.add_argument(
        "--local_files_only",
        action="store_true",
        help="不向 Hugging Face Hub 发起下载/校验请求，仅使用本地目录或已有缓存",
    )
    p.add_argument("--output_dir", type=str, required=True, help="Checkpoints + processor + dataset stats")
    p.add_argument("--val_ratio", type=float, default=0.05, help="Fraction of transitions for validation")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--max_steps", type=int, default=20_000)
    p.add_argument("--save_steps", type=int, default=2000)
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--num_workers", type=int, default=0, help="Keep 0 for HDF5 on many systems")
    p.add_argument(
        "--episode_weight_csv",
        type=str,
        default="",
        help="可选：episode 权重表 CSV（列: episode,weight），用于加权采样降权低质量数据",
    )
    p.add_argument(
        "--weighted_num_samples",
        type=int,
        default=0,
        help="加权采样每个 epoch 抽样数；0 表示等于训练子集样本数",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb_project", type=str, default="", help="Empty = disable W&B")
    p.add_argument("--bf16", action="store_true", help="Use bfloat16 (recommended on Ampere+)")
    p.add_argument(
        "--lora_target",
        type=str,
        default="llm",
        choices=("llm", "all-linear"),
        help="llm: only Llama linear proj (freeze vision); all-linear: match upstream OpenVLA finetune (heavier)",
    )
    p.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Every N optimizer steps print train_loss to stdout on rank0 (0 = disable). W&B unchanged.",
    )
    p.add_argument(
        "--eval_steps",
        type=int,
        default=0,
        help="每 N 个 optimizer step 在验证集上算一次 loss 并写入 W&B；0=仅训练结束后验证",
    )
    p.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.0,
        help="梯度裁剪阈值；0 表示不裁剪。loss 尖峰时可试 0.5~1.0",
    )
    p.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=("constant", "cosine", "cosine_with_min_lr"),
        help="constant=固定 lr；cosine 与 cosine_with_min_lr 相同：warmup 后半余弦衰减至 --min_lr",
    )
    p.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="lr_scheduler=cosine 时的 warmup 步数（optimizer step，非 micro-batch）",
    )
    p.add_argument(
        "--min_lr",
        type=float,
        default=1e-5,
        help="lr_scheduler=cosine 时余弦衰减到的最小学习率（须小于 --learning_rate）",
    )
    p.add_argument(
        "--log_grad_norm",
        action="store_true",
        help="将梯度 L2 范数写入 W&B（不裁剪时也会计算，开销很小）",
    )
    p.add_argument(
        "--log_throughput",
        action="store_true",
        help="记录近似吞吐（样本/秒，含梯度累积与多卡）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accumulation_steps,
        mixed_precision="bf16" if args.bf16 else "no",
    )
    device = accelerator.device

    if not torch.cuda.is_available() and device.type != "cuda":
        accelerator.print("[WARN] No CUDA: training will be very slow.")

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)

    hub_kw: dict = {}
    if args.local_files_only or _is_local_model_directory(args.vla_path):
        hub_kw["local_files_only"] = True
        if accelerator.is_main_process:
            accelerator.print(
                "[INFO] local_files_only=True（本地权重或 --local_files_only，跳过 Hub 网络请求）"
            )

    processor = AutoProcessor.from_pretrained(
        args.vla_path, trust_remote_code=True, **hub_kw
    )
    vla = load_openvla_model(
        args.vla_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        **hub_kw,
    )

    target_modules: str | list[str] = (
        LLM_LORA_TARGET_MODULES if args.lora_target == "llm" else "all-linear"
    )
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=min(args.lora_rank, 16),
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        init_lora_weights="gaussian",
    )
    vla = get_peft_model(vla, lora_config)
    if accelerator.is_main_process:
        vla.print_trainable_parameters()

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    image_transform = processor.image_processor.apply_transform

    episode_weights: dict[str, float] | None = None
    if args.episode_weight_csv:
        episode_weights = _load_episode_weights_csv(args.episode_weight_csv)
        accelerator.print(
            f"[INFO] loaded episode weights: {len(episode_weights)} entries from {args.episode_weight_csv}"
        )

    full_ds = PouringHDF5VLADataset(
        str(data_dir),
        action_tokenizer,
        processor.tokenizer,
        image_transform,
        prompt_builder_fn=VicunaV15ChatPromptBuilder,
        action_source=args.action_source,
        episode_weights=episode_weights,
    )

    if accelerator.is_main_process:
        full_ds.save_dataset_statistics(str(out_dir))
        processor.save_pretrained(str(out_dir))

    n_total = len(full_ds)
    n_val = max(1, int(n_total * args.val_ratio)) if n_total > 10 else 1
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )

    train_sampler = None
    train_shuffle = True
    if args.episode_weight_csv:
        # random_split 返回 Subset，indices 指向 full_ds 的扁平样本索引
        train_indices = getattr(train_ds, "indices", None)
        if train_indices is None:
            raise RuntimeError("weighted sampling expects train_ds to be torch.utils.data.Subset")
        sample_weights = torch.as_tensor(
            [full_ds.index_weight(int(i)) for i in train_indices], dtype=torch.double
        )
        n_samples = int(args.weighted_num_samples) if args.weighted_num_samples > 0 else int(
            sample_weights.numel()
        )
        train_sampler = WeightedRandomSampler(sample_weights, num_samples=n_samples, replacement=True)
        train_shuffle = False
        accelerator.print(
            f"[INFO] weighted sampler enabled: n_train={sample_weights.numel()} num_samples={n_samples}"
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    optimizer = AdamW((p for p in vla.parameters() if p.requires_grad), lr=args.learning_rate)

    lr_scheduler = None
    if args.lr_scheduler in ("cosine", "cosine_with_min_lr"):
        if args.min_lr >= args.learning_rate:
            raise ValueError(
                f"--min_lr ({args.min_lr}) 必须小于 --learning_rate ({args.learning_rate})"
            )
        lr_scheduler = get_scheduler(
            name="cosine_with_min_lr",
            optimizer=optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.max_steps,
            scheduler_specific_kwargs={"min_lr": args.min_lr},
        )

    if lr_scheduler is not None:
        vla, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
            vla, optimizer, train_loader, val_loader, lr_scheduler
        )
    else:
        vla, optimizer, train_loader, val_loader = accelerator.prepare(
            vla, optimizer, train_loader, val_loader
        )

    wandb_run = None
    if args.wandb_project and accelerator.is_main_process:
        import wandb

        cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        cfg["effective_batch_size"] = (
            args.batch_size * args.grad_accumulation_steps * accelerator.num_processes
        )
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=Path(args.output_dir).name,
            config=cfg,
        )

    global_step = 0
    train_iter = iter(train_loader)
    progress = tqdm.tqdm(total=args.max_steps, disable=not accelerator.is_main_process, desc="train")
    step_timer = time.perf_counter()

    vla.train()
    while global_step < args.max_steps:
        grad_norm_val: float | None = None
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        with accelerator.accumulate(vla):
            pixel_values = batch["pixel_values"]
            if isinstance(pixel_values, torch.Tensor):
                pixel_values = pixel_values.to(
                    device=device, dtype=torch.bfloat16 if args.bf16 else torch.float32
                )
            elif isinstance(pixel_values, dict):
                pixel_values = {
                    k: v.to(device=device, dtype=torch.bfloat16 if args.bf16 else torch.float32)
                    for k, v in pixel_values.items()
                }

            out = vla(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                pixel_values=pixel_values,
                labels=batch["labels"].to(device),
            )
            loss = out.loss
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                if args.max_grad_norm and args.max_grad_norm > 0:
                    grad_norm_val = float(
                        accelerator.clip_grad_norm_(vla.parameters(), args.max_grad_norm)
                    )
                elif args.log_grad_norm:
                    grad_norm_val = float(
                        accelerator.clip_grad_norm_(vla.parameters(), float("inf"))
                    )
            optimizer.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            if lr_scheduler is not None:
                lr_scheduler.step()
            global_step += 1
            progress.update(1)
            loss_item = loss.detach().float().item()
            tok_acc = supervised_token_accuracy(out.logits, batch["labels"].to(device))
            lr = optimizer.param_groups[0]["lr"]

            elapsed = time.perf_counter() - step_timer
            step_timer = time.perf_counter()
            samples_per_step = (
                args.batch_size * accelerator.num_processes * args.grad_accumulation_steps
            )

            wb_metrics: dict = {
                "train/loss": loss_item,
                "train/learning_rate": lr,
                "train/action_token_accuracy": tok_acc,
            }
            if grad_norm_val is not None:
                wb_metrics["train/grad_norm"] = grad_norm_val
            if args.log_throughput and elapsed > 0:
                wb_metrics["train/samples_per_sec"] = samples_per_step / elapsed

            if wandb_run is not None:
                wb_metrics["train_loss"] = loss_item
                wandb_run.log(wb_metrics, step=global_step)

            if (
                args.log_interval > 0
                and accelerator.is_main_process
                and global_step % args.log_interval == 0
            ):
                extra = f"  lr={lr:.2e}"
                if grad_norm_val is not None:
                    extra += f"  grad_norm={grad_norm_val:.3f}"
                if not (tok_acc != tok_acc):
                    extra += f"  tok_acc={tok_acc:.4f}"
                if args.log_throughput and elapsed > 0:
                    extra += f"  ~{samples_per_step / elapsed:.1f} samples/s"
                accelerator.print(f"[TRAIN] step={global_step}  loss={loss_item:.4f}{extra}")

            if args.eval_steps > 0 and global_step % args.eval_steps == 0 and len(val_ds) > 0:
                eval_loss = run_validation_pass(accelerator, vla, val_loader, device, args.bf16)
                vla.train()
                if accelerator.is_main_process:
                    accelerator.print(f"[EVAL] step={global_step}  val_loss={eval_loss:.4f}")
                if wandb_run is not None:
                    wandb_run.log({"eval/loss": eval_loss}, step=global_step)

            if global_step % args.save_steps == 0 and accelerator.is_main_process:
                save_path = out_dir / f"checkpoint_step_{global_step}"
                save_path.mkdir(parents=True, exist_ok=True)
                unwrapped = accelerator.unwrap_model(vla)
                unwrapped.save_pretrained(save_path)
                processor.save_pretrained(save_path)
                accelerator.print(f"[SAVE] {save_path}")

        if global_step >= args.max_steps:
            break

    progress.close()

    need_final_val = len(val_ds) > 0 and (
        args.eval_steps <= 0 or (global_step % args.eval_steps != 0)
    )
    if need_final_val:
        mean_loss = run_validation_pass(accelerator, vla, val_loader, device, args.bf16)
        if accelerator.is_main_process:
            accelerator.print(f"[VAL] mean loss = {mean_loss:.4f}")
            if wandb_run is not None:
                wandb_run.log({"val_mean_loss": mean_loss, "eval/loss": mean_loss}, step=global_step)

    if accelerator.is_main_process:
        final_dir = out_dir / "final_lora"
        final_dir.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(vla).save_pretrained(final_dir)
        processor.save_pretrained(final_dir)
        accelerator.print(f"[DONE] LoRA adapter saved to {final_dir}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
