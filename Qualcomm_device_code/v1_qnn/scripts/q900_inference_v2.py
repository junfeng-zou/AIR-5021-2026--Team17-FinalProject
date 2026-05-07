#!/usr/bin/env python3
"""
Q900 Edge Inference V2.0 — OpenVLA VLA Pipeline
=================================================

架构: ONNX Runtime (视觉) + llama-cpp-python (LLM) + 动作解码

与 q900_inference.py 的核心区别:
  - 修复了图像预处理（DINOv2+SigLIP 双骨干正确归一化）
  - 使用 llama-cpp-python 的低层 C API 注入视觉 embedding，
    替代不稳定的 ctypes memmove 方案
  - 支持三阶段任务 Prompt（与训练一致）

依赖 (在 Q900 上安装):
    pip install llama-cpp-python onnxruntime opencv-python numpy

用法:
    python q900_inference_v2.py \\
        --gguf_path /opt/openvla/llm.gguf \\
        --vision_onnx /opt/openvla/vision_projector.onnx \\
        --action_params /opt/openvla/action_head_params.json \\
        --camera_id 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── 禁用 Vulkan 后端（必须在 import llama_cpp 之前）──────────
# llama-cpp-python 0.3.x 若编译时启用 Vulkan，加载时会自动初始化 Vulkan，
# 在不兼容的驱动上会触发 vk::IncompatibleDriverError → SIGABRT
os.environ.setdefault("GGML_VK_DISABLE", "1")

import cv2
import numpy as np


# ── 图像预处理（必须与训练时一致）──────────────────────────────
# OpenVLA fused backbone:
#   通道 0-2: DINOv2 (ImageNet normalization)
#   通道 3-5: SigLIP (mean=0.5, std=0.5)
_DINO_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_DINO_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_SIG_MEAN  = np.array([0.5, 0.5, 0.5],       dtype=np.float32)
_SIG_STD   = np.array([0.5, 0.5, 0.5],       dtype=np.float32)


def preprocess(rgb_uint8: np.ndarray) -> np.ndarray:
    """224x224 RGB uint8 → (1, 6, 224, 224) float32，供 ONNX 使用"""
    x = rgb_uint8.astype(np.float32) / 255.0  # (224,224,3)
    x = np.transpose(x, (2, 0, 1))            # (3,224,224)
    dino = (x - _DINO_MEAN[:, None, None]) / _DINO_STD[:, None, None]
    sglp = (x - _SIG_MEAN[:, None, None])  / _SIG_STD[:, None, None]
    fused = np.concatenate([dino, sglp], axis=0)[np.newaxis]  # (1,6,224,224)
    return fused.astype(np.float32)


# ── ONNX 视觉编码器 ────────────────────────────────────────────
class VisionEncoder:
    """ONNX Runtime 视觉编码器（双骨干 DINOv2+SigLIP → Projector）"""

    def __init__(self, onnx_path: str):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # 优先用 Vulkan（Adreno GPU），回退 CPU
        providers = []
        if "VulkanExecutionProvider" in ort.get_all_providers():
            providers.append("VulkanExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(onnx_path, opts, providers=providers)
        ep = self.session.get_providers()[0]
        print(f"  Vision encoder: {Path(onnx_path).name}  [{ep}]")

    def encode(self, rgb_uint8: np.ndarray) -> np.ndarray:
        """输入: (224,224,3) uint8 → 输出: (256,4096) float32"""
        x = preprocess(rgb_uint8)
        out = self.session.run(None, {"pixel_values": x})[0]  # (1,256,4096)
        return out.squeeze(0)                                   # (256,4096)


# ── 动作解码器 ─────────────────────────────────────────────────
class ActionDecoder:
    """token IDs → 7-DoF 物理动作"""

    def __init__(self, params_path: str, unnorm_key: str = "dobot_pouring"):
        with open(params_path) as f:
            params = json.load(f)

        self.vocab_size    = params["vocab_size"]      # 32000
        self.n_bins        = params["n_action_bins"]   # 256
        self.bin_centers   = np.array(params["bin_centers"], dtype=np.float64)  # (256,)

        stats = params["dataset_statistics"][unnorm_key]["action"]
        self.q01  = np.array(stats["q01"],  dtype=np.float64)  # (7,)
        self.q99  = np.array(stats["q99"],  dtype=np.float64)  # (7,)
        self.mask = np.array(stats["mask"], dtype=bool)         # (7,)
        self.action_dim = len(self.q01)

        print(f"  Action decoder: {self.action_dim}D  "
              f"token_range=[{self.vocab_size-self.n_bins}, {self.vocab_size-1}]")

    def decode(self, token_ids: list[int]) -> np.ndarray:
        """
        token_id → bin_index → normalized_value → physical_value

        OpenVLA 编码: bin_index = vocab_size - token_id - 1
        """
        indices = np.array(
            [int(np.clip(self.vocab_size - tid - 1, 0, self.n_bins - 1)) for tid in token_ids],
            dtype=np.int32,
        )
        normalized = self.bin_centers[indices]  # [-1, 1]
        physical = np.where(
            self.mask,
            0.5 * (normalized + 1.0) * (self.q99 - self.q01) + self.q01,
            normalized,
        )
        return physical.astype(np.float32)


# ── LLM 推理（视觉 embedding 注入）────────────────────────────
VICUNA_SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

# 图像占位符 token（OpenVLA 训练时使用的 special token）
IMAGE_TOKEN = "<image>"


class VLAEngine:
    """OpenVLA 推理引擎：ONNX 视觉 + llama-cpp-python LLM"""

    def __init__(
        self,
        gguf_path: str,
        vision_encoder: VisionEncoder,
        action_decoder: ActionDecoder,
        n_ctx: int = 512,
        n_threads: int = 4,
        n_gpu_layers: int = 0,
        use_mmap: bool = False,
    ):
        from llama_cpp import Llama
        import llama_cpp as lc_lib

        self.vision_encoder = vision_encoder
        self.action_decoder = action_decoder
        self.lc_lib = lc_lib

        # 打印诊断信息
        gguf_size_mb = os.path.getsize(gguf_path) / (1024 * 1024)
        print(f"  GGUF 文件大小: {gguf_size_mb:.0f} MB")
        print(f"  llama-cpp-python 版本: {lc_lib.__version__ if hasattr(lc_lib, '__version__') else 'unknown'}")

        try:
            self.llm = Llama(
                model_path=gguf_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                use_mmap=use_mmap,     # 边缘设备建议关闭 mmap
                verbose=True,          # 开启 llama.cpp 原生日志以便排错
                # 注意: logits_all 在新版 llama-cpp-python 已移除，不再传入
            )
        except Exception as e:
            print(f"\n  [ERROR] LLM 加载失败: {e}")
            print(f"  提示: 请检查 llama-cpp-python 版本与 GGUF 文件兼容性")
            print(f"  运行: pip show llama-cpp-python")
            raise SystemExit(1)

        self.n_embd = self.llm.n_embd()
        self._verbose_timing = True  # 开启延迟分解打印
        print(f"  LLM: {Path(gguf_path).name}  n_embd={self.n_embd}  ngl={n_gpu_layers}")

    def _build_prompt(self, task: str) -> str:
        return (
            f"{VICUNA_SYSTEM} "
            f"USER: {IMAGE_TOKEN}\n"
            f"What action should the robot take to {task.lower().strip()}? "
            f"ASSISTANT:"
        )

    def infer(self, rgb_uint8: np.ndarray, task: str) -> np.ndarray:
        """
        主推理入口: 图像 + 任务描述 → 7-DoF 动作

        推理流程:
          1. ONNX 编码图像 → vision_emb (256, 4096)
          2. 分割 prompt 为 [text_before_image] 和 [text_after_image]
          3. KV-prefill: text_before → 视觉 embedding(单次批量) → text_after
          4. Greedy decode: 生成 7 个 action token
          5. detokenize → 物理动作
        """
        import time
        t0 = time.time()

        # Step 1: 视觉编码
        vision_emb = self.vision_encoder.encode(rgb_uint8)  # (256, 4096)
        t1 = time.time()

        # Step 2: 分割 prompt
        prompt = self._build_prompt(task)
        img_pos = prompt.find(IMAGE_TOKEN)
        text_before = prompt[:img_pos]
        text_after  = prompt[img_pos + len(IMAGE_TOKEN):]

        tokens_before = self.llm.tokenize(text_before.encode(), add_bos=True)
        tokens_after  = self.llm.tokenize(text_after.encode(),  add_bos=False)

        # Step 3: KV-prefill
        # 每次 infer 前必须清除 KV-cache，否则上下文会持续累积导致动作相同
        self.llm.reset()
        self.llm.eval(tokens_before)

        # 视觉 embedding 批量注入（256 tokens 一次 decode，不是 256 次）
        self._inject_vision_embeddings(vision_emb, start_pos=len(tokens_before))
        self.llm.eval(tokens_after)
        t2 = time.time()

        # Step 4: Greedy decode，精确生成 action_dim 个 token
        action_tokens = []
        for _ in range(self.action_decoder.action_dim):
            token_id = self._greedy_sample()
            action_tokens.append(token_id)
            self.llm.eval([token_id])
        t3 = time.time()

        # 打印延迟分解（调试用）
        if hasattr(self, '_verbose_timing') and self._verbose_timing:
            print(f"    [timing] vision={1000*(t1-t0):.0f}ms  prefill={1000*(t2-t1):.0f}ms  decode={1000*(t3-t2):.0f}ms")

        # Step 5: 解码为物理动作
        return self.action_decoder.decode(action_tokens)

    def _inject_vision_embeddings(self, vision_emb: np.ndarray, start_pos: int):
        """
        将 (n_vis, n_embd) 的视觉 embedding 一次性批量注入 KV-cache。

        关键优化：256 个视觉 token 打包为单个 llama_batch 一次 decode，
        而非 256 次单独调用（原方案延迟约 20s → 批量后约 1-2s）。
        """
        import ctypes
        lc = self.lc_lib
        n_vis  = vision_emb.shape[0]  # 256
        n_embd = vision_emb.shape[1]  # 4096

        # 展平为连续 float32 内存 (n_vis * n_embd,)
        embd_flat = np.ascontiguousarray(vision_emb, dtype=np.float32)

        # 单次 batch，包含全部 256 个视觉 token
        # llama_batch_init(n_tokens, n_embd, n_seq_max):
        #   n_embd > 0 → 分配 embd 数组；token 数组为 NULL（两者互斥）
        batch = lc.llama_batch_init(n_vis, n_embd, 1)
        try:
            batch.n_tokens = n_vis

            # 一次写入所有 embedding（n_vis * n_embd 个 float32）
            ctypes.memmove(
                batch.embd,
                embd_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                n_vis * n_embd * ctypes.sizeof(ctypes.c_float),
            )

            # 设置每个 token 的位置和 seq_id
            for i in range(n_vis):
                batch.pos[i]      = start_pos + i
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = 0
                batch.logits[i]   = False  # 只有最后一个 decode token 需要 logits

            ret = lc.llama_decode(self.llm._ctx.ctx, batch)
            if ret != 0:
                raise RuntimeError(f"llama_decode (vision batch n={n_vis}) failed: ret={ret}")
        finally:
            lc.llama_batch_free(batch)

    def _greedy_sample(self) -> int:
        """temperature=0 贪心采样（机器人控制必须确定性）"""
        import ctypes
        lc = self.lc_lib
        n_vocab = self.llm.n_vocab()
        # 使用 llama_get_logits_ith 获取最后一个 token 的 logits
        # 这在 logits_all=False（默认）时也能正确工作
        n_tokens = self.llm.n_tokens
        logits_ptr = lc.llama_get_logits_ith(self.llm._ctx.ctx, n_tokens - 1)
        if not logits_ptr:
            # 回退到旧 API
            logits_ptr = lc.llama_get_logits(self.llm._ctx.ctx)
        logits = np.ctypeslib.as_array(logits_ptr, shape=(n_vocab,)).copy()
        return int(np.argmax(logits))


# ── 主流程 ─────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Q900 OpenVLA V2.0 推理")
    p.add_argument("--gguf_path",     default="/opt/openvla/llm.gguf")
    p.add_argument("--vision_onnx",   default="/opt/openvla/vision_projector.onnx")
    p.add_argument("--action_params", default="/opt/openvla/action_head_params.json")
    p.add_argument("--unnorm_key",    default="dobot_pouring")
    p.add_argument("--camera_id",     type=int, default=0)
    p.add_argument("--task",          default="pour cola into cup")
    p.add_argument("--hz",            type=float, default=2.0, help="目标控制频率")
    p.add_argument("--n_gpu_layers",  type=int, default=0,
                   help="LLM 卸载到 QNN HTP 的层数（0=纯CPU，建议逐步增加）")
    p.add_argument("--n_threads",     type=int, default=8)  # Q900 有 8 个 Kryo 核
    p.add_argument("--use_mmap",      action="store_true",
                   help="启用 mmap 加载模型（默认关闭，边缘设备建议关闭）")
    p.add_argument("--n_ctx",         type=int, default=512)
    p.add_argument("--dry_run",       action="store_true", help="不连接机器人，仅打印动作")
    p.add_argument("--num_steps",     type=int, default=0, help="0=无限")
    p.add_argument("--hdf5_path",     default="", help="用 HDF5 替代摄像头（验证用）")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Q900 OpenVLA V2.0 边缘推理")
    print("=" * 60)

    # 加载组件
    print("\n[1/3] 初始化视觉编码器 ...")
    vision_enc = VisionEncoder(args.vision_onnx)

    print("\n[2/3] 初始化动作解码器 ...")
    action_dec = ActionDecoder(args.action_params, args.unnorm_key)

    print("\n[3/3] 加载 LLM ...")
    engine = VLAEngine(
        gguf_path=args.gguf_path,
        vision_encoder=vision_enc,
        action_decoder=action_dec,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_gpu_layers=args.n_gpu_layers,
        use_mmap=args.use_mmap,
    )

    # 数据源
    if args.hdf5_path:
        import h5py
        with h5py.File(args.hdf5_path, "r") as f:
            images = f["observations/images/rgb"][:]
        img_idx = [0]
        def get_frame():
            img = images[img_idx[0] % len(images)]
            img_idx[0] += 1
            return cv2.resize(img, (224, 224))
        def release(): pass
    else:
        cap = cv2.VideoCapture(args.camera_id)
        def get_frame():
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Camera read failed")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return cv2.resize(rgb, (224, 224))
        def release(): cap.release()

    print(f"\n{'=' * 60}")
    print(f"任务: {args.task}")
    print(f"目标频率: {args.hz} Hz  |  GPU 层: {args.n_gpu_layers}  |  dry_run: {args.dry_run}")
    print("Ctrl+C 退出\n")

    dt = 1.0 / args.hz
    step = 0
    latencies = []

    try:
        while args.num_steps == 0 or step < args.num_steps:
            t0 = time.time()

            rgb = get_frame()

            action = engine.infer(rgb, args.task)
            latency = time.time() - t0
            latencies.append(latency)

            # 格式化输出（与 real_openvla_lora_infer.py 日志风格对齐）
            a = action
            print(
                f"[{step:4d}] "
                f"dxyz=[{a[0]*1000:+.1f},{a[1]*1000:+.1f},{a[2]*1000:+.1f}]mm  "
                f"drpy=[{np.rad2deg(a[3]):+.1f},{np.rad2deg(a[4]):+.1f},{np.rad2deg(a[5]):+.1f}]°  "
                f"g={a[6]:+.2f}  "
                f"({latency*1000:.0f}ms)"
            )

            step += 1
            sleep_t = dt - (time.time() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n用户中断。")
    finally:
        release()

    if latencies:
        print(f"\n{'=' * 60}")
        print(f"总步数: {step}")
        print(f"平均延迟: {np.mean(latencies)*1000:.0f} ms")
        print(f"最大延迟: {np.max(latencies)*1000:.0f} ms")
        print(f"实际频率: {1.0/np.mean(latencies):.2f} Hz")
        print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
