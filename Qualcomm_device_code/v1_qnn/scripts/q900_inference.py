#!/usr/bin/env python3
"""
Q900 Edge Inference: OpenVLA VLA Pipeline
==========================================

完整推理 pipeline: 摄像头图像 → Vision ONNX → LLM GGUF → 7-DoF 动作

依赖:
    pip install llama-cpp-python onnxruntime opencv-python numpy

用法:
    # USB 摄像头模式
    python edge_optimization/scripts/q900_inference.py --camera_id 0

    # HDF5 数据验证模式
    python edge_optimization/scripts/q900_inference.py \
        --hdf5_path edge_optimization/data/episode_0006.hdf5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


def _resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(repo_root, path)


# ---------------------------------------------------------------------------
# Vision Encoder
# ---------------------------------------------------------------------------

class VisionEncoder:
    """ONNX Runtime 视觉编码器 (INT8 量化)"""

    def __init__(self, onnx_path: str, use_fused: bool = True):
        import onnxruntime as ort
        self.use_fused = use_fused
        providers = ["CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(onnx_path, opts, providers=providers)
        print(f"  Vision encoder loaded: {onnx_path}")

    def encode(self, rgb_uint8: np.ndarray) -> np.ndarray:
        """将 RGB uint8 图像编码为 vision embedding

        Args:
            rgb_uint8: (224, 224, 3) uint8

        Returns:
            vision_emb: (256, 4096) float32 — 256 个 vision tokens
        """
        # 归一化到 [0, 1], HWC → CHW
        x = rgb_uint8.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))  # (3, 224, 224)

        if self.use_fused:
            # fused backbone: DINOv2 + SigLIP 各接收相同图像, 拼接为 6 通道
            x = np.concatenate([x, x], axis=0)  # (6, 224, 224)

        x = np.expand_dims(x, 0)  # (1, 6, 224, 224)
        embeddings = self.session.run(None, {"pixel_values": x})[0]  # (1, 256, 4096)
        return embeddings.squeeze(0)  # (256, 4096)


# ---------------------------------------------------------------------------
# Action Decoder
# ---------------------------------------------------------------------------

class ActionDecoder:
    """从 LLM 输出的 token IDs 解码为 7-DoF 动作"""

    def __init__(self, params_path: str):
        with open(params_path, "r") as f:
            params = json.load(f)

        self.vocab_size = params["vocab_size"]  # 32000
        self.n_action_bins = params["n_action_bins"]  # 256
        self.bin_centers = np.array(params["bin_centers"], dtype=np.float64)

        stats = params["dataset_statistics"]["dobot_pouring"]["action"]
        self.q01 = np.array(stats["q01"], dtype=np.float64)
        self.q99 = np.array(stats["q99"], dtype=np.float64)
        self.mask = np.array(stats["mask"], dtype=bool)
        self.action_dim = len(self.q01)  # 7
        print(f"  Action decoder loaded: {self.action_dim}D, {self.n_action_bins} bins")

    def decode(self, token_ids: list[int]) -> np.ndarray:
        """token IDs → 7-DoF 动作 (物理单位)"""
        # token_id → bin index
        indices = np.array([
            np.clip(self.vocab_size - tid - 1, 0, self.n_action_bins - 1)
            for tid in token_ids
        ])
        # bin index → normalized action [-1, 1]
        normalized = self.bin_centers[indices]
        # unnormalize
        real = np.where(
            self.mask,
            0.5 * (normalized + 1.0) * (self.q99 - self.q01) + self.q01,
            normalized,
        )
        return real.astype(np.float32)


# ---------------------------------------------------------------------------
# LLM with Vision Embedding Injection
# ---------------------------------------------------------------------------

class VLAInference:
    """OpenVLA 推理: Vision Embedding + LLM (llama-cpp-python)"""

    SYSTEM_PROMPT = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    )

    def __init__(self, gguf_path: str, n_ctx: int = 2048, n_threads: int = 4):
        from llama_cpp import Llama

        self.llm = Llama(
            model_path=gguf_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
            embedding=True,
        )
        self.n_ctx = n_ctx
        print(f"  LLM loaded: {gguf_path} (n_ctx={n_ctx}, n_threads={n_threads})")

    def _build_prompt(self, task: str) -> str:
        """构造 Vicuna v1.5 chat prompt"""
        instruction = task.lower().strip()
        human_msg = f"<image>\nWhat action should the robot take to {instruction}?"
        return (
            f"{self.SYSTEM_PROMPT} "
            f"USER: {human_msg} ASSISTANT:"
        )

    def _inject_vision_embedding(self, vision_emb: np.ndarray):
        """将 vision embedding 注入到 LLM 的 embedding table 中

        策略: 将 unused token IDs (32001..32256) 的 embedding 替换为 vision tokens,
        然后在 prompt 中用这些 token 替换 <image>.
        """
        import ctypes

        # 获取 embedding table 指针
        model = self.llm._model
        embd_ptr = model.llama_model_n_embd(model.model)  # 4096
        n_embd = model.llama_model_n_embd(model.model)

        # 获取 embedding table 的原始指针
        # llama-cpp-python 通过 internal state 访问
        # 我们需要直接修改 embedding 表

        # 使用 token ID 32001-32256 (unused) 来存储 vision embeddings
        # 每个 token 对应一个 vision token (共 256 个)
        self.vision_token_ids = list(range(32001, 32001 + 256))

        # 通过 llama-cpp-python 的内部方法修改 embedding
        for i, token_id in enumerate(self.vision_token_ids):
            emb_vector = vision_emb[i].astype(np.float32)
            # 使用 set_token_embedding 如果可用
            if hasattr(model, 'set_token_embedding'):
                model.set_token_embedding(token_id, emb_vector)
            else:
                # 直接写入 embedding table 内存
                self._write_embedding(model, token_id, emb_vector, n_embd)

    def _write_embedding(self, model, token_id: int, emb: np.ndarray, n_embd: int):
        """直接写入 embedding table 内存"""
        import ctypes

        # 获取 embedding table 指针
        # llama_model_get_embeddings 返回 float* 指针
        get_emb = model.llama_model_get_embeddings
        get_emb.restype = ctypes.POINTER(ctypes.c_float)
        embeddings_ptr = get_emb(model.model)

        if not embeddings_ptr:
            raise RuntimeError("Failed to get embedding table pointer")

        # 写入 vision embedding
        offset = token_id * n_embd
        emb_ctypes = emb.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ctypes.memmove(
            ctypes.byref(embeddings_ptr, offset * ctypes.sizeof(ctypes.c_float)),
            emb_ctypes,
            n_embd * ctypes.sizeof(ctypes.c_float),
        )

    def infer_action(self, rgb_uint8: np.ndarray, task: str,
                     vision_encoder: VisionEncoder, action_decoder: ActionDecoder) -> np.ndarray:
        """完整推理: 图像 → 动作"""
        # 1. Vision encoding
        vision_emb = vision_encoder.encode(rgb_uint8)  # (256, 4096)

        # 2. 注入 vision embedding
        self._inject_vision_embedding(vision_emb)

        # 3. 构造 prompt, 用 vision token IDs 替换 <image>
        prompt = self._build_prompt(task)
        # 将 <image> 替换为 vision token 字符串
        vision_token_str = "".join([f"<unused{i}>" for i in range(32001, 32257)])
        prompt_with_vision = prompt.replace("<image>", vision_token_str)

        # 4. LLM 生成 7 个 action tokens
        output = self.llm.create_completion(
            prompt=prompt_with_vision,
            max_tokens=action_decoder.action_dim,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            repeat_penalty=1.0,
            echo=False,
        )

        # 5. 提取 token IDs
        # create_completion 返回文本, 我们需要 token IDs
        # 使用 tokenize 获取 prompt 长度, 然后从 completion 中提取
        completion_text = output["choices"][0]["text"]

        # 重新 tokenize 整个输出来获取 token IDs
        full_tokens = self.llm.tokenize(prompt_with_vision.encode("utf-8"))
        prompt_len = len(full_tokens)

        # 从 completion logprobs 获取 token IDs (如果可用)
        # 否则用 tokenize
        if "logprobs" in output["choices"][0] and output["choices"][0]["logprobs"]:
            token_ids = output["choices"][0]["logprobs"]["tokens"]
            # 将 token 文本转为 ID
            action_token_ids = []
            for tok_text in token_ids:
                tok_ids = self.llm.tokenize(tok_text.encode("utf-8"))
                if tok_ids:
                    action_token_ids.append(tok_ids[0])
        else:
            # fallback: tokenize completion text
            completion_tokens = self.llm.tokenize(completion_text.encode("utf-8"))
            action_token_ids = completion_tokens[:action_decoder.action_dim]

        # 6. 解码动作
        action = action_decoder.decode(action_token_ids)
        return action


# ---------------------------------------------------------------------------
# 替代方案: 使用 transformers 直接加载合并模型 (更简单但需要更多内存)
# ---------------------------------------------------------------------------

class VLAInferenceTransformers:
    """使用 transformers + merged model 的替代推理方案 (验证用)"""

    def __init__(self, merged_dir: str, device: str = "cpu"):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForVision2Seq.from_pretrained(
            merged_dir, torch_dtype=dtype, low_cpu_mem_usage=True,
            trust_remote_code=True, attn_implementation="eager",
        )
        self.model = self.model.to(device).eval()
        self.processor = AutoProcessor.from_pretrained(merged_dir, trust_remote_code=True)
        self.device = device
        self.dtype = dtype

        # 注入 dataset statistics
        stats_path = os.path.join(merged_dir, "dataset_statistics.json")
        if os.path.isfile(stats_path):
            import sys
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            sys.path.insert(0, repo_root)
            from tools.vla.openvla_lora_runtime import inject_dataset_statistics
            inject_dataset_statistics(self.model, stats_path)

        print(f"  Transformers model loaded: {merged_dir}")

    def infer_action(self, rgb_uint8: np.ndarray, task: str,
                     vision_encoder: None, action_decoder: ActionDecoder) -> np.ndarray:
        """使用 transformers 推理"""
        import torch
        from PIL import Image
        from tools.vla.openvla_lora_runtime import (
            build_vicuna_prompt, decode_normalized_action,
            ensure_rgb_uint8_hwc, maybe_append_empty_token,
            unnormalize_action_q01q99,
        )

        prompt = build_vicuna_prompt(task)
        rgb_u8 = ensure_rgb_uint8_hwc(rgb_uint8)
        rgb_pil = Image.fromarray(rgb_u8)
        inputs = self.processor(prompt, rgb_pil, return_tensors="pt")
        inputs = {k: v.to(device=self.device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.dtype)
        maybe_append_empty_token(inputs, device=torch.device(self.device), dtype_ids=inputs["input_ids"].dtype)
        prompt_len = int(inputs["input_ids"].shape[1])

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=action_decoder.action_dim,
                min_new_tokens=action_decoder.action_dim, do_sample=False,
            )

        action_norm = decode_normalized_action(
            self.model, generated_ids, action_decoder.action_dim, prompt_len=prompt_len,
        )
        action = unnormalize_action_q01q99(self.model, action_norm, "dobot_pouring")
        return np.asarray(action[:7], dtype=np.float32)


# ---------------------------------------------------------------------------
# 数据源
# ---------------------------------------------------------------------------

class CameraSource:
    """USB 摄像头"""

    def __init__(self, camera_id: int = 0):
        import cv2
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")
        print(f"  Camera opened: {camera_id}")

    def get_frame(self) -> np.ndarray:
        import cv2
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return cv2.resize(rgb, (224, 224))

    def release(self):
        self.cap.release()


class HDF5Source:
    """HDF5 数据集"""

    def __init__(self, hdf5_path: str):
        import h5py
        with h5py.File(hdf5_path, "r") as f:
            self.images = f["observations/images/rgb"][:]
        self.idx = 0
        print(f"  HDF5 loaded: {hdf5_path}, {len(self.images)} frames")

    def get_frame(self) -> np.ndarray:
        if self.idx >= len(self.images):
            self.idx = 0  # loop
        img = self.images[self.idx]
        self.idx += 1
        return img

    def release(self):
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Q900 Edge VLA Inference")
    p.add_argument("--gguf_path", type=str,
                   default="edge_optimization/gguf_models/openvla-llm-Q4_K_M.gguf")
    p.add_argument("--vision_onnx", type=str,
                   default="edge_optimization/components/vision_projector/vision_projector.onnx")
    p.add_argument("--action_params", type=str,
                   default="edge_optimization/components/action_head/action_head_params.json")
    p.add_argument("--merged_model", type=str,
                   default="edge_optimization/merged_model",
                   help="合并后模型目录 (transformers 方案)")
    p.add_argument("--camera_id", type=int, default=0)
    p.add_argument("--hdf5_path", type=str, default="",
                   help="HDF5 文件路径 (替代摄像头)")
    p.add_argument("--task", type=str, default="pour water from bottle into cup")
    p.add_argument("--n_threads", type=int, default=4)
    p.add_argument("--n_ctx", type=int, default=2048)
    p.add_argument("--hz", type=float, default=10.0,
                   help="目标控制频率 (Hz)")
    p.add_argument("--num_steps", type=int, default=0,
                   help="推理步数 (0=无限)")
    p.add_argument("--backend", type=str, default="transformers",
                   choices=["gguf", "transformers"],
                   help="推理后端: gguf (llama-cpp-python) 或 transformers (PyTorch)")
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Q900 Edge VLA Inference")
    print("=" * 60)

    # 加载组件
    print("\n[1/4] Loading action decoder ...")
    action_decoder = ActionDecoder(_resolve(args.action_params))

    print("\n[2/4] Loading vision encoder ...")
    vision_encoder = VisionEncoder(_resolve(args.vision_onnx))

    print(f"\n[3/4] Loading LLM ({args.backend}) ...")
    if args.backend == "gguf":
        vla = VLAInference(
            _resolve(args.gguf_path),
            n_ctx=args.n_ctx,
            n_threads=args.n_threads,
        )
    else:
        vla = VLAInferenceTransformers(
            _resolve(args.merged_model),
            device=args.device,
        )

    print("\n[4/4] Setting up data source ...")
    if args.hdf5_path:
        source = HDF5Source(_resolve(args.hdf5_path))
    else:
        source = CameraSource(args.camera_id)

    # 推理循环
    print("\n" + "=" * 60)
    print(f"Task: {args.task}")
    print(f"Target Hz: {args.hz}")
    print(f"Backend: {args.backend}")
    print("=" * 60)
    print("\nPress Ctrl+C to stop.\n")

    dt = 1.0 / args.hz
    step = 0
    latencies = []

    try:
        while True:
            if args.num_steps > 0 and step >= args.num_steps:
                break

            # 采集图像
            rgb = source.get_frame()

            # 推理
            t0 = time.time()
            action = vla.infer_action(rgb, args.task, vision_encoder, action_decoder)
            t1 = time.time()

            latency = t1 - t0
            latencies.append(latency)

            # 输出
            action_str = " ".join([f"{v:+.6f}" for v in action])
            print(f"[step {step:4d}] {action_str}  ({latency*1000:.0f}ms)")

            step += 1

            # 控制频率
            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    finally:
        source.release()

    # 统计
    if latencies:
        print(f"\n{'=' * 60}")
        print(f"Steps: {step}")
        print(f"Avg latency: {np.mean(latencies)*1000:.0f}ms")
        print(f"Min latency: {np.min(latencies)*1000:.0f}ms")
        print(f"Max latency: {np.max(latencies)*1000:.0f}ms")
        print(f"Avg Hz: {1.0/np.mean(latencies):.1f}")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
