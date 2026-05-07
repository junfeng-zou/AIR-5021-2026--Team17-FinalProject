#!/usr/bin/env python3
"""
Q900 NPU Inference: OpenVLA via Qualcomm QNN SDK
=================================================

在 Q900 (Hexagon NPU) 上运行 OpenVLA 推理:
  摄像头图像 → Vision (NPU/CPU) → LLM (NPU) → 7-DoF 动作

依赖 (Q900 设备上):
    pip install qairt opencv-python numpy

用法:
    # Step 1: 在服务器上先用 qairt API 编译模型 (会生成 QNN context binary)
    python q900_npu_inference.py --compile_only \
        --gguf_path edge_optimization/gguf_models/openvla-llm-Q4_K_M.gguf

    # Step 2: 在 Q900 上运行推理
    python q900_npu_inference.py --camera_id 0

    # HDF5 验证模式
    python q900_npu_inference.py --hdf5_path edge_optimization/data/episode_0006.hdf5
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
# Action Decoder (与 q900_inference.py 相同)
# ---------------------------------------------------------------------------

class ActionDecoder:
    """从 LLM 输出的 token IDs 解码为 7-DoF 动作"""

    def __init__(self, params_path: str):
        with open(params_path, "r") as f:
            params = json.load(f)

        self.vocab_size = params["vocab_size"]
        self.n_action_bins = params["n_action_bins"]
        self.bin_centers = np.array(params["bin_centers"], dtype=np.float64)

        stats = params["dataset_statistics"]["dobot_pouring"]["action"]
        self.q01 = np.array(stats["q01"], dtype=np.float64)
        self.q99 = np.array(stats["q99"], dtype=np.float64)
        self.mask = np.array(stats["mask"], dtype=bool)
        self.action_dim = len(self.q01)
        print(f"  Action decoder: {self.action_dim}D, {self.n_action_bins} bins")

    def decode(self, token_ids: list[int]) -> np.ndarray:
        indices = np.array([
            np.clip(self.vocab_size - tid - 1, 0, self.n_action_bins - 1)
            for tid in token_ids
        ])
        normalized = self.bin_centers[indices]
        real = np.where(
            self.mask,
            0.5 * (normalized + 1.0) * (self.q99 - self.q01) + self.q01,
            normalized,
        )
        return real.astype(np.float32)


# ---------------------------------------------------------------------------
# Vision Encoder (ONNX Runtime, CPU 或 QNN)
# ---------------------------------------------------------------------------

class VisionEncoderONNX:
    """ONNX Runtime 视觉编码器 (服务器端/Q900 CPU fallback)"""

    def __init__(self, onnx_path: str):
        import onnxruntime as ort
        providers = ["CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(onnx_path, opts, providers=providers)
        print(f"  Vision encoder (ONNX): {onnx_path}")

    def encode(self, rgb_uint8: np.ndarray) -> np.ndarray:
        x = rgb_uint8.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        x = np.concatenate([x, x], axis=0)  # 6ch fused
        x = np.expand_dims(x, 0)
        return self.session.run(None, {"pixel_values": x})[0].squeeze(0)


# ---------------------------------------------------------------------------
# LLM: qairt GenAI API (NPU)
# ---------------------------------------------------------------------------

class VLAInferenceNPU:
    """使用 qairt GenAI API 在 Hexagon NPU 上运行 LLM"""

    SYSTEM_PROMPT = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    )

    def __init__(self, compiled_model_dir: str, tokenizer_path: str,
                 action_dim: int = 7, n_vocab: int = 32064):
        from qairt import CompiledModel
        from qairt.api.configs.common import BackendType
        from qairt.gen_ai_api.configs.gen_ai_config import GenAIConfig
        from qairt.gen_ai_api.executors.t2t_executor import T2TExecutor

        # 加载编译好的 QNN 模型
        models = [CompiledModel.load(compiled_model_dir)]
        print(f"  Loaded compiled model: {compiled_model_dir}")

        # GenAI 配置
        genai_config = GenAIConfig(
            tokenizer_path=tokenizer_path,
            context_length=2048,
            n_vocab=n_vocab,
            bos_token=1,
            eos_token=2,
        )

        # 创建 T2T Executor (HTP backend)
        self.executor = T2TExecutor(
            models=models,
            genai_config=genai_config,
            backend=BackendType.HTP,
        )
        self.executor.prepare_environment()
        self.action_dim = action_dim
        print(f"  LLM NPU ready (HTP backend)")

    def _build_prompt(self, task: str) -> str:
        instruction = task.lower().strip()
        human_msg = f"<image>\nWhat action should the robot take to {instruction}?"
        return f"{self.SYSTEM_PROMPT} USER: {human_msg} ASSISTANT:"

    def infer_action(self, rgb_uint8: np.ndarray, task: str,
                     vision_encoder, action_decoder: ActionDecoder) -> np.ndarray:
        """完整推理: 图像 → 动作"""

        # 1. Vision encoding (CPU 或 NPU)
        vision_emb = vision_encoder.encode(rgb_uint8)

        # 2. 构造 prompt
        # TODO: 当 qairt API 支持 embedding input 时，注入 vision_emb
        # 目前使用文本 prompt，LLM 将基于文本生成 action tokens
        prompt = self._build_prompt(task)

        # 3. LLM 推理
        result = self.executor.generate(prompt)
        output_text = result.generated_text

        # 4. 提取 action token IDs
        # Genie 返回文本, 需要 tokenize 回 token IDs
        action_token_ids = self._extract_action_tokens(output_text, action_decoder)

        # 5. 解码动作
        return action_decoder.decode(action_token_ids)

    def _extract_action_tokens(self, text: str, action_decoder: ActionDecoder) -> list[int]:
        """从 LLM 输出文本中提取 action token IDs"""
        # Genie 返回的文本可能包含 tokenized 的 action
        # 需要根据实际情况调整解析逻辑
        tokens = text.strip().split()
        token_ids = []
        for t in tokens:
            try:
                tid = int(t)
                token_ids.append(tid)
            except ValueError:
                # 尝试 tokenize
                pass
        if len(token_ids) < action_decoder.action_dim:
            # Fallback: 使用默认 action
            print(f"  [WARN] 只提取到 {len(token_ids)} 个 token, 需要 {action_decoder.action_dim}")
            token_ids.extend([31744] * (action_decoder.action_dim - len(token_ids)))
        return token_ids[:action_decoder.action_dim]

    def cleanup(self):
        self.executor.clean_environment()


# ---------------------------------------------------------------------------
# LLM: Transformers fallback (服务器端验证)
# ---------------------------------------------------------------------------

class VLAInferenceTransformers:
    """使用 transformers + merged model 的推理方案 (服务器端验证用)"""

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

        stats_path = os.path.join(merged_dir, "dataset_statistics.json")
        if os.path.isfile(stats_path):
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
            from tools.vla.openvla_lora_runtime import inject_dataset_statistics
            inject_dataset_statistics(self.model, stats_path)

        print(f"  Transformers model loaded: {merged_dir}")

    def infer_action(self, rgb_uint8: np.ndarray, task: str,
                     vision_encoder, action_decoder: ActionDecoder) -> np.ndarray:
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
# 编译模式: 使用 qairt API 将 GGUF 编译为 QNN context binary
# ---------------------------------------------------------------------------

def compile_model(gguf_path: str, output_dir: str):
    """使用 qairt GenAI API 编译 GGUF 模型为 QNN HTP context binary"""
    from qairt.api.configs.common import BackendType
    from qairt.gen_ai_api.gen_ai_builder_factory import GenAIBuilderFactory

    os.makedirs(output_dir, exist_ok=True)

    print(f"  编译 GGUF: {gguf_path}")
    print(f"  输出目录: {output_dir}")

    # 自动检测 Llama 架构, 创建 HTP builder
    builder = GenAIBuilderFactory.create(gguf_path, BackendType.HTP)

    # 编译为 QNN context binary
    compiled_models = builder.compile(output_dir)

    print(f"  编译完成! 生成 {len(compiled_models)} 个 model(s)")
    for m in compiled_models:
        print(f"    - {m}")

    return compiled_models


# ---------------------------------------------------------------------------
# 数据源
# ---------------------------------------------------------------------------

class CameraSource:
    def __init__(self, camera_id: int = 0):
        import cv2
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")
        print(f"  Camera: {camera_id}")

    def get_frame(self) -> np.ndarray:
        import cv2
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame")
        return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (224, 224))

    def release(self):
        self.cap.release()


class HDF5Source:
    def __init__(self, hdf5_path: str):
        import h5py
        with h5py.File(hdf5_path, "r") as f:
            self.images = f["observations/images/rgb"][:]
        self.idx = 0
        print(f"  HDF5: {hdf5_path}, {len(self.images)} frames")

    def get_frame(self) -> np.ndarray:
        if self.idx >= len(self.images):
            self.idx = 0
        img = self.images[self.idx]
        self.idx += 1
        return img

    def release(self):
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Q900 NPU VLA Inference")
    p.add_argument("--compile_only", action="store_true",
                   help="仅编译模型 (在服务器上运行)")
    p.add_argument("--gguf_path", type=str,
                   default="edge_optimization/gguf_models/openvla-llm-Q4_K_M.gguf")
    p.add_argument("--vision_onnx", type=str,
                   default="edge_optimization/components/vision_projector/vision_projector.onnx")
    p.add_argument("--action_params", type=str,
                   default="edge_optimization/components/action_head/action_head_params.json")
    p.add_argument("--compiled_model_dir", type=str,
                   default="edge_optimization/qnn_models/compiled",
                   help="编译后的 QNN 模型目录")
    p.add_argument("--tokenizer_path", type=str, default="",
                   help="tokenizer.json 路径 (Genie 需要)")
    p.add_argument("--merged_model", type=str,
                   default="edge_optimization/merged_model")
    p.add_argument("--camera_id", type=int, default=0)
    p.add_argument("--hdf5_path", type=str, default="")
    p.add_argument("--task", type=str, default="pour water from bottle into cup")
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument("--num_steps", type=int, default=0)
    p.add_argument("--backend", type=str, default="npu",
                   choices=["npu", "transformers"],
                   help="推理后端: npu (QNN HTP) 或 transformers (PyTorch)")
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Q900 NPU VLA Inference")
    print("=" * 60)

    # 编译模式
    if args.compile_only:
        print("\n[编译模式] GGUF → QNN HTP context binary")
        compile_model(_resolve(args.gguf_path), _resolve(args.compiled_model_dir))
        print("\n编译完成! 将产出文件拷贝到 Q900 设备。")
        return 0

    # 加载组件
    print("\n[1/3] Loading action decoder ...")
    action_decoder = ActionDecoder(_resolve(args.action_params))

    print("\n[2/3] Loading vision encoder ...")
    vision_encoder = VisionEncoderONNX(_resolve(args.vision_onnx))

    print(f"\n[3/3] Loading LLM ({args.backend}) ...")
    if args.backend == "npu":
        tokenizer_path = args.tokenizer_path
        if not tokenizer_path:
            # 尝试从 merged_model 或 components 中找 tokenizer
            for candidate in [
                os.path.join(args.merged_model, "tokenizer.json"),
                "edge_optimization/components/llm_llama2_7b/tokenizer.json",
            ]:
                if os.path.isfile(_resolve(candidate)):
                    tokenizer_path = _resolve(candidate)
                    break
        if not tokenizer_path:
            print("[ERROR] 需要 --tokenizer_path 指定 tokenizer.json")
            return 1

        vla = VLAInferenceNPU(
            compiled_model_dir=_resolve(args.compiled_model_dir),
            tokenizer_path=tokenizer_path,
            action_dim=action_decoder.action_dim,
        )
    else:
        vla = VLAInferenceTransformers(_resolve(args.merged_model), device=args.device)

    # 数据源
    if args.hdf5_path:
        source = HDF5Source(_resolve(args.hdf5_path))
    else:
        source = CameraSource(args.camera_id)

    # 推理循环
    print(f"\n{'=' * 60}")
    print(f"Task: {args.task}")
    print(f"Backend: {args.backend}")
    print(f"Target Hz: {args.hz}")
    print(f"{'=' * 60}")
    print("\nPress Ctrl+C to stop.\n")

    dt = 1.0 / args.hz
    step = 0
    latencies = []

    try:
        while True:
            if args.num_steps > 0 and step >= args.num_steps:
                break

            rgb = source.get_frame()

            t0 = time.time()
            action = vla.infer_action(rgb, args.task, vision_encoder, action_decoder)
            t1 = time.time()

            latency = t1 - t0
            latencies.append(latency)

            action_str = " ".join([f"{v:+.6f}" for v in action])
            print(f"[step {step:4d}] {action_str}  ({latency*1000:.0f}ms)")

            step += 1

            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    finally:
        source.release()
        if hasattr(vla, 'cleanup'):
            vla.cleanup()

    if latencies:
        print(f"\n{'=' * 60}")
        print(f"Steps: {step}")
        print(f"Avg latency: {np.mean(latencies)*1000:.0f}ms")
        print(f"Min latency: {np.min(latencies)*1000:.0f}ms")
        print(f"Max latency: {np.max(latencies)*1000:.0f}ms")
        print(f"Avg Hz: {1.0/np.mean(latencies):.1f}")
        print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
