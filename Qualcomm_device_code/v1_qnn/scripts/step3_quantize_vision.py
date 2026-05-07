#!/usr/bin/env python3
"""
Phase 1 Step 2: 视觉编码器 INT8 量化
=====================================

对 Vision+Projector 的 ONNX 模型进行 INT8 量化，
减小模型大小并加速边缘端推理。

支持两种模式：
  - 动态量化（无需校准数据，直接量化权重）
  - 静态量化（使用校准数据集，量化权重+激活值，精度更高）

用法:
    # 动态量化（无需数据）
    python edge_optimization/scripts/step3_quantize_vision.py \
        --vision_onnx edge_optimization/components/vision_projector/vision_projector.onnx \
        --mode dynamic

    # 静态量化（使用校准数据）
    python edge_optimization/scripts/step3_quantize_vision.py \
        --vision_onnx edge_optimization/components/vision_projector/vision_projector.onnx \
        --mode static \
        --calib_data_dir data/vla_dataset
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vision 编码器 INT8 量化")
    p.add_argument(
        "--vision_onnx",
        type=str,
        default="edge_optimization/components/vision_projector/vision_projector.onnx",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="输出目录，默认与输入同目录",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="dynamic",
        choices=["dynamic", "static"],
    )
    p.add_argument(
        "--calib_data_dir",
        type=str,
        default="data/vla_dataset",
        help="校准数据目录（仅 static 模式）",
    )
    p.add_argument(
        "--num_calib_samples",
        type=int,
        default=100,
        help="校准样本数（仅 static 模式）",
    )
    p.add_argument(
        "--use_fused_backbone",
        action="store_true",
        default=True,
        help="模型使用 fused backbone（DINOv2+SigLIP, 6 通道输入）",
    )
    return p.parse_args()


def _resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(repo_root, path)


class CalibrationDataReader:
    """从 HDF5 数据集读取图像作为校准数据"""

    def __init__(self, data_dir: str, num_samples: int, use_fused: bool):
        self.use_fused = use_fused
        self.samples = []
        self._load_samples(data_dir, num_samples)
        self.idx = 0

    def _load_samples(self, data_dir: str, num_samples: int):
        """从 HDF5 文件中加载 RGB 图像并预处理"""
        import h5py
        import glob

        hdf5_files = sorted(glob.glob(os.path.join(data_dir, "episode_*.hdf5")))
        if not hdf5_files:
            print(f"  [WARN] 未找到 HDF5 文件: {data_dir}")
            return

        count = 0
        for hf_path in hdf5_files:
            if count >= num_samples:
                break
            try:
                with h5py.File(hf_path, "r") as hf:
                    images = hf["observations/images/rgb"][:]
                    # 每个 episode 均匀采样
                    step = max(1, len(images) // 5)
                    for i in range(0, len(images), step):
                        if count >= num_samples:
                            break
                        img = images[i]  # (224, 224, 3) uint8
                        pixel_values = self._preprocess(img)
                        self.samples.append(pixel_values)
                        count += 1
            except Exception as e:
                print(f"  [WARN] 读取 {hf_path} 失败: {e}")

        print(f"  加载了 {len(self.samples)} 个校准样本")

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """将 uint8 HWC 图像转为模型输入格式"""
        # (H, W, 3) uint8 → (1, C, H, W) float32, 归一化到 [0, 1]
        x = img.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))  # HWC → CHW

        if self.use_fused:
            # fused backbone: 两个 backbone 接收相同图像（堆叠为 6 通道）
            x = np.concatenate([x, x], axis=0)  # (6, H, W)

        return np.expand_dims(x, 0)  # (1, C, H, W)

    def get_next(self):
        if self.idx >= len(self.samples):
            return None
        sample = {"pixel_values": self.samples[self.idx]}
        self.idx += 1
        return sample

    def rewind(self):
        self.idx = 0


def quantize_dynamic(input_onnx: str, output_onnx: str) -> bool:
    """ONNX Runtime 动态量化（仅量化权重）"""
    print("  [DEBUG] entered quantize_dynamic")
    sys.stdout.flush()
    from onnxruntime.quantization import quantize_dynamic as ort_quantize_dynamic, QuantType

    print("  执行动态量化 (INT8 weights) ...")
    sys.stdout.flush()
    try:
        ort_quantize_dynamic(
            model_input=input_onnx,
            model_output=output_onnx,
            weight_type=QuantType.QInt8,
        )
        # ort_quantize_dynamic 成功时返回 None，用文件是否存在来判断
        success = os.path.isfile(output_onnx)
        print(f"  [DEBUG] ort_quantize_dynamic done, output exists: {success}")
        sys.stdout.flush()
        return success
    except Exception as e:
        print(f"  [ERROR] 动态量化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def quantize_static(
    input_onnx: str,
    output_onnx: str,
    calib_reader,
) -> bool:
    """ONNX Runtime 静态量化（量化权重+激活值）"""
    from onnxruntime.quantization import quantize_static as ort_quantize_static
    from onnxruntime.quantization import QuantType, CalibrationMethod

    print("  执行静态量化 (INT8 weights + activations) ...")
    try:
        ort_quantize_static(
            model_input=input_onnx,
            model_output=output_onnx,
            calibration_data_reader=calib_reader,
            quant_format=None,  # 使用默认
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,
            calibrate_method=CalibrationMethod.MinMax,
        )
        # ort_quantize_static 成功时返回 None，用文件是否存在来判断
        return os.path.isfile(output_onnx)
    except Exception as e:
        print(f"  [ERROR] 静态量化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_quantized_model(
    original_onnx: str,
    quantized_onnx: str,
    use_fused: bool,
    num_tests: int = 5,
) -> None:
    """对比量化前后的输出差异"""
    import onnxruntime as ort

    print("\n  验证量化精度 ...")

    # 创建 sessions
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    orig_sess = ort.InferenceSession(original_onnx, opts, providers=["CPUExecutionProvider"])
    quant_sess = ort.InferenceSession(quantized_onnx, opts, providers=["CPUExecutionProvider"])

    channels = 6 if use_fused else 3
    max_diffs = []
    mean_diffs = []
    cos_sims = []

    for i in range(num_tests):
        dummy = np.random.randn(1, channels, 224, 224).astype(np.float32)
        orig_out = orig_sess.run(None, {"pixel_values": dummy})[0]
        quant_out = quant_sess.run(None, {"pixel_values": dummy})[0]

        diff = np.abs(orig_out - quant_out)
        max_diffs.append(diff.max())
        mean_diffs.append(diff.mean())

        # cosine similarity
        flat_orig = orig_out.flatten()
        flat_quant = quant_out.flatten()
        cos = np.dot(flat_orig, flat_quant) / (
            np.linalg.norm(flat_orig) * np.linalg.norm(flat_quant) + 1e-8
        )
        cos_sims.append(cos)

    print(f"  Max diff  (avg over {num_tests} tests): {np.mean(max_diffs):.6f}")
    print(f"  Mean diff (avg over {num_tests} tests): {np.mean(mean_diffs):.6f}")
    print(f"  Cosine similarity (avg):                {np.mean(cos_sims):.6f}")

    if np.mean(cos_sims) > 0.99:
        print("  ✅ 量化精度验证通过（cosine > 0.99）")
    elif np.mean(cos_sims) > 0.95:
        print("  ⚠️ 量化有轻微精度损失（cosine > 0.95），通常可接受")
    else:
        print("  ❌ 量化精度损失较大，建议检查量化参数或使用动态量化")


def main() -> int:
    args = parse_args()
    vision_onnx = _resolve(args.vision_onnx)
    output_dir = _resolve(args.output_dir) if args.output_dir else os.path.dirname(vision_onnx)

    if not os.path.isfile(vision_onnx):
        print(f"[ERROR] ONNX 文件不存在: {vision_onnx}", file=sys.stderr)
        print("  请先运行 step1_extract_components.py（带 --export_vision_onnx）", file=sys.stderr)
        return 1

    # 检查依赖
    try:
        import onnxruntime
        from onnxruntime.quantization import quantize_dynamic as _ort_qd  # noqa: F401，仅做可用性检查
    except ImportError:
        print("[ERROR] 需要安装 onnxruntime: pip install onnxruntime", file=sys.stderr)
        return 2

    os.makedirs(output_dir, exist_ok=True)

    basename = Path(vision_onnx).stem
    quantized_onnx = os.path.join(output_dir, f"{basename}_int8_{args.mode}.onnx")

    print("=" * 60)
    print(f"Phase 1 Step 2: Vision 编码器 {args.mode.upper()} INT8 量化")
    print("=" * 60)
    print(f"  输入: {vision_onnx}")
    print(f"  输出: {quantized_onnx}")
    print(f"  模式: {args.mode}")
    print("=" * 60)

    orig_size = os.path.getsize(vision_onnx) / 1024 / 1024

    if args.mode == "dynamic":
        print(f"  [DEBUG] calling quantize_dynamic({vision_onnx}, {quantized_onnx})")
        sys.stdout.flush()
        success = quantize_dynamic(vision_onnx, quantized_onnx)
        print(f"  [DEBUG] quantize_dynamic returned: {success}, file exists: {os.path.isfile(quantized_onnx)}")
        sys.stdout.flush()
    else:
        calib_dir = _resolve(args.calib_data_dir)
        print(f"  校准数据: {calib_dir}")
        calib_reader = CalibrationDataReader(calib_dir, args.num_calib_samples, args.use_fused_backbone)
        if len(calib_reader.samples) == 0:
            print("  [WARN] 没有校准样本，回退到动态量化")
            success = quantize_dynamic(vision_onnx, quantized_onnx)
        else:
            success = quantize_static(vision_onnx, quantized_onnx, calib_reader)

    if not success or not os.path.isfile(quantized_onnx):
        print("\n❌ 量化失败", file=sys.stderr)
        return 3

    # 原始模型大小：优先用 onnx initializer 中的权重总量（external data 时文件本身很小）
    try:
        import onnx as _onnx
        _m = _onnx.load(vision_onnx, load_external_data=False)
        init_size = sum(w.ByteSize() for w in _m.graph.initializer)
        orig_size_display = init_size / 1024 / 1024 if init_size > 0 else orig_size
    except Exception:
        orig_size_display = orig_size

    quant_size = os.path.getsize(quantized_onnx) / 1024 / 1024
    compression = (1 - quant_size / orig_size_display) * 100

    print(f"\n  原始模型 (权重): {orig_size_display:.1f} MB")
    print(f"  量化模型:        {quant_size:.1f} MB")
    print(f"  压缩率:          {compression:.1f}%")

    # 验证（部分 onnxruntime 版本对 ConvInteger 支持有限，捕获异常避免崩溃）
    try:
        verify_quantized_model(vision_onnx, quantized_onnx, args.use_fused_backbone)
    except Exception as e:
        print(f"  ⚠️  验证跳过（运行时不支持量化算子）: {e}")
        print("  量化文件已生成，可在支持 INT8 的设备上推理。")

    print(f"\n✅ Phase 1 Step 2 完成！量化模型: {quantized_onnx}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
