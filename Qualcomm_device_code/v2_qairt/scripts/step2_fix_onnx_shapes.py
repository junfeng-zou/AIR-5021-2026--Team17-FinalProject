#!/usr/bin/env python3
"""
step2_fix_onnx_shapes.py
========================
在 qairt-converter 转换之前，先用 onnxsim 对 vision_projector.onnx 做
常量折叠 + shape 固化，解决 Reshape 动态 shape 导致的转换报错。

原因: qairt-converter 在遇到动态 Reshape (shape 由运行时决定) 时，
会尝试用 onnxsim 做简化，但若 onnxsim 未安装或模型有外部数据，
onnx 的形状推断会产生 "140469248 != -981802364" 这类溢出错误。

解决方案:
  1. 先用 onnxsim.simplify() + overwrite_input_shapes 固化输入维度
  2. 得到 vision_projector_static.onnx（可选保留外部数据）
  3. qairt-converter 对 static 版本进行转换

用法:
    python step2_fix_onnx_shapes.py \
        --onnx edge_optimization/components/vision_projector/vision_projector.onnx \
        --out  edge_optimization_v2.0/output/vision_projector_static.onnx
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path

def fix_shapes(src_onnx: str, dst_onnx: str, input_shapes: dict[str, list]):
    import onnx

    print(f"[step2_fix] 加载模型: {Path(src_onnx).name}  (可能需要 30-60 秒)")
    t0 = time.time()

    # 用 load_model_without_shape_inference 避免加载时 OOM
    # onnx.load 会直接读 external data 到内存
    load_opts = dict()
    model = onnx.load(src_onnx, load_external_data=False)
    t1 = time.time()
    print(f"  加载完成 ({t1-t0:.1f}s, 不含 external data)")

    # 方法1: 直接修改 graph.input 的 dim_value，强制固化 batch 维度
    modified = False
    for inp in model.graph.input:
        if inp.name in input_shapes:
            target_shape = input_shapes[inp.name]
            t = inp.type.tensor_type
            if t.HasField("shape"):
                for i, d in enumerate(t.shape.dim):
                    if i < len(target_shape):
                        val = target_shape[i]
                        if val is not None:
                            d.ClearField("dim_param")  # 清除符号维度
                            d.dim_value = val
                            modified = True
            print(f"  固化输入 '{inp.name}' → shape={target_shape}")

    if not modified:
        print("  WARNING: 未找到目标输入，检查 tensor 名称")

    # 方法2: 用 onnx shape_inference 传播固化后的 shape
    try:
        from onnx import shape_inference
        print("  运行 onnx shape_inference ...")
        model = shape_inference.infer_shapes(model, data_prop=False)
        print(f"  shape_inference 完成 ({time.time()-t1:.1f}s)")
    except Exception as e:
        print(f"  WARNING: shape_inference 失败: {e}")

    # 保存 (不保存 external data，让原始的 .onnx.data 文件继续被引用)
    os.makedirs(os.path.dirname(dst_onnx) or ".", exist_ok=True)
    print(f"  保存到: {dst_onnx}")
    onnx.save(model, dst_onnx, save_as_external_data=False)
    print(f"  ✅ 完成 (total {time.time()-t0:.1f}s)")
    print()
    print("  注意: 保存的文件不含权重，需要在同目录下运行 qairt-converter")
    print("  即 external data 文件仍然是原始 vision_projector.onnx.data")
    return dst_onnx


def try_onnxsim(src_onnx: str, dst_onnx: str, input_shapes: dict[str, list]):
    """尝试用 onnxsim 做完整简化（更彻底，但对大模型可能慢）"""
    try:
        from onnxsim import simplify
        import onnx
        print("[step2_fix] 尝试 onnxsim 完整简化 (可能需要数分钟，OOM 则回退)...")
        model = onnx.load(src_onnx)  # 包含 external data
        overwrite_shapes = {k: [int(x) for x in v] for k, v in input_shapes.items()}
        simplified, check = simplify(
            model,
            overwrite_input_shapes=overwrite_shapes,
            perform_optimization=True,
        )
        if not check:
            print("  WARNING: onnxsim 验证失败，但仍保存")
        onnx.save(simplified, dst_onnx)
        print(f"  ✅ onnxsim 简化完成: {dst_onnx}")
        return True
    except MemoryError:
        print("  ❌ OOM！回退到轻量方案")
        return False
    except Exception as e:
        print(f"  ❌ onnxsim 失败: {e}")
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True, help="输入 ONNX 路径")
    p.add_argument("--out",  required=True, help="输出 ONNX 路径（静态 shape）")
    p.add_argument("--no_onnxsim", action="store_true", help="跳过 onnxsim，只做轻量 shape 固化")
    args = p.parse_args()

    input_shapes = {"pixel_values": [1, 6, 224, 224]}

    if not args.no_onnxsim:
        ok = try_onnxsim(args.onnx, args.out, input_shapes)
        if ok:
            return

    # 回退: 轻量 shape 固化
    fix_shapes(args.onnx, args.out, input_shapes)


if __name__ == "__main__":
    main()
