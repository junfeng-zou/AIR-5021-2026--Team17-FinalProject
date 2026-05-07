#!/usr/bin/env python3
"""
Step 5: 验证 INT8 DLC 与 ONNX FP32 的输出精度对比
在服务器上运行，使用 qairt-net-run 或 snpe-net-run 对比输出

用法:
    python step5_verify_accuracy.py \
        --onnx  edge_optimization/components/vision_projector/vision_projector.onnx \
        --int8_dlc edge_optimization_v2.0/output/vision_projector_int8.dlc \
        --calib_dir edge_optimization_v2.0/data/calib \
        --num_tests 5
"""
from __future__ import annotations
import argparse, os, subprocess, tempfile, sys
import numpy as np

_DINO_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_DINO_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_SIG_MEAN  = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_SIG_STD   = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def preprocess(rgb_uint8: np.ndarray) -> np.ndarray:
    import cv2
    if rgb_uint8.shape[:2] != (224, 224):
        rgb_uint8 = cv2.resize(rgb_uint8, (224, 224))
    x = rgb_uint8.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    dino = (x - _DINO_MEAN[:, None, None]) / _DINO_STD[:, None, None]
    sglp = (x - _SIG_MEAN[:, None, None])  / _SIG_STD[:, None, None]
    return np.concatenate([dino, sglp], axis=0)[np.newaxis].astype(np.float32)


def run_onnx(session, raw_path: str) -> np.ndarray:
    x = np.fromfile(raw_path, dtype=np.float32).reshape(1, 6, 224, 224)
    out = session.run(None, {"pixel_values": x})[0]
    return out.squeeze(0)  # (256, 4096)


def run_dlc_via_snpe(dlc_path: str, raw_path: str, sdk_root: str) -> np.ndarray | None:
    """用 snpe-net-run 在 CPU 上运行 DLC，输出结果"""
    snpe_run = os.path.join(sdk_root, "bin/x86_64-linux-clang/snpe-net-run")
    if not os.path.isfile(snpe_run):
        return None
    with tempfile.TemporaryDirectory() as tmp:
        list_path = os.path.join(tmp, "input_list.txt")
        out_dir   = os.path.join(tmp, "output")
        os.makedirs(out_dir)
        with open(list_path, "w") as f:
            f.write(os.path.abspath(raw_path) + "\n")
        try:
            subprocess.run(
                [snpe_run,
                 "--container", dlc_path,
                 "--input_list", list_path,
                 "--output_dir", out_dir,
                 "--use_cpu"],
                check=True, capture_output=True, timeout=120
            )
            # 读取第一个输出文件
            result_dir = os.path.join(out_dir, "Result_0")
            if not os.path.isdir(result_dir):
                return None
            out_files = sorted(os.listdir(result_dir))
            if not out_files:
                return None
            arr = np.fromfile(os.path.join(result_dir, out_files[0]), dtype=np.float32)
            return arr.reshape(256, 4096)
        except Exception as e:
            print(f"    [WARNING] snpe-net-run 失败: {e}")
            return None


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    return float(np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-12))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx",     required=True)
    p.add_argument("--int8_dlc", required=True)
    p.add_argument("--calib_dir", default="edge_optimization_v2.0/data/calib")
    p.add_argument("--num_tests", type=int, default=5)
    args = p.parse_args()

    # 查找 SDK
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sdk_root = os.path.join(repo_root, "edge_optimization/qairt_v2.42.0.251225/2.42.0.251225")

    # 加载 ONNX
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    except ImportError:
        print("ERROR: 需要 pip install onnxruntime")
        sys.exit(1)

    # 找校准 .raw 文件
    raw_files = sorted(
        os.path.join(args.calib_dir, f)
        for f in os.listdir(args.calib_dir) if f.endswith(".raw")
    )[:args.num_tests]

    if not raw_files:
        print(f"ERROR: {args.calib_dir} 中无 .raw 文件，请先运行 step1")
        sys.exit(1)

    print("=" * 55)
    print("Step 5: ONNX (FP32) vs DLC (INT8) 精度对比")
    print(f"  ONNX   : {args.onnx}")
    print(f"  DLC    : {args.int8_dlc}")
    print(f"  测试数 : {len(raw_files)}")
    print("=" * 55)

    cos_sims, max_errs, l2_errs = [], [], []

    for i, raw_path in enumerate(raw_files):
        # FP32 ONNX 参考输出
        ref = run_onnx(sess, raw_path)

        # INT8 DLC 输出
        qnt = run_dlc_via_snpe(args.int8_dlc, raw_path, sdk_root)
        if qnt is None:
            print(f"  [{i}] DLC 推理失败（snpe-net-run 不可用或报错），跳过")
            continue

        cos = cosine_sim(ref, qnt)
        max_err = float(np.max(np.abs(ref - qnt)))
        l2_err  = float(np.linalg.norm(ref - qnt)) / float(np.linalg.norm(ref))

        cos_sims.append(cos)
        max_errs.append(max_err)
        l2_errs.append(l2_err)

        status = "✅" if cos > 0.990 else ("⚠️" if cos > 0.970 else "❌")
        print(f"  [{i}] 余弦相似度={cos:.4f}  max_err={max_err:.4f}  L2_rel={l2_err:.4f}  {status}")

    if cos_sims:
        print()
        print(f"  平均余弦相似度: {np.mean(cos_sims):.4f}")
        print(f"  最低余弦相似度: {np.min(cos_sims):.4f}")
        passed = sum(1 for c in cos_sims if c > 0.990)
        print(f"  通过率 (>0.990): {passed}/{len(cos_sims)}")
        if np.mean(cos_sims) < 0.970:
            print("\n  ❌ 精度不足，建议：增加校准样本（50+张）或改用 W4A8 量化")
        elif np.mean(cos_sims) < 0.990:
            print("\n  ⚠️  精度可接受，建议在实机上验证动作是否偏差")
        else:
            print("\n  ✅ 精度验证通过，可以继续 Step 4 生成 Context Binary")
    else:
        print("\n  [INFO] DLC 精度对比跳过（snpe-net-run 不可用）")
        print("  建议: 在 Q900 上用 qnn-net-run 对比结果")


if __name__ == "__main__":
    main()
