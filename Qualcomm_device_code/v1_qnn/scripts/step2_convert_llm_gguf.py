#!/usr/bin/env python3
"""
Phase 1 Step 1: LLM → GGUF 格式转换 + 量化
============================================

将提取出的 Llama-2-7B (HuggingFace 格式) 转换为 GGUF 格式，
并生成多种量化版本供边缘设备使用。

依赖: llama.cpp（会自动 clone 并编译）

用法:
    python edge_optimization/scripts/step2_convert_llm_gguf.py \
        --llm_dir edge_optimization/components/llm_llama2_7b \
        --output_dir edge_optimization/gguf_models \
        --quantize Q4_K_M Q8_0
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM → GGUF 转换与量化")
    p.add_argument(
        "--llm_dir",
        type=str,
        default="edge_optimization/components/llm_llama2_7b",
        help="提取出的 HuggingFace Llama 模型目录",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="edge_optimization/gguf_models",
        help="GGUF 输出目录",
    )
    p.add_argument(
        "--llama_cpp_dir",
        type=str,
        default="edge_optimization/llama.cpp",
        help="llama.cpp 源码目录（不存在则自动 clone）",
    )
    p.add_argument(
        "--quantize",
        type=str,
        nargs="+",
        default=["Q4_K_M", "Q8_0"],
        help="量化类型列表，如 Q4_K_M Q8_0 Q4_0 Q5_K_M",
    )
    p.add_argument(
        "--skip_clone",
        action="store_true",
        help="跳过 clone llama.cpp（已有目录时）",
    )
    p.add_argument(
        "--skip_build",
        action="store_true",
        help="跳过编译 llama.cpp（已有可执行文件时）",
    )
    p.add_argument(
        "--outtype",
        type=str,
        default="f16",
        choices=["f16", "f32", "bf16"],
        help="GGUF 基础精度（转换前）",
    )
    return p.parse_args()


def _resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(repo_root, path)


def _run(cmd: list[str], cwd: str | None = None, check: bool = True) -> int:
    """执行命令并实时打印输出"""
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if check and result.returncode != 0:
        print(f"  [ERROR] 命令返回码: {result.returncode}", file=sys.stderr)
    return result.returncode


def ensure_llama_cpp(llama_cpp_dir: str, skip_clone: bool, skip_build: bool) -> str:
    """确保 llama.cpp 存在并已编译，返回 build 目录"""
    if not os.path.isdir(llama_cpp_dir) and not skip_clone:
        print(f"\n  Cloning llama.cpp → {llama_cpp_dir}")
        _run(["git", "clone", "--depth=1", "https://github.com/ggerganov/llama.cpp", llama_cpp_dir])

    if not os.path.isdir(llama_cpp_dir):
        print(f"[ERROR] llama.cpp 目录不存在: {llama_cpp_dir}", file=sys.stderr)
        sys.exit(1)

    build_dir = os.path.join(llama_cpp_dir, "build")
    quantize_bin = os.path.join(build_dir, "bin", "llama-quantize")
    # 有些版本叫 quantize, 有些叫 llama-quantize
    alt_quantize_bin = os.path.join(build_dir, "bin", "quantize")

    if not skip_build:
        print(f"\n  编译 llama.cpp ...")
        os.makedirs(build_dir, exist_ok=True)
        ret = _run(["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"], cwd=build_dir)
        if ret != 0:
            print("[ERROR] cmake 配置失败", file=sys.stderr)
            sys.exit(1)
        import multiprocessing
        jobs = max(1, multiprocessing.cpu_count() // 2)
        ret = _run(["cmake", "--build", ".", "-j", str(jobs), "--target", "llama-quantize"], cwd=build_dir)
        if ret != 0:
            # 旧版可能 target 叫 quantize
            _run(["cmake", "--build", ".", "-j", str(jobs)], cwd=build_dir)

    # 查找 quantize 可执行文件
    for candidate in [quantize_bin, alt_quantize_bin]:
        if os.path.isfile(candidate):
            return candidate

    # 搜索
    for root, dirs, files in os.walk(build_dir):
        for f in files:
            if f in ("llama-quantize", "quantize") and os.access(os.path.join(root, f), os.X_OK):
                return os.path.join(root, f)

    print(f"[WARN] 未找到 quantize 可执行文件，量化步骤将跳过")
    return ""


def convert_hf_to_gguf(llama_cpp_dir: str, llm_dir: str, output_path: str, outtype: str) -> bool:
    """使用 llama.cpp 的 convert 脚本将 HF 模型转为 GGUF"""
    # 新版 llama.cpp 使用 convert_hf_to_gguf.py
    convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    if not os.path.isfile(convert_script):
        # 旧版
        convert_script = os.path.join(llama_cpp_dir, "convert.py")
    if not os.path.isfile(convert_script):
        print(f"[ERROR] 未找到转换脚本: convert_hf_to_gguf.py 或 convert.py", file=sys.stderr)
        return False

    cmd = [
        sys.executable, convert_script,
        llm_dir,
        "--outfile", output_path,
        "--outtype", outtype,
    ]
    ret = _run(cmd)
    return ret == 0


def quantize_gguf(quantize_bin: str, input_gguf: str, output_gguf: str, qtype: str) -> bool:
    """使用 llama-quantize 量化 GGUF 模型"""
    cmd = [quantize_bin, input_gguf, output_gguf, qtype]
    ret = _run(cmd)
    return ret == 0


def main() -> int:
    args = parse_args()
    llm_dir = _resolve(args.llm_dir)
    output_dir = _resolve(args.output_dir)
    llama_cpp_dir = _resolve(args.llama_cpp_dir)

    if not os.path.isdir(llm_dir):
        print(f"[ERROR] LLM 目录不存在: {llm_dir}", file=sys.stderr)
        print("  请先运行 step1_extract_components.py", file=sys.stderr)
        return 1

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Phase 1 Step 1: LLM → GGUF 转换与量化")
    print("=" * 60)
    print(f"  LLM 目录     : {llm_dir}")
    print(f"  输出目录     : {output_dir}")
    print(f"  llama.cpp    : {llama_cpp_dir}")
    print(f"  基础精度     : {args.outtype}")
    print(f"  量化类型     : {args.quantize}")
    print("=" * 60)

    # 1) 确保 llama.cpp
    print("\n[1/3] 准备 llama.cpp ...")
    quantize_bin = ensure_llama_cpp(llama_cpp_dir, args.skip_clone, args.skip_build)

    # 2) HF → GGUF (F16)
    base_gguf = os.path.join(output_dir, f"openvla-llm-{args.outtype}.gguf")
    print(f"\n[2/3] 转换 HF → GGUF ({args.outtype}) ...")
    if os.path.isfile(base_gguf):
        print(f"  已存在，跳过: {base_gguf}")
    else:
        success = convert_hf_to_gguf(llama_cpp_dir, llm_dir, base_gguf, args.outtype)
        if not success:
            print("[ERROR] GGUF 转换失败", file=sys.stderr)
            return 2

    if os.path.isfile(base_gguf):
        sz_gb = os.path.getsize(base_gguf) / 1024 / 1024 / 1024
        print(f"  ✅ 基础 GGUF: {base_gguf} ({sz_gb:.2f} GB)")

    # 3) 量化
    print(f"\n[3/3] 量化 GGUF ...")
    if not quantize_bin:
        print("  [SKIP] quantize 可执行文件未找到，跳过量化步骤")
        print("  请手动编译 llama.cpp 后执行:")
        for qtype in args.quantize:
            out_name = f"openvla-llm-{qtype}.gguf"
            print(f"    llama-quantize {base_gguf} {os.path.join(output_dir, out_name)} {qtype}")
        return 0

    results = {}
    for qtype in args.quantize:
        out_gguf = os.path.join(output_dir, f"openvla-llm-{qtype}.gguf")
        if os.path.isfile(out_gguf):
            print(f"  [{qtype}] 已存在，跳过")
            results[qtype] = out_gguf
            continue

        print(f"\n  [{qtype}] 量化中 ...")
        success = quantize_gguf(quantize_bin, base_gguf, out_gguf, qtype)
        if success and os.path.isfile(out_gguf):
            sz_gb = os.path.getsize(out_gguf) / 1024 / 1024 / 1024
            print(f"  ✅ {qtype}: {out_gguf} ({sz_gb:.2f} GB)")
            results[qtype] = out_gguf
        else:
            print(f"  ❌ {qtype} 量化失败")

    # 总结
    print("\n" + "=" * 60)
    print("Phase 1 Step 1 完成！GGUF 模型:")
    print("=" * 60)
    for f in sorted(os.listdir(output_dir)):
        if f.endswith(".gguf"):
            fp = os.path.join(output_dir, f)
            sz = os.path.getsize(fp) / 1024 / 1024 / 1024
            print(f"  {f:45s} {sz:.2f} GB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
