#!/usr/bin/env python3
"""
q900_inference_v3.py — OpenVLA Q900 推理 (Vision NPU 加速版)
=============================================================

Vision 后端优先级 (自动探测):
  1. QNN Context Binary (.bin) via qnn-net-run  ← 最快，NPU 原生
  2. ONNX Runtime QNN EP                        ← NPU 加速，易安装
  3. ONNX Runtime CPU                           ← 兜底，无需额外依赖

用法:
    # 最快路径: NPU context binary
    python q900_inference_v3.py \
        --context_bin ~/pouring_vla/vision_projector_v73.bin \
        --gguf_path   ~/pouring_vla/openvla-llm-Q4_K_M.gguf \
        --action_params ~/pouring_vla/action_head_params.json \
        --camera_id 0 --task "pour cola into cup"

    # 备选路径: INT8 ONNX (跳过 DLC/bin 步骤)
    python q900_inference_v3.py \
        --vision_onnx ~/pouring_vla/vision_projector_int8_dynamic.onnx \
        --gguf_path   ~/pouring_vla/openvla-llm-Q4_K_M.gguf \
        --action_params ~/pouring_vla/action_head_params.json \
        --camera_id 0
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np

# ── 图像预处理（双骨干，必须与训练完全一致）────────────────────────
_DINO_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_DINO_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_SIG_MEAN  = np.array([0.5, 0.5, 0.5],       dtype=np.float32)
_SIG_STD   = np.array([0.5, 0.5, 0.5],       dtype=np.float32)


def preprocess(rgb_uint8: np.ndarray) -> np.ndarray:
    """(224,224,3) uint8 → (1,6,224,224) float32"""
    import cv2
    if rgb_uint8.shape[:2] != (224, 224):
        rgb_uint8 = cv2.resize(rgb_uint8, (224, 224), interpolation=cv2.INTER_AREA)
    x = rgb_uint8.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    dino = (x - _DINO_MEAN[:, None, None]) / _DINO_STD[:, None, None]
    sglp = (x - _SIG_MEAN[:, None, None])  / _SIG_STD[:, None, None]
    return np.concatenate([dino, sglp], axis=0)[np.newaxis].astype(np.float32)


# ── Vision 编码器: 后端 1 — QNN Context Binary ────────────────────
class QNNContextVisionEncoder:
    """
    通过 qnn-net-run 调用 NPU context binary (.bin)
    输入: (1,6,224,224) float32 → 输出: (256,4096) float32
    """
    def __init__(self, bin_path: str, qnn_sdk_root: str):
        self.bin_path = bin_path
        # 在 Q900 上 qnn-net-run 位于 aarch64-ubuntu-gcc9.4
        for subdir in ["aarch64-ubuntu-gcc9.4", "aarch64-oe-linux-gcc9.3",
                       "aarch64-oe-linux-gcc11.2"]:
            candidate = os.path.join(qnn_sdk_root, "bin", subdir, "qnn-net-run")
            if os.path.isfile(candidate):
                self.qnn_net_run = candidate
                break
        else:
            raise FileNotFoundError(f"qnn-net-run 未找到，SDK: {qnn_sdk_root}")

        # libQnnHtp.so
        for subdir in ["aarch64-ubuntu-gcc9.4", "aarch64-oe-linux-gcc9.3",
                       "aarch64-oe-linux-gcc11.2"]:
            candidate = os.path.join(qnn_sdk_root, "lib", subdir, "libQnnHtp.so")
            if os.path.isfile(candidate):
                self.htp_lib = candidate
                break
        else:
            raise FileNotFoundError(f"libQnnHtp.so 未找到，SDK: {qnn_sdk_root}")

        self._tmpdir = tempfile.mkdtemp(prefix="qnn_vision_")
        self._input_path  = os.path.join(self._tmpdir, "pixel_values.raw")
        self._list_path   = os.path.join(self._tmpdir, "input_list.txt")
        self._output_dir  = os.path.join(self._tmpdir, "output")
        os.makedirs(self._output_dir, exist_ok=True)
        with open(self._list_path, "w") as f:
            f.write(self._input_path + "\n")

        print(f"  [QNN] context binary: {Path(bin_path).name}")
        print(f"  [QNN] qnn-net-run: {self.qnn_net_run}")

    def encode(self, rgb_uint8: np.ndarray) -> np.ndarray:
        # 1. 写输入
        preprocess(rgb_uint8).tofile(self._input_path)

        # 2. 清空上次输出
        import shutil
        if os.path.exists(self._output_dir):
            shutil.rmtree(self._output_dir)
        os.makedirs(self._output_dir)

        # 3. 调用 qnn-net-run
        cmd = [
            self.qnn_net_run,
            "--retrieve_context", self.bin_path,
            "--backend", self.htp_lib,
            "--input_list", self._list_path,
            "--output_dir", self._output_dir,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=10)
        if result.returncode != 0:
            raise RuntimeError(
                f"qnn-net-run 失败 (code={result.returncode}):\n"
                f"{result.stderr.decode()}"
            )

        # 4. 读输出 (Result_0/ 下第一个 .raw 文件)
        result_dir = os.path.join(self._output_dir, "Result_0")
        out_files = sorted(f for f in os.listdir(result_dir) if f.endswith(".raw"))
        if not out_files:
            raise RuntimeError(f"qnn-net-run 没有输出文件: {result_dir}")
        arr = np.fromfile(os.path.join(result_dir, out_files[0]), dtype=np.float32)
        return arr.reshape(256, 4096)


# ── Vision 编码器: 后端 2 & 3 — ONNX Runtime ─────────────────────
class ONNXVisionEncoder:
    """
    ONNX Runtime 视觉编码器
    provider_priority: ["QNNExecutionProvider", "CPUExecutionProvider"]
    """
    def __init__(self, onnx_path: str, use_qnn_ep: bool = True):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = []
        if use_qnn_ep:
            # QNN EP 配置（需要 onnxruntime-qnn 或支持 QNN 的 ORT 构建）
            qnn_ep_opts = {
                "backend_path": "QnnHtp.so",
                "htp_performance_mode": "burst",
                "enable_htp_fp16_precision": "0",
            }
            # 检查 QNN EP 是否可用
            if "QNNExecutionProvider" in ort.get_all_providers():
                providers.append(("QNNExecutionProvider", qnn_ep_opts))

        # Vulkan（Adreno GPU 加速）
        if "VulkanExecutionProvider" in ort.get_all_providers():
            providers.append("VulkanExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(onnx_path, opts, providers=providers)
        ep = self.session.get_providers()[0]
        print(f"  [ONNX] {Path(onnx_path).name}  EP={ep}")

    def encode(self, rgb_uint8: np.ndarray) -> np.ndarray:
        x = preprocess(rgb_uint8)
        out = self.session.run(None, {"pixel_values": x})[0]
        return out.squeeze(0)  # (256, 4096)


# ── 动作解码器 ────────────────────────────────────────────────────
class ActionDecoder:
    def __init__(self, params_path: str, unnorm_key: str = "dobot_pouring"):
        with open(params_path) as f:
            params = json.load(f)
        self.vocab_size  = params["vocab_size"]
        self.n_bins      = params["n_action_bins"]
        self.bin_centers = np.array(params["bin_centers"], dtype=np.float64)
        stats = params["dataset_statistics"][unnorm_key]["action"]
        self.q01  = np.array(stats["q01"],  dtype=np.float64)
        self.q99  = np.array(stats["q99"],  dtype=np.float64)
        self.mask = np.array(stats["mask"], dtype=bool)
        self.action_dim = len(self.q01)
        print(f"  ActionDecoder: dim={self.action_dim}, unnorm_key={unnorm_key}")

    def decode(self, token_ids: list[int]) -> np.ndarray:
        indices = np.clip(
            [self.vocab_size - tid - 1 for tid in token_ids],
            0, self.n_bins - 1
        ).astype(np.int32)
        normalized = self.bin_centers[indices]
        return np.where(
            self.mask,
            0.5 * (normalized + 1.0) * (self.q99 - self.q01) + self.q01,
            normalized,
        ).astype(np.float32)


# ── VLA 推理引擎 ──────────────────────────────────────────────────
VICUNA_SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
IMAGE_TOKEN = "<image>"


class VLAEngine:
    def __init__(self, gguf_path, vision_encoder, action_decoder,
                 n_ctx=512, n_threads=8, n_gpu_layers=0):
        from llama_cpp import Llama
        import llama_cpp as lc_lib
        self.vis = vision_encoder
        self.dec = action_decoder
        self.lc  = lc_lib
        self.llm = Llama(
            model_path=gguf_path, n_ctx=n_ctx,
            n_threads=n_threads, n_gpu_layers=n_gpu_layers,
            verbose=False, logits_all=True,
        )
        self.n_embd = self.llm.n_embd()
        print(f"  LLM: {Path(gguf_path).name}  n_embd={self.n_embd}  ngl={n_gpu_layers}")

    def _build_prompt(self, task: str) -> str:
        return (f"{VICUNA_SYSTEM} USER: {IMAGE_TOKEN}\n"
                f"What action should the robot take to {task.lower().strip()}? ASSISTANT:")

    def infer(self, rgb_uint8: np.ndarray, task: str) -> np.ndarray:
        import ctypes
        t0 = time.time()

        # 1. Vision 编码（NPU 或 ONNX）
        vision_emb = self.vis.encode(rgb_uint8)  # (256, 4096)
        t1 = time.time()

        # 2. 分割 prompt
        prompt = self._build_prompt(task)
        img_pos = prompt.find(IMAGE_TOKEN)
        tok_before = self.llm.tokenize(prompt[:img_pos].encode(), add_bos=True)
        tok_after  = self.llm.tokenize(prompt[img_pos + len(IMAGE_TOKEN):].encode(), add_bos=False)

        # 3. KV-prefill
        self.llm.reset()
        self.llm.eval(tok_before)
        self._inject_vision(vision_emb, start_pos=len(tok_before))
        self.llm.eval(tok_after)
        t2 = time.time()

        # 4. Greedy decode
        action_tokens = []
        for _ in range(self.dec.action_dim):
            tid = self._greedy()
            action_tokens.append(tid)
            self.llm.eval([tid])
        t3 = time.time()

        print(f"    [timing] vision={1000*(t1-t0):.0f}ms  "
              f"prefill={1000*(t2-t1):.0f}ms  "
              f"decode={1000*(t3-t2):.0f}ms  "
              f"total={1000*(t3-t0):.0f}ms")

        return self.dec.decode(action_tokens)

    def _inject_vision(self, vision_emb: np.ndarray, start_pos: int):
        import ctypes
        lc = self.lc
        n_vis, n_embd = vision_emb.shape
        embd_flat = np.ascontiguousarray(vision_emb, dtype=np.float32)
        batch = lc.llama_batch_init(n_vis, n_embd, 1)
        try:
            batch.n_tokens = n_vis
            ctypes.memmove(
                batch.embd,
                embd_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                n_vis * n_embd * ctypes.sizeof(ctypes.c_float),
            )
            for i in range(n_vis):
                batch.pos[i] = start_pos + i
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = 0
                batch.logits[i] = False
            ret = lc.llama_decode(self.llm._ctx.ctx, batch)
            if ret != 0:
                raise RuntimeError(f"llama_decode 失败: ret={ret}")
        finally:
            lc.llama_batch_free(batch)

    def _greedy(self) -> int:
        import ctypes
        lc = self.lc
        ptr = lc.llama_get_logits(self.llm._ctx.ctx)
        logits = np.ctypeslib.as_array(ptr, shape=(self.llm.n_vocab(),)).copy()
        return int(np.argmax(logits))


# ── 自动探测并创建 Vision 编码器 ──────────────────────────────────
def create_vision_encoder(args) -> tuple[object, str]:
    """
    按优先级探测可用的 Vision 后端:
    1. QNN Context Binary (.bin)
    2. ONNX RT + QNN EP
    3. ONNX RT + CPU
    """
    # 后端 1: QNN Context Binary
    if args.context_bin and os.path.isfile(args.context_bin):
        qnn_sdk = args.qnn_sdk_root or _find_qnn_sdk()
        if qnn_sdk:
            try:
                enc = QNNContextVisionEncoder(args.context_bin, qnn_sdk)
                return enc, "QNN_NPU_ContextBinary"
            except Exception as e:
                print(f"  [WARNING] QNN context binary 初始化失败: {e}")
        else:
            print("  [WARNING] 未找到 QNN SDK，跳过 context binary 后端")

    # 后端 2 & 3: ONNX
    onnx_path = args.vision_onnx
    if not onnx_path or not os.path.isfile(onnx_path):
        # 自动在 context_bin 同目录或 pouring_vla 下寻找 ONNX
        for search_dir in [
            os.path.dirname(args.context_bin or ""),
            os.path.expanduser("~/pouring_vla"),
        ]:
            for fname in ["vision_projector_int8_dynamic.onnx",
                          "vision_projector.onnx"]:
                candidate = os.path.join(search_dir, fname)
                if os.path.isfile(candidate):
                    onnx_path = candidate
                    break
            if onnx_path:
                break

    if not onnx_path or not os.path.isfile(onnx_path):
        raise FileNotFoundError("未找到 ONNX 或 context binary 文件，请检查参数")

    try:
        enc = ONNXVisionEncoder(onnx_path, use_qnn_ep=True)
        ep = enc.session.get_providers()[0]
        backend_name = "QNN_EP" if "QNN" in ep else ("Vulkan" if "Vulkan" in ep else "CPU")
        return enc, f"ONNX_{backend_name}"
    except Exception as e:
        print(f"  [WARNING] ONNX + QNN EP 初始化失败: {e}，回退 CPU")
        enc = ONNXVisionEncoder(onnx_path, use_qnn_ep=False)
        return enc, "ONNX_CPU"


def _find_qnn_sdk() -> str | None:
    candidates = [
        os.path.expanduser("~/qairt/2.37.1.250807"),
        os.path.expanduser("~/qairt/2.42.0.251225"),
        "/opt/qcom/aistack/qnn/2.37.1.250807",
    ]
    return next((p for p in candidates if os.path.isdir(p)), None)


# ── 主流程 ────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Q900 OpenVLA V3.0 — NPU Vision 加速")
    p.add_argument("--context_bin",    default="", help="NPU context binary (.bin)")
    p.add_argument("--vision_onnx",    default="", help="ONNX 模型路径（备用）")
    p.add_argument("--gguf_path",      default=os.path.expanduser("~/pouring_vla/openvla-llm-Q4_K_M.gguf"))
    p.add_argument("--action_params",  default=os.path.expanduser("~/pouring_vla/action_head_params.json"))
    p.add_argument("--unnorm_key",     default="dobot_pouring")
    p.add_argument("--camera_id",      type=int, default=0)
    p.add_argument("--task",           default="pour cola into cup")
    p.add_argument("--hz",             type=float, default=2.0)
    p.add_argument("--n_gpu_layers",   type=int, default=0)
    p.add_argument("--n_threads",      type=int, default=8)
    p.add_argument("--n_ctx",          type=int, default=512)
    p.add_argument("--num_steps",      type=int, default=0, help="0=无限")
    p.add_argument("--hdf5_path",      default="", help="用 HDF5 替代摄像头（验证用）")
    p.add_argument("--dry_run",        action="store_true")
    p.add_argument("--qnn_sdk_root",   default="", help="Q900 上的 QAIRT SDK 路径")
    return p.parse_args()


def main():
    args = parse_args()
    import cv2

    print("=" * 60)
    print("Q900 OpenVLA V3.0 — Vision NPU 加速推理")
    print("=" * 60)

    print("\n[1/3] 初始化 Vision 编码器 ...")
    vision_enc, backend = create_vision_encoder(args)
    print(f"  ✅ Vision 后端: {backend}")

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
    )

    # 数据源
    if args.hdf5_path:
        import h5py
        with h5py.File(args.hdf5_path, "r") as f:
            images = f["observations/images/rgb"][:]
        idx = [0]
        def get_frame():
            img = images[idx[0] % len(images)]
            idx[0] += 1
            return cv2.resize(img.astype(np.uint8), (224, 224))
        release = lambda: None
    else:
        cap = cv2.VideoCapture(args.camera_id)
        def get_frame():
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("摄像头读取失败")
            return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (224, 224))
        release = cap.release

    print(f"\n{'=' * 60}")
    print(f"任务: {args.task}")
    print(f"Vision 后端: {backend}")
    print(f"目标频率: {args.hz} Hz  |  dry_run: {args.dry_run}")
    print("Ctrl+C 退出\n")

    dt = 1.0 / args.hz
    step, latencies = 0, []

    try:
        while args.num_steps == 0 or step < args.num_steps:
            t0 = time.time()
            rgb = get_frame()
            action = engine.infer(rgb, args.task)
            latency = time.time() - t0
            latencies.append(latency)

            a = action
            print(
                f"[{step:4d}] "
                f"dxyz=[{a[0]*1000:+.1f},{a[1]*1000:+.1f},{a[2]*1000:+.1f}]mm  "
                f"drpy=[{np.rad2deg(a[3]):+.1f},{np.rad2deg(a[4]):+.1f},{np.rad2deg(a[5]):+.1f}]°  "
                f"g={a[6]:+.2f}  ({latency*1000:.0f}ms)"
            )

            if not args.dry_run:
                pass  # TODO: 接入 Dobot ServoP

            step += 1
            sleep_t = dt - (time.time() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        release()

    if latencies:
        print(f"\n{'=' * 60}")
        print(f"总步数: {step}  |  Vision 后端: {backend}")
        print(f"平均延迟: {np.mean(latencies)*1000:.0f} ms")
        print(f"Vision 占比: 请查看上方 [timing] 输出")
        print(f"实际频率: {1.0/np.mean(latencies):.2f} Hz")
        print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
