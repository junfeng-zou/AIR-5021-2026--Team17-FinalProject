#!/usr/bin/env python3
"""
Q900 Edge Inference - QNN Direct ctypes Edition
================================================
直接用 ctypes 调用 QNN 版 libllama.so，不依赖 llama-cpp-python 包。

用法:
    python q900_inference_qnn.py \
        --gguf_path ~/pouring_vla/openvla-llm-Q4_K_M.gguf \
        --vision_onnx ~/pouring_vla/vision_projector.onnx \
        --action_params ~/pouring_vla/action_head_params.json \
        --hdf5_path ~/pouring_vla/episode_0006.hdf5 \
        --task "pour cola into cup" \
        --n_gpu_layers 32 \
        --num_steps 5
"""

from __future__ import annotations
import argparse, ctypes, json, os, sys, time
from pathlib import Path
import cv2, numpy as np

# ─── 1. QNN 库路径配置 ────────────────────────────────────────────────────
_BUILD_BIN  = os.path.expanduser("~/pouring_vla/llama.cpp/build/bin")
_QNN_RTLIB  = "/home/radxa/qairt/2.37.1.250807/lib/aarch64-oe-linux-gcc11.2"

# 预先加入 LD_LIBRARY_PATH（Python import 之前必须设置）
os.environ["LD_LIBRARY_PATH"] = ":".join([
    _BUILD_BIN, _QNN_RTLIB,
    os.environ.get("LD_LIBRARY_PATH", ""),
])

# 按依赖顺序加载（RTLD_GLOBAL 使后续库能解析前面的符号）
for _so in ["libggml-base.so", "libggml-cpu.so", "libggml-qnn.so", "libggml.so"]:
    ctypes.CDLL(os.path.join(_BUILD_BIN, _so), mode=ctypes.RTLD_GLOBAL)

_lib = ctypes.CDLL(os.path.join(_BUILD_BIN, "libllama.so"), mode=ctypes.RTLD_GLOBAL)

# ─── 2. C 类型定义 ────────────────────────────────────────────────────────
llama_token  = ctypes.c_int32
llama_pos    = ctypes.c_int32
llama_seq_id = ctypes.c_int32

# llama_model_params (llama.cpp ggml ~0.9.x, late 2024)
# 布局: devices ptr(8), n_gpu_layers(4), split_mode(4), main_gpu(4), pad(4),
#       tensor_split ptr(8), progress_cb ptr(8), cb_data ptr(8), kv_overrides ptr(8),
#       vocab_only(1), use_mmap(1), use_mlock(1), check_tensors(1), pad(4)
class LlamaModelParams(ctypes.Structure):
    _fields_ = [
        ("devices",                     ctypes.c_void_p),
        ("n_gpu_layers",                ctypes.c_int32),
        ("split_mode",                  ctypes.c_int32),
        ("main_gpu",                    ctypes.c_int32),
        ("_pad0",                       ctypes.c_int32),
        ("tensor_split",                ctypes.c_void_p),
        ("progress_callback",           ctypes.c_void_p),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("kv_overrides",                ctypes.c_void_p),
        ("vocab_only",                  ctypes.c_bool),
        ("use_mmap",                    ctypes.c_bool),
        ("use_mlock",                   ctypes.c_bool),
        ("check_tensors",               ctypes.c_bool),
    ]

# llama_context_params (trimmed to fields we use)
class LlamaContextParams(ctypes.Structure):
    _fields_ = [
        ("n_ctx",               ctypes.c_uint32),
        ("n_batch",             ctypes.c_uint32),
        ("n_ubatch",            ctypes.c_uint32),
        ("n_seq_max",           ctypes.c_uint32),
        ("n_threads",           ctypes.c_int32),
        ("n_threads_batch",     ctypes.c_int32),
        ("rope_scaling_type",   ctypes.c_int32),
        ("pooling_type",        ctypes.c_int32),
        ("attention_type",      ctypes.c_int32),
        ("_pad0",               ctypes.c_int32),
        ("rope_freq_base",      ctypes.c_float),
        ("rope_freq_scale",     ctypes.c_float),
        ("yarn_ext_factor",     ctypes.c_float),
        ("yarn_attn_factor",    ctypes.c_float),
        ("yarn_beta_fast",      ctypes.c_float),
        ("yarn_beta_slow",      ctypes.c_float),
        ("yarn_orig_ctx",       ctypes.c_uint32),
        ("defrag_thold",        ctypes.c_float),
        ("cb_eval",             ctypes.c_void_p),
        ("cb_eval_user_data",   ctypes.c_void_p),
        ("type_k",              ctypes.c_int32),
        ("type_v",              ctypes.c_int32),
        ("logits_all",          ctypes.c_bool),
        ("embeddings",          ctypes.c_bool),
        ("offload_kqv",         ctypes.c_bool),
        ("flash_attn",          ctypes.c_bool),
        ("no_perf",             ctypes.c_bool),
        ("op_offload",          ctypes.c_bool),
        ("_pad1",               ctypes.c_uint8 * 2),
        ("abort_callback",      ctypes.c_void_p),
        ("abort_callback_data", ctypes.c_void_p),
    ]

# llama_batch
class LlamaBatch(ctypes.Structure):
    _fields_ = [
        ("n_tokens",  ctypes.c_int32),
        ("token",     ctypes.POINTER(llama_token)),
        ("embd",      ctypes.POINTER(ctypes.c_float)),
        ("pos",       ctypes.POINTER(llama_pos)),
        ("n_seq_id",  ctypes.POINTER(ctypes.c_int32)),
        ("seq_id",    ctypes.POINTER(ctypes.POINTER(llama_seq_id))),
        ("logits",    ctypes.POINTER(ctypes.c_int8)),
    ]

# ─── 3. 函数签名绑定 ──────────────────────────────────────────────────────
def _setup_lib(lib):
    lib.llama_backend_init.restype  = None
    lib.llama_backend_init.argtypes = []

    lib.llama_model_default_params.restype  = LlamaModelParams
    lib.llama_model_default_params.argtypes = []

    lib.llama_context_default_params.restype  = LlamaContextParams
    lib.llama_context_default_params.argtypes = []

    lib.llama_model_load_from_file.restype  = ctypes.c_void_p
    lib.llama_model_load_from_file.argtypes = [ctypes.c_char_p, LlamaModelParams]

    lib.llama_model_free.restype  = None
    lib.llama_model_free.argtypes = [ctypes.c_void_p]

    lib.llama_init_from_model.restype  = ctypes.c_void_p
    lib.llama_init_from_model.argtypes = [ctypes.c_void_p, LlamaContextParams]

    lib.llama_free.restype  = None
    lib.llama_free.argtypes = [ctypes.c_void_p]

    lib.llama_get_memory.restype  = ctypes.c_void_p
    lib.llama_get_memory.argtypes = [ctypes.c_void_p]

    lib.llama_memory_clear.restype  = None
    lib.llama_memory_clear.argtypes = [ctypes.c_void_p, ctypes.c_bool]

    lib.llama_model_n_embd.restype  = ctypes.c_int32
    lib.llama_model_n_embd.argtypes = [ctypes.c_void_p]

    lib.llama_model_get_vocab.restype  = ctypes.c_void_p
    lib.llama_model_get_vocab.argtypes = [ctypes.c_void_p]

    lib.llama_vocab_n_tokens.restype  = ctypes.c_int32
    lib.llama_vocab_n_tokens.argtypes = [ctypes.c_void_p]

    lib.llama_tokenize.restype  = ctypes.c_int32
    lib.llama_tokenize.argtypes = [
        ctypes.c_void_p,           # vocab
        ctypes.c_char_p,           # text
        ctypes.c_int32,            # text_len
        ctypes.POINTER(llama_token), # tokens out
        ctypes.c_int32,            # n_tokens_max
        ctypes.c_bool,             # add_special
        ctypes.c_bool,             # parse_special
    ]

    lib.llama_batch_init.restype  = LlamaBatch
    lib.llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]

    lib.llama_batch_free.restype  = None
    lib.llama_batch_free.argtypes = [LlamaBatch]

    lib.llama_decode.restype  = ctypes.c_int32
    lib.llama_decode.argtypes = [ctypes.c_void_p, LlamaBatch]

    lib.llama_get_logits.restype  = ctypes.POINTER(ctypes.c_float)
    lib.llama_get_logits.argtypes = [ctypes.c_void_p]

_setup_lib(_lib)
_lib.llama_backend_init()

# ─── 4. LlamaCtypes: 替代 llama_cpp.Llama ────────────────────────────────
class LlamaCtypes:
    """直接 ctypes 调用 QNN libllama.so，接口兼容原 Llama 类"""

    def __init__(self, model_path: str, n_ctx=512, n_threads=4,
                 n_gpu_layers=0, verbose=False, logits_all=True):
        self._lib = _lib
        self._verbose = verbose

        # --- 加载模型 ---
        mp = _lib.llama_model_default_params()
        mp.n_gpu_layers = n_gpu_layers
        self._model = _lib.llama_model_load_from_file(
            model_path.encode(), mp)
        if not self._model:
            raise RuntimeError(f"Failed to load model: {model_path}")

        # --- 创建上下文 ---
        cp = _lib.llama_context_default_params()
        cp.n_ctx          = n_ctx
        cp.n_threads      = n_threads
        cp.n_threads_batch= n_threads
        cp.logits_all     = logits_all
        self._ctx = _lib.llama_init_from_model(self._model, cp)
        if not self._ctx:
            raise RuntimeError("Failed to create llama context")

        self._vocab   = _lib.llama_model_get_vocab(self._model)
        self._n_ctx   = n_ctx
        self._pos_cur = 0  # 跟踪当前 KV cache 位置

    # --- 兼容接口 ---
    def n_embd(self) -> int:
        return _lib.llama_model_n_embd(self._model)

    def n_vocab(self) -> int:
        return _lib.llama_vocab_n_tokens(self._vocab)

    def tokenize(self, text: bytes, add_bos: bool = True) -> list[int]:
        max_tokens = len(text) + 64
        buf = (llama_token * max_tokens)()
        n = _lib.llama_tokenize(
            self._vocab, text, len(text),
            buf, max_tokens,
            add_bos, False)
        if n < 0:
            raise RuntimeError(f"llama_tokenize failed: {n}")
        return list(buf[:n])

    def reset(self):
        """清除 KV cache（每次推理前调用）"""
        mem = _lib.llama_get_memory(self._ctx)
        _lib.llama_memory_clear(mem, True)
        self._pos_cur = 0

    def eval(self, tokens: list[int]):
        """把一批 token 送入 KV cache（不需要 logits）"""
        n = len(tokens)
        batch = _lib.llama_batch_init(n, 0, 1)
        try:
            batch.n_tokens = n
            for i, tok in enumerate(tokens):
                batch.token[i]     = tok
                batch.pos[i]       = self._pos_cur + i
                batch.n_seq_id[i]  = 1
                batch.seq_id[i][0] = 0
                batch.logits[i]    = (1 if i == n - 1 else 0)
            ret = _lib.llama_decode(self._ctx, batch)
            if ret != 0:
                raise RuntimeError(f"llama_decode failed: {ret}")
            self._pos_cur += n
        finally:
            _lib.llama_batch_free(batch)

    def __del__(self):
        try:
            if hasattr(self, "_ctx")   and self._ctx:   _lib.llama_free(self._ctx)
            if hasattr(self, "_model") and self._model: _lib.llama_model_free(self._model)
        except Exception:
            pass


# ─── 5. 图像预处理（与原脚本完全一致）────────────────────────────────────
_DINO_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_DINO_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_SIG_MEAN  = np.array([0.5,   0.5,   0.5  ], dtype=np.float32)
_SIG_STD   = np.array([0.5,   0.5,   0.5  ], dtype=np.float32)

def preprocess(rgb_uint8):
    x = rgb_uint8.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    dino = (x - _DINO_MEAN[:, None, None]) / _DINO_STD[:, None, None]
    sglp = (x - _SIG_MEAN[:, None, None])  / _SIG_STD[:, None, None]
    return np.concatenate([dino, sglp], axis=0)[np.newaxis].astype(np.float32)

class VisionEncoder:
    def __init__(self, onnx_path):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = []
        if "VulkanExecutionProvider" in ort.get_all_providers():
            providers.append("VulkanExecutionProvider")
        providers.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(onnx_path, opts, providers=providers)
        print(f"  Vision encoder: {Path(onnx_path).name}  [{self.session.get_providers()[0]}]")

    def encode(self, rgb_uint8):
        out = self.session.run(None, {"pixel_values": preprocess(rgb_uint8)})[0]
        return out.squeeze(0)  # (256, 4096)

class ActionDecoder:
    def __init__(self, params_path, unnorm_key="dobot_pouring"):
        with open(params_path) as f:
            params = json.load(f)
        self.vocab_size = params["vocab_size"]
        self.n_bins     = params["n_action_bins"]
        self.bin_centers= np.array(params["bin_centers"], dtype=np.float64)
        stats = params["dataset_statistics"][unnorm_key]["action"]
        self.q01  = np.array(stats["q01"],  dtype=np.float64)
        self.q99  = np.array(stats["q99"],  dtype=np.float64)
        self.mask = np.array(stats["mask"], dtype=bool)
        self.action_dim = len(self.q01)
        print(f"  Action decoder: {self.action_dim}D  "
              f"token_range=[{self.vocab_size-self.n_bins}, {self.vocab_size-1}]")

    def decode(self, token_ids):
        indices  = np.array([int(np.clip(self.vocab_size - tid - 1, 0, self.n_bins-1))
                             for tid in token_ids], dtype=np.int32)
        normalized = self.bin_centers[indices]
        physical   = np.where(
            self.mask,
            0.5*(normalized+1.0)*(self.q99-self.q01)+self.q01,
            normalized)
        return physical.astype(np.float32)

# ─── 6. VLAEngine（核心推理，与原脚本逻辑相同）──────────────────────────
VICUNA_SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
IMAGE_TOKEN = "<image>"

class VLAEngine:
    def __init__(self, gguf_path, vision_encoder, action_decoder,
                 n_ctx=512, n_threads=4, n_gpu_layers=0):
        self.vision_encoder = vision_encoder
        self.action_decoder = action_decoder

        # 用我们的 LlamaCtypes 替换原来的 Llama
        self.llm = LlamaCtypes(
            model_path=gguf_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            logits_all=True,
        )
        self.n_embd = self.llm.n_embd()
        print(f"  LLM (QNN ctypes): {Path(gguf_path).name}  "
              f"n_embd={self.n_embd}  ngl={n_gpu_layers}")

    def _build_prompt(self, task):
        return (f"{VICUNA_SYSTEM} USER: {IMAGE_TOKEN}\n"
                f"What action should the robot take to {task.lower().strip()}? "
                f"ASSISTANT:")

    def _inject_vision_embeddings(self, vision_emb, start_pos):
        """批量注入 256 个视觉 token 的 embedding（与原脚本逻辑相同）"""
        n_vis, n_embd = vision_emb.shape
        embd_flat = np.ascontiguousarray(vision_emb, dtype=np.float32)

        batch = _lib.llama_batch_init(n_vis, n_embd, 1)
        try:
            batch.n_tokens = n_vis
            ctypes.memmove(
                batch.embd,
                embd_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                n_vis * n_embd * ctypes.sizeof(ctypes.c_float),
            )
            for i in range(n_vis):
                batch.pos[i]       = start_pos + i
                batch.n_seq_id[i]  = 1
                batch.seq_id[i][0] = 0
                batch.logits[i]    = 0
            ret = _lib.llama_decode(self.llm._ctx, batch)
            if ret != 0:
                raise RuntimeError(f"llama_decode (vision) failed: {ret}")
            self.llm._pos_cur = start_pos + n_vis
        finally:
            _lib.llama_batch_free(batch)

    def _greedy_sample(self):
        n_vocab = self.llm.n_vocab()
        logits_ptr = _lib.llama_get_logits(self.llm._ctx)
        logits = np.ctypeslib.as_array(logits_ptr, shape=(n_vocab,)).copy()
        return int(np.argmax(logits))

    def infer(self, rgb_uint8, task):
        t0 = time.time()
        vision_emb = self.vision_encoder.encode(rgb_uint8)   # (256, 4096)
        t1 = time.time()

        prompt     = self._build_prompt(task)
        img_pos    = prompt.find(IMAGE_TOKEN)
        text_before = prompt[:img_pos]
        text_after  = prompt[img_pos + len(IMAGE_TOKEN):]

        tokens_before = self.llm.tokenize(text_before.encode(), add_bos=True)
        tokens_after  = self.llm.tokenize(text_after.encode(),  add_bos=False)

        self.llm.reset()
        self.llm.eval(tokens_before)
        self._inject_vision_embeddings(vision_emb, start_pos=len(tokens_before))
        self.llm.eval(tokens_after)
        t2 = time.time()

        action_tokens = []
        for _ in range(self.action_decoder.action_dim):
            tok = self._greedy_sample()
            action_tokens.append(tok)
            self.llm.eval([tok])
        t3 = time.time()

        print(f"    [timing] vision={1000*(t1-t0):.0f}ms  "
              f"prefill={1000*(t2-t1):.0f}ms  decode={1000*(t3-t2):.0f}ms")

        return self.action_decoder.decode(action_tokens)


# ─── 7. 主流程 ────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf_path",     default="/opt/openvla/llm.gguf")
    p.add_argument("--vision_onnx",   default="/opt/openvla/vision_projector.onnx")
    p.add_argument("--action_params", default="/opt/openvla/action_head_params.json")
    p.add_argument("--unnorm_key",    default="dobot_pouring")
    p.add_argument("--camera_id",     type=int, default=0)
    p.add_argument("--task",          default="pour cola into cup")
    p.add_argument("--hz",            type=float, default=2.0)
    p.add_argument("--n_gpu_layers",  type=int, default=0)
    p.add_argument("--n_threads",     type=int, default=8)
    p.add_argument("--n_ctx",         type=int, default=512)
    p.add_argument("--num_steps",     type=int, default=0)
    p.add_argument("--hdf5_path",     default="")
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("Q900 OpenVLA QNN Direct-ctypes 推理")
    print("=" * 60)

    print("\n[1/3] 初始化视觉编码器 ...")
    vision_enc = VisionEncoder(args.vision_onnx)

    print("\n[2/3] 初始化动作解码器 ...")
    action_dec = ActionDecoder(args.action_params, args.unnorm_key)

    print("\n[3/3] 加载 LLM (QNN) ...")
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
            img = images[idx[0] % len(images)]; idx[0] += 1
            return cv2.resize(img, (224, 224))
        release = lambda: None
    else:
        cap = cv2.VideoCapture(args.camera_id)
        def get_frame():
            ret, frame = cap.read()
            if not ret: raise RuntimeError("Camera read failed")
            return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (224, 224))
        release = cap.release

    print(f"\n{'='*60}\n任务: {args.task}  | GPU层: {args.n_gpu_layers}\nCtrl+C 退出\n")

    dt, step, latencies = 1.0/args.hz, 0, []
    try:
        while args.num_steps == 0 or step < args.num_steps:
            t0  = time.time()
            rgb = get_frame()
            act = engine.infer(rgb, args.task)
            lat = time.time() - t0
            latencies.append(lat)
            a = act
            print(f"[{step:4d}] "
                  f"dxyz=[{a[0]*1000:+.1f},{a[1]*1000:+.1f},{a[2]*1000:+.1f}]mm  "
                  f"drpy=[{np.rad2deg(a[3]):+.1f},{np.rad2deg(a[4]):+.1f},{np.rad2deg(a[5]):+.1f}]°  "
                  f"g={a[6]:+.2f}  ({lat*1000:.0f}ms)")
            step += 1
            sl = dt - (time.time() - t0)
            if sl > 0: time.sleep(sl)
    except KeyboardInterrupt:
        print("\n用户中断。")
    finally:
        release()

    if latencies:
        print(f"\n{'='*60}")
        print(f"总步数: {step}")
        print(f"平均延迟: {np.mean(latencies)*1000:.0f} ms  ({1/np.mean(latencies):.2f} Hz)")
        print(f"{'='*60}")


if __name__ == "__main__":
    sys.exit(main())
