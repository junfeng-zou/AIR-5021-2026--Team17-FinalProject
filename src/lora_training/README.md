# OpenVLA LoRA 微调使用说明

本目录在 **pouring_VLA** 采集的 HDF5 数据上，对 **OpenVLA-7B** 做 **PEFT LoRA** 微调。数据格式需与 `tools/vla/vla_data_writer.py` 一致（`episode_*.hdf5`，内含 `observations/images/rgb`、`actions`、`attrs["task"]` 等）。

## 1. 环境与 Isaac Sim 隔离

训练依赖 **PyTorch + Transformers + PEFT**，与 Isaac Sim / Isaac Lab 的 Python 环境**不要混用**。请单独建 **Conda** 环境（示例环境名 `openvla_lora`，可按需修改）：

```bash
conda create -n openvla_lora python=3.10 -y
conda activate openvla_lora
pip install -U pip
pip install -r LoRA_train/requirements.txt
```

若需 **GPU 版 PyTorch**，可先按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 用 conda/pip 装好 `torch` 再执行 `pip install -r LoRA_train/requirements.txt`（或去掉 `requirements.txt` 里已满足的依赖以免覆盖）。

若 **未** 提前准备权重，首次需能从 Hugging Face 拉取 `openvla/openvla-7b`（`huggingface-cli login` 或环境变量 `HF_TOKEN`）。

### 使用本机已下载的 OpenVLA-7B

`train_lora.py` 的 **`--vla_path`** 可以是 **本地目录**（与 HuggingFace 模型快照结构一致：含 `config.json`、分词与处理器文件、`model*.safetensors` 或 `pytorch_model*.bin` 等）。无需再访问外网加载基座。

```bash
python LoRA_train/train_lora.py \
  --vla_path /你的路径/openvla-7b \
  --data_dir data/vla_dataset \
  --output_dir runs/openvla_pouring_lora \
  --bf16
```

常见来源：

- 用 `huggingface-cli download openvla/openvla-7b --local-dir /你的路径/openvla-7b` 下载的目录；
- 或本机缓存里的某次 snapshot，例如  
  `~/.cache/huggingface/hub/models--openvla--openvla-7b/snapshots/<一串 commit hash>/`  
  （该目录内即为可直接作为 `--vla_path` 的模型根目录）。

仍建议保留 `trust_remote_code=True`（脚本已使用），以便加载 OpenVLA 自带的建模代码。

## 2. 数据路径

- 默认读取目录：`data/vla_dataset/`（或通过 `--data_dir` 指定）。
- 会匹配 `episode_*.hdf5` 及目录下其它 `.hdf5`（自动去重）。
- 每个样本一步：`224×224` RGB + 语言指令 + 7 维动作（`[-1,1]`）。

## 3. 启动训练

在仓库根目录 `pouring_VLA/` 下执行：

**单卡：**

```bash
python LoRA_train/train_lora.py \
  --data_dir data/vla_dataset \
  --output_dir runs/openvla_pouring_lora \
  --bf16
```

**多卡（Accelerate）：**

```bash
accelerate config   # 首次配置并行方式
accelerate launch LoRA_train/train_lora.py \
  --data_dir data/vla_dataset \
  --output_dir runs/openvla_pouring_lora \
  --bf16
```

常用参数说明：

| 参数 | 含义 |
|------|------|
| `--data_dir` | HDF5 所在目录 |
| `--output_dir` | 输出根目录（含 `final_lora/`、checkpoint、`dataset_statistics.json`） |
| `--vla_path` | 模型 ID 或本地路径，默认 `openvla/openvla-7b` |
| `--batch_size` | 每步微批大小 |
| `--grad_accumulation_steps` | 梯度累积步数，**有效 batch ≈ batch_size × 该值 × GPU 数** |
| `--learning_rate` | 默认 `5e-4`，与 OpenVLA 官方 LoRA 量级一致 |
| `--max_steps` | 优化步数上限 |
| `--save_steps` | 保存 checkpoint 间隔 |
| `--lora_rank` | LoRA 秩，默认 32 |
| `--bf16` | Ampere 及以上建议开启，省显存、加速 |
| `--wandb_project` | 填写项目名则启用 Weights & Biases；留空不记录 |
| `--log_interval` | 主进程每隔 N 个 **optimizer step** 在终端打印一行 `[TRAIN] step=… loss=…`（默认 10；设为 `0` 关闭）。短跑可看 loss 是否异常/NaN |
| `--num_workers` | DataLoader 进程数；HDF5 建议保持 **0** 避免多进程读盘问题 |
| `--lora_target` | `llm`（默认）：只对 Llama 的 `q/k/v/o/gate/up/down_proj` 加 LoRA，视觉塔冻结；`all-linear`：与上游 OpenVLA 脚本一致，视觉+投影+LLM 全插 LoRA（更吃显存） |

### 实现要点（与 OpenVLA 对齐）

- **`<image>`**：人类轮指令里包含 `<image>`，且 **不再** 在 prompt 构建时删掉该占位符，便于模型把 `pixel_values` 对齐到序列位置。
- **动作 token**：对文本只 `tokenize` 到 `ASSISTANT:` 为止；**7 个离散动作 id + EOS** 直接按 `vocab_size - digitize(...)` 拼到 `input_ids`，避免 `decode → 再 encode` 造成的动作 token 错位。
- **验证集**：`val_loader` 经 `accelerate.prepare`，各进程跑自己的分片并用 `accelerator.reduce` 汇总加权平均 loss（多卡一致）。

## 4. 输出说明

- `dataset_statistics.json`：动作分位数统计，供推理时与 OpenVLA 流程对齐做反归一化等。
- `final_lora/`：训练结束时的 **LoRA 适配器**（非完整 7B 合并权重）。
- `checkpoint_step_*`：中间 checkpoint。
- 推理时在基座上加载适配器（例如 `peft.PeftModel.from_pretrained`），或按需合并权重。

## 5. 推荐参数：RTX 3090（24GB）与 A100 80GB

以下为在 **单卡、LoRA `all-linear`、OpenVLA-7B、224 图像** 场景下的**起点配置**。若 OOM，优先 **减小 `batch_size`**，再用 **`grad_accumulation_steps` 补有效 batch**；仍不够可把 **`--lora_rank` 改为 16**。

### RTX 3090（24GB）

显存较紧，宜 **小 micro-batch + 较大梯度累积**，并开启 **`--bf16`**。

```bash
python LoRA_train/train_lora.py \
  --data_dir data/vla_dataset \
  --output_dir runs/pouring_lora_3090 \
  --bf16 \
  --batch_size 2 \
  --grad_accumulation_steps 16 \
  --learning_rate 5e-4 \
  --lora_rank 32 \
  --lora_dropout 0.0 \
  --max_steps 20000 \
  --save_steps 2000 \
  --num_workers 0 \
  --wandb_project openVLA_LoRA
```

- **有效 batch（单卡）**：2 × 16 = **32**（与较大 batch 训练较接近）。
- 若仍 OOM：试 `--batch_size 1`、`--grad_accumulation_steps 24`，或 `--lora_rank 16`。

### NVIDIA A100 80GB

显存充裕，可提高 **micro-batch**，减少累积步数以加快迭代（与 OpenVLA 文档中 80GB 机器可开大 batch 的思路一致）。

```bash
python LoRA_train/train_lora.py \
  --data_dir data/vla_dataset \
  --output_dir runs/pouring_lora_a10080 \
  --bf16 \
  --batch_size 16 \
  --grad_accumulation_steps 2 \
  --learning_rate 5e-4 \
  --lora_rank 32 \
  --lora_dropout 0.0 \
  --max_steps 20000 \
  --save_steps 2000 \
  --num_workers 0  \
  --wandb_project openVLA_LoRA
```

- **有效 batch（单卡）**：`batch_size` × `grad_accumulation_steps` = 16 × 2 = **32**。这是「几次前向累加后再更新一次」的等效批量，**不是**把 `--batch_size` 设为 32；单次前向仍是 16 条。
- 若显存仍有大量余量：可把 **单次前向** 加大，例如 `--batch_size 20`～`24` 且 `--grad_accumulation_steps 1`，或直接试 `--batch_size 32 --grad_accumulation_steps 1`（OOM 再降）；也可略增 `max_steps`。若 loss 不稳定可适当 **略降学习率**（如 `3e-4`）。

### 多卡 A100（可选）

有效 batch 再乘以 GPU 数；可按需 **同比减小 `grad_accumulation_steps`**，避免总 batch 过大。

```bash
# 例如 2×A100 80GB
accelerate launch --num_processes 2 LoRA_train/train_lora.py \
  --data_dir data/vla_dataset \
  --output_dir runs/pouring_lora_2xa100 \
  --bf16 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --lora_rank 32 \
  --max_steps 20000 \
  --save_steps 2000
```

## 6. 常见问题

- **CUDA OOM**：减小 `batch_size` → 增大 `grad_accumulation_steps` → 降低 `lora_rank` → 确认已加 `--bf16`。
- **下载模型失败**：检查网络、`HF_ENDPOINT`（国内镜像）与 `huggingface-cli login`。
- **HDF5 读取慢**：保持 `num_workers=0`；数据放在本地 NVMe 优于网络盘。
- **与官方 OpenVLA 全流程差异**：官方常用 RLDS + `torchrun`；本脚本直接读项目 HDF5，**动作离散化与 Vicuna 提示格式**与 OpenVLA 仓库一致，便于对接同一套推理代码思路。

更多实现细节见 `train_lora.py` 顶部注释与各模块源码。
