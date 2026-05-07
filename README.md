# AIR 5051 (2026) — Team 17 Final Project

**Course**: AIR 5051 (2026)  
**Team**: Team 17  
**Project**: Pouring VLA — Vision-Language-Action Model for Robotic Pouring

---

## Project Overview

This project develops a **Vision-Language-Action (VLA)** system for autonomous liquid pouring with a robotic arm. We fine-tune **OpenVLA** with LoRA on a custom pouring dataset, and deploy the optimized model on a **Qualcomm QCS9100** edge device for real-time inference.

### Key Contributions
- Custom RLDS-format dataset collection via teleoperation (Dobot Nova5 arm)
- LoRA fine-tuning of OpenVLA on the pouring task
- Edge deployment pipeline: model quantization and inference on Qualcomm QCS9100 (two approaches: QNN and QAIRT DLC)

---

## Repository Structure

```
├── src/                          # Source code
│   ├── data_collection/          # Teleoperation data collection scripts
│   ├── dataset_builder/          # RLDS dataset builder (OpenX format)
│   ├── lora_training/            # OpenVLA LoRA fine-tuning
│   ├── inference/                # PC-side VLA inference
│   ├── openvla/                  # OpenVLA framework (with our modifications)
│   └── tools/
│       ├── gripper/              # Custom gripper firmware & controller
│       ├── teleop/               # Gamepad teleoperation module
│       └── vla/                  # VLA data processing utilities
│
├── Qualcomm_device_code/         # Edge device deployment
│   ├── v1_qnn/                   # V1: QNN + llama.cpp approach
│   └── v2_qairt/                 # V2: QAIRT DLC approach (QCS9100)
│
└── docs/                         # Documentation
    ├── Team17_FinalReport.pdf    # Final report
    ├── Proposal.pdf              # Project proposal
    ├── data_collection_guide.md  # Data collection guide
    └── real_vla_inference_guide.md  # Inference deployment guide
```

---

## Quick Start

### 1. Data Collection

```bash
# On the robot server side
python src/data_collection/real_teleop_server.py

# On the operator side
python src/data_collection/real_teleop_collect.py
```

### 2. LoRA Fine-tuning

Fine-tuning uses the OpenVLA official `finetune.py` script with `torchrun` for multi-GPU support.

```bash
# Install dependencies
pip install peft==0.11.1 draccus accelerate

# Single GPU
torchrun --standalone --nnodes 1 --nproc-per-node 1 \
  src/openvla/vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /path/to/rlds_dataset \
  --dataset_name dobot_pouring \
  --run_root_dir runs/ \
  --adapter_tmp_dir adapter-tmp/ \
  --lora_rank 32 \
  --batch_size 4 \
  --grad_accumulation_steps 4 \
  --learning_rate 4e-4 \
  --image_aug True \
  --wandb_project openvla-pouring-v1 \
  --wandb_entity <your-wandb-entity> \
  --save_steps 1000 \
  --max_steps 16000
```

Key parameters:
| Parameter | Value | Description |
|-----------|-------|-------------|
| `--lora_rank` | 32 | LoRA rank |
| `--batch_size` | 4 | Per-GPU batch size |
| `--grad_accumulation_steps` | 4 | Effective batch size = 16 |
| `--learning_rate` | 4e-4 | AdamW learning rate |
| `--max_steps` | 16000 | Total training steps |

### 3. PC-side Inference

```bash
python src/inference/real_openvla_lora_infer.py
```

### 4. Qualcomm Edge Deployment

See [`Qualcomm_device_code/v2_qairt/README.md`](Qualcomm_device_code/v2_qairt/README.md) for the full quantization and deployment pipeline on QCS9100.

---

## Hardware

- **Robot Arm**: Dobot Nova5
- **Edge Device**: Qualcomm QCS9100 (Snapdragon 8cx Gen 3)
- **Gripper**: Custom STC8G-based gripper with Pico2W wireless control
- **Camera**: RGB-D camera for visual input

---

## Dependencies

- Python 3.10+
- PyTorch 2.x
- OpenVLA (see `src/openvla/`)
- Qualcomm AI Runtime (QAIRT) SDK v2.42

---

## License

See individual component licenses. OpenVLA framework follows its original license in `src/openvla/LICENSE`.
