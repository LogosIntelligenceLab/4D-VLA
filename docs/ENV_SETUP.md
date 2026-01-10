# Environment Setup

This repo uses two separate environments:

- **Training / finetune / (non-simulator) eval**: `envs/4d-vla-train.yaml`
- **RLDS → jsonl preprocessing (TFDS/dlimp)**: `envs/4d-vla-rlds.yaml`

If you only want to run training/finetuning, you only need the **train** env.

## 0) Prerequisites

- Linux recommended.
- `conda`/`mamba` installed.
- For GPU training: a working NVIDIA driver + a PyTorch wheel matching your CUDA runtime.

## 1) Train / Finetune Environment (`4d-vla-train`)

Create and activate:

```bash
conda env create -f envs/4d-vla-train.yaml
conda activate 4d-vla-train
```

Install PyTorch **for your CUDA** (do this after activating the env):

```bash
# Pick the right command from https://pytorch.org/get-started/locally/
pip install --upgrade pip
python -c "import torch; print(torch.__version__)"
```

If you see `OSError: [Errno 18] Invalid cross-device link` during installs, pin pip temp/cache to this repo:

```bash
export TMPDIR="$PWD/.tmp" PIP_CACHE_DIR="$PWD/.pip-cache"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
```

The conda env file already installs the project, but if you need to re-install:

```bash
pip install -r internvl_chat/requirements.txt
pip install -e ./internvl_chat
```

Sanity checks:

```bash
python -c "import torch; import internvl; import transformers; print('ok')"
```

## 2) RLDS / TFDS Preprocessing Environment (`4d-vla-rlds`)

This is for scripts like `internvl_chat/tools/prepare_libero_memory_bank_mb.py` which rely on TFDS/RLDS/dlimp.

Create and activate:

```bash
conda env create -f envs/4d-vla-rlds.yaml
conda activate 4d-vla-rlds
```

Notes:

- The env pins `numpy==1.26.4` because `tensorflow-cpu==2.15.0` requires `numpy<2`.

Sanity checks:

```bash
python -c "import tensorflow as tf; import tensorflow_datasets as tfds; print(tf.__version__, tfds.__version__)"
python internvl_chat/tools/prepare_libero_memory_bank_mb.py --help
```

## 3) (Optional) LIBERO Simulation Eval Dependencies

To run `experiments/robot/libero/run_libero_eval_ablation.py`, you need extra simulator deps:

```bash
pip install libero robosuite
```

LIBERO/robosuite usually also require MuJoCo and a working OpenGL setup (headless rendering configs vary by machine).
If these deps are missing, the eval code will fail fast with a clear error at import/runtime.

## Troubleshooting

- **`bitsandbytes` / CUDA errors**: make sure your PyTorch CUDA wheel matches your driver/CUDA runtime; `bitsandbytes==0.41.0` is GPU-oriented.
- **`deepspeed` build/import issues**: ensure `ninja`/`cmake` are available (included in `4d-vla-train.yaml`) and that your CUDA toolchain is configured for your system.
