# InternVL-Chat

This folder contains the implementation of the InternVL-Chat.

## 🛠️ Installation

See [INSTALLATION.md](../INSTALLATION.md)

In addition, using this codebase requires executing the following steps:

- Install other requirements:

  ```bash
  pip install --upgrade pip  # enable PEP 660 support
  pip install -e .
  ```

## 📖 Documents

- InternVL 2.0

  - Introduction [\[link\]](https://internvl.readthedocs.io/en/latest/internvl2.0/introduction.html)
  - Quick Start [\[link\]](https://internvl.readthedocs.io/en/latest/internvl2.0/quick_start.html)
  - Finetune [\[link\]](https://internvl.readthedocs.io/en/latest/internvl2.0/finetune.html)
  - Evaluation [\[link\]](https://internvl.readthedocs.io/en/latest/internvl2.0/evaluation.html)
  - Deployment [\[link\]](https://internvl.readthedocs.io/en/latest/internvl2.0/deployment.html)

- InternVL 1.5

  - Introduction [\[link\]](https://internvl.readthedocs.io/en/latest/internvl1.5/introduction.html)
  - Quick Start [\[link\]](https://internvl.readthedocs.io/en/latest/internvl1.5/quick_start.html)
  - Finetune [\[link\]](https://internvl.readthedocs.io/en/latest/internvl1.5/finetune.html)
  - Evaluation [\[link\]](https://internvl.readthedocs.io/en/latest/internvl1.5/evaluation.html)
  - Deployment [\[link\]](https://internvl.readthedocs.io/en/latest/internvl1.5/deployment.html)

- InternVL 1.2

  - Introduction [\[link\]](https://internvl.readthedocs.io/en/latest/internvl1.2/introduction.html)
  - Quick Start [\[link\]](https://internvl.readthedocs.io/en/latest/internvl1.2/quick_start.html)
  - Reproduce [\[link\]](https://internvl.readthedocs.io/en/latest/internvl1.2/reproduce.html)
  - Finetune [\[link\]](https://internvl.readthedocs.io/en/latest/internvl1.2/finetune.html)
  - Evaluation [\[link\]](https://internvl.readthedocs.io/en/latest/internvl1.2/evaluation.html)

- InternVL 1.1

  - Introduction [\[link\]](https://internvl.readthedocs.io/en/latest/internvl1.1/introduction.html)
  - Quick Start [\[link\]](https://internvl.readthedocs.io/en/latest/internvl1.1/quick_start.html)
  - Evaluation [\[link\]](https://internvl.readthedocs.io/en/latest/internvl1.1/evaluation.html)

## 4D-VLA: Pretrain

This repo adds a robot/VLA pretrain entrypoint + launcher:

```bash
bash internvl_chat/run_pretrain.sh
```

Common overrides (examples):

```bash
GPUS=8 PER_DEVICE_BATCH_SIZE=32 GLOBAL_BATCH_SIZE=512 \
MODEL_NAME_OR_PATH=../hugg_models/InternVL2-4B \
META_PATH=shell/exp_data/droid_pretrain_mb_20ws_5bs_simple.json \
OUTPUT_DIR=work_dirs/4dvla/pretrain_droid_mb_20ws_5bs \
bash internvl_chat/run_pretrain.sh
```
