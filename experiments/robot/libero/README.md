# LIBERO Eval (4D-VLA)

This folder contains LIBERO simulation evaluation entrypoints and scripts.

## Entrypoint

- `experiments/robot/libero/run_libero_eval_ablation.py`

## Scripts

- Spatial suite (parallel): `experiments/robot/libero/scripts/eval_ablation_exp4_parallel_spatial_seed.sh`
- Goal suite (parallel): `experiments/robot/libero/scripts/eval_ablation_exp4_parallel_goal_seed.sh`

## Usage

```bash
PRETRAINED_CHECKPOINT=/path/to/checkpoint \
  bash experiments/robot/libero/scripts/eval_ablation_exp4_parallel_goal_seed.sh 0
```

Common env overrides:

- `GPUS=0,1,2`
- `TASK_IDS=0,1,2,3,4,5,6,7,8,9`
- `LOG_DIR=experiments/logs/libero_goal`

## Dependencies

Eval requires external simulation dependencies:

- `libero`
- `robosuite`

If missing, the code will raise a clear `ModuleNotFoundError` when you start eval.

