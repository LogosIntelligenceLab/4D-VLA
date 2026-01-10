# Evaluation scripts

These scripts are thin launchers for running LIBERO evaluation from the repo root.

## Quick start

```bash
# Run MB-SFT LIBERO-10 eval for seeds 0/1/2
bash vla-scripts/exp_eval/eval_whole_mb.sh
```

## Common overrides

```bash
# Override checkpoint path used by vla-scripts/exp_eval/sft_mb_paral_long_seed.sh
export PRETRAINED_CHECKPOINT=/path/to/your/checkpoint_dir

# Optional: customize exp_name suffix used for logs/videos
export EXP_NAME_SUFFIX=my_exp
```

## Dependencies

- Python deps: `draccus`, `tqdm`, `torch`, `torchvision`, `transformers`
- Simulation deps: `libero`, `robosuite`

