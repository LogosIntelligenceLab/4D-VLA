"""
run_libero_eval_ablation_mb_sft.py

LIBERO eval with a simple memory-bank history selection (MB-SFT).
Intended to be launched by scripts under `vla-scripts/exp_eval/`.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image
from transformers import AutoTokenizer

try:
    import draccus  # type: ignore
except ModuleNotFoundError:
    draccus = None

try:
    import tqdm  # type: ignore
except ModuleNotFoundError:
    tqdm = None

try:
    from libero.libero import benchmark  # type: ignore
except ModuleNotFoundError:
    benchmark = None

try:
    import wandb  # type: ignore
except ModuleNotFoundError:
    wandb = None

import torch

from experiments.robot.memory_bank import SimpleMemoryBank


def _add_repo_root_to_sys_path() -> None:
    this = Path(__file__).resolve()
    for p in (this.parent, *this.parents):
        if (p / ".git").exists() or (p / "pyproject.toml").exists():
            p_str = str(p)
            if p_str not in sys.path:
                sys.path.insert(0, p_str)
            break


_add_repo_root_to_sys_path()

from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
try:
    from experiments.robot.openvla_utils import get_processor  # type: ignore # noqa: E402
except ModuleNotFoundError:
    get_processor = None
from experiments.robot.robot_utils import (  # noqa: E402
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


def _wrap_draccus_or_error():
    if draccus is None:
        def decorator(fn):
            def wrapped(*args, **kwargs):
                raise ModuleNotFoundError(
                    "Missing dependency: draccus. Install it (e.g. `pip install draccus`) "
                    "or install repo requirements."
                )
            return wrapped
        return decorator
    return draccus.wrap()


def _torch_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_bv_depth(depth: torch.Tensor) -> torch.Tensor:
    # Expect per-frame depth as (V=1, H, W); return (B=1, V=1, H, W).
    if depth.dim() == 2:
        depth = depth.unsqueeze(0)  # (1, H, W)
    if depth.dim() != 3:
        raise ValueError(f"Unexpected depth shape: {tuple(depth.shape)} (expected (1,H,W) or (H,W))")
    return depth.unsqueeze(0)


@dataclass
class GenerateConfig:
    # Model
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    center_crop: bool = True
    use_image_prompt: bool = False
    prompt_version: str = "v1"
    use_2view: bool = False
    use_anyview: bool = False
    use_nromaction: bool = False
    use_hist: bool = False
    hist_version: str = "v1"
    hist_len: int = 1
    use_multi_size: bool = False
    all_224: bool = False
    sample_interval: int = 1
    use_action_hist: bool = False
    use_depth: bool = False
    use_proprio: bool = False
    is_ft: bool = False

    task_id: int = 0
    exp_name: str = "debug"

    # LIBERO
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    num_task_for_eval: int = 10

    # Logging
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    use_wandb: bool = False
    wandb_project: str = "YOUR_WANDB_PROJECT"
    wandb_entity: str = "YOUR_WANDB_ENTITY"

    # Repro + MB
    seed: int = 0
    window_size: int = 20
    bank_size: int = 5


@_wrap_draccus_or_error()
def eval_libero(cfg: GenerateConfig) -> None:
    if tqdm is None:
        raise ModuleNotFoundError("Missing dependency: tqdm.")
    if benchmark is None:
        raise ModuleNotFoundError("Missing dependency: libero. Install LIBERO (and its deps) to run simulation eval.")

    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    if cfg.model_family == "openvla":
        assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    set_seed_everywhere(cfg.seed)
    cfg.unnorm_key = cfg.task_suite_name

    model = get_model(cfg)

    processor = None
    if cfg.model_family == "openvla":
        if get_processor is None:
            raise ModuleNotFoundError(
                "Missing dependency for model_family=openvla (likely `prismatic`). "
                "Either install OpenVLA/prismatic deps or use --model_family internvl."
            )
        processor = get_processor(cfg)
    elif cfg.model_family == "internvl":
        processor = AutoTokenizer.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True, use_fast=False)
    else:
        raise NotImplementedError(f"Model family {cfg.model_family} not implemented.")

    if cfg.use_wandb and wandb is None:
        raise ModuleNotFoundError("Missing dependency: wandb (set --use_wandb False or `pip install wandb`).")

    incontext = None
    if cfg.use_image_prompt:
        if cfg.prompt_version == "v1":
            jsonl_file_path = "data/image_prompt.jsonl"
            image_prompt_folder = "data/image_prompt"
        elif cfg.prompt_version == "v2":
            jsonl_file_path = "data/image_prompt_v2.jsonl"
            image_prompt_folder = "data/image_prompt_v2"
        else:
            raise NotImplementedError

        text = None
        with open(jsonl_file_path, "r", encoding="utf-8") as jsonl_file:
            for line in jsonl_file:
                line_data = json.loads(line.strip())
                if cfg.task_suite_name in str(line_data.keys()):
                    text = line_data
                    break
        if text is None:
            raise ValueError(f"Cannot find prompt for task_suite_name={cfg.task_suite_name} in {jsonl_file_path}")

        image = None
        for image_file in os.listdir(image_prompt_folder):
            if cfg.task_suite_name in image_file:
                image = Image.open(os.path.join(image_prompt_folder, image_file)).convert("RGB")
                break
        if image is None:
            raise ValueError(f"Cannot find prompt image for task_suite_name={cfg.task_suite_name} in {image_prompt_folder}")

        incontext = {"text": text, "image": image}

    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    if cfg.use_wandb:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name} ({num_tasks_in_suite} tasks)")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    resize_size = get_image_resize_size(cfg)

    total_episodes, total_successes = 0, 0
    task_id = cfg.task_id
    for task_id in tqdm.tqdm(range(task_id, task_id + 1)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256, seed=cfg.seed)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []

            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400
            else:
                raise ValueError(f"Unknown task_suite_name={cfg.task_suite_name}")

            print(f"Starting episode {task_episodes + 1}...")
            log_file.write(f"Starting episode {task_episodes + 1}...\n")

            action_for_error = np.zeros([7], dtype=np.float32)
            response_for_error = "[5000, 5000, 5000, 5000, 5000, 5000, 1]"

            memory_bank = SimpleMemoryBank(window_size=cfg.window_size, bank_size=cfg.bank_size)
            primed_bank = False
            device = _torch_device()

            while t < max_steps + cfg.num_steps_wait:
                if t < cfg.num_steps_wait:
                    obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                    t += 1
                    continue

                img, wrist, depth, intrinsics, pose = get_libero_image(
                    obs, resize_size, cfg.use_2view, cfg.use_depth, env
                )

                if cfg.use_hist:
                    # Prime bank once to ensure we have enough frames at the beginning.
                    if not primed_bank:
                        for _ in range(cfg.bank_size):
                            memory_bank.add(img[None, :], depth=depth, fid=t, from_img=True)
                        primed_bank = True
                    else:
                        memory_bank.add(img[None, :], depth=depth, fid=t, from_img=True)

                replay_images.append(img)

                observation = {
                    "full_image": img,
                    "wrist_image": wrist if cfg.use_2view else None,
                    "state": np.concatenate(
                        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), [0], obs["robot0_gripper_qpos"])
                    ),
                }

                if cfg.use_proprio:
                    observation["proprio"] = np.concatenate(
                        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), [0], obs["robot0_gripper_qpos"])
                    )

                if cfg.use_image_prompt and incontext is not None:
                    observation["incontext"] = incontext

                if cfg.use_hist:
                    imgs = memory_bank.get_image()
                    if imgs:
                        observation["image_primary_hist"] = np.concatenate(imgs, axis=0)
                        fids = memory_bank.get_fid()
                    else:
                        observation["image_primary_hist"] = img[None, ...]
                        fids = [t]
                    # Support both names: internvl utils historically used `frame_id`.
                    observation["frame_id"] = fids
                    observation["frame_idx"] = torch.tensor([fids], dtype=torch.long, device=device)

                    if cfg.use_depth:
                        depths = memory_bank.get_depth()
                        if not depths:
                            raise RuntimeError("use_depth=True but memory bank has no depth history.")
                        if any(d is None for d in depths):
                            raise RuntimeError("use_depth=True but some depth entries are None in memory bank.")
                        depths_t = [d.to(device) for d in depths]  # each: (1, H, W)
                        # Hist excludes current (last). Pack as (B=1, V_hist, H, W).
                        if len(depths_t) > 1:
                            depth_hist = torch.cat(depths_t[:-1], dim=0)  # (V_hist, H, W)
                            observation["depth_hist"] = depth_hist.unsqueeze(0)  # (1, V_hist, H, W)
                            observation["depth_mask_hist"] = torch.ones_like(observation["depth_hist"])
                        else:
                            observation["depth_hist"] = torch.zeros((1, 0, depths_t[0].shape[-2], depths_t[0].shape[-1]), device=device)
                            observation["depth_mask_hist"] = torch.zeros_like(observation["depth_hist"])

                if cfg.use_depth:
                    if depth is None or intrinsics is None or pose is None:
                        raise RuntimeError("use_depth=True but LIBERO image extraction returned missing depth/intrinsics/pose.")
                    depth_bv = _ensure_bv_depth(depth).to(device)  # (1, 1, H, W)
                    observation["depth"] = depth_bv
                    observation["depth_mask"] = torch.ones_like(depth_bv)
                    observation["intrinsics"] = intrinsics.to(device)
                    observation["pose"] = pose.to(device)

                try:
                    with torch.no_grad():
                        action, response = get_action(
                            cfg,
                            model,
                            observation,
                            task_description,
                            processor=processor,
                            t=t,
                            ablation=True,
                            is_ft=cfg.is_ft,
                            task_suite_name=cfg.task_suite_name,
                        )
                except Exception:
                    action = action_for_error
                    response = response_for_error

                action_for_error = action
                response_for_error = response

                action = normalize_gripper_action(action, binarize=True)
                if cfg.model_family == "openvla":
                    action = invert_gripper_action(action)
                elif cfg.model_family == "internvl":
                    if not cfg.is_ft:
                        action = invert_gripper_action(action)
                else:
                    raise NotImplementedError(f"Model family {cfg.model_family} not implemented.")

                obs, reward, done, info = env.step(action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1

            task_episodes += 1
            total_episodes += 1

            save_rollout_video(
                replay_images,
                total_episodes,
                success=done,
                task_description=task_description,
                log_file=log_file,
                steps=t,
                exp_name=cfg.exp_name,
            )

            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()

        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )

    log_file.close()
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()
