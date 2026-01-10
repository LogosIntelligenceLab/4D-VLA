"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import os
import sys
import json
from PIL import Image
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
from transformers import AutoTokenizer

try:
    import draccus  # type: ignore
except ModuleNotFoundError:
    draccus = None
import numpy as np
import tqdm
try:
    from libero.libero import benchmark  # type: ignore
except ModuleNotFoundError:
    benchmark = None
from collections import deque
from scipy.spatial.transform import Rotation as R
import time
try:
    import wandb  # type: ignore
except ModuleNotFoundError:
    wandb = None

try:
    from robosuite.utils.camera_utils import (  # type: ignore
        get_camera_extrinsic_matrix,
        get_camera_intrinsic_matrix,
        CameraMover,
    )
except ModuleNotFoundError:
    get_camera_extrinsic_matrix = None
    get_camera_intrinsic_matrix = None
    CameraMover = None


# Append current directory so that interpreter can find experiments.robot
def _add_repo_root_to_sys_path() -> None:
    this = Path(__file__).resolve()
    for p in (this.parent, *this.parents):
        if (p / ".git").exists() or (p / "pyproject.toml").exists():
            p_str = str(p)
            if p_str not in sys.path:
                sys.path.insert(0, p_str)
            break


_add_repo_root_to_sys_path()
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
# from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
import torch


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    use_image_prompt: bool = False                   # 
    prompt_version: str = 'v1'
    use_2view: bool = False                          # Use 2-view input
    use_anyview: bool = False                        # Use anyview input
    use_nromaction: bool = False                     # Use normalized action
    use_hist: bool = False                           # Use history
    hist_version: str = 'v1'
    hist_len: int = 1                                # History length
    use_multi_size: bool = False                     # Use multi-size input
    all_224: bool = False                            # Use all 224 input
    sample_interval: int = 1                         # Sample interval
    use_action_hist: bool = False                    # Use action history
    use_depth: bool = False                          # Use depth image
    use_proprio: bool = False                        # Use proprioception
    is_ft: bool = False                              # Is fine-tuning

    task_id: int = 0
    exp_name: str = "debug"                         # Experiment name


    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    num_task_for_eval: int = 10                      # Number of tasks to evaluate

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 0                                    # Random Seed (for reproducibility)

    # fmt: on


def set_new_pose(camera_mover, env, new_pos, new_rot):
    new_quat = get_new_quat(new_rot[0], new_rot[1], new_rot[2])
    camera_mover.set_camera_pose(pos=new_pos, quat=new_quat)
    for _ in range(10):
        obs, reward, done, info = env.step(get_libero_dummy_action("llava"))

def get_new_quat(x: int, y: int, z: int):
    '''
    input: x, y, z in degrees
    output: quaternion
    '''
    euler_angles = [x, y, z]

    # Convert the Euler angles to a quaternion
    rotation = R.from_euler('xyz', euler_angles, degrees=True)
    quaternion = rotation.as_quat()
    return quaternion

def check_the_move(camera_mover, env, new_pos, new_rot, debug=True, idx=0, task_name="spatial"):
    new_quat = get_new_quat(new_rot[0], new_rot[1], new_rot[2])
    camera_mover.set_camera_pose(pos=new_pos, quat=new_quat)
    obs, reward, done, info = env.step(get_libero_dummy_action("llava"))
    agent_view = obs["agentview_image"][::-1]
    agent_view = Image.fromarray(agent_view)
    if debug:
        agent_view.save(f"change_pose_{task_name}/agent_view_newpose_debug.png")
    else:
        os.makedirs("change_pose_debug", exist_ok=True)
        agent_view.save(f"change_pose_{task_name}/agent_view_newpose{idx}_pos{new_pos}_rot{new_rot}.png")

def _wrap_draccus_or_error():
    if draccus is None:
        def decorator(fn):
            def wrapped(*args, **kwargs):
                raise ModuleNotFoundError(
                    "Missing dependency: draccus. Install it (e.g. `pip install draccus`) "
                    "or install repo requirements (`pip install -r internvl_chat/requirements.txt`)."
                )
            return wrapped
        return decorator
    return draccus.wrap()

@_wrap_draccus_or_error()
def eval_libero(cfg: GenerateConfig) -> None:
    if benchmark is None:
        raise ModuleNotFoundError(
            "Missing dependency: libero. Install LIBERO (and its deps) to run simulation eval."
        )
    if CameraMover is None:
        raise ModuleNotFoundError(
            "Missing dependency: robosuite. Install it to run LIBERO simulation eval (e.g. `pip install robosuite`)."
        )
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    if cfg.model_family == "openvla":
        assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
    elif cfg.model_family == "internvl":
        processor = AutoTokenizer.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True, use_fast=False)
    else:
        raise NotImplementedError(f"Model family {cfg.model_family} not implemented.")

    if cfg.use_wandb and wandb is None:
        raise ModuleNotFoundError("Missing dependency: wandb (set --use_wandb False or `pip install wandb`).")
    
    if cfg.use_image_prompt:
        if cfg.prompt_version == 'v1':
            jsonl_file_path = 'data/image_prompt.jsonl'
            image_prompt_folder = 'data/image_prompt'
        elif cfg.prompt_version == 'v2':
            jsonl_file_path = 'data/image_prompt_v2.jsonl'
            image_prompt_folder = 'data/image_prompt_v2'
        else:
            raise NotImplementedError
        with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file:
            for line in jsonl_file:
                line_data = json.loads(line.strip())
                if cfg.task_suite_name in str(line_data.keys()):
                    text = line_data
                    break

        for image_file in os.listdir(image_prompt_folder):
            if cfg.task_suite_name in image_file:
                image = Image.open(os.path.join(image_prompt_folder, image_file)).convert('RGB')
                break
        incontext = {'text': text, 'image': image}


    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    # assert cfg.num_task_for_eval <= num_tasks_in_suite, "Number of tasks to evaluate exceeds total number of tasks!"
    task_id = cfg.task_id
    for task_id in tqdm.tqdm(range(task_id, task_id+1)):
        # if task_id < 3:
        #     continue
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256, seed=cfg.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])


            # Initialize history for history-based models
            if cfg.use_hist:
                image_primary_hist = deque(maxlen=cfg.hist_len)
                action_hist = deque(maxlen=cfg.hist_len-1)
                proprio_hist = deque(maxlen=cfg.hist_len)
                depth_hist = deque(maxlen=cfg.hist_len)
                depth_mask_hist = deque(maxlen=cfg.hist_len)


            # Setup
            t = 0
            replay_images = []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            action_for_error = np.zeros([7])
            response_for_error = '[5000, 5000, 5000, 5000, 5000, 5000, 1]'

            while t < max_steps + cfg.num_steps_wait:
                # try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < cfg.num_steps_wait:
                    obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                    t += 1
                    continue

                # Get preprocessed image
                img, wrist, depth, intrinsics, pose = get_libero_image(obs, resize_size, cfg.use_2view, cfg.use_depth, env)
                if cfg.use_hist:
                    if (t == cfg.num_steps_wait):  # [0 0] 0
                        for _ in range(cfg.hist_len):
                            image_primary_hist.append(img)
                            proprio_hist.append(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]))
                            action_hist.append([0, 0, 0, 0, 0, 0, 0])
                            depth_hist.append(np.zeros_like(depth))
                            depth_mask_hist.append(np.zeros_like(depth))

                        priopro = np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), [0], obs["robot0_gripper_qpos"])
                        )
                        proprio_hist.append(priopro)
                        depth_hist.append(depth)
                        depth_mask_hist.append(np.ones_like(depth))

                    else:
                        image_primary_hist.append(img)  # [0 0] 1, [0 1] 2
                        priopro = np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), [0], obs["robot0_gripper_qpos"])
                        )
                        proprio_hist.append(priopro)  # [0 0] 1, [0 1] 2
                        depth_hist.append(depth)
                        depth_mask_hist.append(np.ones_like(depth))

                # Save preprocessed image for replay video
                replay_images.append(img)

                # Prepare observations dict
                # Note: OpenVLA does not take proprio state as input
                observation = {
                    "full_image": img,
                    "wrist_image": wrist if cfg.use_2view else None,
                    "state": np.concatenate(
                        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), [0], obs["robot0_gripper_qpos"])
                    ),
                }

                if cfg.use_proprio:
                    priopro = np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), [0], obs["robot0_gripper_qpos"])
                    )
                    observation['proprio'] = priopro

                if cfg.use_image_prompt:
                    observation['incontext'] = incontext

                if cfg.use_hist:
                    observation['image_primary_hist'] = np.array(image_primary_hist)
                    observation['action_hist'] = np.array(action_hist)
                    observation['proprio_hist'] = np.stack(proprio_hist, axis=0)
                    observation['depth_hist'] = np.array(depth_hist)[0:9]

                if cfg.use_depth:
                    observation['depth'] = depth
                    observation['depth_mask'] = np.ones_like(depth)
                    observation['intrinsics'] = intrinsics
                    observation['pose'] = pose
                    if cfg.use_hist:
                        observation['depth_mask'] = np.array(depth_mask_hist)[9:10]
                        observation['depth_mask_hist'] = np.array(depth_mask_hist)[:9]

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

                action_for_error = action
                response_for_error = response

                if cfg.use_hist:
                    if (t >= cfg.num_steps_wait):
                        action_hist.append(action)

                # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                action = normalize_gripper_action(action, binarize=True)

                # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                if cfg.model_family == "openvla":
                    action = invert_gripper_action(action)
                elif cfg.model_family == "internvl":
                    if (not cfg.is_ft):
                        action = invert_gripper_action(action)
                    else:
                        pass
                else:
                    raise NotImplementedError(f"Model family {cfg.model_family} not implemented.")

                # Execute action in environment
                obs, reward, done, info = env.step(action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1

                # except Exception as e:
                #     print(f"Caught exception: {e}")
                #     log_file.write(f"Caught exception: {e}\n")
                #     break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file, steps=t, exp_name=cfg.exp_name
            )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # Log final results
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

    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
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
