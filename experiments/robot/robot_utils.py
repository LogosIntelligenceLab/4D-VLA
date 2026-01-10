"""Utils for evaluating robot policies in various environments."""

import os
import random
import time

import numpy as np
import torch

import torchvision.transforms as T

try:
    from experiments.robot.openvla_utils import (  # type: ignore
        get_vla,
        get_vla_action,
    )
except ModuleNotFoundError:
    get_vla = None
    get_vla_action = None
from experiments.robot.internvl_utils import (
    get_vla_internvl,
    get_vla_action_internvl,
)
from experiments.robot.internvl_utils_ablation import (
    # get_vla_internvl,
    get_vla_action_internvl_ablation,
)
from experiments.robot.internvl_utils_final import (
    # get_vla_internvl,
    get_vla_action_internvl_final,
)


# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def set_seed_everywhere(seed: int):
    """Sets the random seed for Python, NumPy, and PyTorch functions."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_model(cfg, wrap_diffusion_policy_for_droid=False, debug=False):
    if debug:
        return None
    """Load model for evaluation."""
    if cfg.model_family == "openvla":
        if get_vla is None:
            raise ModuleNotFoundError(
                "Missing dependency for model_family=openvla (likely `prismatic`). "
                "Either install OpenVLA/prismatic deps or use --model_family internvl."
            )
        model = get_vla(cfg)
    elif cfg.model_family == "internvl":
        model = get_vla_internvl(cfg)
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    print(f"Loaded model: {type(model)}")
    return model


def get_image_resize_size(cfg):
    """
    Gets image resize size for a model class.
    If `resize_size` is an int, then the resized image will be a square.
    Else, the image will be a rectangle.
    """
    if cfg.model_family == "openvla":
        resize_size = 224
    elif cfg.model_family == "internvl":
        resize_size = 448
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return resize_size


def get_action(cfg, model, obs, task_label, processor=None, t=0, ablation=False, is_ft=False, task_suite_name=None, is_final=False):
    """Queries the model to get an action."""
    if cfg.model_family == "openvla":
        if get_vla_action is None:
            raise ModuleNotFoundError(
                "Missing dependency for model_family=openvla (likely `prismatic`). "
                "Either install OpenVLA/prismatic deps or use --model_family internvl."
            )
        action = get_vla_action(
            model, processor, cfg.pretrained_checkpoint, obs, task_label, cfg.unnorm_key, center_crop=cfg.center_crop
        )
        assert action.shape == (ACTION_DIM,)
        response = None
    elif cfg.model_family == "internvl":
        if ablation:
            action, response = get_vla_action_internvl_ablation(model, 
                                         processor, 
                                         obs, 
                                         task_label, 
                                         center_crop=cfg.center_crop, 
                                         use_image_prompt=cfg.use_image_prompt, 
                                         prompt_version=cfg.prompt_version,
                                         use_anyview=cfg.use_anyview,
                                         use_nromaction=cfg.use_nromaction,
                                         use_2view=cfg.use_2view,
                                         use_hist=cfg.use_hist,
                                         hist_version=cfg.hist_version,
                                         hist_len=cfg.hist_len,
                                         use_multi_size=cfg.use_multi_size,
                                         all_224=cfg.all_224,
                                         sample_interval=cfg.sample_interval,
                                         use_action_hist=cfg.use_action_hist,
                                         use_depth=cfg.use_depth,
                                         use_proprio=cfg.use_proprio,
                                         is_ft=is_ft,
                                         t=t,
                                         task_suite_name=task_suite_name)
        elif is_final:
            action, response = get_vla_action_internvl_final(model, 
                                         processor, 
                                         obs, 
                                         task_label, 
                                         center_crop=cfg.center_crop, 
                                         use_image_prompt=cfg.use_image_prompt, 
                                         prompt_version=cfg.prompt_version,
                                         use_anyview=cfg.use_anyview,
                                         use_nromaction=cfg.use_nromaction,
                                         use_2view=cfg.use_2view,
                                         use_hist=cfg.use_hist,
                                         hist_version=cfg.hist_version,
                                         hist_len=cfg.hist_len,
                                         use_multi_size=cfg.use_multi_size,
                                         all_224=cfg.all_224,
                                         sample_interval=cfg.sample_interval,
                                         use_action_hist=cfg.use_action_hist,
                                         use_depth=cfg.use_depth,
                                         use_proprio=cfg.use_proprio,
                                         is_ft=is_ft,
                                         t=t,
                                         task_suite_name=task_suite_name)
        else:
            action, response = get_vla_action_internvl(model, 
                                            processor, 
                                            obs, 
                                            task_label, 
                                            center_crop=cfg.center_crop, 
                                            use_image_prompt=cfg.use_image_prompt, 
                                            prompt_version=cfg.prompt_version,
                                            use_anyview=cfg.use_anyview,
                                            use_nromaction=cfg.use_nromaction,
                                            use_2view=cfg.use_2view,
                                            use_hist=cfg.use_hist,
                                            hist_version=cfg.hist_version,
                                            hist_len=cfg.hist_len,
                                            use_multi_size=cfg.use_multi_size,
                                            all_224=cfg.all_224,
                                            sample_interval=cfg.sample_interval,
                                            use_action_hist=cfg.use_action_hist,
                                            use_depth=cfg.use_depth,
                                            use_proprio=cfg.use_proprio,
                                            t=t)
        assert action.shape == (ACTION_DIM,)
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return action, response


def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])

    return action


def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action
