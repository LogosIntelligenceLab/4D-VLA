import json
import os
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

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

def get_vla(cfg):
    """Loads and returns a VLA model from checkpoint."""
    print("[*] Instantiating Pretrained VLA model")
    print("[*] Loading in BF16 with Flash-Attention Enabled")

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(DEVICE)

    dataset_statistics_path = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )

    return vla

def get_processor(cfg):
    """Get VLA model's Hugging Face processor."""
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)
    return processor

def crop_and_resize(image, crop_scale):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use PyTorch transformations to implement this.

    Args:
        image: PIL Image or PyTorch Tensor of shape (H, W, C) with values between [0, 1].
        crop_scale: The area of the center crop with respect to the original image.
    """
    # Convert to PyTorch tensor if needed
    if not isinstance(image, torch.Tensor):
        image = transforms.ToTensor()(image)

    # Get original dimensions
    _, original_height, original_width = image.shape

    # Calculate crop dimensions
    crop_height = int(original_height * torch.sqrt(torch.tensor(crop_scale)))
    crop_width = int(original_width * torch.sqrt(torch.tensor(crop_scale)))

    # Center crop
    transform_crop = transforms.CenterCrop((crop_height, crop_width))
    cropped_image = transform_crop(image)

    # Resize back to original size
    transform_resize = transforms.Resize((original_height, original_width))
    resized_image = transform_resize(cropped_image)

    return resized_image

def get_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False):
    """Generates an action with the VLA policy."""
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    if center_crop:
        crop_scale = 0.9
        image = crop_and_resize(image, crop_scale)
        image = transforms.ToPILImage()(image)

    if "openvla-v01" in base_vla_name:
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    else:
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

    action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    return action
