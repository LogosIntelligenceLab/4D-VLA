import gc
import json
import logging
import math
import os
import random
import sys
import traceback
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Optional
from scipy.spatial.transform import Rotation as R
import cv2

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import transformers
from internvl.dist_utils import init_dist
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from internvl.model.internvl_chat import (InternVisionConfig,
                                          InternVisionModel,
                                          InternVLChatConfig,
                                          InternVLChatModel3D,
                                          ActionHead_p1,
                                          SpatialAwareModule,
                                          InternVLChatModel3DProprioFTRPECat,
                                          ProprioHead,
                                          InternVLChatModel3DProprioRPECat
                                          )
from internvl.patch import (concat_pad_data_collator,
                            replace_llama_rmsnorm_with_fused_rmsnorm,
                            replace_train_sampler)
from internvl.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                      IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                      IMG_START_TOKEN, QUAD_END_TOKEN,
                                      QUAD_START_TOKEN, REF_END_TOKEN,
                                      REF_START_TOKEN, ACT_START_TOKEN,
                                      ACT_END_TOKEN, STATE_PRED,
                                      PROP_START_TOKEN, PROP_END_TOKEN, PROP_CONTEXT_TOKEN,
                                      TIME_EMBED_TOKEN)
from internvl.train.dataset import (ConcatDataset, TCSLoader,
                                    WeightedConcatDataset, build_transform,
                                    dynamic_preprocess, preprocess,
                                    # preprocess_internlm, 
                                    preprocess_mpt)
                                    # preprocess_phi3)
from internvl.conversation import get_conv_template
from internvl.train.trainer_monkey_patch import replace_create_optimizer
from PIL import Image, ImageFile, PngImagePlugin, UnidentifiedImageError
from torch.utils.data import Dataset
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          HfArgumentParser, Trainer, TrainingArguments,
                          set_seed)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import (enable_default_handler,
                                        enable_explicit_format, set_verbosity)

from internvl.train.vla_trainer import VLATrainer

# Apply necessary patches for the transformers library
replace_llama_rmsnorm_with_fused_rmsnorm()
replace_train_sampler()

# Try to import petrel_client for image loading, fallback to PIL if unavailable
try:
    from petrel_client.client import Client
    from petrel_client.common.config import Config
    has_tcs_loader = True
except ImportError as E:
    print('petrel_client is not installed. Using PIL to load images.')
    has_tcs_loader = False

# Set constants for image processing and logging
IGNORE_INDEX = -100
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

def _maybe_apply_droid_path_prefix(data_item: dict) -> None:
    """
    DROID-style jsonl stores per-sample `path_prefix` and image/depth basenames, e.g.:
      - path_prefix: ".../images/camera_ext1/left"
      - image: ["left_image_2.jpg", ..., "left_image_step_2.jpg"]
      - image_depth: ["depth_image_2.png", ..., "depth_image_step_2.png"]

    Pretrain normalizes these to:
      - image: join(path_prefix, name) and ensure "step_" exists in basename
      - image_depth: join(path_prefix.replace('left','depth'), name) and ensure "step_" exists
    """
    if not isinstance(data_item, dict):
        return
    prefix = data_item.get("path_prefix")
    images = data_item.get("image")
    depths = data_item.get("image_depth")
    if not prefix or not isinstance(images, list) or not isinstance(depths, list):
        return

    data_item["image"] = [os.path.join(prefix, d) for d in images]
    depth_prefix = str(prefix).replace("left", "depth")
    data_item["image_depth"] = [os.path.join(depth_prefix, d) for d in depths]

    new_images = []
    new_depths = []
    for img_path, dep_path in zip(data_item["image"], data_item["image_depth"]):
        img_dir, img_name = os.path.split(img_path)
        dep_dir, dep_name = os.path.split(dep_path)
        if "step" not in img_name:
            try:
                fid = int(os.path.splitext(img_name)[0].split("_")[-1])
            except Exception:
                fid = None
            if fid is not None:
                img_name = img_name.replace(str(fid), f"step_{fid}")
                dep_name = dep_name.replace(str(fid), f"step_{fid}")
        new_images.append(os.path.join(img_dir, img_name))
        new_depths.append(os.path.join(dep_dir, dep_name))
    data_item["image"] = new_images
    data_item["image_depth"] = new_depths


def _frame_id_list_from_item(data_item: dict, num_image: int) -> list[int]:
    frame_id = data_item.get("frame_id")
    if isinstance(frame_id, list) and len(frame_id) == num_image and all(isinstance(x, (int, float)) for x in frame_id):
        return [int(x) for x in frame_id]

    def _parse_one(p: str) -> int:
        base = os.path.basename(p)
        stem = os.path.splitext(base)[0]
        # common patterns: fid123.png / left_image_step_123.jpg / *_123
        if "fid" in stem:
            try:
                return int(stem.split("fid")[-1])
            except Exception:
                pass
        if "step_" in stem:
            try:
                return int(stem.split("step_")[-1])
            except Exception:
                pass
        try:
            return int(stem.split("_")[-1])
        except Exception:
            raise ValueError(f"cannot parse frame_id from image name: {base}")

    images = data_item.get("image")
    if not isinstance(images, list) or len(images) != num_image:
        return list(range(num_image))
    return [_parse_one(p) for p in images]


def _pose_and_intrinsics_from_camera_info(camera_info: dict, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Supports both:
      - DROID: camera_extrinsic / camera_intrinsic
      - Libero: agentview_extrinsic / agentview_intrinsic or fovy-based intrinsics
    Returns (pose_4x4, K_4x4).
    """
    if not isinstance(camera_info, dict):
        raise ValueError("camera_info must be a dict")

    if "camera_extrinsic" in camera_info and "camera_intrinsic" in camera_info:
        pose = np.asarray(camera_info["camera_extrinsic"])
        K = np.eye(4, dtype=np.float32)
        K[:3, :3] = np.asarray(camera_info["camera_intrinsic"])
        return pose, K

    if "agentview_extrinsic" in camera_info:
        pose = np.asarray(camera_info["agentview_extrinsic"])
        if "agentview_intrinsic" in camera_info:
            K = np.eye(4, dtype=np.float32)
            K[:3, :3] = np.asarray(camera_info["agentview_intrinsic"])
            return pose, K

        # fallback: compute from fovy (kept for backward-compat)
        fovy = camera_info.get("fovy")
        if isinstance(fovy, list) and fovy:
            fovy = fovy[0]
        if fovy is None:
            raise KeyError("camera_info missing agentview_intrinsic and fovy")

        width, height = image_size, image_size
        fovy_rad = np.deg2rad(float(fovy))
        fy = height / (2 * np.tan(fovy_rad / 2))
        fx = fy * (width / height)
        cx, cy = width / 2, height / 2
        K = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        return pose, K

    raise KeyError("camera_info missing supported extrinsic/intrinsic keys")


def _num_vit_tokens_from_chw(
    pixel_chw: torch.Tensor,
    *,
    patch_size: int,
    down_sample_ratio: float,
) -> int:
    """
    Return the number of image tokens produced by `InternVLChatModel*.extract_feature(...)` for a single image.

    This matches the actual token count after:
      - ViT patchify (H/patch, W/patch)
      - pixel_shuffle with `scale_factor=down_sample_ratio` (typically 0.5)
    """
    if pixel_chw.ndim != 3:
        raise ValueError(f"expected CHW tensor, got shape={tuple(pixel_chw.shape)}")
    _, h, w = pixel_chw.shape
    if patch_size <= 0:
        raise ValueError(f"patch_size must be > 0, got {patch_size}")
    if down_sample_ratio <= 0:
        raise ValueError(f"down_sample_ratio must be > 0, got {down_sample_ratio}")

    grid_h = h // patch_size
    grid_w = w // patch_size
    inv = int(round(1.0 / float(down_sample_ratio)))
    if abs(inv * float(down_sample_ratio) - 1.0) > 1e-6:
        raise ValueError(f"unsupported down_sample_ratio={down_sample_ratio} (1/ratio must be an integer)")
    if grid_h % inv != 0 or grid_w % inv != 0:
        raise ValueError(
            f"token grid ({grid_h},{grid_w}) not divisible by inv={inv} "
            f"(patch_size={patch_size}, down_sample_ratio={down_sample_ratio})"
        )
    return (grid_h // inv) * (grid_w // inv)

def preprocess_phi3(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token_list: list,
        text_only: bool = False,
        group_by_length: bool = False,
        use_packed_ds: bool = False,
        ds_name: str = None,
        num_image: int = 1,
        is_drop: bool = False,
) -> Dict:
    # assert num_image == 1
    conv = get_conv_template(template_name)
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())

    if not text_only:
        new_conversations = []
        img = "<image>\n"
        if num_image > 1:
            conversations[0] = conversations[0].replace(
                'Current:\n<image>\n',
                'History:\n' + f'{img * (num_image - 1)}' + 'Current:\n<image>\n',
            )
        for conversation in conversations:
            for i in range(num_image):
                image_tokens = f'{TIME_EMBED_TOKEN}{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[i]}{IMG_END_TOKEN}'
                conversation = conversation.replace('<image>', image_tokens, 1)
            
            # HACK
            # proprio_tokens = f'{PROP_START_TOKEN}{PROP_CONTEXT_TOKEN * 10}{PROP_END_TOKEN}'
            # conversation = conversation.replace('<proprio>', proprio_tokens, 1)

            new_conversations.append(conversation)
        conversations = new_conversations

    # Tokenize conversations
    tokenizer.padding_side = 'right'
    input_ids = tokenizer(
        conversations,
        return_tensors='pt',
        padding=False if group_by_length or use_packed_ds else 'max_length',
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1]  # <|end|>\n<|assistant|>
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(int(tokenizer.pad_token_id)).sum())

        turns = conversation.split(conv.sep)
        re_turns = [conv.sep.join(turns[:3])]  # system + user + gpt
        for conv_idx in range(3, len(turns), 2):
            re_turns.append(conv.sep.join(turns[conv_idx:conv_idx + 2]))  # user + gpt
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        endoftext_id = tokenizer.convert_tokens_to_ids('<|endoftext|>')
        target[target == endoftext_id] = IGNORE_TOKEN_ID

        for i, turn in enumerate(re_turns):
            if turn == '':
                break
            if i == 0:
                turn_len = len(tokenizer(turn).input_ids)
            else:
                turn_len = len(tokenizer(turn).input_ids) - 1
            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if i == 0:
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1
            else:
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            # Ignore the user instructions
            target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
            # print(f'[question {i}]', tokenizer.decode(input_ids[:, cur_len: cur_len + instruction_len][0]))
            # print(f'[answer {i}]', tokenizer.decode(input_ids[:, cur_len + instruction_len: cur_len + turn_len][0]))
            # print(f'[label {i}]', target[cur_len + instruction_len: cur_len + turn_len])
            cur_len += turn_len

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            print(repr(tokenizer.decode(z)))

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(
                    f'WARNING: tokenization mismatch: {cur_len} vs. {total_len}.'
                    f' #turn = {len(turns) - 1}. (ignored). This dataset is {ds_name}.'
                )
                sys.stdout.flush()

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

def preprocess_internlm(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token_list: list,
        text_only: bool = False,
        group_by_length: bool = False,
        use_packed_ds: bool = False,
        ds_name: str = None,
        num_image: int = 1
) -> Dict:
    conv = get_conv_template(template_name)
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'
            sentence['value'] = sentence['value'].strip()
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())

    if not text_only:
        new_conversations = []
        for conversation in conversations:
            for i in range(num_image):
                image_tokens = f'{PROP_START_TOKEN}{PROP_CONTEXT_TOKEN}{PROP_END_TOKEN}{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[i]}{IMG_END_TOKEN}'
                conversation = conversation.replace('<image>', image_tokens, 1)
            
            proprio_tokens = f'{PROP_START_TOKEN}{PROP_CONTEXT_TOKEN * 9}{PROP_END_TOKEN}'
            conversation = conversation.replace('<proprio>', proprio_tokens, 1)
            
            new_conversations.append(conversation)
        conversations = new_conversations

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors='pt',
        padding=False if group_by_length or use_packed_ds else 'max_length',
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())  # 浦语里面 pad_token_id = eos_token_id
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID  # <s>
        parts = conversation.split(conv.roles[1])  # [UNUSED_TOKEN_146]assistant\n
        info = parts[0] + conv.roles[1]
        temp_len = len(tokenizer(info).input_ids) - 1  # 去除tokenizer的<s>
        target[cur_len: cur_len + temp_len] = IGNORE_TOKEN_ID
        cur_len = cur_len + temp_len

        for index in range(1, len(parts) - 1):
            info = parts[index]
            part1, part2 = info.split(conv.roles[0])
            temp_len = len(tokenizer(part1).input_ids) - 1
            cur_len = cur_len + temp_len
            part = conv.roles[0] + part2 + conv.roles[1]
            temp_len = len(tokenizer(part).input_ids) - 1
            target[cur_len: cur_len + temp_len] = IGNORE_TOKEN_ID
            cur_len = cur_len + temp_len
        last_info = parts[-1]
        temp_len = len(tokenizer(last_info).input_ids) - 1
        cur_len = cur_len + temp_len

        target[cur_len:] = IGNORE_TOKEN_ID
        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            print(repr(tokenizer.decode(z)))

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(f'WARNING: tokenization mismatch: {cur_len} vs. {total_len}. This dataset is {ds_name}.')
                sys.stdout.flush()

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

@dataclass
class ModelArguments:
    """
    Arguments for specifying model, tokenizer, and configurations.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    vision_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    llm_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    mlp_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    freeze_llm: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the LLM decoder.'},
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the vision backbone of the model.'},
    )
    freeze_mlp: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the MLP layers of the model.'},
    )
    unfreeze_vit_layers: int = field(
        default=0,
        metadata={'help': 'Specify the number of ViT layers to unfreeze. Default is 0.'},
    )
    vision_select_layer: int = field(
        default=-1,
        metadata={'help': 'Specify the layer of ViT feature map to use. Default is last layer.'},
    )
    use_backbone_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the backbone model. Default is 0.'}
    )
    use_llm_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the LLM. Default is 0.'}
    )
    unfreeze_lm_head: bool = field(
        default=False,
        metadata={'help': "Set to True to unfreeze the language model's head."},
    )
    use_custom_trainer: bool = field(
        default=False,
        metadata={'help': 'Set to True to enable the use of a custom trainer.'},
    )
    grad_checkpoint: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use gradient checkpointing.'},
    )
    drop_path_rate: float = field(
        default=0.0,
        metadata={'help': 'Set the drop path rate for the ViT model. Default is 0.'},
    )
    ps_version: str = field(
        default='v2',
        metadata={'help': 'Specify the version of pixel shuffle implementation. Default is `v1`.'
                          'Please use `v2` to fix the bug of transposed image.'}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments for specifying data input for training and evaluation.
    """
    max_seq_length: Optional[int] = field(
        default=2048,
        metadata={
            'help': (
                'The maximum total input sequence length after tokenization. Sequences longer '
                'than this will be truncated, sequences shorter will be padded.'
            )
        },
    )
    force_image_size: Optional[int] = field(
        default=448,
        metadata={'help': 'Set the desired size for the image. Default is 224.'},
    )
    force_hist_image_size: Optional[int] = field(
        default=224,
        metadata={'help': 'Set the desired size for the history image. Default is 224.'},
    )
    hist_len: Optional[int] = field(
        default=1,
        metadata={'help': 'Set the length of the history. Default is 1.'},
    )
    use_multi_size_image: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use multi-size images.'},
    )
    down_sample_ratio: Optional[float] = field(
        default=0.5,
        metadata={'help': 'Set the desired down-sampling ratio for the image. Default is 1.0.'},
    )
    pad2square: Optional[bool] = field(
        default=False,
        metadata={'help': 'Pad the image to a square shape if set to True.'},
    )
    conv_style: Optional[str] = field(
        default='internlm2-chat', metadata={'help': 'Prompt style for a conversation.'}
    )
    meta_path: Optional[str] = field(
        default=None,
        metadata={'help': 'The path of the meta file of datasets.'},
    )
    use_data_resampling: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use data resampling.'},
    )
    dynamic_image_size: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use dynamic image size.'},
    )
    use_thumbnail: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to add a thumbnail image.'},
    )
    min_dynamic_patch: Optional[int] = field(
        default=1,
        metadata={'help': 'The minimum number of dynamic patches. Default is 1.'},
    )
    max_dynamic_patch: Optional[int] = field(
        default=12,
        metadata={'help': 'The maximum number of dynamic patches. Default is 6.'},
    )
    normalize_type: Optional[str] = field(
        default='imagenet',
        metadata={'help': 'The normalize type for the image. Default is imagenet.'},
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        template_name,
        meta,
        tokenizer,
        tcs_loader,
        ds_name,
        num_image_token,
        num_image_token_hist,
        patch_size: int,
        down_sample_ratio: float,
        image_size=224,
        hist_image_size=224,
        hist_len=1,
        use_multi_size_image=False,
        is_train=True,
        pad2square=False,
        group_by_length=False,
        dynamic_image_size=False,
        use_thumbnail=False,
        min_dynamic_patch=1,
        max_dynamic_patch=6,
        min_num_frame=4,  # for video data
        max_num_frame=12,  # for video data
        sampling_method='rand',  # for video data
        repeat_time=1,
        normalize_type='imagenet',
        random_seed=0,
    ):
        super(LazySupervisedDataset, self).__init__()
        self.ds_name = ds_name
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        self.num_image_token_hist = num_image_token_hist
        self.patch_size = int(patch_size)
        self.down_sample_ratio = float(down_sample_ratio)
        logger.info(f'[Dataset] num_image_token: {num_image_token}')
        logger.info(f'[Dataset] num_image_token_hist: {num_image_token_hist}')
        logger.info(f'[Dataset] dynamic_image_size: {dynamic_image_size}')
        logger.info(f'[Dataset] use_thumbnail: {use_thumbnail}')
        logger.info(f'[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}')

        self.image_size = image_size
        self.hist_image_size = hist_image_size
        self.hist_len = hist_len
        self.use_multi_size_image = use_multi_size_image
        self.is_train = is_train
        self.pad2square = pad2square
        self.max_num_frame = max_num_frame
        self.min_num_frame = min_num_frame
        self.sampling_method = sampling_method

        logger.info('Formatting inputs...Skip in lazy mode')
        assert meta['annotation'].endswith('jsonl'), f'annotation must be jsonl, but got {meta["annotation"]}'

        with open(meta['annotation'], 'r') as f:
            self.raw_data = f.readlines()
            if repeat_time < 1:
                # If repeat_time is less than 1, select a portion of the data
                self.raw_data = self.raw_data[:int(len(self.raw_data) * repeat_time)]
            if repeat_time > 1:
                assert isinstance(repeat_time, int)
                # Repeat the list if repeat_time is greater than 1
                self.raw_data = self.raw_data * repeat_time

        self.rng = np.random.default_rng(seed=random_seed)
        self.rng.shuffle(self.raw_data)

        gc.collect()
        self.root = meta['root']
        self.cached_data_dict = {}
        self.tcs_loader = tcs_loader
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.normalize_type = normalize_type

        # If the precomputed length does not exist, roughly estimate the length of
        # each sample to improve the efficiency of group_by_length.
        if self.group_by_length:
            self.conv2length = {}  # Using a dictionary to speed up token length calculation
            self.length = []
            for data_item in self.raw_data:
                data_item = json.loads(data_item)
                if 'length' in data_item:
                    token_length = data_item['length']  # Use precomputed length if available
                else:
                    # Compute token length using the tokenizer
                    conversations = '\n'.join([temp['value'] for temp in data_item['conversations']])
                    str_length = len(conversations)
                    if str_length not in self.conv2length:
                        token_length = tokenizer(
                            conversations, return_tensors='pt', padding=False, truncation=False,
                        ).input_ids.size(1)
                        self.conv2length[str_length] = token_length + num_image_token * (
                                    max_dynamic_patch + use_thumbnail)
                    else:
                        token_length = self.conv2length[str_length]
                self.length.append(token_length)
        gc.collect()

    def __len__(self):
        return len(self.raw_data)

    def get_preprocess_function(self):
        # Select the appropriate preprocessing function based on the template name
        if self.template_name == 'Hermes-2':
            preprocess_function = preprocess_mpt
        elif self.template_name == 'internlm2-chat':
            preprocess_function = preprocess_internlm
        elif self.template_name == 'phi3-chat':
            preprocess_function = preprocess_phi3
        else:
            preprocess_function = preprocess
        return preprocess_function

    def load_image(self, image_path):
        # Load the image using tcs_loader if available, otherwise use PIL
        if self.tcs_loader is not None and 's3://' in image_path:
            return self.tcs_loader(image_path)
        return Image.open(image_path).convert('RGB')
    

    def load_depth(self, depth_path):
        # Load the PNG depth image as uint16
        depth_image = Image.open(depth_path)
        
        # Ensure the image is loaded in the correct mode
        depth_array = np.array(depth_image, dtype=np.uint16) / 256.
    
        return depth_array
    
    def load_image_depth(self, image_path):
        # Load the image using tcs_loader if available, otherwise use PIL
        if self.tcs_loader is not None and 's3://' in image_path:
            return self.tcs_loader(image_path)
        return Image.open(image_path).convert('RGB')

    def get_image_path(self, image_path):
        if image_path.startswith('s3://'):  # for ceph
            image_path = self.root + image_path
        else:  # for local image
            image_path = os.path.join(self.root, image_path)
        return image_path

    def get_transform(self):
        # Build transformation function
        transform = build_transform(is_train=self.is_train, input_size=self.image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type)
        return transform
    
    def get_transform_hist(self):
        # Build transformation function
        transform = build_transform(is_train=self.is_train, input_size=self.hist_image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type)
        return transform
    

    def multi_modal_multi_size_image_rgbd_proprio_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()
        transform_hist = self.get_transform_hist()

        camera_info = data_item['camera_info']
        pose_matrix, K = _pose_and_intrinsics_from_camera_info(camera_info, self.image_size)
        _maybe_apply_droid_path_prefix(data_item)

        images, num_tiles = [], []
        images_depth = []
        depth_mask_list = []
        num_image = len(data_item['image'])
        if num_image < (self.hist_len + 1):
            raise ValueError(f"num_image={num_image} must be >= hist_len+1 ({self.hist_len + 1})")
        if 'image_depth' in data_item and isinstance(data_item['image_depth'], list):
            if len(data_item['image_depth']) != num_image:
                raise ValueError(
                    f"len(image_depth)={len(data_item['image_depth'])} must match len(image)={num_image}"
                )
        frame_id = _frame_id_list_from_item(data_item, num_image)
        for (image_path, image_depth_path) in zip(data_item['image'], data_item['image_depth']):
            # Merge the image path
            image_path = self.get_image_path(image_path)
            image_depth_path = self.get_image_path(image_depth_path)

            # Load the image using tcs_loader if available, otherwise use PIL
            image = self.load_image(image_path)
            if image_depth_path.split('/')[-1].split('.')[0] == 'zero':
                image_depth = np.zeros((self.image_size, self.image_size))
            else:
                image_depth = self.load_depth(image_depth_path)

            depth_image_size = self.image_size

            image_depth_resized = cv2.resize(image_depth, (depth_image_size, depth_image_size), interpolation=cv2.INTER_LINEAR) 


            if image_depth_path.split('/')[-1].split('.')[0] == 'zero':
                depth_mask = np.zeros((depth_image_size, depth_image_size))
            else:
                depth_mask = np.ones((depth_image_size, depth_image_size))

            depth_mask_list.append(depth_mask)


            if self.dynamic_image_size:
                per_frame_max = max(1, self.max_dynamic_patch // num_image)
                image_tiles = dynamic_preprocess(
                    image,
                    min_num=self.min_dynamic_patch,
                    max_num=per_frame_max,
                    image_size=self.image_size,
                    use_thumbnail=self.use_thumbnail,
                )
                if len(image_tiles) != 1:
                    raise ValueError(
                        f"dynamic_preprocess produced {len(image_tiles)} tiles for a single frame; "
                        "this dataloader currently assumes 1 tile per frame. "
                        "Fix: ensure `max_dynamic_patch // num_frames == 1` or set dynamic_image_size/use_thumbnail off."
                    )
                images += image_tiles
                num_tiles.append(len(image_tiles))
            else:
                images.append(image)
                num_tiles.append(1)
            images_depth.append(image_depth_resized)


        pixel_values = [transform(images[i]) for i in range(self.hist_len, num_image)]
        pixel_values = torch.stack(pixel_values)

        pixel_values_hist = [transform_hist(images[i]) for i in range(self.hist_len)]
        pixel_values_hist = torch.stack(pixel_values_hist)

        pixel_depth = torch.stack([torch.tensor(images_depth[i]) for i in range(self.hist_len, num_image)])
        pixel_depth_hist = torch.stack([torch.tensor(images_depth[i]) for i in range(self.hist_len)])

        depth_mask = torch.stack([torch.tensor(depth_mask_list[i]) for i in range(self.hist_len, num_image)])
        depth_mask_hist = torch.stack([torch.tensor(depth_mask_list[i]) for i in range(self.hist_len)])

        num_patches = pixel_values.size(0)
        num_patches_hist = pixel_values_hist.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary.
        # IMPORTANT: keep IMG_CONTEXT token counts consistent with actual ViT token counts.
        num_image_tokens_hist = [
            _num_vit_tokens_from_chw(pixel_values_hist[i], patch_size=self.patch_size, down_sample_ratio=self.down_sample_ratio)
            for i in range(self.hist_len)
        ]
        num_image_tokens_curr = [
            _num_vit_tokens_from_chw(pixel_values[i], patch_size=self.patch_size, down_sample_ratio=self.down_sample_ratio)
            for i in range(num_image - self.hist_len)
        ]
        num_image_tokens = num_image_tokens_hist + num_image_tokens_curr
        # num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
        # num_image_tokens = [self.num_image_token for _ in range(self.hist_len, num_image)]
        # num_image = 1

        # is_drop = False
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
                                  ds_name=self.ds_name, num_image=num_image)
        
        action = np.array(data_item['action'])
        # action[0:3] = action[0:3] * 2
        # action[3:6] = action[3:6] * 5

        action[-1] = 1 - action[-1]

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            action_labels=action,
            attention_mask=ret['attention_mask'][0],
            pixel_values=pixel_values,
            pixel_values_hist=pixel_values_hist,
            depth=pixel_depth.unsqueeze(0), 
            depth_hist=pixel_depth_hist.unsqueeze(0),
            depth_mask=depth_mask.unsqueeze(0),
            depth_mask_hist=depth_mask_hist.unsqueeze(0),
            # depth=torch.tensor(image_depth_resized, dtype=torch.float).unsqueeze(0), 
            pose=torch.tensor(pose_matrix, dtype=torch.float).unsqueeze(0),
            intrinsics=torch.tensor(K, dtype=torch.float).unsqueeze(0),
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            frame_idx=torch.tensor(frame_id, dtype=torch.long),
            image_flags_hist=torch.tensor([1] * num_patches_hist, dtype=torch.long)
        )
        return ret
        

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        i = i % len(self.raw_data)
        while True:
            try:
                data_item = json.loads(self.raw_data[i])
                ret = self.multi_modal_multi_size_image_rgbd_proprio_get_item(data_item)
                break
            except Exception as e:
                print(e, self.ds_name, flush=True)
                if not isinstance(e, UnidentifiedImageError):
                    traceback.print_exc()
                data_item = json.loads(self.raw_data[i])
                if 'image' in data_item:
                    if type(data_item['image']) == list:
                        images = [self.root + item for item in data_item['image']]
                        print(f'Failed to load image: {images}, the dataset is: {self.ds_name}')
                    else:
                        if data_item['image'].startswith('s3://'):
                            data_path = self.root + data_item['image']
                        else:
                            data_path = os.path.join(self.root, data_item['image'])
                        print(f'Failed to load image: {data_path}, the dataset is: {self.ds_name}')
                elif 'video' in data_item:
                    data_path = os.path.join(self.root, data_item['video'])
                    print(f'Failed to load video: {data_path}, the dataset is: {self.ds_name}')
                i = random.randint(0, len(self.raw_data) - 1)
        return ret


def build_datasets(
    data_args,
    tokenizer,
    tcs_loader,
    model,
    group_by_length=False,
    dynamic_image_size=False,
    use_thumbnail=False,
    min_dynamic_patch=1,
    max_dynamic_patch=12,
    normalize_type='imagenet',
):
    datasets = []
    lengths = []
    ds_collections = json.loads(open(data_args.meta_path).read())
    for ds_idx, ds_name in enumerate(ds_collections.keys()):
        repeat_time = ds_collections[ds_name]['repeat_time']
        if 'max_dynamic_patch' in ds_collections[ds_name]:
            max_num = ds_collections[ds_name]['max_dynamic_patch']
            logger.info(f'max_dynamic_patch is set to {max_num} according to the meta file')
        else:
            max_num = max_dynamic_patch
        dataset = LazySupervisedDataset(
            data_args.conv_style, ds_collections[ds_name],
            tokenizer,
            tcs_loader,
            ds_name=ds_name,
            num_image_token=model.num_image_token,
            num_image_token_hist=model.num_image_token_hist,
            patch_size=model.config.vision_config.patch_size,
            down_sample_ratio=data_args.down_sample_ratio,
            image_size=data_args.force_image_size,
            hist_len=data_args.hist_len,
            hist_image_size=data_args.force_hist_image_size,
            use_multi_size_image=data_args.use_multi_size_image,
            is_train=ds_collections[ds_name]['data_augment'],
            pad2square=data_args.pad2square,
            group_by_length=group_by_length,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_num,
            repeat_time=repeat_time,
            normalize_type=normalize_type,
            random_seed=ds_idx,
        )
        logger.info(f'Add dataset: {ds_name} with length: {len(dataset)}')
        datasets.append(dataset)
        if data_args.use_data_resampling:
            lengths.append(math.sqrt(len(dataset)))
        else:
            lengths.append(len(dataset))
    if data_args.use_data_resampling:
        total_length = sum(lengths)
        weights = [l / total_length for l in lengths]
        train_dataset = WeightedConcatDataset(datasets, weights)
    else:
        train_dataset = ConcatDataset(datasets)
    return train_dataset


def main():
    # Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # If use DeepSpeed zero3, init_dist must before HfArgumentParser
    launcher = os.environ.get('LAUNCHER', 'slurm')
    init_dist(launcher=launcher, backend='nccl')
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry('InternV-Chat', model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
        + f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
    )
    logger.info(f'Training/evaluation parameters {training_args}')

    # Detecting last checkpoint and eventually continue from last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f'Output directory ({training_args.output_dir}) already exists and is not empty. '
                'Use --overwrite_output_dir to overcome.'
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f'Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change '
                'the `--output_dir` or add `--overwrite_output_dir` to train from scratch.'
            )
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model, tokenizer, and image processor
    tokenizer_path = model_args.model_name_or_path or model_args.llm_path
    logger.info(f'Loading Tokenizer: {tokenizer_path}')
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, add_eos_token=False, trust_remote_code=True, use_fast=False)
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = data_args.max_seq_length
    token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                  QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                  REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN,
                  STATE_PRED, 
                  PROP_START_TOKEN, PROP_END_TOKEN, PROP_CONTEXT_TOKEN,
                  TIME_EMBED_TOKEN]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    prop_context_token_id = tokenizer.convert_tokens_to_ids(PROP_CONTEXT_TOKEN)
    time_embed_token_id = tokenizer.convert_tokens_to_ids(TIME_EMBED_TOKEN) 
    tcs_loader = TCSLoader('~/petreloss.conf') if has_tcs_loader else None

    if model_args.model_name_or_path is not None:
        logger.info('Loading InternVLChatModel...')
        config = InternVLChatConfig.from_pretrained(model_args.model_name_or_path)
        config.vision_config.drop_path_rate = model_args.drop_path_rate
        if config.llm_config.model_type == 'internlm2':
            config.llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
            logger.info('Using flash_attention_2 for InternLM')
        else:
            config.llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
            logger.info('Using flash_attention_2 for LLaMA')
        config.template = data_args.conv_style
        config.select_layer = model_args.vision_select_layer
        config.dynamic_image_size = data_args.dynamic_image_size
        config.use_thumbnail = data_args.use_thumbnail
        config.ps_version = model_args.ps_version
        config.min_dynamic_patch = data_args.min_dynamic_patch
        config.max_dynamic_patch = data_args.max_dynamic_patch
        model = InternVLChatModel3DProprioFTRPECat.from_pretrained(
            model_args.model_name_or_path, torch_dtype=torch.bfloat16, config=config)

    # else:
    #     logger.info('Loading ViT-6B...')
    #     vision_config = InternVisionConfig.from_pretrained(model_args.vision_path)
    #     vision_config.drop_path_rate = model_args.drop_path_rate
    #     vision_model = InternVisionModel.from_pretrained(
    #         model_args.vision_path, torch_dtype=torch.bfloat16, config=vision_config)
    #     logger.info('Loading LLaMA...')
    #     llm_config = AutoConfig.from_pretrained(model_args.llm_path, trust_remote_code=True)
    #     if llm_config.model_type == 'internlm2':
    #         model_type = InternLM2ForCausalLM
    #         llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
    #         logger.info('Using flash_attention_2 for InternLM')
    #     else:
    #         model_type = AutoModelForCausalLM
    #         llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
    #         logger.info('Using flash_attention_2 for LLaMA')
    #     llm = model_type.from_pretrained(
    #         model_args.llm_path, torch_dtype=torch.bfloat16,
    #         config=llm_config, trust_remote_code=True)
    #     logger.info('Building InternVLChatConfig...')
    #     internvl_chat_config = InternVLChatConfig(
    #         vision_config.to_dict(), llm_config.to_dict(), downsample_ratio=data_args.down_sample_ratio,
    #         pad2square=data_args.pad2square, template=data_args.conv_style,
    #         select_layer=model_args.vision_select_layer, dynamic_image_size=data_args.dynamic_image_size,
    #         use_thumbnail=data_args.use_thumbnail, ps_version=model_args.ps_version,
    #         min_dynamic_patch=data_args.min_dynamic_patch, max_dynamic_patch=data_args.max_dynamic_patch)
    #     internvl_chat_config.force_image_size = data_args.force_image_size
    #     logger.info('Building InternVLChatModel...')
    #     model = InternVLChatModel(internvl_chat_config, vision_model, llm)
    model.img_context_token_id = img_context_token_id
    model.prop_context_token_id = prop_context_token_id
    model.time_embed_token_id = time_embed_token_id

    assert model.config.downsample_ratio == data_args.down_sample_ratio

    if model_args.mlp_path is not None:
        logger.info('Loading pretrained MLP projector...')
        state_dict = torch.load(model_args.mlp_path, map_location='cpu')
        message = model.mlp1.load_state_dict(state_dict)
        logger.info(message)

    # model.action_head = ActionHead_p1(model.llm_hidden_size)
    # model.spatial_agg = SpatialAwareModule()
    # model.proprio_head = ProprioHead(model.llm_hidden_size, input_dim=48)
    # model.time_embedding = nn.Embedding(50, 512)
    # model.time_mlp = nn.Linear(512, model.llm_hidden_size)
    
    logger.info('Finished')

    patch_size = model.config.vision_config.patch_size
    logger.info(f'model.config.force_image_size: {model.config.force_image_size}')
    logger.info(f'data_args.force_image_size: {data_args.force_image_size}')
    logger.info(f'model.config.vision_config.image_size: {model.config.vision_config.image_size}')
    if model.config.vision_config.image_size != data_args.force_image_size:
        logger.info(f'Resizing position embedding from '
                    f'{model.config.vision_config.image_size} '
                    f'to {data_args.force_image_size}...')
        model.vision_model.resize_pos_embeddings(old_size=model.config.vision_config.image_size,
                                                 new_size=data_args.force_image_size,
                                                 patch_size=patch_size)
        model.config.vision_config.image_size = data_args.force_image_size
    model.config.force_image_size = data_args.force_image_size
    model.num_image_token = int((data_args.force_image_size // patch_size) ** 2 * (data_args.down_sample_ratio ** 2))
    model.num_image_token_hist = int((data_args.force_hist_image_size // patch_size) ** 2 * (data_args.down_sample_ratio ** 2))

    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    model.language_model.config.use_cache = False
    model.vision_model.gradient_checkpointing = True
    model.vision_model.encoder.gradient_checkpointing = True
    if model_args.grad_checkpoint:
        model.language_model._set_gradient_checkpointing()

    train_dataset = build_datasets(
        data_args, tokenizer, tcs_loader, model, group_by_length=training_args.group_by_length,
        dynamic_image_size=data_args.dynamic_image_size, use_thumbnail=data_args.use_thumbnail,
        min_dynamic_patch=data_args.min_dynamic_patch, max_dynamic_patch=data_args.max_dynamic_patch,
        normalize_type=data_args.normalize_type)

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_backbone:
        # model.vision_model = model.vision_model.eval()
        _freeze_params(model.vision_model)

    if model_args.freeze_llm:
        model.language_model = model.language_model.eval()
        _freeze_params(model.language_model)

    if model_args.unfreeze_lm_head:
        model.language_model.lm_head.requires_grad = True

    if model_args.use_backbone_lora:
        model.wrap_backbone_lora(r=model_args.use_backbone_lora, lora_alpha=2 * model_args.use_backbone_lora)
        model.config.use_backbone_lora = model_args.use_backbone_lora

    if model_args.use_llm_lora:
        model.wrap_llm_lora(r=model_args.use_llm_lora, lora_alpha=2 * model_args.use_llm_lora)
        model.config.use_llm_lora = model_args.use_llm_lora

    if model_args.freeze_mlp:
        _freeze_params(model.mlp1)

    if model_args.unfreeze_vit_layers != 0:
        layers = model.vision_model.encoder.layers[model_args.unfreeze_vit_layers:]
        for k, v in layers.named_parameters():
            logger.info(f'Unfreezing ViT layer: {k}')
            v.requires_grad = True

    # print trainable parameters
    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name)

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    # Initialize our Trainer
    if model_args.use_custom_trainer:
        replace_create_optimizer()

    trainer = VLATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=concat_pad_data_collator
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        try:
            metrics['train_samples'] = len(train_dataset)
        except:
            metrics['train_samples'] = -1

        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()


if __name__ == '__main__':
    main()
