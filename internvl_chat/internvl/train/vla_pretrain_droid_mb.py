from __future__ import annotations

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
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import transformers
from internvl.dist_utils import init_dist
from internvl.model.internvl_chat import (
    ActionHead_p1,
    InternVLChatConfig,
    InternVLChatModel3DProprioDroidMB,
    SpatialAwareModule,
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
                                    preprocess,
                                    preprocess_internlm,
                                    preprocess_mpt)
                                    # preprocess_phi3)
from internvl.conversation import get_conv_template
from PIL import Image, ImageFile, PngImagePlugin, UnidentifiedImageError
from torch.utils.data import Dataset
from transformers import (AutoTokenizer,
                          HfArgumentParser, TrainingArguments,
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
    # if num_image != 5:
    #     raise
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
        conversations[0] = conversations[0].replace('Current:\n<image>\n', 'History:\n'+f'{img*(num_image-1)}'+'Current:\n<image>\n')
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


@dataclass
class ModelArguments:
    """
    Arguments for specifying model, tokenizer, and configurations.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    mlp_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Optional path to a pretrained MLP projector state_dict (loaded into `model.mlp1`).'}
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
    vision_select_layer: int = field(
        default=-1,
        metadata={'help': 'Specify the layer of ViT feature map to use. Default is last layer.'},
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
        metadata={
            'help': (
                'Pixel shuffle implementation version. Use `v2` to fix the transposed-image bug present in `v1`.'
            )
        },
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
        metadata={'help': 'The (square) image resolution used by the vision encoder.'},
    )
    force_hist_image_size: Optional[int] = field(
        default=None,
        metadata={
            'help': (
                'History image size (for models that use a separate history image encoder path). '
                'NOTE: the current RGBD pretrain path stacks all frames into one `pixel_values` tensor, '
                'so history frames must share the same resolution as current frames; keep this equal to '
                '`force_image_size` for now.'
            )
        },
    )
    hist_len: Optional[int] = field(
        default=1,
        metadata={
            'help': (
                'Number of history frames (excluding current). For list-based datasets, the expected number of '
                'frames is `hist_len + 1`.'
            )
        },
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
        patch_size: int,
        down_sample_ratio: float,
        image_size=224,
        hist_image_size=224,
        hist_len=1,
        use_multi_size_image=False,
        is_train=True,
        pad2square=False,
        group_by_length=False,
        repeat_time=1,
        normalize_type='imagenet',
        random_seed=0,
    ):
        super(LazySupervisedDataset, self).__init__()
        self.ds_name = ds_name
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        logger.info(f'[Dataset] num_image_token: {num_image_token}')
        self.patch_size = int(patch_size)
        self.down_sample_ratio = float(down_sample_ratio)

        self.image_size = image_size
        self.hist_image_size = hist_image_size
        self.hist_len = hist_len
        self.use_multi_size_image = use_multi_size_image
        self.is_train = is_train
        self.pad2square = pad2square

        logger.info('Formatting inputs...Skip in lazy mode')
        assert meta['annotation'].endswith('jsonl'), f'annotation must be jsonl, but got {meta["annotation"]}'

        internvl_chat_root = Path(__file__).resolve().parents[2]

        annotation_path = Path(meta['annotation'])
        if not annotation_path.is_absolute() and not str(annotation_path).startswith('s3://'):
            annotation_path = internvl_chat_root / annotation_path
        annotation_path = annotation_path.resolve()

        with open(annotation_path, 'r') as f:
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
        root = meta['root']
        if isinstance(root, str) and not root.startswith('s3://'):
            root_path = Path(root)
            if not root_path.is_absolute():
                root_path = internvl_chat_root / root_path
            self.root = str(root_path.resolve())
        else:
            self.root = root
        self.cached_data_dict = {}
        self.tcs_loader = tcs_loader
        self.group_by_length = group_by_length
        self.normalize_type = normalize_type

        # If the precomputed length does not exist, roughly estimate the length of
        # each sample to improve the efficiency of group_by_length.
        if self.group_by_length:
            self.conv2length = {}  # Using a dictionary to speed up token length calculation
            self.length = []
            expected_num_image = (self.hist_len or 0) + 1
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
                        # Roughly account for <image> replacement tokens (multi-frame: hist_len + 1).
                        self.conv2length[str_length] = token_length + num_image_token * expected_num_image
                    else:
                        token_length = self.conv2length[str_length]
                self.length.append(token_length)
        gc.collect()

    def __len__(self):
        return len(self.raw_data)

    def _num_image_tokens_from_pixel_values(self, pixel_values: torch.Tensor) -> int:
        """
        Compute the number of visual tokens produced by ViT + downsample for a single image tensor.

        In this DROID pretrain path we keep the original aspect ratio (no resize), so the number of
        visual tokens depends on per-sample (H, W). The prompt must use the same count of
        `IMG_CONTEXT_TOKEN` as the model produces, otherwise the forward will crash when scattering
        ViT embeddings into `input_embeds`.
        """
        if pixel_values.ndim != 3:
            raise ValueError(f'Expected pixel_values with shape (C,H,W), got {tuple(pixel_values.shape)}')
        h, w = int(pixel_values.shape[-2]), int(pixel_values.shape[-1])
        if self.patch_size <= 0:
            raise ValueError(f'Invalid patch_size={self.patch_size}')
        grid_h = h // self.patch_size
        grid_w = w // self.patch_size
        if grid_h <= 0 or grid_w <= 0:
            raise ValueError(f'Invalid image size for patching: (H,W)=({h},{w}) patch_size={self.patch_size}')
        out_h = int(grid_h * self.down_sample_ratio)
        out_w = int(grid_w * self.down_sample_ratio)
        if out_h <= 0 or out_w <= 0:
            raise ValueError(
                f'Invalid down_sample_ratio={self.down_sample_ratio} for grid (H,W)=({grid_h},{grid_w})'
            )
        return out_h * out_w

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
    
    def get_image_path(self, image_path):
        # Keep both local paths and s3:// paths usable.
        if image_path.startswith('s3://'):
            return image_path
        if isinstance(self.root, str) and self.root.startswith('s3://'):
            return self.root.rstrip('/') + '/' + image_path.lstrip('/')
        return os.path.join(self.root, image_path)

    def get_transform(self):
        # Build transformation function
        transform = build_transform(is_train=self.is_train, input_size=self.image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type, resize=False)
        return transform
    
    def get_transform_hist(self):
        # Build transformation function
        transform = build_transform(is_train=self.is_train, input_size=self.hist_image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type)
        return transform


    def multi_modal_multi_size_image_rgbd_proprio_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()
        if self.hist_image_size != self.image_size:
            raise NotImplementedError(
                'RGBD path currently stacks all frames into one `pixel_values` tensor, so history frames must have the '
                'same resolution as current frames. Set `force_hist_image_size == force_image_size`.'
            )

        camera_info = data_item['camera_info']
        agentview_extrinsic = camera_info['camera_extrinsic']
        agentview_intrinsic = camera_info['camera_intrinsic']
        # robothand_extrinsic = camera_info['robothand_extrinsic']
        # rotation_matrix = R.from_quat(quaternion).as_matrix()
        # pose_matrix = np.eye(4)
        # pose_matrix[:3, :3] = rotation_matrix
        # pose_matrix[:3, 3] = position
        pose_matrix = np.array(agentview_extrinsic)
        pose_matrix = torch.tensor(pose_matrix, dtype=torch.float)

        K = np.eye(4)
        K[:3, :3] = np.array(agentview_intrinsic)


        data_item['image'] = [os.path.join(data_item['path_prefix'], d) for d in data_item['image']]
        data_item['image_depth'] = [os.path.join(data_item['path_prefix'].replace('left', 'depth'), d) for d in data_item['image_depth']]

        num_image = len(data_item['image'])
        expected_num_image = (self.hist_len or 0) + 1
        if expected_num_image <= 0:
            raise ValueError(f'Invalid hist_len={self.hist_len} (expected >=0).')
        new_data_image = []
        new_data_depth = []
        for d_i, d_d in zip(data_item['image'], data_item['image_depth']):
            d_i_path = os.path.join(*d_i.split('/')[:-1])
            d_d_path = os.path.join(*d_d.split('/')[:-1])
            d_i_name = d_i.split('/')[-1]
            d_d_name = d_d.split('/')[-1]
            d_id = int(d_i_name.split('.')[0].split('_')[-1])
            if 'step' not in d_i_name:
                
                d_i_new_name = d_i_name.replace(str(d_id), f"step_{d_id}")
                d_d_new_name = d_d_name.replace(str(d_id), f"step_{d_id}")
                d_i = os.path.join(d_i_path, d_i_new_name)
                d_d = os.path.join(d_d_path, d_d_new_name)
                # print(d_i_new_name)
            else:
                d_i = os.path.join(d_i_path, d_i_name)
                d_d = os.path.join(d_d_path, d_d_name)
            new_data_image.append(d_i)
            new_data_depth.append(d_d)
        data_item['image'] = new_data_image
        data_item['image_depth'] = new_data_depth

        if len(data_item['image']) != len(data_item['image_depth']):
            raise ValueError(
                f'RGB/D frame count mismatch: len(image)={len(data_item["image"])} '
                f'len(image_depth)={len(data_item["image_depth"])}'
            )
        if num_image < expected_num_image:
            # Left-pad with the first frame to satisfy `hist_len + 1`.
            num_diff = expected_num_image - num_image
            data_item['image'] = [data_item['image'][0]] * num_diff + data_item['image']
            data_item['image_depth'] = [data_item['image_depth'][0]] * num_diff + data_item['image_depth']
        elif num_image > expected_num_image:
            # Keep the most recent `hist_len + 1` frames.
            data_item['image'] = data_item['image'][-expected_num_image:]
            data_item['image_depth'] = data_item['image_depth'][-expected_num_image:]
        num_image = expected_num_image
        images, num_tiles = [], []
        images_depth = []
        frame_id = [int(d.split('.')[0].split('_')[-1]) for d in data_item['image']]
        for (image_path, image_depth_path) in zip(data_item['image'], data_item['image_depth']):
            # Merge the image path
            image_path = self.get_image_path(image_path)
            image_depth_path = self.get_image_path(image_depth_path)

            # Load the image using tcs_loader if available, otherwise use PIL
            image = self.load_image(image_path)
            image_depth = self.load_depth(image_depth_path)
            image_depth_resized = image_depth
            # image_depth_resized = cv2.resize(image_depth, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)  # FIXME

            # Use the original image as a single patch (no dynamic patching in this pretrain path).
            images.append(image)
            images_depth.append(image_depth_resized)
            num_tiles.append(1)


        # pixel_values = [transform(images[i]) for i in range(self.hist_len, num_image)]
        # pixel_values = torch.stack(pixel_values)
        pixel_values = [transform(images[i]) for i in range(0, num_image)]
        pixel_values = torch.stack(pixel_values)



        # pixel_values_hist = [transform_hist(images[i]) for i in range(self.hist_len)]
        # pixel_values_hist = torch.stack(pixel_values_hist)

        num_patches = pixel_values.size(0)
        # num_patches_hist = pixel_values_hist.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        # Must match actual ViT token count for each frame (aspect-ratio dependent).
        num_image_tokens = [self._num_image_tokens_from_pixel_values(pixel_values[i]) for i in range(num_image)]
        # num_image = 1

        # is_drop = random.random() < 0.5
        is_drop = True
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
                                  ds_name=self.ds_name, num_image=num_image, is_drop=is_drop)

        # proprio = torch.tensor(np.array(data_item['proprio']))
        # prop_flags = torch.tensor([1], dtype=torch.long)

        # is_translation = random.random() < 0.5
        # if is_translation:
        #     # uniform sample from -0.5 to 0.5
        #     translation = torch.rand(3) - 0.5
        #     proprio[:, :3] += translation
        #     pose_matrix[:3, 3] += translation
        
        # if is_drop:
        #     proprio[:-1] = torch.zeros_like(proprio[:-1])
        #     prop_flags = torch.tensor([0, 0, 0, 0, 1], dtype=torch.long)
        # else:
        #     proprio = proprio
        #     prop_flags = torch.tensor([1, 1, 1, 1, 1], dtype=torch.long)

        images_depth = torch.tensor(np.array(images_depth), dtype=pixel_values.dtype).unsqueeze(0)
        # image_depth_resized = torch.tensor(image_depth_resized, dtype=torch.float)
        depth_mask = images_depth > 0

        
        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            action_labels=np.array(data_item['action']),
            # proprio=proprio,
            attention_mask=ret['attention_mask'][0],
            pixel_values=pixel_values,
            # pixel_values_hist=pixel_values_hist,
            # pixel_values_hist=None,
            depth=images_depth,
            depth_mask=depth_mask, 
            pose=pose_matrix.unsqueeze(0),
            intrinsics=torch.tensor(K, dtype=torch.float).unsqueeze(0),
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            frame_idx=torch.tensor(frame_id, dtype=torch.long),
            # prop_flags=prop_flags,
            # image_flags_hist=torch.tensor([1] * num_patches_hist, dtype=torch.long)
        )
        return ret
        

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        i = i % len(self.raw_data)
        while True:
            try:
                data_item = json.loads(self.raw_data[i])
                images = data_item.get('image')
                if not isinstance(images, list) or len(images) == 0:
                    raise NotImplementedError(
                        'This pretrain entrypoint expects list-based multi-frame samples (data_item["image"] is a list).'
                    )
                if not self.use_multi_size_image:
                    raise ValueError('Expected `use_multi_size_image=True` for DROID multi-frame pretrain.')
                if 'image_depth' not in data_item:
                    raise NotImplementedError(
                        'This pretrain entrypoint expects RGBD samples (data_item["image_depth"] exists).'
                    )
                ret = self.multi_modal_multi_size_image_rgbd_proprio_get_item(data_item)
                break
            except Exception as e:
                print(e, self.ds_name, flush=True)
                if not isinstance(e, UnidentifiedImageError):
                    traceback.print_exc()
                # For unsupported sample formats, fail fast instead of looping forever.
                if isinstance(e, (NotImplementedError, ValueError, KeyError, TypeError)):
                    raise
                data_item = json.loads(self.raw_data[i])
                if 'image' in data_item:
                    if isinstance(data_item['image'], list):
                        image_paths = [self.get_image_path(item) for item in data_item['image']]
                        print(f'Failed to load image: {image_paths}, the dataset is: {self.ds_name}')
                    else:
                        data_path = self.get_image_path(data_item['image'])
                        print(f'Failed to load image: {data_path}, the dataset is: {self.ds_name}')
                i = random.randint(0, len(self.raw_data) - 1)
        return ret


def build_datasets(
    data_args,
    tokenizer,
    tcs_loader,
    model,
    group_by_length=False,
    normalize_type='imagenet',
):
    datasets = []
    lengths = []
    with open(data_args.meta_path, 'r') as f:
        ds_collections = json.load(f)
    patch_size = int(model.config.vision_config.patch_size)
    down_sample_ratio = float(model.config.downsample_ratio)
    for ds_idx, ds_name in enumerate(ds_collections.keys()):
        repeat_time = ds_collections[ds_name]['repeat_time']
        dataset = LazySupervisedDataset(
            data_args.conv_style, ds_collections[ds_name],
            tokenizer,
            tcs_loader,
            ds_name=ds_name,
            num_image_token=model.num_image_token,
            patch_size=patch_size,
            down_sample_ratio=down_sample_ratio,
            image_size=data_args.force_image_size,
            hist_len=data_args.hist_len,
            hist_image_size=data_args.force_hist_image_size,
            use_multi_size_image=data_args.use_multi_size_image,
            is_train=ds_collections[ds_name]['data_augment'],
            pad2square=data_args.pad2square,
            group_by_length=group_by_length,
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

    if data_args.force_hist_image_size is None:
        data_args.force_hist_image_size = data_args.force_image_size

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
    tokenizer_path = model_args.model_name_or_path
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

    tcs_loader = None
    if has_tcs_loader:
        conf_path = os.environ.get('PETRELOSS_CONF', os.path.expanduser('~/petreloss.conf'))
        conf_path = str(Path(conf_path).expanduser())
        if Path(conf_path).is_file():
            tcs_loader = TCSLoader(conf_path)
        else:
            logger.info(
                f'petrel_client is installed but config not found: {conf_path}. '
                'Set PETRELOSS_CONF to enable loading s3:// paths.'
            )

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
        config.ps_version = model_args.ps_version
        model = InternVLChatModel3DProprioDroidMB.from_pretrained(
            model_args.model_name_or_path, torch_dtype=torch.bfloat16, config=config)
    model.img_context_token_id = img_context_token_id
    model.prop_context_token_id = prop_context_token_id
    model.time_embed_token_id = time_embed_token_id

    assert model.config.downsample_ratio == data_args.down_sample_ratio

    if model_args.mlp_path is not None:
        logger.info('Loading pretrained MLP projector...')
        state_dict = torch.load(model_args.mlp_path, map_location='cpu')
        message = model.mlp1.load_state_dict(state_dict)
        logger.info(message)

    model.action_head = ActionHead_p1(model.llm_hidden_size)
    model.spatial_agg = SpatialAwareModule()
    # model.proprio_head = ProprioHead(model.llm_hidden_size, input_dim=48)
    model.time_embedding = torch.nn.Embedding(50, 512)
    model.time_mlp = torch.nn.Linear(512, model.llm_hidden_size)
    
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

    if model_args.freeze_mlp:
        _freeze_params(model.mlp1)

    # print trainable parameters
    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name)

    # set seed for torch dataloaders
    set_seed(training_args.seed)

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
