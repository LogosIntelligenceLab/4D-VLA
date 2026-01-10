# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import warnings
from typing import Any, List, Optional, Tuple, Union, Dict

import torch.distributed as dist
import torch.utils.checkpoint
import transformers
from internvl.conversation import get_conv_template
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from internvl.model.phi3.modeling_phi3 import Phi3ForCausalLM
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)
# from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from .configuration_internvl_chat import InternVLChatConfig
from .modeling_intern_vit import InternVisionModel
from .spatial_aware_module import SpatialAwareModule
from .unproject import backprojector_dataloader
from .position_encodings import SinusoidalPosEmb

from einops import rearrange
from dataclasses import dataclass
import torch.nn.functional as F
import math

logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))

@dataclass
class CustomCausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    log_loss: Optional[Dict[str, float]] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

class ActionHead_p1(nn.Module):
    def __init__(self, hidden_size):
        super(ActionHead_p1, self).__init__()
        
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 7)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight) 
        
        self.relu = nn.ReLU()
        

    def forward(self, pred_states):
        states = self.fc1(pred_states)
        states = self.relu(states)
        action_prediction = self.fc2(states)

        return action_prediction
    
class ProprioHead(nn.Module):
    def __init__(self, hidden_size, input_dim=72):
        super(ProprioHead, self).__init__()
        
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, pred_states):
        return self.mlp1(pred_states)


def positional_encoding(x, n=10):
    """
    x: 输入位置，形状为 (batch_size, seq_len)
    n: 编码的阶数
    返回: 位置编码，形状为 (batch_size, seq_len, 2 * n)
    """
    positions = x.unsqueeze(-1)  # 形状变为 (batch_size, seq_len, 1)

    # 生成 n 阶的乘法器，从 1 到 n
    multiplier = torch.arange(0, n, dtype=x.dtype, device=x.device)  # 形状 (n,)

    # 扩展维度以与 positions 相乘
    multiplier = multiplier.unsqueeze(0)  # 形状 (1, n)

    # 计算 sin(nx) 和 cos(nx)
    sin_encodings = torch.sin(positions * (2**multiplier)*torch.pi)  # 形状 (batch_size, seq_len, n)
    cos_encodings = torch.cos(positions * (2**multiplier)*torch.pi)  # 形状 (batch_size, seq_len, n)

    # 将 sin 和 cos 编码在最后一个维度上进行拼接
    encoding = torch.cat([sin_encodings, cos_encodings], dim=-1)  # 形状 (batch_size, seq_len, 2 * n)

    return torch.flatten(encoding, start_dim=-2, end_dim=-1)

class ActionLoss(nn.Module):
    def __init__(self):
        super(ActionLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, action_predictions, target_pos, target_rot, target_grasp):
        """
        计算总损失，包括回归任务 (xyz, roll, pitch, yaw) 和二分类任务 (grasp)

        参数:
        - action_predictions: 模型预测的动作值，大小为 [batch_size, 7]
        - target_pos: 真实的 xyz 目标，大小为 [batch_size, 3]
        - target_rot: 真实的 roll, pitch, yaw 目标，大小为 [batch_size, 3]
        - target_grasp: 真实的 grasp 标签 (0 或 1)，大小为 [batch_size, 1]

        返回:
        - 总损失
        """
        predicted_pos = action_predictions[:, :3]
        predicted_rot = action_predictions[:, 3:6]
        predicted_grasp = action_predictions[:, 6:]
        predicted_grasp_prob = torch.sigmoid(predicted_grasp)


        pos_loss = self.mse_loss(predicted_pos, target_pos)
        rot_loss = self.mse_loss(predicted_rot, target_rot)
        # pos_loss = self.mae_loss(predicted_pos, target_pos)
        # rot_loss = self.mae_loss(predicted_rot, target_rot)

        grasp_loss = self.bce_loss(predicted_grasp_prob, target_grasp)

        total_loss = pos_loss + rot_loss + grasp_loss
        log_loss = {
            'pos_loss': pos_loss.item(),
            'rot_loss': rot_loss.item(),
            'grasp_loss': grasp_loss.item()
        }

        return total_loss, log_loss

def get_sinusoidal_embedding(indices, d_model=512):
    """
    indices: [B, N]，表示每个样本、每个时序位置的整数索引
    d_model: 输出嵌入的维度（偶数）
    
    return: [B, N, d_model]
            其中 d_model 里，偶数维用 sin(...)，奇数维用 cos(...)
    """
    # indices 转为 float，后面要做乘法
    positions = indices.float()  # [B, N]
    
    # 准备好 [0, 2, 4, ..., d_model-2] 这些下标
    dim = torch.arange(0, d_model, 2, dtype=torch.float, device=indices.device)  
    # 根据 dim / d_model 生成衰减频率，这里用 exp(-log(10000)*(dim/d_model))
    freq = torch.exp(-math.log(10000) * (dim / d_model))  # [d_model/2]

    # 计算角度 angles = pos * freq
    # positions.unsqueeze(-1): [B, N, 1]
    # freq.unsqueeze(0).unsqueeze(0): [1, 1, d_model/2]
    angles = positions.unsqueeze(-1) * freq.unsqueeze(0).unsqueeze(0)  # [B, N, d_model/2]

    # 分别计算 sin 和 cos
    sin = torch.sin(angles)  # [B, N, d_model/2]
    cos = torch.cos(angles)  # [B, N, d_model/2]

    # 最终我们要拼成 [B, N, d_model]
    # 假设 d_model=512，那么前 256 维放 sin，后 256 维放 cos，
    # 或者按奇偶维度交错写入。
    # 下面演示交错写法：向空的 [B, N, d_model] 里偶数下标放 sin，奇数下标放 cos
    
    embeddings = torch.zeros(indices.size(0), indices.size(1), d_model, device=indices.device)
    embeddings[..., 0::2] = sin
    embeddings[..., 1::2] = cos
    
    return embeddings  # [B, N, d_model]
    
class InternVLChatModel3DProprioRPESin(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'InternLM2DecoderLayer',
                         'Phi3DecoderLayer', 'Qwen2DecoderLayer']
    _supports_flash_attn_2 = True

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.num_image_token_hist = int((224 // patch_size) ** 2 * (config.downsample_ratio ** 2))

        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.llm_arch_name = config.llm_config.architectures[0]

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Phi3ForCausalLM':
                self.language_model = Phi3ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.img_context_token_id = None
        self.prop_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        if hasattr(config, 'system_message'):
            self.system_message = config.system_message
        else:
            self.system_message = self.conv_template.system_message
        self.num_samples = 0

        if config.use_backbone_lora:
            self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora)

        if config.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)

        self.llm_hidden_size = llm_hidden_size
        # self.action_head = ActionHead(llm_hidden_size)
        self.action_loss = ActionLoss()
        # self.pos_emb = SinusoidalPosEmb(18)


    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        # Determine the target modules based on the architecture of the language model
        if self.llm_arch_name == 'InternLM2ForCausalLM':
            target_modules = ['attention.wqkv', 'attention.wo', 'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']
        elif self.llm_arch_name == 'Phi3ForCausalLM':
            target_modules = ['mlp.down_proj', 'mlp.gate_up_proj', 'self_attn.o_proj', 'self_attn.qkv_proj']
        elif self.llm_arch_name in ['Qwen2ForCausalLM', 'LlamaForCausalLM']:
            target_modules = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                              'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']
        else:
            raise NotImplemented
        lora_config = LoraConfig(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM'
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            pixel_values_hist: torch.FloatTensor = None,
            frame_idx: torch.LongTensor = None,
            depth: torch.FloatTensor = None,
            depth_hist: torch.FloatTensor = None,
            depth_mask: torch.FloatTensor = None,
            depth_mask_hist: torch.FloatTensor = None,
            pose: torch.FloatTensor = None,
            intrinsics: torch.FloatTensor = None,
            proprio: torch.FloatTensor = None,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            prop_flags: Optional[torch.LongTensor] = None,
            image_flags_hist: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            action_labels: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CustomCausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_batch_size = pixel_values.shape[0] + (pixel_values_hist.shape[0] if pixel_values_hist is not None else 0)

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        # assert depth is not None
        if (depth_mask is None) and (depth is not None):
            depth_mask = torch.ones_like(depth)
        vit_embeds = self.extract_feature(pixel_values, depth, depth_mask, pose, intrinsics)  # bs*2, 256, 3072
        vit_embeds = vit_embeds[image_flags == 1]


        if pixel_values_hist is not None:
            if (depth_mask_hist is None) and (depth_hist is not None):
                depth_mask_hist = torch.ones_like(depth_hist)
            vit_embeds_hist = self.extract_feature(pixel_values_hist, depth_hist, depth_mask_hist, pose, intrinsics)
            vit_embeds_hist = vit_embeds_hist[image_flags_hist == 1]  # bs*4, 64, 3072
            N_curr = vit_embeds.shape[1]
            N_hist = vit_embeds_hist.shape[1]
            vit_embeds_ = rearrange(vit_embeds, '(b t) n c -> b (t n) c', b=B)  # bs, 1*256, 3072
            vit_embeds_hist_ = rearrange(vit_embeds_hist, '(b t) n c -> b (t n) c', b=B)  # bs, 4*64, 3072
            vit_embeds = torch.cat([vit_embeds_hist_, vit_embeds_], dim=1)

            frame_idx = frame_idx[:,-1:] - frame_idx
            frame_hist, frame_curr = frame_idx[:, :-1], frame_idx[:, -1:]
            embed_hist = get_sinusoidal_embedding(frame_hist, d_model=512)  # [B, N_hist, 512]
            embed_curr = get_sinusoidal_embedding(frame_curr, d_model=512)
            embed_hist = embed_hist.unsqueeze(2).expand(-1, -1, N_hist, -1).reshape(B, -1, 512)
            embed_curr = embed_curr.unsqueeze(2).expand(-1, -1, N_curr, -1).reshape(B, -1, 512)
            # embed_hist = self.time_embedding(frame_hist).unsqueeze(2).expand(-1, -1, N_hist, -1).reshape(B, -1, 512)
            # embed_curr = self.time_embedding(frame_curr).unsqueeze(2).expand(-1, -1, N_curr, -1).reshape(B, -1, 512)
            time_embed_ = torch.cat([embed_hist, embed_curr], dim=1).to(vit_embeds.dtype)
            time_embed_ = self.time_mlp(time_embed_)
            vit_embeds = time_embed_ + vit_embeds
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
            ignore_flag = False
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
            ignore_flag = True

        if (proprio is not None) and (sum(prop_flags) > 0):
            prop_flags = prop_flags.squeeze(-1)
            proprio = positional_encoding(proprio, n=4)
            proprio_embed = self.proprio_head(proprio)

            proprio_embed = proprio_embed.reshape(-1, C)
            proprio_embed = proprio_embed[prop_flags == 1]
            
            selected = (input_ids == self.prop_context_token_id)
            try:
                input_embeds[selected] = input_embeds[selected] * 0.0 + proprio_embed # .reshape(-1, C)
            except Exception as e:
                print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                    f'proprio_embed.shape={proprio_embed.shape}')
                n_token = selected.sum()
                input_embeds[selected] = input_embeds[selected] * 0.0 + proprio_embed[:n_token]


        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        hidden_states = outputs.hidden_states[-1]
        # pos_pred_states = hidden_states[:, -3, :]  # Hidden state for <pos_pred>
        # rot_pred_states = hidden_states[:, -2, :]  # Hidden state for <rot_pred>
        # grasp_pred_states = hidden_states[:, -1, :]  # Hidden state for <grasp_pred>
        # action_prediction = self.action_head(pos_pred_states, rot_pred_states, grasp_pred_states)

        last_index = (attention_mask == 1).sum(dim=1) - 1 
        batch_indices = torch.arange(hidden_states.size(0)).unsqueeze(1)  # [batch_size, 1]
        last_valid_index = last_index.unsqueeze(1)
        pred_states = hidden_states[batch_indices, last_valid_index, :].squeeze(1)
        
        action_prediction = self.action_head(pred_states)
        logits = outputs.logits

        loss = None
        if action_labels is not None:
            target_pos = action_labels[:, :3]
            target_rot = action_labels[:, 3:6]
            target_grasp = action_labels[:, 6:]
            loss, log_loss = self.action_loss(action_prediction, target_pos, target_rot, target_grasp)
            if ignore_flag:
                loss = loss * 0.0

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CustomCausalLMOutputWithPast(
            loss=loss,
            log_loss=log_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values, depths=None, depth_mask=None, poses=None, intrinsics=None):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]  # bs, h/14, h/14, 1024

        # h = w = int(vit_embeds.shape[1] ** 0.5)
        b, c, h, w = pixel_values.shape
        h = h // self.patch_size
        w = w // self.patch_size

        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)

        if depths is not None:
            # Project
            B, H, W, C = vit_embeds.shape
            # assert intrinsics.dim() == 4
            # (B, 32, 32, 3)
            vit_embeds = rearrange(vit_embeds, 'b h w c -> b c h w')
            feat_xyz, xyz, multi_scale_depth_masks = backprojector_dataloader([vit_embeds], depths, depth_mask, poses.unsqueeze(1), intrinsics.unsqueeze(1))
            # (B, V*H*W, C)
            vit_embeds = self.spatial_agg([vit_embeds], [feat_xyz.flatten(0, 1)], [multi_scale_depth_masks.flatten(0, 1)], (B, 1))[0]
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)

        
        # bs, h*downsample_ratio/14, w*downsample_ratio/14, 1024/(downsample_ratio^2)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep)[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False, pixel_values_hist=None, num_patches_list_hist=None):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list_hist:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token_hist * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            pixel_values_hist=pixel_values_hist,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            pixel_values_hist: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)


            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            
            if pixel_values_hist is not None:
                vit_embeds_hist = self.extract_feature(pixel_values_hist)
                vit_embeds_ = rearrange(vit_embeds, '(b t) n c -> b (t n) c', b=B)
                vit_embeds_hist_ = rearrange(vit_embeds_hist, '(b t) n c -> b (t n) c', b=B)
                vit_embeds = torch.cat([vit_embeds_hist_, vit_embeds_], dim=1)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs
