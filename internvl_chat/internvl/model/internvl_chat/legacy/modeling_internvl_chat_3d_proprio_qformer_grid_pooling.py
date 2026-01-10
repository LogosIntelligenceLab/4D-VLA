# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import warnings
from typing import Any, List, Optional, Tuple, Union, Dict
import math
import torch.nn.functional as F
from einops import rearrange
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

logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))

def _pad_or_truncate(x: torch.Tensor, need: int) -> torch.Tensor:
    """确保返回 (need, C)。多则截断，少则零填充。"""
    if x is None or need == 0:
        return x.new_zeros((need, x.shape[-1])) if need > 0 else x
    have = x.shape[0]
    if have == need:
        return x
    if have > need:
        return x[:need]
    pad = torch.zeros((need - have, x.shape[-1]), dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=0)

def _maybe_project_to_lm(x: torch.Tensor, projector, target_dim: int) -> torch.Tensor:
    """若末维与 LLM 词向量维不同，则用 projector 投影。"""
    if x.shape[-1] != target_dim:
        x = projector(x)
    return x

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

class InternVLChatModel3DProprioQformerGridPooling(PreTrainedModel):
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

    def qformer_sample_history(
        self,
        vit_embeds_hist: torch.Tensor,          # (B,T,N,Cv) 或 (B*T,N,Cv)
        image_flags_hist: Optional[torch.Tensor],  # (B,T) 或 (B*T,1) 或 None
        query_tokens: torch.Tensor,             # (1,Q,Dq) 或 (B*?,Q,Dq) 也可
        qformer,                                # 你的 self.Qformer
        projector,                              # 你的 self.llm_proj 或 mm_projector，输出到 C_lm
        per_frame: bool = True,                 # True: 每帧采Q个；False: 整段采Q个
        B: Optional[int] = None,                # 无法从张量推断时可手动传入
        T: Optional[int] = None,                # 同上
    ) -> dict:
        """
        返回:
        {
            "flat": (B*K, C_lm),     # 用于直接写回 <HIST> 占位符；K= T*Q (per_frame) 或 Q (global)
            "btqc": (B, T*Q, C_lm) 或 (B, Q, C_lm),  # 结构化结果，便于调试/可视化
            "Q": int, "T": int
        }
        """
        if vit_embeds_hist is None:
            return {"flat": None, "btqc": None, "Q": 0, "T": 0}

        dev = vit_embeds_hist.device
        # 统一到 (B,T,N,Cv)
        if vit_embeds_hist.dim() == 3:
            # (B*T,N,Cv) -> 需要 B,T
            if (B is None) or (T is None):
                if image_flags_hist is not None:
                    if image_flags_hist.dim() == 2 and B is not None:
                        T = image_flags_hist.shape[1]
                    else:
                        # (B*T,1) 这种
                        if B is None:
                            raise ValueError("qformer_sample_history: 传入 (B*T,N,C) 时需提供 B 或可推断的 image_flags_hist")
                        T = int(image_flags_hist.numel() // B)
                else:
                    raise ValueError("qformer_sample_history: 无法从输入推断 (B,T)，请显式传入 B/T 或提供 image_flags_hist")
            BxT, N, Cv = vit_embeds_hist.shape
            assert BxT == B * T, f"shape不匹配: {BxT} != {B}*{T}"
            vit = vit_embeds_hist.view(B, T, N, Cv)
        elif vit_embeds_hist.dim() == 4:
            vit = vit_embeds_hist
            B, T, N, Cv = vit.shape
        else:
            raise ValueError(f"unexpected vit_embeds_hist dim={vit_embeds_hist.dim()}")

        # 构建注意力mask (B,T,N)
        if image_flags_hist is not None:
            if image_flags_hist.dim() == 2 and image_flags_hist.shape[0] == B:  # (B,T)
                valid_ft = image_flags_hist.to(dev).bool()
            else:  # (B*T,1)
                valid_ft = image_flags_hist.view(B, T).to(dev).bool()
            attn_mask = valid_ft.unsqueeze(-1).expand(B, T, N)  # (B,T,N)
        else:
            attn_mask = torch.ones((B, T, N), dtype=torch.bool, device=dev)

        Q = query_tokens.size(1)
        if per_frame:
            # 每帧 Q 个
            enc = rearrange(vit, 'b t n c -> (b t) n c')                    # (B*T,N,Cv)
            enc_mask = rearrange(attn_mask, 'b t n -> (b t) n').long()      # (B*T,N)
            q = query_tokens.expand(B * T, -1, -1)                          # (B*T,Q,Dq)
            out = qformer.bert(
                query_embeds=q,
                encoder_hidden_states=enc,
                encoder_attention_mask=enc_mask,
                return_dict=True,
            ).last_hidden_state[:, :Q, :]                                   # (B*T,Q,Dq)
            hist = projector(out)                                           # (B*T,Q,C_lm)
            btqc = rearrange(hist, '(b t) q c -> b (t q) c', b=B)           # (B,T*Q,C_lm)
            flat = btqc.reshape(-1, btqc.shape[-1])                         # (B*T*Q,C_lm)
            return {"flat": flat, "btqc": btqc, "Q": Q, "T": T}
        else:
            # 整段历史视为一段，整体采 Q 个
            enc = rearrange(vit, 'b t n c -> b (t n) c')                    # (B,T*N,Cv)
            enc_mask = rearrange(attn_mask, 'b t n -> b (t n)').long()      # (B,T*N)
            q = query_tokens.expand(B, -1, -1)                              # (B,Q,Dq)
            out = qformer.bert(
                query_embeds=q,
                encoder_hidden_states=enc,
                encoder_attention_mask=enc_mask,
                return_dict=True,
            ).last_hidden_state[:, :Q, :]                                   # (B,Q,Dq)
            btqc = projector(out)                                           # (B,Q,C_lm)
            flat = btqc.reshape(-1, btqc.shape[-1])                         # (B*Q,C_lm)
            return {"flat": flat, "btqc": btqc, "Q": Q, "T": T}

    def _pad_or_truncate(self,  x, need):
        """x: (S, C) -> 截断/补零到 need 行"""
        if x is None:
            return torch.zeros(need, self.language_model.config.hidden_size,
                               device=input_ids.device, dtype=input_embeds.dtype)
        S, C = x.shape
        if S == need:
            return x
        elif S > need:
            return x[:need]
        else:
            pad = torch.zeros(need - S, C, device=x.device, dtype=x.dtype)
            return torch.cat([x, pad], dim=0)


    def _content_pool_tokens(self, vis_embed, n=1):
        """
        Key part 2: 对单/多帧的 patch 特征做自适应平均池化，得到 n^2 个 content tokens
        vis_embed: (Bv, N, C)
        return: (Bv, n*n, C)  （如需严格论文样式，可在此处加投影 self.visproj）
        """
        if vis_embed is None or vis_embed.numel() == 0:
            return vis_embed
        Bv, N, C = vis_embed.shape
        H = int(round(math.sqrt(N)))
        W = max(1, N // H)
        # (Bv, N, C) -> (Bv, C, H, W)
        x = vis_embed.view(Bv, H, W, C).permute(0, 3, 1, 2).contiguous()
        x = F.adaptive_avg_pool2d(x, (n, n))              # (Bv, C, n, n)
        x = x.permute(0, 2, 3, 1).contiguous().view(Bv, n * n, C)  # (Bv, n^2, C)
        # 如需线性投影到 LLM 空间:
        # x = self.visproj(x)
        return x

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            pixel_values_hist: torch.FloatTensor = None,
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

        # ===== 0) 文本嵌入 =====
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()  # (B, L, C_lm)
        B, L, C_lm = input_embeds.shape

        # ===== 1) 当前图像特征（不使用深度）=====
        vit_embeds_cur = self.extract_feature(
            pixel_values,
            # depth=None, depth_mask=None,
            # pose=pose, intrinsics=intrinsics
        )  # 期望 (M, N_img, C_v) 或 (B, N_img, C_v) 或 (B*视角, N_img, C_v)

        # 基于 image_flags 选择有效的视角/帧（保持与占位符数一致）
        if image_flags is not None:
            image_flags = image_flags.squeeze(-1).to(dtype=torch.bool)  # (M,)
            vit_embeds_cur = vit_embeds_cur[image_flags]

        # 投影到 LLM 维度，保持分布一致
        # vit_embeds_cur = _maybe_project_to_lm(vit_embeds_cur, self.llm_proj, C_lm)
        vit_embeds_cur_flat = vit_embeds_cur.reshape(-1, C_lm)           # (sum_img_tokens, C_lm)

        # ===== 2) 历史 token：用 Q‑Former 做采样 =====
        hist_flat = None
        if pixel_values_hist is not None:
            # 提取历史帧 patch 特征
            vit_embeds_hist = self.extract_feature(
                pixel_values_hist,
                # depth=None, depth_mask=None,
                # pose=pose, intrinsics=intrinsics,
                is_hist=True
            )  # 期望形如 (B*T, N_img, C_lm) 或 (B', N_img, C_lm)

            if image_flags_hist is not None:
                image_flags_hist = image_flags_hist.to(dtype=torch.bool).squeeze(-1) if image_flags_hist.dim() == 2 else image_flags_hist
                vit_embeds_hist = vit_embeds_hist[image_flags_hist == 1]

            # -- 2.1 Q-Former（你的“Key part 1/上下文”实现原样保留） --
            # 下面这段保持你原来的做法（每帧 Q 个查询）：
            # 推断历史窗口长度 T 仅用于 expand query_tokens
            # if pixel_values_hist.dim() == 5:
            #     T = pixel_values_hist.shape[1]
            # else:
            #     T = getattr(self.config, "history_len", 9)
            vit_for_qformer = self.mlp_hist(self.ln_q(vit_embeds_hist))
            # vit_for_qformer = vit_embeds_hist  # (Bh, N, C)
            image_atts = torch.ones(vit_for_qformer.size()[:-1], dtype=torch.long, device=vit_for_qformer.device)
            Q = self.query_tokens.size(1)
            query_tokens = self.query_tokens[:, :Q, :].expand(vit_for_qformer.size(0), -1, -1)  # (Bh, Q, Dq)

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=vit_for_qformer,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            ctx_hist = self.llm_proj(query_output.last_hidden_state[:, :Q, :])  # (Bh, Q, C_lm)

            # -- 2.2 Key part 2：给历史帧也加 content token --
            content_hist = self._content_pool_tokens(vit_embeds_hist, n=2)  # (Bh, n^2, C_lm)

            # -- 2.3 拼接历史的 context+content，并展平 --
            hist_tokens = torch.cat([ctx_hist, content_hist], dim=1)  # (Bh, Q + n^2, C_lm)
            hist_flat = hist_tokens.reshape(-1, C_lm)                 # (Shist, C_lm)

        # ===== 3) 把视觉 token 写回到文本序列占位符位置 =====
        # 展平
        input_ids_flat = input_ids.view(B * L)
        input_embeds_flat = input_embeds.view(B * L, C_lm)

        # 当前图像占位符（如 <IMG>）
        sel_img = (input_ids_flat == self.img_context_token_id)
        need_img = int(sel_img.sum().item())
        vit_embeds_cur_flat = _pad_or_truncate(vit_embeds_cur_flat, need_img)
        if need_img > 0:
            input_embeds_flat[sel_img] = vit_embeds_cur_flat

        # 历史占位符（如 <HIST>）
        if hist_flat is not None:
            sel_hist = (input_ids_flat == self.qf_context_token_id)
            need_hist = int(sel_hist.sum().item())
            hist_flat = _pad_or_truncate(hist_flat, need_hist)
            if need_hist > 0:
                input_embeds_flat[sel_hist] = hist_flat

        # 回到 (B, L, C_lm)
        input_embeds = input_embeds_flat.view(B, L, C_lm)

        # ===== 4) （可选）本体感知写回 =====
        if (proprio is not None) and (prop_flags is not None) and (prop_flags.sum() > 0):
            prop_flags = prop_flags.squeeze(-1).bool()
            proprio_pe = positional_encoding(proprio, n=4)
            proprio_embed = self.proprio_head(proprio_pe)    # (B*?, C_lm) 或 (B, ?, C_lm)
            proprio_embed = proprio_embed.reshape(-1, C_lm)

            sel_prop = (input_ids_flat == self.prop_context_token_id)
            need_prop = int(sel_prop.sum().item())
            proprio_embed = _pad_or_truncate(proprio_embed, need_prop)
            if need_prop > 0:
                input_embeds_flat[sel_prop] = proprio_embed
                input_embeds = input_embeds_flat.view(B, L, C_lm)

        # ===== 5) 送入 LLM =====
        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=(return_dict if return_dict is not None else True),
        )

        hidden_states = outputs.hidden_states[-1]  # (B, L, C_lm)

        # 取每个样本最后一个有效 token 的隐状态
        if attention_mask is None:
            last_index = torch.full((B,), L - 1, device=hidden_states.device, dtype=torch.long)
        else:
            last_index = (attention_mask == 1).sum(dim=1) - 1
        batch_indices = torch.arange(B, device=hidden_states.device)
        pred_states = hidden_states[batch_indices, last_index, :]          # (B, C_lm)

        # 行为头预测
        action_prediction = self.action_head(pred_states)
        logits = outputs.logits

        # ===== 6) 行为损失 =====
        loss = None
        log_loss = None
        if action_labels is not None:
            target_pos = action_labels[:, :3]
            target_rot = action_labels[:, 3:6]
            target_grasp = action_labels[:, 6:]
            loss, log_loss = self.action_loss(action_prediction, target_pos, target_rot, target_grasp)

        if not (return_dict if return_dict is not None else True):
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

    def extract_feature(self, pixel_values, depths=None, depth_mask=None, poses=None, intrinsics=None, is_hist=False):
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
        if is_hist:
            return vit_embeds
        else:
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
