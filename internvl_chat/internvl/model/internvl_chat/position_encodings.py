"""
Various positional encodings for transformer-style models.

This file contains a mix of absolute (sine/learned) and rotary position encodings,
including 2D and 3D variants.
"""

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from:
# https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class PositionEmbeddingSine(nn.Module):
    """Sine/cosine 2D absolute position embedding for images.

    Input `x` is expected to be shaped as `(B, C, H, W)`. Optional `mask` is
    shaped as `(B, H, W)` where True indicates padded / invalid positions.
    """

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: float = 10000,
        normalize: bool = False,
        scale: Optional[float] = None,
    ):
        super().__init__()
        self.num_pos_feats = int(num_pos_feats)
        self.temperature = float(temperature)
        self.normalize = bool(normalize)

        if scale is not None and not normalize:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = float(2 * math.pi if scale is None else scale)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"x should be 4D (B,C,H,W), got shape={tuple(x.shape)}")

        bsz, _, height, width = x.shape
        if mask is None:
            mask = torch.zeros((bsz, height, width), device=x.device, dtype=torch.bool)
        else:
            if mask.shape != (bsz, height, width):
                raise ValueError(
                    f"mask should have shape (B,H,W)={(bsz, height, width)}, got {tuple(mask.shape)}"
                )
            if mask.dtype != torch.bool:
                mask = mask.to(dtype=torch.bool)
            if mask.device != x.device:
                mask = mask.to(device=x.device)

        not_mask = ~mask
        y_embed = not_mask.cumsum(dim=1, dtype=torch.float32)
        x_embed = not_mask.cumsum(dim=2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, device=x.device, dtype=torch.float32)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[..., None] / dim_t  # (B,H,W,C)
        pos_y = y_embed[..., None] / dim_t

        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2).contiguous()
        return pos

    def __repr__(self, _repr_indent: int = 4) -> str:
        head = f"Positional encoding {self.__class__.__name__}"
        body = [
            f"num_pos_feats: {self.num_pos_feats}",
            f"temperature: {self.temperature}",
            f"normalize: {self.normalize}",
            f"scale: {self.scale}",
        ]
        return "\n".join([head] + [" " * _repr_indent + line for line in body])


class PositionEmbeddingSine3D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: float = 10000,
        normalize: bool = False,
        scale: Optional[float] = None,
        add_temporal: bool = True,
    ):
        super().__init__()
        self.num_pos_feats = int(num_pos_feats)
        self.temperature = float(temperature)
        self.normalize = bool(normalize)
        self.add_temporal = bool(add_temporal)

        if scale is not None and not normalize:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = float(2 * math.pi if scale is None else scale)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # expected shape: (B, T, C, H, W)
        if x.dim() != 5:
            raise ValueError(f"x should be 5D (B,T,C,H,W), got shape={tuple(x.shape)}")

        bsz, time, _, height, width = x.shape
        if mask is None:
            mask = torch.zeros((bsz, time, height, width), device=x.device, dtype=torch.bool)
        else:
            if mask.shape != (bsz, time, height, width):
                raise ValueError(
                    f"mask should have shape (B,T,H,W)={(bsz, time, height, width)}, got {tuple(mask.shape)}"
                )
            if mask.dtype != torch.bool:
                mask = mask.to(dtype=torch.bool)
            if mask.device != x.device:
                mask = mask.to(device=x.device)

        not_mask = ~mask
        z_embed = not_mask.cumsum(dim=1, dtype=torch.float32)  # temporal
        y_embed = not_mask.cumsum(dim=2, dtype=torch.float32)  # height
        x_embed = not_mask.cumsum(dim=3, dtype=torch.float32)  # width

        if self.normalize:
            eps = 1e-6
            z_embed = z_embed / (z_embed[:, -1:, :, :] + eps) * self.scale
            y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, device=x.device, dtype=torch.float32)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        dim_t_z = torch.arange(self.num_pos_feats * 2, device=x.device, dtype=torch.float32)
        dim_t_z = self.temperature ** (
            2 * torch.div(dim_t_z, 2, rounding_mode="floor") / (self.num_pos_feats * 2)
        )

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_z = z_embed[..., None] / dim_t_z

        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=5).flatten(4)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=5).flatten(4)
        pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=5).flatten(4)

        if not self.add_temporal:
            pos_z = 0.0

        pos = (torch.cat((pos_y, pos_x), dim=4) + pos_z).permute(0, 1, 4, 2, 3).contiguous()
        return pos


class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 1:
            raise ValueError(f"x should be 1D (B,), got shape={tuple(x.shape)}")

        device = x.device
        half_dim = self.dim // 2
        if half_dim <= 0:
            raise ValueError(f"dim must be >= 2, got dim={self.dim}")

        # Handle half_dim==1 without division by zero.
        if half_dim == 1:
            frequencies = torch.ones((1,), device=device, dtype=torch.float32)
        else:
            step = math.log(10000.0) / (half_dim - 1)
            frequencies = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * (-step))

        args = x.to(dtype=torch.float32)[:, None] * frequencies[None, :]
        return torch.cat((args.sin(), args.cos()), dim=-1)


class RotaryPositionEncoding(nn.Module):
    def __init__(self, feature_dim, pe_type='Rotary1D'):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.pe_type = pe_type

    @staticmethod
    def embed_rotary(x, cos, sin):
        x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x).contiguous()
        x = x * cos + x2 * sin
        return x

    def forward(self, x_position):
        if x_position.dim() != 2:
            raise ValueError(f"x_position should be 2D (B,N), got shape={tuple(x_position.shape)}")
        bsize, npoint = x_position.shape
        if self.feature_dim % 2 != 0:
            raise ValueError(f"feature_dim must be even for Rotary1D, got feature_dim={self.feature_dim}")

        div_term = torch.exp(
            torch.arange(0, self.feature_dim, 2, dtype=torch.float32, device=x_position.device)
            * (-math.log(10000.0) / self.feature_dim)
        ).view(1, 1, -1)  # [1, 1, d/2]

        x_pos = x_position.to(dtype=torch.float32).unsqueeze(-1)  # [B, N, 1]
        sinx = torch.sin(x_pos * div_term)  # [B, N, d/2]
        cosx = torch.cos(x_pos * div_term)

        sin_pos, cos_pos = map(
            lambda feat: torch.stack([feat, feat], dim=-1).view(bsize, npoint, -1),
            [sinx, cosx]
        )
        position_code = torch.stack([cos_pos, sin_pos] , dim=-1)

        if position_code.requires_grad:
            position_code = position_code.detach()

        return position_code


class RotaryPositionEncoding3D(RotaryPositionEncoding):

    def __init__(self, feature_dim, pe_type='Rotary3D'):
        super(RotaryPositionEncoding3D, self).__init__(feature_dim, pe_type)

    @torch.no_grad()
    def forward(self, XYZ):
        '''
        @param XYZ: [B,N,3]
        @return:
        '''
        if XYZ.dim() != 3 or XYZ.size(-1) != 3:
            raise ValueError(f"XYZ should have shape (B,N,3), got shape={tuple(XYZ.shape)}")
        if self.feature_dim % 6 != 0:
            raise ValueError(
                f"feature_dim must be divisible by 6 for Rotary3D, got feature_dim={self.feature_dim}"
            )

        bsize, npoint, _ = XYZ.shape
        x_position, y_position, z_position = XYZ[..., 0:1], XYZ[..., 1:2], XYZ[..., 2:3]
        div_term = torch.exp(
            torch.arange(0, self.feature_dim // 3, 2, dtype=torch.float32, device=XYZ.device)
            * (-math.log(10000.0) / (self.feature_dim // 3))
        )
        div_term = div_term.view(1, 1, -1)  # [1, 1, d//6]

        sinx = torch.sin(x_position * div_term)  # [B, N, d//6]
        cosx = torch.cos(x_position * div_term)
        siny = torch.sin(y_position * div_term)
        cosy = torch.cos(y_position * div_term)
        sinz = torch.sin(z_position * div_term)
        cosz = torch.cos(z_position * div_term)

        sinx, cosx, siny, cosy, sinz, cosz = map(
            lambda feat: torch.stack([feat, feat], -1).view(bsize, npoint, -1),
            [sinx, cosx, siny, cosy, sinz, cosz]
        )

        position_code = torch.stack([
            torch.cat([cosx, cosy, cosz], dim=-1),  # cos_pos
            torch.cat([sinx, siny, sinz], dim=-1)  # sin_pos
        ], dim=-1)

        if position_code.requires_grad:
            position_code = position_code.detach()

        return position_code


class PositionEmbeddingLearned(nn.Module):
    """Absolute pos embedding, learned."""

    def __init__(self, dim=3, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(dim, num_pos_feats, kernel_size=1),
            nn.GroupNorm(1, num_pos_feats),
            nn.ReLU(),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, xyz):
        """Forward pass, xyz is (B, N, 3or6), output (B, F, N)."""
        shape_len = xyz.dim()
        if shape_len == 5:
            B, V, H, W, _ = xyz.shape
            xyz = xyz.flatten(1, 3).permute(0, 2, 1)
        elif shape_len == 3:
            xyz = xyz.permute(0, 2, 1)
        else:
            raise ValueError("xyz should be 3 or 5 dimensional")
        position_embedding = self.position_embedding_head(xyz)
        if shape_len == 5:
            return position_embedding.permute(0, 2, 1).reshape(B, V, H, W, -1)
        else:
            return position_embedding.permute(0, 2, 1)
        
        
class PositionEmbeddingLearnedMLP(nn.Module):
    """Absolute pos embedding, learned."""

    def __init__(self, dim=3, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Linear(dim, num_pos_feats),
            nn.LayerNorm(num_pos_feats),
            nn.ReLU(),
            nn.Linear(num_pos_feats, num_pos_feats))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, xyz):
        """Forward pass, xyz is (B, N, 3or6), output (B, N, F)."""
        shape_len = xyz.dim()
        if shape_len == 5:
            B, V, H, W, _ = xyz.shape
            xyz = xyz.flatten(1, 3)
        elif shape_len == 3 or shape_len == 2:
            xyz = xyz
        else:
            raise ValueError("xyz should be 2, 3 or 5 dimensional")
        position_embedding = self.position_embedding_head(xyz)
        if shape_len == 5:
            return position_embedding.reshape(B, V, H, W, -1)
        else:
            return position_embedding
