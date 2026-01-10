# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

# Keep this package's public API minimal: only the modules required by
# `run_pretrain.sh` and `run_finetune.sh` are re-exported here.

from .configuration_intern_vit import InternVisionConfig
from .configuration_internvl_chat import InternVLChatConfig
from .modeling_intern_vit import InternVisionModel
from .modeling_internvl_chat import InternVLChatModel
from .modeling_internvl_chat_3d import InternVLChatModel3D
from .modeling_internvl_chat_action_head_p1 import ActionHead_p1
from .spatial_aware_module import SpatialAwareModule
from .modeling_internvl_chat_3d_proprio import InternVLChatModel3DProprio, ProprioHead
from .modeling_internvl_chat_3d_proprio_droid_mb import InternVLChatModel3DProprioDroidMB
from .modeling_internvl_chat_3d_proprio_ft_rpe_cat import InternVLChatModel3DProprioFTRPECat
from .modeling_internvl_chat_3d_proprio_ft_rpe_cat_wodepth import (
    InternVLChatModel3DProprioFTRPECatwodepth,
)
from .modeling_internvl_chat_3d_proprio_mb_rpe_cat import InternVLChatModel3DProprioRPECat

import importlib

__all__ = [
    'InternVisionConfig',
    'InternVisionModel',
    'InternVLChatConfig',
    'InternVLChatModel',
    'InternVLChatModel3D',
    'ActionHead_p1',
    'SpatialAwareModule',
    'InternVLChatModel3DProprio',
    'ProprioHead',
    'InternVLChatModel3DProprioDroidMB',
    'InternVLChatModel3DProprioFTRPECat',
    'InternVLChatModel3DProprioFTRPECatwodepth',
    'InternVLChatModel3DProprioRPECat',
]


# Legacy / experimental exports (moved under `internvl.model.internvl_chat.legacy`).
# Keep them importable to avoid breaking old entrypoints, but load them lazily.
_LEGACY_EXPORTS = {
    'MemoryInternVLChatModel': ('.legacy.modeling_internvl_chat_memory', 'MemoryInternVLChatModel'),
    'InternVLChatModelAH': ('.legacy.modeling_internvl_chat_action_head', 'InternVLChatModelAH'),
    'ActionHead': ('.legacy.modeling_internvl_chat_action_head', 'ActionHead'),
    'InternVLChatModelAH_p1': ('.modeling_internvl_chat_action_head_p1', 'InternVLChatModelAH_p1'),
    'InternVLChatModelAH_p1_hist': ('.legacy.modeling_internvl_chat_action_head_p1_hist', 'InternVLChatModelAH_p1_hist'),
    'ActionHead_p1_hist': ('.legacy.modeling_internvl_chat_action_head_p1_hist', 'ActionHead_p1_hist'),
    'InternVLChatModelAH_p1_Prio': ('.legacy.modeling_internvl_chat_action_head_p1_priopro', 'InternVLChatModelAH_p1_Prio'),
    'ActionHead_p1_prio': ('.legacy.modeling_internvl_chat_action_head_p1_priopro', 'ActionHead_p1_prio'),
    'InternVLChatModelDiffusionHead': ('.legacy.modeling_internvl_chat_diffusion_head', 'InternVLChatModelDiffusionHead'),
    'DiffusionHead': ('.legacy.modeling_internvl_chat_diffusion_head', 'DiffusionHead'),
    'InternVLChatModel3DProprioQformer': ('.legacy.modeling_internvl_chat_3d_proprio_qformer', 'InternVLChatModel3DProprioQformer'),
    'InternVLChatModel3DProprioQformerGridPooling': ('.legacy.modeling_internvl_chat_3d_proprio_qformer_grid_pooling', 'InternVLChatModel3DProprioQformerGridPooling'),
    'InternVLChatModel3DProprioAPE': ('.legacy.modeling_internvl_chat_3d_proprio_mb_ape', 'InternVLChatModel3DProprioAPE'),
    'InternVLChatModel3DProprioRPE': ('.legacy.modeling_internvl_chat_3d_proprio_mb_rpe', 'InternVLChatModel3DProprioRPE'),
    'InternVLChatModel3DProprioRPESin': ('.legacy.modeling_internvl_chat_3d_proprio_mb_rpe_sin', 'InternVLChatModel3DProprioRPESin'),
    'InternVLChatModelRLBench': ('.legacy.rlbench_mb_rpt_cat', 'InternVLChatModelRLBench'),
    'InternVLChatModel3DProprioFT': ('.legacy.modeling_internvl_chat_3d_proprio_ft', 'InternVLChatModel3DProprioFT'),
    'InternVLChatModel3DProprioFTCalvin': ('.legacy.modeling_internvl_chat_3d_proprio_ft_calvin', 'InternVLChatModel3DProprioFTCalvin'),
    'InternVLChatModel3DProprioDroid': ('.legacy.modeling_internvl_chat_3d_proprio_droid', 'InternVLChatModel3DProprioDroid'),
    'InternVLChatModel3DProprioRPECat_MV': ('.legacy.modeling_internvl_chat_3d_proprio_rpe_cat', 'InternVLChatModel3DProprioRPECat_MV'),
    'InternVLChatModelProprio': ('.legacy.modeling_internvl_chat_proprio', 'InternVLChatModelProprio'),
}


def __getattr__(name: str):
    if name not in _LEGACY_EXPORTS:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    module_rel, attr = _LEGACY_EXPORTS[name]
    module = importlib.import_module(module_rel, package=__name__)
    value = getattr(module, attr)
    globals()[name] = value
    return value
