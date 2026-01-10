# import torch
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm

class DINOv2FeatureExtractor(nn.Module):
    def __init__(self, device='cuda', model_name='vit_base_patch14_dinov2', img_size=224):
        super().__init__()
        self.device = device
        self.img_size = img_size
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.tfm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            self.normalize,
        ])

        self.use_dino = False
        if True:
            try:
                self.model = timm.create_model(model_name, pretrained=True)
                self.model.reset_classifier(0)  # 去掉分类头
                self.model.eval().to(self.device)
                self.use_dino = True
            except Exception as e:
                print(f"[WARN] timm dinov2 load failed: {e}")

        if not self.use_dino:
            # 回退：resnet50 全局池化作为特征（也 L2 归一化）
            import torchvision.models as tvm
            self.model = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
            self.model.fc = nn.Identity()
            self.model.eval().to(self.device)

    @torch.no_grad()
    def extract(self, np_images):  # np_images: [T, H, W, 3], uint8
        bs = 64
        feats = []
        for i in range(0, len(np_images), bs):
            batch = np_images[i:i+bs]
            _imgs = []
            for img in batch:
                _imgs.append(self.tfm(img[:, :, ::-1] if img.shape[-1]==3 else img))  # 如果是BGR转RGB
            _imgs = torch.stack(_imgs, dim=0).to(self.device)
            f = self.model.forward_features(_imgs) if hasattr(self.model, 'forward_features') else self.model(_imgs)
            # 有的 ViT 返回 dict
            if isinstance(f, dict) and 'x_norm_clstoken' in f:
                f = f['x_norm_clstoken']
            elif isinstance(f, dict) and 'pooled' in f:
                f = f['pooled']
            f = f.mean(dim=(1,2)) if f.dim()==4 else f  # 兼容卷积backbone
            f = F.normalize(f, dim=-1)
            feats.append(f)
        feats = torch.cat(feats, dim=0)  # [T, D]
        return feats  # 已 L2 归一化

def _frame_to_vec(seg):
    """
    seg: [W, D] 或 [W, P, D]
    return: [W, D’] 且已 L2 归一化
    """
    if seg.dim() == 3:
        # [W, P, D] -> 先对 patch 维做平均/汇聚
        seg = seg.mean(dim=1)              # [W, D]
        seg = F.normalize(seg, dim=-1)     # L2 归一化
    elif seg.dim() == 2:
        seg = F.normalize(seg, dim=-1)     # [W, D], 直接归一化
    else:
        raise ValueError(f"Unexpected seg.dim()={seg.dim()}, expected 2 or 3.")
    return seg

@torch.no_grad()
def temporal_reduction_dino(feats, window_size=16, sim_thresh=0.83):
    """
    feats: [T, D] 或 [T, P, D]（建议已在特征维做过 L2 归一化；若未归一化，这里会处理）
    返回: keep_mask: [T] bool
    逻辑：按窗口计算每帧与同窗其它帧的平均余弦相似度，低于阈值的保留；
          每个窗口强制保留中间帧，防止全被滤掉（与 LongVU 代码风格一致）。
    """
    if isinstance(feats, list):
        feats = torch.stack(feats, dim=0)
    T = feats.shape[0]
    device = feats.device
    keep = torch.zeros(T, dtype=torch.bool, device=device)

    for s in range(0, T, window_size):
        e = min(s + window_size, T)
        seg = feats[s:e]               # [W, D] or [W, P, D]
        if seg.shape[0] == 1:
            keep[s] = True
            continue

        seg_vec = _frame_to_vec(seg)   # [W, D’], 已归一化
        S = seg_vec @ seg_vec.T        # [W, W]，余弦相似度矩阵
        W = seg_vec.shape[0]
        # 平均相似度（排除自身）
        avg_sim = (S.sum(dim=1) - 1.0) / max(W - 1, 1)

        # 低于阈值保留
        keep[s:e] = avg_sim < sim_thresh
        # 中间帧强制保底
        # mid = s + W // 2
        # keep[mid] = True

    return keep

@torch.no_grad()
def anchor_window_dedup(feats, keep_mask, window_size=16, anchor_thresh=0.90):
    """
    二级去重：在一级 keep 的基础上，再滑窗，以第一帧为 anchor，
    丢弃与 anchor 余弦相似度 >= anchor_thresh 的帧。
    返回: selected_fids (list[int])
    """
    if isinstance(feats, list):
        feats = torch.stack(feats, dim=0)
    idxs = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)  # [K]
    if idxs.numel() == 0:
        return []

    selected = []
    for s in range(0, idxs.numel(), window_size):
        e = min(s + window_size, idxs.numel())
        win = idxs[s:e]               # 原时间轴索引
        seg = feats[win]              # [W, D] or [W, P, D]
        seg_vec = _frame_to_vec(seg)  # [W, D’], 已归一化
        anchor = seg_vec[0]           # [D’]
        sims = seg_vec @ anchor       # [W]

        keep_local = sims < anchor_thresh
        keep_local[0] = True          # anchor 自己保留
        selected.extend(win[keep_local].tolist())

    return selected