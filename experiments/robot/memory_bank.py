from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class _MbFeatureCfg:
    use_imagenet_weights: bool = True


class SimpleFeatureExtractor:
    def __init__(self, cfg: Optional[_MbFeatureCfg] = None, device: Optional[torch.device] = None):
        self.cfg = cfg or _MbFeatureCfg()
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        try:
            from torchvision.models import ResNet50_Weights, resnet50  # type: ignore

            weights = ResNet50_Weights.IMAGENET1K_V1 if self.cfg.use_imagenet_weights else None
            try:
                backbone = resnet50(weights=weights)
            except Exception:
                backbone = resnet50(weights=None)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Missing dependency: torchvision (required for memory-bank feature extraction)."
            ) from e

        self.resnet50 = torch.nn.Sequential(*list(backbone.children())[:-1]).to(self.device).eval()

    def extract_image_feature(self, image_batch: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Args:
            image_batch: (B, H, W, C) in uint8/float, or torch tensor.
        Returns:
            (B, 2048) float32 numpy array, L2-normalized.
        """
        if isinstance(image_batch, np.ndarray):
            batch = torch.from_numpy(image_batch)
        else:
            batch = image_batch

        if batch.dim() == 3:
            batch = batch.unsqueeze(0)

        if batch.dtype != torch.float32 and batch.dtype != torch.float16 and batch.dtype != torch.bfloat16:
            batch = batch.to(torch.float32)

        # (B, H, W, C) -> (B, C, H, W)
        if batch.shape[-1] == 3:
            batch = batch.permute(0, 3, 1, 2)

        # Normalize to [0,1]
        if batch.max() > 1.5:
            batch = batch / 255.0

        # Resize to 224x224 for ResNet
        batch = F.interpolate(batch, size=(224, 224), mode="bilinear", align_corners=False)

        mean = torch.tensor([0.485, 0.456, 0.406], device=batch.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=batch.device).view(1, 3, 1, 1)
        batch = (batch - mean) / std

        batch = batch.to(self.device)
        with torch.no_grad():
            feat = self.resnet50(batch)  # (B, 2048, 1, 1)
        feat = feat.squeeze(-1).squeeze(-1)  # (B, 2048)
        feat = feat / (feat.norm(dim=1, keepdim=True) + 1e-8)
        return feat.detach().cpu().to(torch.float32).numpy()


class SimpleMemoryBank:
    def __init__(self, bank_size: int = 5, window_size: int = 20, feature_cfg: Optional[_MbFeatureCfg] = None):
        self.bank_size = int(bank_size)
        self.window_size = int(window_size)

        self._features: list[np.ndarray] = []
        self._fids: list[int] = []
        self._images: list[np.ndarray] = []  # each: (1, H, W, C)
        self._depths: list[object] = []  # keep as-is (typically torch.Tensor), each: (1, H, W) or similar
        self.feature_extractor = SimpleFeatureExtractor(cfg=feature_cfg)

    def add(self, image: np.ndarray, depth, fid: int, from_img: bool = True):
        """
        Args:
            image: (1, H, W, C) or (H, W, C)
            depth: depth map associated with this image (kept as-is, can be None)
            fid: frame id / timestep
        """
        if image.ndim == 3:
            image = image[None, ...]

        if from_img:
            new_feature = self.feature_extractor.extract_image_feature(image)[0]  # (2048,)
        else:
            new_feature = np.asarray(image, dtype=np.float32).reshape(-1)

        fid = int(fid)

        # Drop frames outside window
        while self._fids and (fid - self._fids[0] > self.window_size):
            self._features.pop(0)
            self._fids.pop(0)
            self._images.pop(0)
            self._depths.pop(0)

        self._features.append(new_feature.astype(np.float32))
        self._fids.append(fid)
        self._images.append(image)
        self._depths.append(depth)

        # Keep at most bank_size by removing the most redundant (excluding newest).
        while len(self._features) > self.bank_size:
            if len(self._features) <= 2:
                # Keep newest + one older
                self._features.pop(0)
                self._fids.pop(0)
                self._images.pop(0)
                self._depths.pop(0)
                continue

            feats = np.stack(self._features, axis=0)  # (N, D)
            sims = (feats[:-1] * feats[1:]).sum(axis=1)  # cosine sim for adjacent normalized feats
            remove_idx = int(np.argmax(sims))
            # Never remove the newest frame.
            if remove_idx >= len(self._features) - 1:
                remove_idx = len(self._features) - 2

            self._features.pop(remove_idx)
            self._fids.pop(remove_idx)
            self._images.pop(remove_idx)
            self._depths.pop(remove_idx)

    def get_length(self) -> int:
        return len(self._images)

    def get_fid(self) -> list[int]:
        return list(self._fids)

    def get_image(self) -> list[np.ndarray]:
        return list(self._images)

    def get_depth(self):
        return list(self._depths)

    def clear(self):
        self._features.clear()
        self._fids.clear()
        self._images.clear()
        self._depths.clear()
