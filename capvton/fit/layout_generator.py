"""
Fit-aware Layout Generator

Agnostic mask + DensePose + fit_embedding → target_mask + SDF map.

경량 U-Net 구조 (256×192 해상도에서 동작).
FiLM conditioning으로 fit 조건을 각 레벨에 주입.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from capvton.fit.film_adapter import FiLMLayer


# ──────────────────────────────────────────────
# Building Blocks
# ──────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv → GroupNorm → GELU → Conv → GroupNorm → GELU"""

    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.act = nn.GELU()

        # Residual projection
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        return x + residual


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int = 128):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.film = FiLMLayer(cond_dim, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        x = self.film(x, cond)
        skip = x
        x = self.pool(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, cond_dim: int = 128):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)
        self.film = FiLMLayer(cond_dim, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Padding if sizes mismatch
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.film(x, cond)
        return x


# ──────────────────────────────────────────────
# Layout Generator Network
# ──────────────────────────────────────────────

class FitLayoutGenerator(nn.Module):
    """
    Fit-aware Layout Generator.

    입력:
      - agnostic_mask: (B, 1, H, W) — 기존 garment agnostic mask
      - densepose_seg: (B, 3, H, W) — DensePose segmentation (RGB)
      - fit_embedding: (B, 128)     — from Fit Predictor

    출력:
      - target_mask: (B, 1, H, W)  — sigmoid, soft 의류 영역 마스크
      - sdf_map: (B, 1, H, W)      — tanh × max_dist, 부호거리함수

    내부 해상도: 256×192 (1/4 다운스케일), 최종 업스케일
    """

    def __init__(
        self,
        in_channels: int = 4,     # mask(1) + densepose(3)
        cond_dim: int = 128,      # fit embedding dimension
        base_ch: int = 64,        # 기본 채널 수
        target_h: int = 256,      # 내부 처리 해상도
        target_w: int = 192,
        output_h: int = 1024,     # 최종 출력 해상도
        output_w: int = 768,
        max_sdf_dist: float = 50.0,  # SDF 최대 거리 (px)
    ):
        super().__init__()

        self.target_h = target_h
        self.target_w = target_w
        self.output_h = output_h
        self.output_w = output_w
        self.max_sdf_dist = max_sdf_dist

        # Encoder
        self.enc1 = DownBlock(in_channels, base_ch, cond_dim)       # → base_ch, H/2
        self.enc2 = DownBlock(base_ch, base_ch * 2, cond_dim)       # → base_ch*2, H/4
        self.enc3 = DownBlock(base_ch * 2, base_ch * 4, cond_dim)   # → base_ch*4, H/8

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 4)
        self.bottleneck_film = FiLMLayer(cond_dim, base_ch * 4)

        # Decoder
        self.dec3 = UpBlock(base_ch * 4, base_ch * 4, base_ch * 2, cond_dim)  # H/4
        self.dec2 = UpBlock(base_ch * 2, base_ch * 2, base_ch, cond_dim)       # H/2
        self.dec1 = UpBlock(base_ch, base_ch, base_ch, cond_dim)               # H

        # Output heads
        self.mask_head = nn.Conv2d(base_ch, 1, 1)   # sigmoid
        self.sdf_head = nn.Conv2d(base_ch, 1, 1)    # tanh × max_dist

    def forward(
        self,
        agnostic_mask: torch.Tensor,  # (B, 1, H, W)
        densepose_seg: torch.Tensor,  # (B, 3, H, W)
        fit_embedding: torch.Tensor,  # (B, 128)
    ) -> Dict[str, torch.Tensor]:
        # Resize to internal resolution
        x = torch.cat([agnostic_mask, densepose_seg], dim=1)  # (B, 4, H, W)
        x = F.interpolate(x, size=(self.target_h, self.target_w), mode="bilinear", align_corners=False)

        # Encoder
        x, skip1 = self.enc1(x, fit_embedding)
        x, skip2 = self.enc2(x, fit_embedding)
        x, skip3 = self.enc3(x, fit_embedding)

        # Bottleneck
        x = self.bottleneck(x)
        x = self.bottleneck_film(x, fit_embedding)

        # Decoder
        x = self.dec3(x, skip3, fit_embedding)
        x = self.dec2(x, skip2, fit_embedding)
        x = self.dec1(x, skip1, fit_embedding)

        # Output heads
        mask_out = torch.sigmoid(self.mask_head(x))         # (B, 1, target_h, target_w)
        sdf_out = torch.tanh(self.sdf_head(x)) * self.max_sdf_dist  # (B, 1, target_h, target_w)

        # Upsample to output resolution
        mask_out = F.interpolate(mask_out, (self.output_h, self.output_w), mode="bilinear", align_corners=False)
        sdf_out = F.interpolate(sdf_out, (self.output_h, self.output_w), mode="bilinear", align_corners=False)

        return {
            "target_mask": mask_out,
            "sdf_map": sdf_out,
        }

    def to_latent_cond(
        self,
        target_mask: torch.Tensor,
        sdf_map: torch.Tensor,
        latent_h: int = 128,
        latent_w: int = 96,
    ) -> torch.Tensor:
        """
        원본 해상도 출력 → latent space 해상도로 변환.
        U-Net conv_in에 concat할 조건 (2-ch).

        Args:
            target_mask: (B, 1, H, W)
            sdf_map: (B, 1, H, W)
            latent_h, latent_w: H/8, W/8

        Returns:
            (B, 2, latent_h, latent_w) — mask + normalized SDF
        """
        mask_lat = F.interpolate(target_mask, (latent_h, latent_w), mode="bilinear", align_corners=False)
        sdf_lat = F.interpolate(sdf_map, (latent_h, latent_w), mode="bilinear", align_corners=False)
        # SDF 정규화: [-1, 1]
        sdf_lat = sdf_lat / (self.max_sdf_dist + 1e-8)
        return torch.cat([mask_lat, sdf_lat], dim=1)


# ──────────────────────────────────────────────
# Layout Generator Loss
# ──────────────────────────────────────────────

class LayoutGeneratorLoss(nn.Module):
    """
    Layout Generator 학습용 손실 함수.

    L = λ_mask × BCE(mask) + λ_sdf × L1(sdf) + λ_boundary × BoundaryConsistency
    """

    def __init__(
        self,
        lambda_mask: float = 1.0,
        lambda_sdf: float = 0.5,
        lambda_boundary: float = 0.3,
    ):
        super().__init__()
        self.lambda_mask = lambda_mask
        self.lambda_sdf = lambda_sdf
        self.lambda_boundary = lambda_boundary

    def forward(
        self,
        pred_mask: torch.Tensor,   # (B, 1, H, W), sigmoid output
        pred_sdf: torch.Tensor,    # (B, 1, H, W), raw SDF
        gt_mask: torch.Tensor,     # (B, 1, H, W), binary
        gt_sdf: torch.Tensor,      # (B, 1, H, W), ground truth SDF
    ) -> Dict[str, torch.Tensor]:
        # Mask loss: BCE
        mask_loss = F.binary_cross_entropy(pred_mask, gt_mask, reduction="mean")

        # SDF loss: L1
        sdf_loss = F.l1_loss(pred_sdf, gt_sdf, reduction="mean")

        # Boundary consistency: mask 경계와 SDF=0 라인의 일관성
        # SDF=0 부근(|SDF|<2px)에서 mask가 0.4~0.6이어야 함
        boundary_region = (gt_sdf.abs() < 2.0).float()
        if boundary_region.sum() > 0:
            boundary_target = 0.5  # 경계에서 mask ≈ 0.5
            boundary_loss = (
                (pred_mask - boundary_target).abs() * boundary_region
            ).sum() / (boundary_region.sum() + 1e-8)
        else:
            boundary_loss = torch.tensor(0.0, device=pred_mask.device)

        total = (
            self.lambda_mask * mask_loss
            + self.lambda_sdf * sdf_loss
            + self.lambda_boundary * boundary_loss
        )

        return {
            "total": total,
            "mask_loss": mask_loss,
            "sdf_loss": sdf_loss,
            "boundary_loss": boundary_loss,
        }


# ──────────────────────────────────────────────
# SDF Computation Utility
# ──────────────────────────────────────────────

def compute_sdf(mask: np.ndarray, max_dist: float = 50.0) -> np.ndarray:
    """
    바이너리 마스크로부터 SDF(Signed Distance Function)를 계산.

    Args:
        mask: (H, W) uint8, 0 or 255
        max_dist: 최대 거리 (클리핑)

    Returns:
        sdf: (H, W) float32, 양수=마스크 내부, 음수=외부
    """
    import cv2

    if mask.max() > 1:
        mask = (mask > 127).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)

    # 외부 거리 (마스크 밖 → 경계까지)
    dist_outside = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
    # 내부 거리 (마스크 안 → 경계까지)
    dist_inside = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # SDF: 내부 양수, 외부 음수
    sdf = dist_inside - dist_outside
    sdf = np.clip(sdf, -max_dist, max_dist)

    return sdf.astype(np.float32)
