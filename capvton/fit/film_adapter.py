"""
FiLM Adapter — Fit-aware Feature-wise Linear Modulation

fit_embedding(128-d)을 U-Net의 각 해상도 블록에 FiLM(scale/shift)으로 주입.
기존 가중치를 그대로 유지하면서 핏 조건을 전달.
초기화: scale=1, shift=0 (identity) → 기존 성능 유지.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """
    단일 FiLM 레이어: condition → (scale, shift) → x * scale + shift.

    초기화: scale_fc.weight=0, bias=0 → scale=1, shift=0 (identity).
    """

    def __init__(self, cond_dim: int, feature_dim: int):
        super().__init__()
        self.scale_fc = nn.Linear(cond_dim, feature_dim)
        self.shift_fc = nn.Linear(cond_dim, feature_dim)

        # Zero-init → identity 변환에서 시작
        nn.init.zeros_(self.scale_fc.weight)
        nn.init.zeros_(self.scale_fc.bias)
        nn.init.zeros_(self.shift_fc.weight)
        nn.init.zeros_(self.shift_fc.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) — feature map
            cond: (B, cond_dim) — fit embedding

        Returns:
            (B, C, H, W) — modulated feature map
        """
        # scale: 1-centered (zero-init → starts at 1.0)
        scale = self.scale_fc(cond).unsqueeze(-1).unsqueeze(-1) + 1.0
        shift = self.shift_fc(cond).unsqueeze(-1).unsqueeze(-1)
        return x * scale + shift


class FitFiLMAdapter(nn.Module):
    """
    U-Net 전체에 대한 FiLM Adapter.

    CaP-VTON의 Generative UNet 구조:
      down_blocks: [320, 640, 1280, 1280]
      mid_block:   1280
      up_blocks:   [1280, 1280, 640, 320]

    각 블록의 ResNet 출력 직후에 FiLM을 적용.
    """

    def __init__(
        self,
        fit_embed_dim: int = 128,
        block_out_channels: List[int] = None,
    ):
        super().__init__()

        if block_out_channels is None:
            block_out_channels = [320, 640, 1280, 1280]

        # Down blocks
        self.down_films = nn.ModuleList([
            FiLMLayer(fit_embed_dim, ch) for ch in block_out_channels
        ])

        # Mid block
        self.mid_film = FiLMLayer(fit_embed_dim, block_out_channels[-1])

        # Up blocks (reversed)
        up_channels = list(reversed(block_out_channels))
        self.up_films = nn.ModuleList([
            FiLMLayer(fit_embed_dim, ch) for ch in up_channels
        ])

        # fit_embedding 차원 변환 (외부에서 다른 크기가 올 수 있으므로)
        self.cond_proj = None  # lazy init

    def _ensure_cond_proj(self, cond_dim: int, target_dim: int, device: torch.device):
        if self.cond_proj is None and cond_dim != target_dim:
            self.cond_proj = nn.Linear(cond_dim, target_dim).to(device)
            nn.init.eye_(self.cond_proj.weight[:target_dim, :target_dim])
            nn.init.zeros_(self.cond_proj.bias)

    def modulate_down(
        self,
        block_idx: int,
        hidden_states: torch.Tensor,
        fit_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Down block 출력에 FiLM 적용."""
        if block_idx >= len(self.down_films):
            return hidden_states
        return self.down_films[block_idx](hidden_states, fit_embedding)

    def modulate_mid(
        self,
        hidden_states: torch.Tensor,
        fit_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Mid block 출력에 FiLM 적용."""
        return self.mid_film(hidden_states, fit_embedding)

    def modulate_up(
        self,
        block_idx: int,
        hidden_states: torch.Tensor,
        fit_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Up block 출력에 FiLM 적용."""
        if block_idx >= len(self.up_films):
            return hidden_states
        return self.up_films[block_idx](hidden_states, fit_embedding)

    def forward(self, x: torch.Tensor, fit_embedding: torch.Tensor) -> torch.Tensor:
        """Generic forward (단일 FiLM 적용 — 테스트용)."""
        return self.mid_film(x, fit_embedding)


class FitEmbeddingEncoder(nn.Module):
    """
    FitReport → fit_embedding(128-d) 변환기.

    Rule-based predictor 사용 시: tightness vector + overall → 128-d
    ML predictor 사용 시: 이미 128-d를 출력하므로 identity.
    """

    def __init__(
        self,
        input_dim: int = 17,  # 8 tightness + 8 confidence + 1 overall
        embed_dim: int = 128,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, fit_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fit_vector: (B, 17) — from FitReport.to_embedding_input()
        Returns:
            (B, 128) fit embedding
        """
        return self.mlp(fit_vector)
