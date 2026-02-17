"""
Learning-Based Fit Predictor — 회귀+분류 멀티헤드 모델

Phase 2에서 학습 데이터 확보 후 사용.
Rule-based와 동일 인터페이스 (predict / recommend_size).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from capvton.fit.schema import (
    BODY_PARTS,
    CATEGORY_RELEVANT_PARTS,
    EaseVector,
    FitClass,
    FitReport,
    GarmentCategory,
    GarmentMeasurements,
    GarmentSuperCategory,
    PartFitResult,
    RiskLevel,
    UserMeasurements,
    normalize_user,
)


# ──────────────────────────────────────────────
# Model Architecture
# ──────────────────────────────────────────────

NUM_PARTS = 8    # BODY_PARTS 개수
NUM_CLASSES = 5  # too_tight / tight / regular / loose / too_loose
NUM_CATEGORIES = len(GarmentCategory)


class FitPredictorNet(nn.Module):
    """
    학습 기반 Fit Predictor.

    입력: u_norm(12) ⊕ g_norm(15) ⊕ f_norm(10) ⊕ cat_emb(16) ⊕ uncertainty(12) = 65-d
    출력:
      - tightness: (B, K) 부위별 연속값
      - fit_class_logits: (B, K, 5) 부위별 5-class 로짓
      - overall_score: (B, 1) 전체 적합도
      - fit_embedding: (B, 128) 레이아웃 생성기 / FiLM 어댑터 입력용
    """

    def __init__(
        self,
        user_dim: int = 12,
        garment_dim: int = 15,
        ease_dim: int = 10,
        uncertainty_dim: int = 12,
        cat_embed_dim: int = 16,
        hidden_dim: int = 256,
        embed_dim: int = 128,
        num_parts: int = NUM_PARTS,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.15,
    ):
        super().__init__()

        self.num_parts = num_parts
        self.num_classes = num_classes

        # Category embedding
        self.cat_embedding = nn.Embedding(NUM_CATEGORIES, cat_embed_dim)

        input_dim = user_dim + garment_dim + ease_dim + cat_embed_dim + uncertainty_dim

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # Regression head (부위별 tightness)
        self.regression_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_parts),
            nn.Tanh(),  # [-1, +1]
        )

        # Classification head (부위별 5-class)
        self.classification_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_parts * num_classes),
        )

        # Overall score head
        self.overall_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        u_norm: torch.Tensor,      # (B, 12)
        g_norm: torch.Tensor,      # (B, 15)
        f_norm: torch.Tensor,      # (B, 10)
        cat_id: torch.LongTensor,  # (B,)
        uncertainty: torch.Tensor,  # (B, 12)
    ) -> Dict[str, torch.Tensor]:
        cat_emb = self.cat_embedding(cat_id)  # (B, 16)
        x = torch.cat([u_norm, g_norm, f_norm, cat_emb, uncertainty], dim=-1)

        # Shared encoding
        fit_embedding = self.encoder(x)  # (B, 128)

        # Heads
        tightness = self.regression_head(fit_embedding)  # (B, K)
        cls_logits = self.classification_head(fit_embedding)  # (B, K*5)
        cls_logits = cls_logits.view(-1, self.num_parts, self.num_classes)  # (B, K, 5)
        overall = self.overall_head(fit_embedding).squeeze(-1)  # (B,)

        return {
            "fit_embedding": fit_embedding,
            "tightness": tightness,
            "fit_class_logits": cls_logits,
            "overall_score": overall,
        }


# ──────────────────────────────────────────────
# Loss Function
# ──────────────────────────────────────────────

class FitPredictorLoss(nn.Module):
    """
    Multi-task loss with uncertainty-aware weighting.

    L = λ_reg × L_regression + λ_cls × L_classification + λ_ovr × L_overall

    추정치수 부위는 가중치 감소 (0.5).
    """

    def __init__(
        self,
        lambda_reg: float = 1.0,
        lambda_cls: float = 1.0,
        lambda_ovr: float = 0.5,
        estimated_weight: float = 0.5,
    ):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.lambda_cls = lambda_cls
        self.lambda_ovr = lambda_ovr
        self.estimated_weight = estimated_weight

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target_tightness: torch.Tensor,      # (B, K)
        target_class: torch.LongTensor,       # (B, K) — 0~4
        target_overall: torch.Tensor,         # (B,)
        part_mask: torch.Tensor,              # (B, K) — 1=관련 부위, 0=무관
        uncertainty_mask: torch.Tensor,       # (B, K) — 1=추정, 0=측정
    ) -> Dict[str, torch.Tensor]:
        # Part-wise weight: measured=1.0, estimated=0.5, irrelevant=0.0
        weight = part_mask * (1.0 - uncertainty_mask * (1.0 - self.estimated_weight))

        # Regression loss: Smooth L1
        reg_loss = F.smooth_l1_loss(
            pred["tightness"], target_tightness, reduction="none"
        )  # (B, K)
        reg_loss = (reg_loss * weight).sum() / (weight.sum() + 1e-8)

        # Classification loss: Cross-Entropy (per part)
        B, K, C = pred["fit_class_logits"].shape
        cls_logits = pred["fit_class_logits"].view(B * K, C)
        cls_target = target_class.view(B * K)
        cls_weight = weight.view(B * K)
        cls_loss = F.cross_entropy(cls_logits, cls_target, reduction="none")
        cls_loss = (cls_loss * cls_weight).sum() / (cls_weight.sum() + 1e-8)

        # Overall loss: BCE
        ovr_loss = F.binary_cross_entropy(
            pred["overall_score"], target_overall, reduction="mean"
        )

        total = (
            self.lambda_reg * reg_loss
            + self.lambda_cls * cls_loss
            + self.lambda_ovr * ovr_loss
        )

        return {
            "total": total,
            "regression": reg_loss,
            "classification": cls_loss,
            "overall": ovr_loss,
        }


# ──────────────────────────────────────────────
# Inference Wrapper (Rule-based와 동일 인터페이스)
# ──────────────────────────────────────────────

class MLFitPredictor:
    """
    학습된 FitPredictorNet을 래핑하여 predict() / recommend_size() 제공.
    """

    CLASS_MAP = [FitClass.TOO_TIGHT, FitClass.TIGHT, FitClass.REGULAR,
                 FitClass.LOOSE, FitClass.TOO_LOOSE]

    def __init__(
        self,
        model: FitPredictorNet,
        device: str = "cpu",
        mc_dropout_samples: int = 10,
    ):
        self.model = model.to(device)
        self.device = device
        self.mc_samples = mc_dropout_samples

    @torch.no_grad()
    def predict(
        self,
        user: UserMeasurements,
        garment: GarmentMeasurements,
    ) -> FitReport:
        user.fill_estimated()
        ease = EaseVector.compute(user, garment)

        # 텐서 준비
        u_norm = torch.from_numpy(normalize_user(user)).unsqueeze(0).to(self.device)
        g_vec = torch.from_numpy(garment.to_vector()).unsqueeze(0).to(self.device)
        f_vec = torch.from_numpy(ease.to_vector()).unsqueeze(0).to(self.device)
        cat_id = torch.tensor(
            [list(GarmentCategory).index(garment.category)], device=self.device
        )
        unc = torch.from_numpy(user.uncertainty_mask()).unsqueeze(0).to(self.device)

        # Forward (eval mode)
        self.model.eval()
        output = self.model(u_norm, g_vec, f_vec, cat_id, unc)

        # MC Dropout for uncertainty (inference시 dropout 유지)
        if self.mc_samples > 1:
            self.model.train()  # enable dropout
            mc_outputs = []
            for _ in range(self.mc_samples):
                mc_out = self.model(u_norm, g_vec, f_vec, cat_id, unc)
                mc_outputs.append(mc_out["tightness"].cpu().numpy())
            self.model.eval()
            mc_std = np.std(mc_outputs, axis=0).squeeze(0)  # (K,)
        else:
            mc_std = np.zeros(NUM_PARTS)

        # 결과 파싱
        tightness = output["tightness"].cpu().numpy().squeeze(0)  # (K,)
        cls_logits = output["fit_class_logits"].cpu().numpy().squeeze(0)  # (K, 5)
        overall = output["overall_score"].cpu().item()

        super_cat = garment.super_category.value
        relevant = CATEGORY_RELEVANT_PARTS.get(super_cat, BODY_PARTS[:6])

        parts: Dict[str, PartFitResult] = {}
        risk_parts: List[str] = []

        for i, part_name in enumerate(BODY_PARTS):
            if part_name not in relevant:
                continue
            if i >= len(tightness):
                continue

            t_val = float(tightness[i])
            probs = F.softmax(torch.from_numpy(cls_logits[i]), dim=-1).numpy()
            cls_idx = int(np.argmax(probs))
            fit_cls = self.CLASS_MAP[cls_idx]

            # Uncertainty-aware risk
            is_estimated = part_name in user.estimated_fields or any(
                f in user.estimated_fields
                for f in {"shoulder_width": ["shoulder"],
                          "arm_length": ["sleeve_length"],
                          "thigh": ["thigh"],
                          "inseam": ["inseam"]}.get(part_name, [])
            )
            uncertainty = float(mc_std[i]) if i < len(mc_std) else 0.0

            if fit_cls in (FitClass.TOO_TIGHT, FitClass.TOO_LOOSE):
                risk_level = RiskLevel.RISK
                risk_parts.append(part_name)
            elif is_estimated and uncertainty > 0.15:
                risk_level = RiskLevel.CAUTION
            else:
                risk_level = RiskLevel.OK

            confidence = max(0.0, 1.0 - uncertainty * 2)
            if is_estimated:
                confidence *= 0.7

            parts[part_name] = PartFitResult(
                tightness=t_val,
                fit_class=fit_cls,
                risk_level=risk_level,
                confidence=confidence,
                ease_raw=ease.values.get(part_name, 0.0),
            )

        size_label = garment.size_label or "?"
        notes = []
        if user.estimated_fields:
            notes.append(
                f"⚠️ 추정 기반 항목: {', '.join(user.estimated_fields)}"
            )

        return FitReport(
            overall_score=overall,
            size_recommendation=size_label,
            parts=parts,
            risk_parts=risk_parts,
            notes=notes,
        )

    def recommend_size(
        self,
        user: UserMeasurements,
        garment_sizes: Dict[str, GarmentMeasurements],
    ) -> FitReport:
        reports = {}
        for size_label, garment in garment_sizes.items():
            garment.size_label = size_label
            reports[size_label] = self.predict(user, garment)

        best_size = max(reports, key=lambda s: reports[s].overall_score)
        best = reports[best_size]
        best.size_recommendation = f"{best_size} 추천"
        best.all_sizes_scores = {
            s: round(r.overall_score, 3) for s, r in reports.items()
        }
        return best

    def get_fit_embedding(
        self,
        user: UserMeasurements,
        garment: GarmentMeasurements,
    ) -> np.ndarray:
        """Layout Generator / FiLM adapter 입력용 128-d 벡터."""
        user.fill_estimated()
        ease = EaseVector.compute(user, garment)

        u_norm = torch.from_numpy(normalize_user(user)).unsqueeze(0).to(self.device)
        g_vec = torch.from_numpy(garment.to_vector()).unsqueeze(0).to(self.device)
        f_vec = torch.from_numpy(ease.to_vector()).unsqueeze(0).to(self.device)
        cat_id = torch.tensor(
            [list(GarmentCategory).index(garment.category)], device=self.device
        )
        unc = torch.from_numpy(user.uncertainty_mask()).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(u_norm, g_vec, f_vec, cat_id, unc)
        return output["fit_embedding"].cpu().numpy().squeeze(0)
