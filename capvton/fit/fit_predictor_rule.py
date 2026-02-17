"""
Rule-Based Fit Predictor (MVP) — 즉시 배포 가능

의류공학 표준 ease 임계값 기반으로 부위별 핏 판정 + 사이즈 추천.
학습 데이터 불필요. 카테고리/성별별 규칙 테이블만으로 동작.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from capvton.fit.schema import (
    BODY_PARTS,
    CATEGORY_RELEVANT_PARTS,
    EaseVector,
    FitClass,
    FitReport,
    GarmentMeasurements,
    GarmentSuperCategory,
    PartFitResult,
    RiskLevel,
    UserMeasurements,
)


# ──────────────────────────────────────────────
# Ease Standards (의류공학 기반 임계값)
# ──────────────────────────────────────────────
# 각 값: (min_ease_ratio, max_ease_ratio) for that fit class
# ease_ratio = (garment_circ - body_circ) / body_circ

EASE_STANDARDS = {
    # ─── 상의 ───
    "tshirt": {
        "chest":         {"too_tight": (-1.0, -0.05), "tight": (-0.05, 0.02), "regular": (0.02, 0.12), "loose": (0.12, 0.25), "too_loose": (0.25, 1.0)},
        "waist":         {"too_tight": (-1.0, -0.05), "tight": (-0.05, 0.03), "regular": (0.03, 0.15), "loose": (0.15, 0.30), "too_loose": (0.30, 1.0)},
        "shoulder":      {"too_tight": (-1.0, -0.03), "tight": (-0.03, 0.00), "regular": (0.00, 0.05), "loose": (0.05, 0.12), "too_loose": (0.12, 1.0)},
        "sleeve_length": {"too_tight": (-1.0, -0.10), "tight": (-0.10, -0.03), "regular": (-0.03, 0.05), "loose": (0.05, 0.12), "too_loose": (0.12, 1.0)},
        "length":        {"too_tight": (-1.0, -0.10), "tight": (-0.10, -0.03), "regular": (-0.03, 0.05), "loose": (0.05, 0.15), "too_loose": (0.15, 1.0)},
    },
    "shirt": {
        "chest":         {"too_tight": (-1.0, -0.03), "tight": (-0.03, 0.03), "regular": (0.03, 0.10), "loose": (0.10, 0.20), "too_loose": (0.20, 1.0)},
        "waist":         {"too_tight": (-1.0, -0.03), "tight": (-0.03, 0.03), "regular": (0.03, 0.12), "loose": (0.12, 0.25), "too_loose": (0.25, 1.0)},
        "shoulder":      {"too_tight": (-1.0, -0.02), "tight": (-0.02, 0.01), "regular": (0.01, 0.04), "loose": (0.04, 0.10), "too_loose": (0.10, 1.0)},
        "sleeve_length": {"too_tight": (-1.0, -0.08), "tight": (-0.08, -0.02), "regular": (-0.02, 0.04), "loose": (0.04, 0.10), "too_loose": (0.10, 1.0)},
        "length":        {"too_tight": (-1.0, -0.08), "tight": (-0.08, -0.02), "regular": (-0.02, 0.05), "loose": (0.05, 0.12), "too_loose": (0.12, 1.0)},
    },
    "jacket": {
        "chest":         {"too_tight": (-1.0, -0.02), "tight": (-0.02, 0.05), "regular": (0.05, 0.15), "loose": (0.15, 0.30), "too_loose": (0.30, 1.0)},
        "waist":         {"too_tight": (-1.0, -0.02), "tight": (-0.02, 0.05), "regular": (0.05, 0.18), "loose": (0.18, 0.35), "too_loose": (0.35, 1.0)},
        "shoulder":      {"too_tight": (-1.0, -0.02), "tight": (-0.02, 0.02), "regular": (0.02, 0.06), "loose": (0.06, 0.15), "too_loose": (0.15, 1.0)},
        "sleeve_length": {"too_tight": (-1.0, -0.08), "tight": (-0.08, -0.02), "regular": (-0.02, 0.05), "loose": (0.05, 0.12), "too_loose": (0.12, 1.0)},
        "length":        {"too_tight": (-1.0, -0.10), "tight": (-0.10, -0.03), "regular": (-0.03, 0.05), "loose": (0.05, 0.15), "too_loose": (0.15, 1.0)},
    },
    # ─── 하의 ───
    "pants": {
        "waist":  {"too_tight": (-1.0, -0.05), "tight": (-0.05, 0.00), "regular": (0.00, 0.08), "loose": (0.08, 0.18), "too_loose": (0.18, 1.0)},
        "hip":    {"too_tight": (-1.0, -0.03), "tight": (-0.03, 0.02), "regular": (0.02, 0.10), "loose": (0.10, 0.22), "too_loose": (0.22, 1.0)},
        "thigh":  {"too_tight": (-1.0, -0.05), "tight": (-0.05, 0.00), "regular": (0.00, 0.10), "loose": (0.10, 0.25), "too_loose": (0.25, 1.0)},
        "length": {"too_tight": (-1.0, -0.08), "tight": (-0.08, -0.02), "regular": (-0.02, 0.03), "loose": (0.03, 0.10), "too_loose": (0.10, 1.0)},
        "inseam": {"too_tight": (-1.0, -0.08), "tight": (-0.08, -0.02), "regular": (-0.02, 0.03), "loose": (0.03, 0.10), "too_loose": (0.10, 1.0)},
    },
    "jeans": {
        "waist":  {"too_tight": (-1.0, -0.05), "tight": (-0.05, 0.00), "regular": (0.00, 0.06), "loose": (0.06, 0.15), "too_loose": (0.15, 1.0)},
        "hip":    {"too_tight": (-1.0, -0.03), "tight": (-0.03, 0.02), "regular": (0.02, 0.08), "loose": (0.08, 0.18), "too_loose": (0.18, 1.0)},
        "thigh":  {"too_tight": (-1.0, -0.05), "tight": (-0.05, 0.00), "regular": (0.00, 0.08), "loose": (0.08, 0.20), "too_loose": (0.20, 1.0)},
        "length": {"too_tight": (-1.0, -0.08), "tight": (-0.08, -0.02), "regular": (-0.02, 0.03), "loose": (0.03, 0.08), "too_loose": (0.08, 1.0)},
        "inseam": {"too_tight": (-1.0, -0.08), "tight": (-0.08, -0.02), "regular": (-0.02, 0.03), "loose": (0.03, 0.08), "too_loose": (0.08, 1.0)},
    },
    # ─── 원피스 ───
    "dress": {
        "chest":         {"too_tight": (-1.0, -0.05), "tight": (-0.05, 0.02), "regular": (0.02, 0.12), "loose": (0.12, 0.28), "too_loose": (0.28, 1.0)},
        "waist":         {"too_tight": (-1.0, -0.05), "tight": (-0.05, 0.02), "regular": (0.02, 0.15), "loose": (0.15, 0.30), "too_loose": (0.30, 1.0)},
        "hip":           {"too_tight": (-1.0, -0.03), "tight": (-0.03, 0.03), "regular": (0.03, 0.12), "loose": (0.12, 0.25), "too_loose": (0.25, 1.0)},
        "shoulder":      {"too_tight": (-1.0, -0.03), "tight": (-0.03, 0.00), "regular": (0.00, 0.05), "loose": (0.05, 0.12), "too_loose": (0.12, 1.0)},
        "sleeve_length": {"too_tight": (-1.0, -0.10), "tight": (-0.10, -0.03), "regular": (-0.03, 0.05), "loose": (0.05, 0.12), "too_loose": (0.12, 1.0)},
        "length":        {"too_tight": (-1.0, -0.12), "tight": (-0.12, -0.05), "regular": (-0.05, 0.08), "loose": (0.08, 0.18), "too_loose": (0.18, 1.0)},
    },
}

# 기본 표준 (테이블에 없는 카테고리용)
_DEFAULT_STANDARDS = EASE_STANDARDS["tshirt"]


def _get_standards(category: str) -> dict:
    """카테고리명으로 ease 표준 테이블 조회."""
    return EASE_STANDARDS.get(category, _DEFAULT_STANDARDS)


class RuleBasedFitPredictor:
    """
    규칙/통계 기반 Fit Predictor (MVP).

    사용법:
        predictor = RuleBasedFitPredictor()
        report = predictor.predict(user, garment)
        best = predictor.recommend_size(user, garment_sizes)
    """

    def __init__(self, preference: str = "regular"):
        """
        Args:
            preference: 사용자 선호 핏 ("tight", "regular", "loose")
        """
        self.preference = preference

    def predict(
        self,
        user: UserMeasurements,
        garment: GarmentMeasurements,
    ) -> FitReport:
        """
        단일 사이즈에 대한 핏 판정.

        Args:
            user: 사용자 신체 치수
            garment: 의류 실측 (특정 사이즈)

        Returns:
            FitReport: 전체/부위별 판정 결과
        """
        user.fill_estimated()
        ease = EaseVector.compute(user, garment)
        standards = _get_standards(garment.category.value)
        super_cat = garment.super_category.value
        relevant_parts = CATEGORY_RELEVANT_PARTS.get(super_cat, BODY_PARTS[:6])

        parts: Dict[str, PartFitResult] = {}
        risk_parts: List[str] = []

        for part in relevant_parts:
            if part not in ease.values:
                continue

            e = ease.values[part]
            part_standards = standards.get(part)
            if part_standards is None:
                continue

            # 어느 fit class에 해당하는지 판정
            fit_cls = FitClass.REGULAR
            for cls_name, (lo, hi) in part_standards.items():
                if lo <= e < hi:
                    fit_cls = FitClass(cls_name)
                    break

            # tightness 연속값: -1 ~ +1 매핑
            tightness = self._ease_to_tightness(e, part_standards)

            # 리스크 판정
            is_estimated = part in self._estimated_related_fields(user, part)
            if fit_cls in (FitClass.TOO_TIGHT, FitClass.TOO_LOOSE):
                risk_level = RiskLevel.RISK
                risk_parts.append(part)
            elif is_estimated and fit_cls in (FitClass.TIGHT, FitClass.LOOSE):
                risk_level = RiskLevel.CAUTION
            else:
                risk_level = RiskLevel.OK

            confidence = 0.6 if is_estimated else 1.0

            parts[part] = PartFitResult(
                tightness=tightness,
                fit_class=fit_cls,
                risk_level=risk_level,
                confidence=confidence,
                ease_raw=e,
            )

        # 전체 점수: 리스크 없는 비율 + 선호 매칭 보너스
        if len(parts) == 0:
            overall = 0.5
        else:
            # 기본: 리스크 없는 비율
            ok_ratio = sum(1 for p in parts.values() if p.risk_level == RiskLevel.OK) / len(parts)
            # 선호 매칭: regular → FitClass.REGULAR에 가까울수록 보너스
            pref_bonus = self._preference_bonus(parts)
            overall = min(1.0, ok_ratio * 0.7 + pref_bonus * 0.3)

        # Notes
        notes = []
        if user.estimated_fields:
            notes.append(
                f"⚠️ 추정 기반 항목: {', '.join(user.estimated_fields)}. "
                "정확한 치수 입력 시 더 정확한 결과를 받으실 수 있습니다."
            )
        for rp in risk_parts:
            if parts[rp].fit_class == FitClass.TOO_TIGHT:
                notes.append(f"🔴 {rp}: 매우 타이트합니다. 한 사이즈 업을 고려하세요.")
            elif parts[rp].fit_class == FitClass.TOO_LOOSE:
                notes.append(f"🟡 {rp}: 매우 여유롭습니다. 한 사이즈 다운을 고려하세요.")

        size_label = garment.size_label or "?"
        return FitReport(
            overall_score=overall,
            size_recommendation=f"{size_label}",
            parts=parts,
            risk_parts=risk_parts,
            notes=notes,
        )

    def recommend_size(
        self,
        user: UserMeasurements,
        garment_sizes: Dict[str, GarmentMeasurements],
    ) -> FitReport:
        """
        여러 사이즈 중 최적 사이즈 추천.

        Args:
            user: 사용자 신체 치수
            garment_sizes: {"S": GarmentMeasurements, "M": ..., "L": ...}

        Returns:
            최적 사이즈의 FitReport (all_sizes_scores 포함)
        """
        reports: Dict[str, FitReport] = {}
        for size_label, garment in garment_sizes.items():
            garment.size_label = size_label
            report = self.predict(user, garment)
            reports[size_label] = report

        # 최고 점수 사이즈 선택
        best_size = max(reports, key=lambda s: reports[s].overall_score)
        best_report = reports[best_size]
        best_report.size_recommendation = f"{best_size} 추천"
        best_report.all_sizes_scores = {
            s: round(r.overall_score, 3) for s, r in reports.items()
        }

        # 근접 사이즈 안내
        scores_sorted = sorted(reports.items(), key=lambda x: x[1].overall_score, reverse=True)
        if len(scores_sorted) > 1:
            second = scores_sorted[1]
            if second[1].overall_score > 0.7:
                best_report.notes.append(
                    f"ℹ️ {second[0]} 사이즈도 적합합니다 (점수: {second[1].overall_score:.0%})."
                )

        return best_report

    # ──────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────

    @staticmethod
    def _ease_to_tightness(ease: float, standards: dict) -> float:
        """ease 값을 -1~+1 tightness로 매핑."""
        # regular 범위의 중심을 0으로
        reg = standards.get("regular", (0.03, 0.12))
        center = (reg[0] + reg[1]) / 2
        # regular 범위의 반폭을 1 단위로
        half_range = max((reg[1] - reg[0]) / 2, 0.01)
        tightness = (ease - center) / (half_range * 4)  # ×4 → ±1 범위 확장
        return float(np.clip(tightness, -1, 1))

    @staticmethod
    def _estimated_related_fields(user: UserMeasurements, part: str) -> List[str]:
        """부위에 관련된 추정 필드 목록."""
        mapping = {
            "shoulder": ["shoulder_width"],
            "sleeve_length": ["arm_length"],
            "thigh": ["thigh"],
            "inseam": ["inseam"],
        }
        related = mapping.get(part, [])
        return [f for f in related if f in user.estimated_fields]

    def _preference_bonus(self, parts: Dict[str, PartFitResult]) -> float:
        """사용자 선호 핏과의 일치도 보너스 (0~1)."""
        target_map = {
            "tight": FitClass.TIGHT,
            "regular": FitClass.REGULAR,
            "loose": FitClass.LOOSE,
        }
        target = target_map.get(self.preference, FitClass.REGULAR)
        matches = sum(
            1 for p in parts.values()
            if p.fit_class == target
        )
        return matches / max(len(parts), 1)
