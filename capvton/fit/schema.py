"""
Fit-Aware Virtual Try-On — Core Data Schemas

측정치 스키마 정의: 사용자 신체, 의류 실측, Fit Report
Pydantic 기반으로 유효성 검증 + 직렬화 제공.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    UNISEX = "unisex"


class GarmentCategory(str, Enum):
    # Upper body
    TSHIRT = "tshirt"
    SHIRT = "shirt"
    JACKET = "jacket"
    SWEATER = "sweater"
    HOODIE = "hoodie"
    BLOUSE = "blouse"
    COAT = "coat"
    # Lower body
    PANTS = "pants"
    JEANS = "jeans"
    SKIRT = "skirt"
    SHORTS = "shorts"
    # Dresses / Overall
    DRESS = "dress"
    JUMPSUIT = "jumpsuit"


class GarmentSuperCategory(str, Enum):
    UPPER = "upper_body"
    LOWER = "lower_body"
    DRESS = "dresses"


class FitClass(str, Enum):
    TOO_TIGHT = "too_tight"
    TIGHT = "tight"
    REGULAR = "regular"
    LOOSE = "loose"
    TOO_LOOSE = "too_loose"


class RiskLevel(str, Enum):
    OK = "ok"
    CAUTION = "caution"  # 추정치수 기반 불확실
    RISK = "risk"        # 확실한 핏 문제


# ──────────────────────────────────────────────
# Body Part Keys (부위 식별)
# ──────────────────────────────────────────────

BODY_PARTS = [
    "chest", "waist", "hip", "shoulder",
    "sleeve_length", "length", "thigh", "inseam",
]

# 카테고리별 관련 부위
CATEGORY_RELEVANT_PARTS: Dict[str, List[str]] = {
    "upper_body": ["chest", "waist", "shoulder", "sleeve_length", "length"],
    "lower_body": ["waist", "hip", "thigh", "length", "inseam"],
    "dresses": ["chest", "waist", "hip", "shoulder", "sleeve_length", "length"],
}


# ──────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────

@dataclass
class UserMeasurements:
    """사용자 신체 측정치. 필수 항목 + 선택 항목."""
    gender: Gender
    height: float  # cm
    chest: float   # cm (둘레)
    waist: float   # cm (둘레)
    hip: float     # cm (둘레)

    # 권장 (미입력 시 추정)
    shoulder_width: Optional[float] = None  # cm
    arm_length: Optional[float] = None      # cm
    inseam: Optional[float] = None          # cm

    # 선택
    thigh: Optional[float] = None       # cm (둘레)
    neck: Optional[float] = None        # cm
    weight: Optional[float] = None      # kg
    age_group: Optional[str] = None     # "20s", "30s", ...

    # 메타: 어떤 항목이 추정치인지 추적
    estimated_fields: List[str] = field(default_factory=list)

    def fill_estimated(self) -> "UserMeasurements":
        """미입력 항목을 통계 기반으로 추정하고, estimated_fields에 기록."""
        is_male = self.gender == Gender.MALE
        estimated = []

        if self.shoulder_width is None:
            ratio = 0.259 if is_male else 0.243
            # 가슴둘레에 의한 보정: chest가 클수록 어깨도 넓은 경향
            chest_correction = (self.chest - (96 if is_male else 88)) * 0.05
            self.shoulder_width = self.height * ratio + chest_correction
            estimated.append("shoulder_width")

        if self.arm_length is None:
            ratio = 0.327 if is_male else 0.317
            self.arm_length = self.height * ratio
            estimated.append("arm_length")

        if self.inseam is None:
            self.inseam = self.height * 0.45
            estimated.append("inseam")

        if self.thigh is None:
            self.thigh = self.hip * 0.62
            estimated.append("thigh")

        self.estimated_fields = estimated
        return self

    def to_vector(self) -> np.ndarray:
        """정규화 전 원시 벡터 변환 (12-d)."""
        self.fill_estimated()
        return np.array([
            1.0 if self.gender == Gender.MALE else 0.0,
            self.height,
            self.chest,
            self.waist,
            self.hip,
            self.shoulder_width or 0.0,
            self.arm_length or 0.0,
            self.inseam or 0.0,
            self.thigh or 0.0,
            self.neck or 0.0,
            self.weight or 0.0,
            _age_to_float(self.age_group),
        ], dtype=np.float32)

    def uncertainty_mask(self) -> np.ndarray:
        """추정 항목에 1, 측정 항목에 0인 마스크 (12-d)."""
        self.fill_estimated()
        fields = [
            "gender", "height", "chest", "waist", "hip",
            "shoulder_width", "arm_length", "inseam", "thigh",
            "neck", "weight", "age_group",
        ]
        return np.array(
            [1.0 if f in self.estimated_fields else 0.0 for f in fields],
            dtype=np.float32,
        )


@dataclass
class GarmentMeasurements:
    """의류 실측 스펙."""
    category: GarmentCategory
    super_category: GarmentSuperCategory

    # 공통 필수
    length: float  # cm (총기장)

    # 상의 필수
    chest_width: Optional[float] = None     # cm (단면, ×2=둘레)
    shoulder: Optional[float] = None        # cm
    sleeve_length: Optional[float] = None   # cm

    # 하의 필수
    waist_width: Optional[float] = None     # cm (단면)
    hip_width: Optional[float] = None       # cm (단면)

    # 권장
    hem_width: Optional[float] = None       # cm (밑단 단면)
    sleeve_width: Optional[float] = None    # cm (소매부리 단면)
    thigh_width: Optional[float] = None     # cm (허벅지 단면)
    rise: Optional[float] = None            # cm (밑위)
    inseam: Optional[float] = None          # cm
    armhole_depth: Optional[float] = None   # cm (소매산)

    # 사이즈별 스펙이 여러 개일 때 사용
    size_label: Optional[str] = None  # "S", "M", "L", ...

    def to_circumference(self, field_name: str) -> Optional[float]:
        """단면 → 둘레 변환 (필요한 필드에 ×2)."""
        val = getattr(self, field_name, None)
        if val is None:
            return None
        # 단면 → 둘레 (shoulder, length, sleeve_length는 이미 실측)
        if field_name in ("chest_width", "waist_width", "hip_width",
                          "hem_width", "sleeve_width", "thigh_width"):
            return val * 2
        return val

    def to_vector(self) -> np.ndarray:
        """정규화 전 원시 벡터 (15-d)."""
        return np.array([
            _cat_to_float(self.category),
            _super_cat_to_float(self.super_category),
            self.length or 0.0,
            self.chest_width or 0.0,
            self.shoulder or 0.0,
            self.sleeve_length or 0.0,
            self.waist_width or 0.0,
            self.hip_width or 0.0,
            self.hem_width or 0.0,
            self.sleeve_width or 0.0,
            self.thigh_width or 0.0,
            self.rise or 0.0,
            self.inseam or 0.0,
            self.armhole_depth or 0.0,
            _size_to_float(self.size_label),
        ], dtype=np.float32)


@dataclass
class EaseVector:
    """부위별 여유분 (원시 비율값)."""
    values: Dict[str, float] = field(default_factory=dict)
    # 각 값: (garment_circ - body_circ) / body_circ (원시 비율, 클리핑 없음)

    @staticmethod
    def compute(user: UserMeasurements, garment: GarmentMeasurements) -> "EaseVector":
        """사용자 치수와 의류 치수로부터 부위별 ease 계산. 원시 비율값."""
        user.fill_estimated()
        ease = {}

        # 가슴
        if garment.chest_width is not None:
            g_circ = garment.chest_width * 2
            ease["chest"] = _raw_ease(g_circ, user.chest)

        # 허리
        if garment.waist_width is not None:
            g_circ = garment.waist_width * 2
            ease["waist"] = _raw_ease(g_circ, user.waist)

        # 엉덩이
        if garment.hip_width is not None:
            g_circ = garment.hip_width * 2
            ease["hip"] = _raw_ease(g_circ, user.hip)

        # 어깨
        if garment.shoulder is not None and user.shoulder_width is not None:
            ease["shoulder"] = _raw_ease(garment.shoulder, user.shoulder_width)

        # 소매기장: 카테고리별 기대 소매 비율에 따라 비교
        if garment.sleeve_length is not None and user.arm_length is not None:
            ref_sleeve = _reference_sleeve_length(user, garment.category)
            if ref_sleeve > 0:
                ease["sleeve_length"] = _raw_ease(garment.sleeve_length, ref_sleeve)

        # 총기장 (카테고리별 기준 대비)
        ref_length = _reference_length(user, garment.super_category)
        if ref_length > 0:
            ease["length"] = _raw_ease(garment.length, ref_length)

        # 허벅지
        if garment.thigh_width is not None and user.thigh is not None:
            g_circ = garment.thigh_width * 2
            ease["thigh"] = _raw_ease(g_circ, user.thigh)

        # 인심
        if garment.inseam is not None and user.inseam is not None:
            ease["inseam"] = _raw_ease(garment.inseam, user.inseam)

        return EaseVector(values=ease)

    def to_vector(self, parts: Optional[List[str]] = None) -> np.ndarray:
        """정규화된 벡터 (10-d, [-1,+1] 범위). ML 모델 입력용."""
        if parts is None:
            parts = BODY_PARTS + ["rise", "hem"]
        return np.array(
            [_normalize_ease(self.values.get(p, 0.0)) for p in parts],
            dtype=np.float32,
        )

    def to_raw_vector(self, parts: Optional[List[str]] = None) -> np.ndarray:
        """원시 비율 벡터 (10-d). 없는 부위는 0."""
        if parts is None:
            parts = BODY_PARTS + ["rise", "hem"]
        return np.array(
            [self.values.get(p, 0.0) for p in parts],
            dtype=np.float32,
        )


@dataclass
class PartFitResult:
    """단일 부위의 핏 판정 결과."""
    tightness: float        # -1(매우 타이트) ~ +1(매우 오버)
    fit_class: FitClass
    risk_level: RiskLevel
    confidence: float       # 0~1 (추정치수이면 낮아짐)
    ease_raw: float         # 원시 ease 비율


@dataclass
class FitReport:
    """전체 핏 판정 결과."""
    overall_score: float                       # 0~1 (1 = 완벽)
    size_recommendation: str                   # "M 추천"
    parts: Dict[str, PartFitResult]            # 부위별 상세
    risk_parts: List[str]                      # 리스크 부위 목록
    all_sizes_scores: Dict[str, float] = field(default_factory=dict)  # {S:0.6, M:0.9, ...}
    notes: List[str] = field(default_factory=list)  # UX 메시지

    @property
    def tightness_vector(self) -> Dict[str, float]:
        return {k: v.tightness for k, v in self.parts.items()}

    @property
    def fit_class_dict(self) -> Dict[str, str]:
        return {k: v.fit_class.value for k, v in self.parts.items()}

    def to_embedding_input(self) -> np.ndarray:
        """Fit embedding 입력용 벡터 (tightness + confidence)."""
        parts_ordered = BODY_PARTS
        t = [self.parts[p].tightness if p in self.parts else 0.0 for p in parts_ordered]
        c = [self.parts[p].confidence if p in self.parts else 0.5 for p in parts_ordered]
        return np.array(t + c + [self.overall_score], dtype=np.float32)


# ──────────────────────────────────────────────
# Normalization Statistics (성별·카테고리별)
# ──────────────────────────────────────────────

BODY_STATS = {
    Gender.MALE: {
        "mean": np.array([0, 174, 96, 82, 96, 45, 57, 78, 56, 38, 72, 30], dtype=np.float32),
        "std":  np.array([1, 7,  7,  8,  6,  3,  3,  4,  4,  2,  12, 10], dtype=np.float32),
    },
    Gender.FEMALE: {
        "mean": np.array([1, 162, 88, 70, 94, 40, 51, 73, 54, 33, 58, 30], dtype=np.float32),
        "std":  np.array([1, 6,  6,  7,  6,  2,  3,  4,  4,  2,  10, 10], dtype=np.float32),
    },
}


def normalize_user(u: UserMeasurements) -> np.ndarray:
    """z-score 정규화."""
    raw = u.to_vector()
    stats = BODY_STATS.get(u.gender, BODY_STATS[Gender.MALE])
    return (raw - stats["mean"]) / (stats["std"] + 1e-8)


# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────

def _raw_ease(garment_val: float, body_val: float) -> float:
    """원시 ease 비율: (garment - body) / body."""
    if body_val <= 0:
        return 0.0
    return (garment_val - body_val) / body_val


def _normalize_ease(raw_ease: float) -> float:
    """원시 ease 비율 → [-1, +1] 정규화. 중심 5%, 범위 ±20%."""
    return float(np.clip((raw_ease - 0.05) / 0.20, -1, 1))


def _reference_length(user: UserMeasurements, super_cat: GarmentSuperCategory) -> float:
    """카테고리별 '기장 기준' 계산 (cm)."""
    h = user.height
    if super_cat == GarmentSuperCategory.UPPER:
        return h * 0.38  # 상의: 대략 허리~엉덩이 (키의 38%)
    elif super_cat == GarmentSuperCategory.LOWER:
        return user.inseam if user.inseam else h * 0.45
    elif super_cat == GarmentSuperCategory.DRESS:
        return h * 0.60  # 원피스: 무릎 부근
    return h * 0.40


def _reference_sleeve_length(user: UserMeasurements, category: GarmentCategory) -> float:
    """
    카테고리별 '적정 소매 기장' 기준 (cm).

    반소매(tshirt 등): 상완 길이의 약 60% (어깨~상완 중간)
    긴소매(shirt, jacket 등): 풀 팔 길이
    """
    arm = user.arm_length if user.arm_length else user.height * 0.32
    # 상완 길이 ≈ 팔 전체의 ~50%
    upper_arm = arm * 0.50

    SHORT_SLEEVE_CATS = {
        GarmentCategory.TSHIRT,
    }
    if category in SHORT_SLEEVE_CATS:
        return upper_arm * 0.70  # 반소매: 상완의 70% (≈20cm for 175cm male)
    else:
        # 긴소매
        return arm * 0.95  # 손목 약간 위


def _age_to_float(age_group: Optional[str]) -> float:
    if age_group is None:
        return 0.0
    mapping = {"10s": 15, "20s": 25, "30s": 35, "40s": 45, "50s": 55, "60s": 65}
    return float(mapping.get(age_group, 30))


def _cat_to_float(cat: GarmentCategory) -> float:
    return float(list(GarmentCategory).index(cat))


def _super_cat_to_float(cat: GarmentSuperCategory) -> float:
    return float(list(GarmentSuperCategory).index(cat))


def _size_to_float(size: Optional[str]) -> float:
    if size is None:
        return 0.0
    mapping = {"XS": 0, "S": 1, "M": 2, "L": 3, "XL": 4, "XXL": 5}
    return float(mapping.get(size.upper(), 2))
