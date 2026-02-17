"""
Pseudo-Fit Data Augmentation

공개 데이터셋(DressCode, VITON-HD)에 다중 사이즈 GT가 없으므로,
부위별 다른 팽창/수축을 적용하여 pseudo-fit 레이아웃(마스크+SDF)을 합성.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from capvton.fit.layout_generator import compute_sdf


# ──────────────────────────────────────────────
# DensePose Body Part IDs (Fine segmentation)
# ──────────────────────────────────────────────
# 0: background
# 1-2: Torso (front/back)
# 3-4: Right/Left hand
# 5-6: Left/Right foot
# 7-8: Upper right/left leg
# 9-10: Lower right/left leg
# 11-12: Upper left/right arm
# 13-14: Lower left/right arm
# 15-16: Head front/back
# 17-18: Right/Left shoulder
# 19-20: Right/Left hip
# 21-22: Right/Left upper arm (back)
# 23-24: Right/Left lower arm (back)

DENSEPOSE_PARTS = {
    "torso": [1, 2],
    "upper_arm": [11, 12, 21, 22],
    "lower_arm": [13, 14, 23, 24],
    "upper_leg": [7, 8],
    "lower_leg": [9, 10],
    "hip": [19, 20],
}

# Ease → 부위 매핑 (어떤 ease가 어떤 DensePose 영역에 영향)
EASE_TO_DENSE_PARTS = {
    "chest": [1, 2],           # 상체 torso
    "waist": [1, 2],           # torso 하단
    "hip": [19, 20, 7, 8],    # 힙 + 상단 다리
    "shoulder": [17, 18],      # 어깨
    "sleeve_length": [11, 12, 13, 14, 21, 22, 23, 24],  # 팔 전체
    "thigh": [7, 8],           # 허벅지
}


# ──────────────────────────────────────────────
# Augmentation Functions
# ──────────────────────────────────────────────

def generate_random_ease() -> Dict[str, float]:
    """
    랜덤 ease 벡터 생성 (자연스러운 범위 내).

    Returns:
        {부위: ease 값} — ease ∈ [-0.3, +0.4] 범위
    """
    return {
        "chest": random.uniform(-0.15, 0.30),
        "waist": random.uniform(-0.15, 0.35),
        "hip": random.uniform(-0.10, 0.25),
        "shoulder": random.uniform(-0.05, 0.12),
        "sleeve_length": random.uniform(-0.10, 0.15),
        "length": random.uniform(-0.10, 0.15),
        "thigh": random.uniform(-0.10, 0.25),
    }


def generate_pseudo_fit_layout(
    original_mask: np.ndarray,
    densepose_imap: np.ndarray,
    target_ease: Dict[str, float],
    garment_type: str = "upper",
    max_kernel: int = 25,
    max_shift_px: int = 40,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    부위별로 다른 팽창/수축을 적용하여 pseudo-fit 레이아웃 생성.

    Args:
        original_mask: (H, W) uint8, 0/255 — 원본 의류 마스크
        densepose_imap: (H, W) uint8 — DensePose I-map (body part index 0-24)
        target_ease: {부위: ease 값}
        garment_type: "upper" / "lower" / "overall"
        max_kernel: 최대 morphology 커널 크기
        max_shift_px: 기장 최대 이동 픽셀

    Returns:
        deformed_mask: (H, W) uint8 0/255
        sdf_map: (H, W) float32
        applied_ease: 실제 적용된 ease (자연스러운 범위 클리핑 후)
    """
    H, W = original_mask.shape[:2]
    deformed = original_mask.copy()
    if deformed.max() > 1:
        deformed = (deformed > 127).astype(np.uint8)
    else:
        deformed = deformed.astype(np.uint8)

    applied_ease = {}

    # ─── 부위별 팽창/수축 ───
    for part_name, ease_val in target_ease.items():
        if part_name == "length":
            continue  # 기장은 별도 처리

        dense_parts = EASE_TO_DENSE_PARTS.get(part_name, [])
        if not dense_parts:
            continue

        # DensePose 영역에서 해당 부위 마스크 추출
        part_region = np.zeros((H, W), dtype=np.uint8)
        for dp_id in dense_parts:
            part_region |= (densepose_imap == dp_id).astype(np.uint8)

        # 원본 마스크와 교차: 해당 부위에서 의류가 있는 영역만
        intersection = deformed & part_region

        if intersection.sum() < 50:  # 너무 작은 영역은 스킵
            continue

        # 커널 크기: ease 절대값에 비례
        k_size = max(3, int(abs(ease_val) * max_kernel))
        if k_size % 2 == 0:
            k_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

        if ease_val > 0:
            # 오버핏: dilate
            expanded = cv2.dilate(intersection, kernel, iterations=1)
        else:
            # 타이트핏: erode
            expanded = cv2.erode(intersection, kernel, iterations=1)

        # 해당 영역만 갱신 (다른 부위는 건드리지 않음)
        region_mask = part_region.astype(bool)
        deformed[region_mask] = expanded[region_mask]
        applied_ease[part_name] = ease_val

    # ─── 기장 조절 (상/하 이동) ───
    if "length" in target_ease:
        shift = int(target_ease["length"] * max_shift_px)
        deformed = _shift_mask_vertically(deformed, shift, garment_type)
        applied_ease["length"] = target_ease["length"]

    # ─── 후처리 ───
    # Convex hull 제약: 너무 불규칙한 형태 방지
    deformed = _smooth_mask(deformed)

    # 0/255로 변환
    deformed_mask = (deformed * 255).astype(np.uint8)

    # SDF 계산
    sdf_map = compute_sdf(deformed_mask, max_dist=50.0)

    return deformed_mask, sdf_map, applied_ease


def _shift_mask_vertically(mask: np.ndarray, shift_px: int, garment_type: str) -> np.ndarray:
    """
    마스크를 수직으로 이동 (기장 조절).

    shift_px > 0: 아래로 이동 (기장 길어짐)
    shift_px < 0: 위로 이동 (기장 짧아짐)
    """
    H, W = mask.shape
    result = np.zeros_like(mask)

    if garment_type in ("upper", "overall"):
        # 상의/원피스: 하단만 이동
        # 상단 50%는 유지, 하단 50%를 shift
        mid_y = H // 2
        result[:mid_y, :] = mask[:mid_y, :]

        if shift_px >= 0:
            # 아래로 확장
            src_start = mid_y
            dst_start = mid_y + shift_px
            if dst_start < H:
                length = min(H - mid_y, H - dst_start)
                result[dst_start:dst_start + length, :] = mask[src_start:src_start + length, :]
                # 빈 영역 채우기 (확장)
                result[mid_y:dst_start, :] = mask[mid_y:mid_y + 1, :]  # 경계 row 복제
        else:
            # 위로 축소
            src_start = mid_y + abs(shift_px)
            if src_start < H:
                length = H - src_start
                result[mid_y:mid_y + length, :] = mask[src_start:src_start + length, :]

    elif garment_type == "lower":
        # 하의: 하단만 이동 (허리 라인 유지)
        top_y = H // 4
        result[:top_y, :] = mask[:top_y, :]

        if shift_px >= 0:
            # 길어짐: 하단 확장
            end_row = H - 1
            overflow = min(shift_px, H - 1)
            # 기존 마스크를 아래로 이동
            for y in range(top_y, H):
                new_y = min(y + shift_px, H - 1)
                result[new_y, :] |= mask[y, :]
        else:
            # 짧아짐: 하단 축소
            for y in range(top_y, H):
                new_y = max(y + shift_px, top_y)
                result[new_y, :] |= mask[y, :]

    return result


def _smooth_mask(mask: np.ndarray, blur_size: int = 5) -> np.ndarray:
    """마스크를 부드럽게 정제."""
    # 가우시안 블러 → 이진화
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (blur_size, blur_size), 0)
    return (blurred > 0.5).astype(np.uint8)


# ──────────────────────────────────────────────
# Batch Augmentation (데이터로더용)
# ──────────────────────────────────────────────

def augment_batch(
    original_masks: List[np.ndarray],
    densepose_imaps: List[np.ndarray],
    garment_types: List[str],
    num_augments_per_sample: int = 3,
) -> List[Dict]:
    """
    배치 단위로 pseudo-fit 증강 수행.

    Returns:
        증강된 샘플 리스트: [{"mask": ..., "sdf": ..., "ease": ...}, ...]
    """
    results = []
    for mask, imap, gtype in zip(original_masks, densepose_imaps, garment_types):
        for _ in range(num_augments_per_sample):
            ease = generate_random_ease()
            deformed_mask, sdf_map, applied_ease = generate_pseudo_fit_layout(
                mask, imap, ease, gtype
            )
            results.append({
                "mask": deformed_mask,
                "sdf": sdf_map,
                "ease": applied_ease,
                "garment_type": gtype,
            })
    return results
