"""
Body Estimator — OpenPose 키포인트로부터 추정 치수를 보정하는 유틸리티.

사용자가 사진만 제공했을 때, DensePose/OpenPose 결과로 대략적인 체형 비율을 추정.
이 모듈은 "사진 기반 보정"을 위한 것이며, 정확한 치수를 대체하지 않음.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image


# OpenPose BODY_25 키포인트 인덱스
KEYPOINT = {
    "nose": 0, "neck": 1,
    "r_shoulder": 2, "r_elbow": 3, "r_wrist": 4,
    "l_shoulder": 5, "l_elbow": 6, "l_wrist": 7,
    "mid_hip": 8,
    "r_hip": 9, "r_knee": 10, "r_ankle": 11,
    "l_hip": 12, "l_knee": 13, "l_ankle": 14,
    "r_eye": 15, "l_eye": 16,
    "r_ear": 17, "l_ear": 18,
}


def estimate_pixel_lengths(keypoints: np.ndarray) -> Dict[str, float]:
    """
    키포인트 좌표(px)로부터 주요 신체 분절 길이(px)를 계산.

    Args:
        keypoints: (N, 3) 배열 — [x, y, confidence]. N ≥ 19.

    Returns:
        딕셔너리: {분절 이름: 길이(px)}. confidence < 0.1인 키포인트 관련 분절은 제외. 
    """
    if keypoints.shape[0] < 19 or keypoints.shape[1] < 3:
        return {}

    def _dist(a_idx: int, b_idx: int) -> Optional[float]:
        a, b = keypoints[a_idx], keypoints[b_idx]
        if a[2] < 0.1 or b[2] < 0.1:
            return None
        return float(np.linalg.norm(a[:2] - b[:2]))

    segments = {}

    # 어깨 너비 (좌-우 어깨)
    d = _dist(KEYPOINT["l_shoulder"], KEYPOINT["r_shoulder"])
    if d is not None:
        segments["shoulder_px"] = d

    # 상체 길이 (목 → mid_hip)
    d = _dist(KEYPOINT["neck"], KEYPOINT["mid_hip"])
    if d is not None:
        segments["torso_px"] = d

    # 팔 길이 (어깨→팔꿈치→손목)
    for side in ["l", "r"]:
        upper = _dist(KEYPOINT[f"{side}_shoulder"], KEYPOINT[f"{side}_elbow"])
        lower = _dist(KEYPOINT[f"{side}_elbow"], KEYPOINT[f"{side}_wrist"])
        if upper is not None and lower is not None:
            segments[f"arm_{side}_px"] = upper + lower

    # 다리 길이 (힙→무릎→발목)
    for side in ["l", "r"]:
        upper = _dist(KEYPOINT[f"{side}_hip"], KEYPOINT[f"{side}_knee"])
        lower = _dist(KEYPOINT[f"{side}_knee"], KEYPOINT[f"{side}_ankle"])
        if upper is not None and lower is not None:
            segments[f"leg_{side}_px"] = upper + lower

    # 전체 높이 (머리 top 추정 → 발목)
    # 머리: nose 위로 약 (코~목 거리)만큼 추가
    head_top = _dist(KEYPOINT["nose"], KEYPOINT["neck"])
    if head_top is not None:
        nose_y = keypoints[KEYPOINT["nose"]][1]
        estimated_top_y = nose_y - head_top * 0.7  # 대략 머리 꼭대기
        # 발목 중 더 낮은 쪽
        ankles = []
        for side in ["l", "r"]:
            if keypoints[KEYPOINT[f"{side}_ankle"]][2] > 0.1:
                ankles.append(keypoints[KEYPOINT[f"{side}_ankle"]][1])
        if ankles:
            bottom_y = max(ankles)
            segments["full_height_px"] = bottom_y - estimated_top_y

    return segments


def compute_pixel_to_cm_scale(
    segments: Dict[str, float],
    known_height_cm: float,
) -> float:
    """
    알려진 키(cm)와 픽셀 높이로부터 scale factor (cm/px)를 계산.

    Args:
        segments: estimate_pixel_lengths의 출력
        known_height_cm: 사용자가 입력한 키 (cm)

    Returns:
        cm_per_pixel scale factor. 계산 불가 시 0.0.
    """
    if "full_height_px" not in segments or segments["full_height_px"] <= 0:
        return 0.0
    return known_height_cm / segments["full_height_px"]


def refine_measurements_from_keypoints(
    keypoints: np.ndarray,
    known_height_cm: float,
    current_shoulder: Optional[float] = None,
    current_arm_length: Optional[float] = None,
) -> Dict[str, Tuple[float, float]]:
    """
    키포인트 기반으로 추정 치수를 보정.

    Returns:
        딕셔너리: {항목: (보정된 값 cm, 불확실성 σ cm)}
    """
    segments = estimate_pixel_lengths(keypoints)
    scale = compute_pixel_to_cm_scale(segments, known_height_cm)
    if scale <= 0:
        return {}

    results = {}

    # 어깨 너비 보정
    if "shoulder_px" in segments:
        estimated_cm = segments["shoulder_px"] * scale
        # 사진 각도에 의한 불확실성: ±3cm
        sigma = 3.0
        if current_shoulder is not None:
            # 가중 평균: 기존 추정 50% + 사진 기반 50%
            refined = 0.5 * current_shoulder + 0.5 * estimated_cm
            sigma = 2.0  # 두 소스 결합으로 불확실성 감소
        else:
            refined = estimated_cm
        results["shoulder_width"] = (refined, sigma)

    # 팔 길이 보정
    arm_pxs = [segments.get(f"arm_{s}_px") for s in ["l", "r"]]
    arm_pxs = [a for a in arm_pxs if a is not None]
    if arm_pxs:
        avg_arm_px = np.mean(arm_pxs)
        estimated_cm = avg_arm_px * scale
        sigma = 2.5
        if current_arm_length is not None:
            refined = 0.5 * current_arm_length + 0.5 * estimated_cm
            sigma = 2.0
        else:
            refined = estimated_cm
        results["arm_length"] = (refined, sigma)

    return results
