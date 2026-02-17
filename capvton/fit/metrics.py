"""
Fit-Aware VTON Evaluation Metrics

핏 판정 정확도, 실루엣 일관성, 이미지 품질 proxy metric.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


# ──────────────────────────────────────────────
# 1. 핏 판정 정확도
# ──────────────────────────────────────────────

def partwise_accuracy(
    pred_classes: Dict[str, str],
    gt_classes: Dict[str, str],
) -> Dict[str, float]:
    """
    부위별 fit class 정확도.

    Args:
        pred_classes: {부위: 예측 클래스}
        gt_classes: {부위: GT 클래스}

    Returns:
        {부위: 1.0 or 0.0} + {"overall": accuracy}
    """
    results = {}
    correct = 0
    total = 0
    for part in gt_classes:
        if part in pred_classes:
            match = pred_classes[part] == gt_classes[part]
            results[part] = 1.0 if match else 0.0
            correct += match
            total += 1
    results["overall"] = correct / max(total, 1)
    return results


def tightness_mae(
    pred_tightness: Dict[str, float],
    gt_tightness: Dict[str, float],
) -> Dict[str, float]:
    """부위별 tightness Mean Absolute Error."""
    results = {}
    errors = []
    for part in gt_tightness:
        if part in pred_tightness:
            err = abs(pred_tightness[part] - gt_tightness[part])
            results[part] = err
            errors.append(err)
    results["mean"] = np.mean(errors) if errors else 0.0
    return results


def size_recommendation_accuracy(
    pred_sizes: List[str],
    gt_sizes: List[str],
    top_k: int = 1,
) -> float:
    """사이즈 추천 Top-K 정확도."""
    correct = sum(1 for p, g in zip(pred_sizes, gt_sizes) if p == g)
    return correct / max(len(gt_sizes), 1)


def risk_detection_f1(
    pred_risks: List[str],
    gt_risks: List[str],
) -> Dict[str, float]:
    """리스크 부위 검출 F1-score."""
    pred_set = set(pred_risks)
    gt_set = set(gt_risks)

    if len(gt_set) == 0 and len(pred_set) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


# ──────────────────────────────────────────────
# 2. 실루엣 일관성
# ──────────────────────────────────────────────

def mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    마스크 IoU (Intersection over Union).

    Args:
        pred_mask, gt_mask: (H, W) binary (0/1 or 0/255)
    """
    pred = (pred_mask > 127).astype(bool) if pred_mask.max() > 1 else pred_mask.astype(bool)
    gt = (gt_mask > 127).astype(bool) if gt_mask.max() > 1 else gt_mask.astype(bool)

    intersection = (pred & gt).sum()
    union = (pred | gt).sum()

    return float(intersection / (union + 1e-8))


def boundary_f_score(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    tolerance: int = 2,
) -> float:
    """
    경계선 F-score.

    경계 픽셀 기준으로 precision/recall 계산 (tolerance 이내 매칭).
    """
    pred = (pred_mask > 127).astype(np.uint8) if pred_mask.max() > 1 else pred_mask.astype(np.uint8)
    gt = (gt_mask > 127).astype(np.uint8) if gt_mask.max() > 1 else gt_mask.astype(np.uint8)

    # 경계 추출
    pred_boundary = _extract_boundary(pred)
    gt_boundary = _extract_boundary(gt)

    if pred_boundary.sum() == 0 or gt_boundary.sum() == 0:
        return 0.0

    # tolerance 이내 매칭
    gt_dilated = cv2.dilate(gt_boundary, np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8))
    pred_dilated = cv2.dilate(pred_boundary, np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8))

    # Precision: pred 경계 중 GT 근처에 있는 비율
    precision = (pred_boundary & gt_dilated).sum() / (pred_boundary.sum() + 1e-8)
    # Recall: GT 경계 중 pred 근처에 있는 비율
    recall = (gt_boundary & pred_dilated).sum() / (gt_boundary.sum() + 1e-8)

    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def sdf_l1_error(pred_sdf: np.ndarray, gt_sdf: np.ndarray) -> float:
    """SDF L1 평균 오차."""
    return float(np.abs(pred_sdf - gt_sdf).mean())


def hemline_pixel_error(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> float:
    """
    밑단선(hemline) 위치 픽셀 오차.

    마스크에서 최하단 경계의 y좌표 차이.
    """
    pred_rows = np.where(pred_mask.max(axis=1) > 0)[0]
    gt_rows = np.where(gt_mask.max(axis=1) > 0)[0]

    if len(pred_rows) == 0 or len(gt_rows) == 0:
        return float("inf")

    pred_bottom = pred_rows[-1]
    gt_bottom = gt_rows[-1]

    return float(abs(pred_bottom - gt_bottom))


# ──────────────────────────────────────────────
# 3. 사용자 만족 Proxy Metrics
# ──────────────────────────────────────────────

def wrinkle_density(
    image: np.ndarray,
    mask: np.ndarray,
    fit_class: str,
) -> Dict[str, float]:
    """
    주름 밀도 측정 (edge response density).

    tight 영역에서는 높아야 하고, loose 영역에서는 중간.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    mask_bool = (mask > 127) if mask.max() > 1 else mask.astype(bool)

    # Canny edge detection
    edges = cv2.Canny(gray, 50, 150)

    # 마스크 내 edge 밀도
    masked_edges = edges * mask_bool.astype(np.uint8)
    area = mask_bool.sum()
    if area == 0:
        return {"density": 0.0, "expected": "n/a"}

    density = float(masked_edges.sum()) / float(area * 255)

    # 기대값 (fit_class에 따라)
    expected = {
        "too_tight": "high",
        "tight": "medium-high",
        "regular": "medium",
        "loose": "low-medium",
        "too_loose": "low",
    }.get(fit_class, "medium")

    return {"density": density, "expected_level": expected}


def silhouette_gap(
    body_contour: np.ndarray,
    garment_contour: np.ndarray,
    densepose_seg: np.ndarray,
    target_parts: List[int],
) -> float:
    """
    신체 윤곽과 의류 윤곽 사이의 평균 거리.

    loose 판정 영역에서는 이 값이 커야 함.
    """
    # 타겟 부위 영역에서만 계산
    part_mask = np.zeros_like(densepose_seg[:, :, 0], dtype=bool)
    for pid in target_parts:
        part_mask |= (densepose_seg[:, :, 0] == pid)

    if not part_mask.any():
        return 0.0

    # 신체 경계 (DensePose에서)
    body_boundary = _extract_boundary(part_mask.astype(np.uint8))
    # 의류 경계
    garment_boundary = _extract_boundary((garment_contour > 127).astype(np.uint8))

    if body_boundary.sum() == 0 or garment_boundary.sum() == 0:
        return 0.0

    # 신체 경계 픽셀에서 의류 경계까지의 최소 거리
    body_pts = np.argwhere(body_boundary > 0)
    garment_dist = cv2.distanceTransform(
        1 - garment_boundary, cv2.DIST_L2, 5
    )

    distances = garment_dist[body_pts[:, 0], body_pts[:, 1]]
    return float(np.mean(distances))


# ──────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────

def _extract_boundary(mask: np.ndarray) -> np.ndarray:
    """바이너리 마스크에서 경계 픽셀 추출."""
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    boundary = dilated - mask
    return boundary
