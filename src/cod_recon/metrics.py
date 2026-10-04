"""COD segmentation and realistic-scene evaluation metrics."""

from __future__ import annotations

import numpy as np


def mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.abs(prediction - target).mean())


def s_measure(prediction: np.ndarray, target: np.ndarray) -> float:
    """Object-aware plus region-aware S-measure (standard COD formulation)."""
    y = target >= 0.5
    x = np.clip(prediction, 0, 1)
    foreground = float(y.mean())
    if foreground == 0:
        return float(1 - x.mean())
    if foreground == 1:
        return float(x.mean())
    object_score = 2 * x[y].mean() / (x[y].mean() ** 2 + 1 + (x[y] ** 2).mean() + 1e-8)
    bg = x[~y]
    object_score_bg = 2 * (1 - bg.mean()) / ((1 - bg.mean()) ** 2 + 1 + ((1 - bg) ** 2).mean() + 1e-8)
    object_term = foreground * object_score + (1 - foreground) * object_score_bg
    h, w = y.shape
    cy, cx = round(h / 2), round(w / 2)
    boxes = ((0, cy, 0, cx), (0, cy, cx, w), (cy, h, 0, cx), (cy, h, cx, w))
    region_term, area_sum = 0.0, 0.0
    for y0, y1, x0, x1 in boxes:
        gt, pred = y[y0:y1, x0:x1], x[y0:y1, x0:x1]
        if gt.size == 0:
            continue
        area = gt.size / y.size
        gt_mean, pred_mean = gt.mean(), pred.mean()
        gt_var = ((gt - gt_mean) ** 2).mean()
        pred_var = ((pred - pred_mean) ** 2).mean()
        cov = ((gt - gt_mean) * (pred - pred_mean)).mean()
        score = 4 * gt_mean * pred_mean * cov / (gt_mean**2 + pred_mean**2 + 1e-8)
        region_term += area * score
        area_sum += area
    return float(np.clip(0.5 * object_term + 0.5 * region_term / max(area_sum, 1e-8), 0, 1))


def weighted_f_measure(prediction: np.ndarray, target: np.ndarray, beta2: float = 1.0) -> float:
    """Weighted F-measure using distance-weighted false negatives."""
    import cv2

    gt = target >= 0.5
    pred = np.clip(prediction, 0, 1)
    if not gt.any():
        return float(1 - pred.mean())
    if gt.all():
        return float(pred.mean())
    distance = cv2.distanceTransform((~gt).astype(np.uint8), cv2.DIST_L2, 3)
    sigma = max(1.0, min(gt.shape) / 50)
    weights = 1 + 5 * np.exp(-distance**2 / (2 * sigma**2))
    errors = np.abs(pred - gt)
    weighted_errors = cv2.GaussianBlur((errors * weights).astype(np.float32), (0, 0), sigmaX=sigma)
    fn = float((weighted_errors * gt).sum())
    fp = float((weighted_errors * ~gt).sum())
    tp = float((pred * gt).sum())
    recall = 1 - fn / (float(gt.sum()) + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    return float((1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-8))


def e_measure(prediction: np.ndarray, target: np.ndarray) -> float:
    gt = target >= 0.5
    pred = np.clip(prediction, 0, 1)
    pred = (pred - pred.mean()) * 2
    gt = gt.astype(np.float32) - float(gt.mean())
    alignment = 2 * pred * gt / (pred**2 + gt**2 + 1e-8)
    enhanced = ((alignment + 1) ** 2) / 4
    return float(np.clip(enhanced.mean(), 0, 1))


def segmentation_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {"S_alpha": s_measure(prediction, target), "weighted_F_beta": weighted_f_measure(prediction, target),
            "E_phi": e_measure(prediction, target), "MAE": mae(prediction, target)}
