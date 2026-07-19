from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinaryMetrics:
    mean_iou: float
    mean_dice: float


def confusion_matrix(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    valid = target != 255
    values = 2 * target[valid].astype(np.int64) + prediction[valid].astype(np.int64)
    return np.bincount(values, minlength=4).reshape(2, 2)


def metrics_from_confusion(matrix: np.ndarray) -> BinaryMetrics:
    true_positive = np.diag(matrix).astype(np.float64)
    union = matrix.sum(axis=1) + matrix.sum(axis=0) - true_positive
    denominator = matrix.sum(axis=1) + matrix.sum(axis=0)
    iou = np.divide(true_positive, union, out=np.zeros(2), where=union > 0)
    dice = np.divide(2 * true_positive, denominator, out=np.zeros(2), where=denominator > 0)
    return BinaryMetrics(float(iou.mean()), float(dice.mean()))
