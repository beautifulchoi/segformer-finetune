import numpy as np

from HF_pipeline.metrics import BinaryMetrics, confusion_matrix, metrics_from_confusion


def test_binary_metrics_ignore_padding_and_compute_iou_and_dice():
    prediction = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    target = np.array([[0, 1], [255, 0]], dtype=np.uint8)

    matrix = confusion_matrix(prediction, target)
    metrics = metrics_from_confusion(matrix)

    assert matrix.tolist() == [[2, 0], [0, 1]]
    assert metrics == BinaryMetrics(mean_iou=1.0, mean_dice=1.0)
