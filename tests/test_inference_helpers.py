import numpy as np

from inference import binary_metrics, normalize_mask, output_stem


def test_binary_helpers_report_perfect_overlap_and_binary_values(tmp_path):
    target = np.array([[0, 255], [255, 0]], dtype=np.uint8)

    assert np.array_equal(normalize_mask(target), np.array([[0, 1], [1, 0]], dtype=np.uint8))
    assert binary_metrics(target, target) == (1.0, 1.0)
    assert output_stem(tmp_path / 'nested' / 'image.jpg', 3) == '0003_image'
