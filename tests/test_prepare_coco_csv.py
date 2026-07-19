import numpy as np
import pytest

from tools.prepare_coco_csv import (
    rasterize_yolo_segments,
    split_names,
    validate_binary_mask,
)


def test_rasterize_yolo_segments_returns_foreground_polygon():
    mask = rasterize_yolo_segments('0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75', 8, 8)

    assert mask.dtype == np.uint8
    assert set(np.unique(mask)) == {0, 1}
    assert mask[3, 3] == 1
    assert mask[0, 0] == 0


def test_split_names_is_deterministic_and_covers_requested_count():
    names = [f'{index:012d}.jpg' for index in range(10)]

    first = split_names(names, 6, 17)
    second = split_names(names, 6, 17)

    assert first == second
    assert [len(first[split]) for split in ('train', 'val', 'test')] == [4, 1, 1]
    assert set().union(*first.values()) == set(names[:6])


def test_validate_binary_mask_rejects_unexpected_values():
    with pytest.raises(ValueError, match='binary'):
        validate_binary_mask(np.array([[0, 2]], dtype=np.uint8))
