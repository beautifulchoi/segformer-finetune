import csv

import numpy as np
import pytest
from PIL import Image

from mmseg.datasets.csv_binary import CsvBinaryDataset
from mmseg.datasets.pipelines.csv_loading import LoadBinaryAnnotations


def _write_manifest(root, rows):
    with (root / 'manifest.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['image', 'mask', 'split'])
        writer.writeheader()
        writer.writerows(rows)


def test_dataset_filters_split_and_resolves_csv_relative_paths(tmp_path):
    image = tmp_path / 'image.png'
    mask = tmp_path / 'mask.png'
    test_image = tmp_path / 'test_image.png'
    test_mask = tmp_path / 'test_mask.png'
    Image.new('RGB', (4, 4), (20, 30, 40)).save(image)
    Image.fromarray(np.array([[0, 1, 0, 0]] * 4, dtype=np.uint8) * 255).save(mask)
    Image.new('RGB', (4, 4), (50, 60, 70)).save(test_image)
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(test_mask)
    _write_manifest(tmp_path, [
        {'image': 'image.png', 'mask': 'mask.png', 'split': 'val'},
        {'image': 'test_image.png', 'mask': 'test_mask.png', 'split': 'test'},
    ])

    dataset = CsvBinaryDataset(pipeline=[], csv_path=str(tmp_path / 'manifest.csv'), split='val')

    assert len(dataset) == 1
    assert dataset.CLASSES == ('background', 'foreground')
    assert dataset.img_infos[0]['filename'] == str(image.resolve())


def test_dataset_rejects_missing_files(tmp_path):
    _write_manifest(tmp_path, [{'image': 'missing.png', 'mask': 'mask.png', 'split': 'train'}])

    with pytest.raises(FileNotFoundError, match='missing'):
        CsvBinaryDataset(pipeline=[], csv_path=str(tmp_path / 'manifest.csv'), split='train')


def test_binary_loader_normalizes_255_mask(tmp_path):
    mask = tmp_path / 'mask.png'
    Image.fromarray(np.array([[0, 255], [255, 0]], dtype=np.uint8)).save(mask)
    results = {
        'seg_prefix': str(tmp_path),
        'ann_info': {'seg_map': 'mask.png'},
        'seg_fields': [],
    }

    loaded = LoadBinaryAnnotations()(results)

    assert set(np.unique(loaded['gt_semantic_seg']).tolist()) == {0, 1}
    assert loaded['gt_semantic_seg'].dtype == np.uint8
