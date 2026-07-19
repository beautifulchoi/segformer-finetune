import csv

import numpy as np
import pytest
import torch
from PIL import Image

from HF_pipeline.data import CsvSegmentationDataset
from HF_pipeline.model import build_processor


def _write_manifest(root, rows):
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "mask", "split"])
        writer.writeheader()
        writer.writerows(rows)


def test_dataset_reads_csv_split_and_returns_binary_labels(tmp_path):
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (8, 6), (20, 30, 40)).save(image_path)
    Image.fromarray(np.array([[0, 255] * 4] * 6, dtype=np.uint8)).save(mask_path)
    _write_manifest(
        tmp_path,
        [{"image": image_path.name, "mask": mask_path.name, "split": "train"}],
    )

    dataset = CsvSegmentationDataset(
        tmp_path / "manifest.csv", "train", build_processor()
    )

    sample = dataset[0]

    assert tuple(sample["pixel_values"].shape) == (3, 512, 512)
    assert tuple(sample["labels"].shape) == (512, 512)
    assert set(sample["labels"].unique().tolist()) == {0, 1}


def test_processor_resizes_image_and_mask_together():
    image = np.zeros((700, 900, 3), dtype=np.uint8)
    mask = np.zeros((700, 900), dtype=np.uint8)
    mask[100:600, 200:700] = 1

    processor = build_processor()
    transformed = processor(
        images=image[:, :, ::-1],
        segmentation_maps=mask,
        return_tensors="pt",
    )

    assert tuple(transformed["pixel_values"].shape) == (1, 3, 512, 512)
    assert tuple(transformed["labels"].shape) == (1, 512, 512)
    assert transformed["labels"].dtype == torch.int64
    assert set(np.unique(transformed["labels"].numpy()).tolist()).issubset({0, 1, 255})


def test_dataset_rejects_invalid_mask_values(tmp_path):
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (4, 4)).save(image_path)
    Image.fromarray(np.full((4, 4), 7, dtype=np.uint8)).save(mask_path)
    _write_manifest(
        tmp_path,
        [{"image": image_path.name, "mask": mask_path.name, "split": "val"}],
    )

    with pytest.raises(ValueError, match="invalid mask values"):
        CsvSegmentationDataset(
            tmp_path / "manifest.csv", "val", build_processor()
        )
