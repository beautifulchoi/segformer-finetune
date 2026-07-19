from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TypedDict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import SegformerImageProcessor


class SegformerSample(TypedDict):
    pixel_values: torch.Tensor
    labels: torch.Tensor
    image_path: str
    mask_path: str
    original_size: tuple[int, int]


@dataclass(frozen=True)
class CsvRecord:
    image_path: Path
    mask_path: Path
    split: str


def _resolve_path(raw_path: str, csv_path: Path, data_root: Optional[Path]) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = [csv_path.parent / path]
    if data_root is not None:
        candidates.insert(0, data_root / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def _read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"unable to read mask: {path}")
    values = set(np.unique(mask).tolist())
    if values.issubset({0, 255}):
        return (mask == 255).astype(np.uint8)
    if values.issubset({0, 1}):
        return mask.astype(np.uint8)
    raise ValueError(f"invalid mask values in {path}: {sorted(values)}")


def read_csv_records(
    csv_path: Path,
    split: str,
    data_root: Optional[Path] = None,
    image_col: str = "image",
    mask_col: str = "mask",
    split_col: str = "split",
) -> list[CsvRecord]:
    csv_path = csv_path.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV manifest not found: {csv_path}")
    records: list[CsvRecord] = []
    seen: set[tuple[Path, Path]] = set()
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {image_col, mask_col, split_col}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain columns {sorted(required)}")
        for row_number, row in enumerate(reader, 2):
            row_split = row[split_col]
            if row_split not in {"train", "val", "test"}:
                raise ValueError(f"invalid split {row_split!r} at CSV row {row_number}")
            image_path = _resolve_path(row[image_col], csv_path, data_root)
            mask_path = _resolve_path(row[mask_col], csv_path, data_root)
            if not image_path.is_file():
                raise FileNotFoundError(f"missing image at CSV row {row_number}: {image_path}")
            if not mask_path.is_file():
                raise FileNotFoundError(f"missing mask at CSV row {row_number}: {mask_path}")
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            mask = _read_mask(mask_path)
            if image is None:
                raise ValueError(f"unable to read image at CSV row {row_number}: {image_path}")
            if image.shape[:2] != mask.shape:
                raise ValueError(f"image/mask size mismatch at CSV row {row_number}")
            pair = (image_path, mask_path)
            if pair in seen:
                raise ValueError(f"duplicate image/mask pair at CSV row {row_number}")
            seen.add(pair)
            if row_split == split:
                records.append(CsvRecord(image_path, mask_path, row_split))
    if not records:
        raise ValueError(f"CSV split {split!r} is empty")
    return records


class CsvSegmentationDataset(Dataset[SegformerSample]):
    def __init__(
        self,
        csv_path: Path,
        split: str,
        processor: SegformerImageProcessor,
        data_root: Optional[Path] = None,
    ) -> None:
        self.records = read_csv_records(csv_path, split, data_root)
        self.processor = processor

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> SegformerSample:
        record = self.records[index]
        image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
        mask = _read_mask(record.mask_path)
        if image is None:
            raise ValueError(f"unable to read image: {record.image_path}")
        original_size = (image.shape[0], image.shape[1])
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.processor(
            images=image_rgb,
            segmentation_maps=mask,
            return_tensors="pt",
            do_reduce_labels=False,
        )
        return {
            "pixel_values": transformed["pixel_values"][0],
            "labels": transformed["labels"][0].long(),
            "image_path": str(record.image_path),
            "mask_path": str(record.mask_path),
            "original_size": original_size,
        }


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unable to read image: {path}")
    return image


def load_mask(path: Path) -> np.ndarray:
    return _read_mask(path)
