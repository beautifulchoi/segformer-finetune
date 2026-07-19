from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image

from .custom import CustomDataset
from .builder import DATASETS
from mmseg.utils import get_root_logger


@DATASETS.register_module()
class CsvBinaryDataset(CustomDataset):
    CLASSES = ('background', 'foreground')
    PALETTE = [[0, 0, 0], [255, 0, 0]]

    def __init__(self,
                 pipeline,
                 csv_path,
                 split,
                 image_col='image',
                 mask_col='mask',
                 split_col='split',
                 data_root=None,
                 test_mode=False,
                 palette=None):
        self.csv_path = Path(csv_path).expanduser().resolve()
        self.requested_split = split
        self.image_col = image_col
        self.mask_col = mask_col
        self.split_col = split_col
        self.csv_data_root = Path(data_root).expanduser().resolve() if data_root else None
        super().__init__(
            pipeline=pipeline,
            img_dir='',
            ann_dir='',
            split=None,
            data_root=None,
            test_mode=test_mode,
            ignore_index=255,
            reduce_zero_label=False,
            classes=self.CLASSES,
            palette=palette or self.PALETTE)

    def _resolve_path(self, raw_path):
        path = Path(raw_path).expanduser()
        if path.is_absolute():
            return path.resolve()
        candidates = []
        if self.csv_data_root is not None:
            candidates.append(self.csv_data_root / path)
        candidates.append(self.csv_path.parent / path)
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        return candidates[0].resolve()

    @staticmethod
    def _validate_pair(image_path, mask_path, row_number):
        if not image_path.is_file():
            raise FileNotFoundError(f'missing image at CSV row {row_number}: {image_path}')
        if not mask_path.is_file():
            raise FileNotFoundError(f'missing mask at CSV row {row_number}: {mask_path}')
        with Image.open(image_path) as image:
            image.load()
            image_size = image.size
        with Image.open(mask_path) as mask_image:
            mask_image.load()
            mask = np.asarray(mask_image)
            mask_size = mask_image.size
        if mask.ndim != 2:
            raise ValueError(f'mask must be single-channel at CSV row {row_number}: {mask_path}')
        values = set(np.unique(mask).tolist())
        if not (values.issubset({0, 1}) or values.issubset({0, 255})):
            raise ValueError(f'mask has invalid values at CSV row {row_number}: {sorted(values)}')
        if image_size != mask_size:
            raise ValueError(f'image/mask size mismatch at CSV row {row_number}')

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split):
        if not self.csv_path.is_file():
            raise FileNotFoundError(f'CSV manifest not found: {self.csv_path}')
        infos = []
        counts = Counter()
        camera_counts = Counter()
        pairs = set()
        with self.csv_path.open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            required = {self.image_col, self.mask_col, self.split_col}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f'CSV must contain columns {sorted(required)}')
            for row_number, row in enumerate(reader, 2):
                row_split = row[self.split_col]
                if row_split not in {'train', 'val', 'test'}:
                    raise ValueError(f'invalid split {row_split!r} at CSV row {row_number}')
                image_path = self._resolve_path(row[self.image_col])
                mask_path = self._resolve_path(row[self.mask_col])
                pair = (str(image_path), str(mask_path))
                if pair in pairs:
                    raise ValueError(f'duplicate image/mask pair at CSV row {row_number}')
                pairs.add(pair)
                self._validate_pair(image_path, mask_path, row_number)
                counts[row_split] += 1
                lower_name = image_path.name.lower()
                for camera in ('cam1', 'cam2', 'cam3'):
                    if camera in lower_name:
                        camera_counts[f'{row_split}:{camera}'] += 1
                if row_split == self.requested_split:
                    infos.append(dict(filename=str(image_path), ann=dict(seg_map=str(mask_path))))
        if not infos:
            raise ValueError(f'CSV split {self.requested_split!r} is empty')
        print_log(f'CSV split counts: {dict(counts)}', logger=get_root_logger())
        if camera_counts:
            print_log(f'CSV camera counts: {dict(camera_counts)}', logger=get_root_logger())
        print_log(f'Loaded {len(infos)} images for split {self.requested_split}', logger=get_root_logger())
        return infos
