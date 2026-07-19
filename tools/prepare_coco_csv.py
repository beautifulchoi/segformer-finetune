from __future__ import annotations

import argparse
import csv
import random
import shutil
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

DATA_URL: Final = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128-seg.zip'
UINT8_MASK = NDArray[np.uint8]


def validate_binary_mask(mask: UINT8_MASK) -> None:
    values = set(np.unique(mask).tolist())
    if not (values.issubset({0, 1}) or values.issubset({0, 255})):
        raise ValueError(f'mask is not binary; found values {sorted(values)}')


def rasterize_yolo_segments(label_text: str, width: int, height: int) -> UINT8_MASK:
    canvas = Image.new('L', (width, height), 0)
    drawer = ImageDraw.Draw(canvas)
    for line_number, line in enumerate(label_text.splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 7 or (len(fields) - 1) % 2:
            raise ValueError(f'invalid polygon at label line {line_number}')
        coordinates = [float(value) for value in fields[1:]]
        points = [
            (
                round(max(0.0, min(1.0, x)) * (width - 1)),
                round(max(0.0, min(1.0, y)) * (height - 1)),
            )
            for x, y in zip(coordinates[::2], coordinates[1::2])
        ]
        if len(points) < 3:
            raise ValueError(f'polygon at label line {line_number} has fewer than 3 points')
        drawer.polygon(points, fill=1)
    return np.asarray(canvas, dtype=np.uint8)


def split_names(names: list[str], max_images: int, seed: int) -> dict[str, list[str]]:
    selected = sorted(names)[:max_images]
    if len(selected) < 3:
        raise ValueError('at least three images are required for train/val/test splits')
    random.Random(seed).shuffle(selected)
    train_count = max(1, round(len(selected) * 2 / 3))
    if len(selected) - train_count < 2:
        train_count = len(selected) - 2
    remaining = len(selected) - train_count
    val_count = remaining // 2
    return {
        'train': selected[:train_count],
        'val': selected[train_count:train_count + val_count],
        'test': selected[train_count + val_count:],
    }


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open('wb') as output:
        shutil.copyfileobj(response, output)


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image.load()
        return image.size


def _validate_manifest(csv_path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    pairs: set[tuple[str, str]] = set()
    with csv_path.open(newline='', encoding='utf-8') as stream:
        rows = csv.DictReader(stream)
        required = {'image', 'mask', 'split'}
        if rows.fieldnames is None or not required.issubset(rows.fieldnames):
            raise ValueError(f'CSV must contain columns {sorted(required)}')
        for row_number, row in enumerate(rows, 2):
            split = row['split']
            if split not in {'train', 'val', 'test'}:
                raise ValueError(f'invalid split {split!r} at row {row_number}')
            pair = (row['image'], row['mask'])
            if pair in pairs:
                raise ValueError(f'duplicate image/mask pair at row {row_number}')
            pairs.add(pair)
            image_path = (csv_path.parent / row['image']).resolve()
            mask_path = (csv_path.parent / row['mask']).resolve()
            if not image_path.is_file() or not mask_path.is_file():
                raise FileNotFoundError(f'missing pair at row {row_number}: {image_path}, {mask_path}')
            if _image_size(image_path) != _image_size(mask_path):
                raise ValueError(f'image/mask size mismatch at row {row_number}')
            with Image.open(mask_path) as mask_image:
                mask = np.asarray(mask_image.convert('L'), dtype=np.uint8)
            validate_binary_mask(mask)
            counts[split] += 1
    if not pairs or any(counts[split] == 0 for split in ('train', 'val', 'test')):
        raise ValueError('CSV must contain non-empty train, val, and test splits')
    return counts


def prepare_dataset(args: argparse.Namespace) -> Counter[str]:
    output_dir = Path(args.output_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    archive = Path(args.archive).resolve() if args.archive else cache_dir / 'coco128-seg.zip'
    manifest = output_dir / 'manifest.csv'
    if manifest.exists() and not args.force:
        raise FileExistsError(f'{manifest} exists; pass --force only to regenerate it')
    if not archive.exists():
        print(f'downloading {args.url}')
        _download(args.url, archive)
    source_dir = cache_dir / 'coco128-seg'
    if not source_dir.exists():
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(cache_dir)
    image_dir = source_dir / 'images' / 'train2017'
    label_dir = source_dir / 'labels' / 'train2017'
    names = [
        image.name for image in image_dir.glob('*.jpg')
        if (label_dir / f'{image.stem}.txt').is_file()
    ]
    splits = split_names(names, args.max_images, args.seed)
    output_image_dir = output_dir / 'images'
    output_mask_dir = output_dir / 'masks'
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for split, split_names_list in splits.items():
        for name in split_names_list:
            source_image = image_dir / name
            source_label = label_dir / f'{source_image.stem}.txt'
            destination_image = output_image_dir / name
            destination_mask = output_mask_dir / f'{source_image.stem}.png'
            shutil.copy2(source_image, destination_image)
            width, height = _image_size(source_image)
            mask = rasterize_yolo_segments(source_label.read_text(encoding='utf-8'), width, height)
            validate_binary_mask(mask)
            Image.fromarray(mask).save(destination_mask)
            rows.append({'image': f'images/{name}', 'mask': f'masks/{source_image.stem}.png', 'split': split})
    output_dir.mkdir(parents=True, exist_ok=True)
    with manifest.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['image', 'mask', 'split'])
        writer.writeheader()
        writer.writerows(rows)
    counts = _validate_manifest(manifest)
    print(f'wrote {manifest}')
    print(f'split counts: {dict(counts)}')
    return counts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Create a CSV binary mask dataset from COCO128-seg.')
    parser.add_argument('--output-dir', default='data/coco_binary')
    parser.add_argument('--cache-dir', default='data/cache')
    parser.add_argument('--archive')
    parser.add_argument('--url', default=DATA_URL)
    parser.add_argument('--max-images', type=int, default=30)
    parser.add_argument('--seed', type=int, default=20260719)
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    prepare_dataset(_parse_args())
