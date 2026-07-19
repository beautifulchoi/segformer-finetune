from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import torch
from PIL import Image

UInt8Array = NDArray[np.uint8]


@dataclass(frozen=True)
class InferenceItem:
    image: Path
    mask: Path | None


def normalize_mask(mask: UInt8Array) -> UInt8Array:
    values = set(np.unique(mask).tolist())
    if values.issubset({0, 255}):
        return (mask == 255).astype(np.uint8)
    if values.issubset({0, 1}):
        return mask.astype(np.uint8)
    raise ValueError(f'mask contains invalid values: {sorted(values)}')


def binary_metrics(prediction: UInt8Array, target: UInt8Array) -> tuple[float, float]:
    pred = normalize_mask(prediction).astype(bool)
    truth = normalize_mask(target).astype(bool)
    if pred.shape != truth.shape:
        raise ValueError(f'metric shape mismatch: {pred.shape} != {truth.shape}')
    true_positive = np.logical_and(pred, truth).sum()
    false_positive = np.logical_and(pred, np.logical_not(truth)).sum()
    false_negative = np.logical_and(np.logical_not(pred), truth).sum()
    union = true_positive + false_positive + false_negative
    dice_denominator = 2 * true_positive + false_positive + false_negative
    iou = 1.0 if union == 0 else float(true_positive / union)
    dice = 1.0 if dice_denominator == 0 else float(2 * true_positive / dice_denominator)
    return iou, dice


def output_stem(image_path: Path, index: int) -> str:
    return f'{index:04d}_{image_path.stem}'


def _resolve(raw_path: str, base_dir: Path, data_root: Path | None) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
    else:
        candidates = [data_root / path] if data_root is not None else []
        candidates.append(base_dir / path)
        resolved = next((candidate.resolve() for candidate in candidates if candidate.is_file()), candidates[0].resolve())
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def items_from_input(input_path: Path, split: str, image_col: str, mask_col: str, split_col: str, data_root: Path | None) -> list[InferenceItem]:
    if input_path.is_file() and input_path.suffix.lower() == '.csv':
        items = []
        with input_path.open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or image_col not in reader.fieldnames or split_col not in reader.fieldnames:
                raise ValueError(f'CSV must contain columns {image_col!r} and {split_col!r}')
            has_masks = mask_col in reader.fieldnames
            for row in reader:
                if row[split_col] != split:
                    continue
                image = _resolve(row[image_col], input_path.parent, data_root)
                mask = _resolve(row[mask_col], input_path.parent, data_root) if has_masks and row[mask_col] else None
                items.append(InferenceItem(image, mask))
        if not items:
            raise ValueError(f'CSV split {split!r} is empty')
        return items
    if input_path.is_file():
        return [InferenceItem(input_path.resolve(), None)]
    if input_path.is_dir():
        images = sorted(path for path in input_path.rglob('*') if path.suffix.lower() in {'.jpg', '.jpeg', '.png'})
        if not images:
            raise ValueError(f'no supported images found in {input_path}')
        return [InferenceItem(path.resolve(), None) for path in images]
    raise FileNotFoundError(input_path)


def _overlay(image_path: Path, prediction: UInt8Array) -> Image.Image:
    with Image.open(image_path) as source:
        image = source.convert('RGB')
    if prediction.shape != (image.height, image.width):
        prediction = np.asarray(Image.fromarray(prediction).resize(image.size, Image.Resampling.NEAREST), dtype=np.uint8)
    pixels = np.asarray(image).copy()
    foreground = prediction.astype(bool)
    pixels[foreground] = (0.55 * pixels[foreground] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
    return Image.fromarray(pixels)


def run(config_path: Path, checkpoint_path: Path, input_path: Path, output_dir: Path, device: str, split: str, image_col: str, mask_col: str, split_col: str, data_root: Path | None) -> tuple[float, float] | None:
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(f'requested {device}, but CUDA is unavailable')
    from mmseg.apis import inference_segmentor, init_segmentor

    items = items_from_input(input_path, split, image_col, mask_col, split_col, data_root)
    model = init_segmentor(str(config_path), str(checkpoint_path), device=device)
    output_dir.mkdir(parents=True, exist_ok=True)
    totals = np.zeros(3, dtype=np.float64)
    scored = 0
    for index, item in enumerate(items):
        result = inference_segmentor(model, str(item.image))
        prediction = normalize_mask(np.asarray(result[0], dtype=np.uint8))
        stem = output_stem(item.image, index)
        Image.fromarray(prediction).save(output_dir / f'{stem}_mask.png')
        _overlay(item.image, prediction).save(output_dir / f'{stem}_overlay.png')
        if item.mask is not None:
            with Image.open(item.mask) as target_image:
                target = normalize_mask(np.asarray(target_image.convert('L'), dtype=np.uint8))
            if prediction.shape != target.shape:
                prediction = np.asarray(Image.fromarray(prediction).resize((target.shape[1], target.shape[0]), Image.Resampling.NEAREST), dtype=np.uint8)
            iou, dice = binary_metrics(prediction, target)
            totals += (iou, dice, 1.0)
            scored += 1
        print(f'{item.image.name}: mask={output_dir / f"{stem}_mask.png"}')
    if scored == 0:
        return None
    return float(totals[0] / totals[2]), float(totals[1] / totals[2])


def _parse_args():
    parser = argparse.ArgumentParser(description='Run binary SegFormer inference.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--split', default='test')
    parser.add_argument('--image-col', default='image')
    parser.add_argument('--mask-col', default='mask')
    parser.add_argument('--split-col', default='split')
    parser.add_argument('--data-root')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    metrics = run(
        Path(args.config), Path(args.checkpoint), Path(args.input), Path(args.output_dir),
        args.device, args.split, args.image_col, args.mask_col, args.split_col,
        Path(args.data_root).resolve() if args.data_root else None)
    if metrics is not None:
        print(f'test IoU={metrics[0]:.6f} Dice={metrics[1]:.6f}')
