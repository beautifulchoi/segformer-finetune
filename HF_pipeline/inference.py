from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image

from .data import load_image, load_mask, read_csv_records
from .metrics import confusion_matrix, metrics_from_confusion
from .model import build_model, build_processor
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


def predict_one(
    model: SegformerForSemanticSegmentation,
    processor: SegformerImageProcessor,
    image_path: Path,
    device: torch.device,
    amp: bool,
) -> np.ndarray:
    image = load_image(image_path)
    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    enabled = amp and device.type == "cuda"
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=enabled
    ):
        logits = model(pixel_values=pixel_values).logits
    logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
    return logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)


def save_outputs(image_path: Path, prediction: np.ndarray, output_dir: Path, index: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{index:04d}_{image_path.stem}"
    Image.fromarray(prediction).save(output_dir / f"{stem}_mask.png")
    image = cv2.cvtColor(load_image(image_path), cv2.COLOR_BGR2RGB)
    overlay = image.copy()
    foreground = prediction == 1
    overlay[foreground] = (
        0.45 * overlay[foreground] + 0.55 * np.array([255, 0, 0])
    ).astype(np.uint8)
    Image.fromarray(overlay).save(output_dir / f"{stem}_overlay.png")


def input_records(arguments) -> list[tuple[Path, Optional[Path]]]:
    if arguments.csv is not None:
        records = read_csv_records(arguments.csv, arguments.split)
        return [(record.image_path, record.mask_path) for record in records]
    if arguments.image is None:
        raise ValueError("provide --csv or --image")
    if arguments.image.is_dir():
        paths = sorted(
            path
            for path in arguments.image.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        )
        return [(path, None) for path in paths]
    return [(arguments.image, arguments.mask)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--image", type=Path)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("HF_pipeline/outputs"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-insecure-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = build_model(
        str(arguments.checkpoint),
        arguments.local_files_only,
        arguments.allow_insecure_download,
    )
    nn.Module.to(model, device)
    processor = build_processor()
    records = input_records(arguments)
    matrix = np.zeros((2, 2), dtype=np.int64)
    for index, (image_path, mask_path) in enumerate(records):
        prediction = predict_one(model, processor, image_path, device, arguments.amp)
        save_outputs(image_path, prediction, arguments.output_dir, index)
        if mask_path is not None:
            matrix += confusion_matrix(prediction, load_mask(mask_path))
    if any(mask_path is not None for _, mask_path in records):
        metrics = metrics_from_confusion(matrix)
        print(f"mIoU={metrics.mean_iou:.5f} mDice={metrics.mean_dice:.5f}")


if __name__ == "__main__":
    main()
