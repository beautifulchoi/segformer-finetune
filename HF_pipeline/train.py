from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import CsvSegmentationDataset
from .model import MODEL_ID, build_model, build_processor
from .training import run_training


@dataclass(frozen=True)
class TrainArguments:
    csv_path: Path
    output_dir: Path
    model_id: str
    device: str
    max_steps: int
    batch_size: int
    workers: int
    eval_interval: int
    checkpoint_interval: int
    log_interval: int
    gradient_accumulation_steps: int
    seed: int
    amp: bool
    local_files_only: bool
    allow_insecure_download: bool


def parse_args() -> TrainArguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", dest="csv_path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("HF_pipeline/work_dirs/segformer_b0"))
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=160000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eval-interval", type=int, default=16000)
    parser.add_argument("--checkpoint-interval", type=int, default=4000)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-insecure-download", action="store_true")
    values = parser.parse_args()
    return TrainArguments(**vars(values))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    arguments = parse_args()
    set_seed(arguments.seed)
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    processor = build_processor()
    train_dataset = CsvSegmentationDataset(arguments.csv_path, "train", processor)
    val_dataset = CsvSegmentationDataset(arguments.csv_path, "val", processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=arguments.batch_size,
        shuffle=True,
        num_workers=arguments.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=arguments.workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=arguments.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=arguments.workers > 0,
    )
    model = build_model(
        arguments.model_id,
        arguments.local_files_only,
        arguments.allow_insecure_download,
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    with (arguments.output_dir / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(asdict(arguments), stream, default=str, indent=2)
    run_training(
        model,
        processor,
        train_loader,
        val_loader,
        device,
        arguments.output_dir,
        arguments.max_steps,
        arguments.eval_interval,
        arguments.checkpoint_interval,
        arguments.log_interval,
        gradient_accumulation_steps=arguments.gradient_accumulation_steps,
        amp=arguments.amp,
    )


if __name__ == "__main__":
    main()
