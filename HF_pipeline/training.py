from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from .metrics import BinaryMetrics, confusion_matrix, metrics_from_confusion


def parameter_groups(
    model: nn.Module, learning_rate: float, weight_decay: float
) -> list[dict[str, object]]:
    grouped: dict[tuple[float, float], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        head = name.startswith("decode_head.")
        no_decay = name.endswith("bias") or any(
            token in name.lower() for token in ("norm", "layer_norm", "batch_norm")
        )
        key = (learning_rate * 10 if head else learning_rate, 0.0 if no_decay else weight_decay)
        grouped.setdefault(key, []).append(parameter)
    return [
        {"params": parameters, "lr": lr, "weight_decay": decay}
        for (lr, decay), parameters in grouped.items()
    ]


def polynomial_factor(
    step: int, max_steps: int, warmup_steps: int, warmup_ratio: float, power: float
) -> float:
    if step < warmup_steps:
        return warmup_ratio + (1.0 - warmup_ratio) * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return max(0.0, (1.0 - progress) ** power)


def create_optimizer_and_scheduler(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    max_steps: int,
    warmup_steps: int,
) -> tuple[Optimizer, LambdaLR]:
    optimizer = AdamW(
        parameter_groups(model, learning_rate, weight_decay),
        betas=(0.9, 0.999),
    )
    scheduler = LambdaLR(
        optimizer,
        lambda step: polynomial_factor(step, max_steps, warmup_steps, 1e-6, 1.0),
    )
    return optimizer, scheduler


def evaluate(
    model: SegformerForSemanticSegmentation,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
) -> BinaryMetrics:
    model.eval()
    matrix = np.zeros((2, 2), dtype=np.int64)
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            enabled = amp and device.type == "cuda"
            context = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)
            with context:
                output = model(pixel_values=pixel_values)
            logits = F.interpolate(output.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            prediction = logits.argmax(dim=1).cpu().numpy()
            matrix += confusion_matrix(prediction, labels.cpu().numpy())
    return metrics_from_confusion(matrix)


def save_artifacts(
    model: SegformerForSemanticSegmentation,
    processor: SegformerImageProcessor,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    directory: Path,
    step: int,
    best_metrics: BinaryMetrics,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory)
    processor.save_pretrained(directory)
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_mean_iou": best_metrics.mean_iou,
            "best_mean_dice": best_metrics.mean_dice,
        },
        directory / "training_state.pt",
    )


def append_metrics(path: Path, step: int, loss: float, metrics: BinaryMetrics) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "step": step,
                    "loss": loss,
                    "mean_iou": metrics.mean_iou,
                    "mean_dice": metrics.mean_dice,
                }
            )
            + "\n"
        )


def run_training(
    model: SegformerForSemanticSegmentation,
    processor: SegformerImageProcessor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    max_steps: int,
    eval_interval: int,
    checkpoint_interval: int,
    log_interval: int,
    learning_rate: float = 6e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 1500,
    gradient_accumulation_steps: int = 1,
    amp: bool = True,
) -> BinaryMetrics:
    nn.Module.to(model, device)
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, learning_rate, weight_decay, max_steps, warmup_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    train_iterator = iter(train_loader)
    best_metrics = BinaryMetrics(0.0, 0.0)
    loss_window: list[float] = []
    progress = tqdm(
        range(1, max_steps + 1),
        desc="HF SegFormer training",
        unit="step",
        dynamic_ncols=True,
    )
    for step in progress:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(gradient_accumulation_steps):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            enabled = amp and device.type == "cuda"
            context = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)
            with context:
                loss = model(pixel_values=pixel_values, labels=labels).loss
                scaled_loss = loss / gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            step_loss += float(loss.detach().cpu())
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        average_loss = step_loss / gradient_accumulation_steps
        loss_window.append(average_loss)
        if step % log_interval == 0 or step == max_steps:
            progress.set_postfix(
                loss=f"{sum(loss_window) / len(loss_window):.5f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
            loss_window.clear()
        if step % eval_interval == 0 or step == max_steps:
            metrics = evaluate(model, val_loader, device, amp)
            append_metrics(metrics_path, step, average_loss, metrics)
            progress.set_postfix(
                loss=f"{average_loss:.5f}",
                mIoU=f"{metrics.mean_iou:.5f}",
                mDice=f"{metrics.mean_dice:.5f}",
            )
            if metrics.mean_iou >= best_metrics.mean_iou:
                best_metrics = metrics
                save_artifacts(
                    model, processor, optimizer, scheduler, output_dir / "best_model", step, best_metrics
                )
        if step % checkpoint_interval == 0 or step == max_steps:
            save_artifacts(
                model, processor, optimizer, scheduler, output_dir / "last_model", step, best_metrics
            )
            checkpoint_dir = output_dir / "checkpoints" / f"step-{step:07d}"
            save_artifacts(model, processor, optimizer, scheduler, checkpoint_dir, step, best_metrics)
    return best_metrics
