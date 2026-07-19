from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from transformers import (
    SegformerConfig,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"
IMAGE_MEAN = [123.675, 116.28, 103.53]
IMAGE_STD = [58.395, 57.12, 57.375]


class ModelMetadata(TypedDict):
    num_labels: int
    id2label: dict[int, str]
    label2id: dict[str, int]
    ignore_index: int


def model_metadata() -> ModelMetadata:
    return {
        "num_labels": 2,
        "id2label": {0: "background", 1: "foreground"},
        "label2id": {"background": 0, "foreground": 1},
        "ignore_index": 255,
    }


def build_model(
    model_id: str = MODEL_ID, local_files_only: bool = False
) -> SegformerForSemanticSegmentation:
    config = build_config(model_id, local_files_only)
    return SegformerForSemanticSegmentation.from_pretrained(
        model_id,
        config=config,
        ignore_mismatched_sizes=True,
        local_files_only=local_files_only,
    )


def build_model_from_config(config: SegformerConfig) -> SegformerForSemanticSegmentation:
    return SegformerForSemanticSegmentation(config)


def build_config(model_id: str = MODEL_ID, local_files_only: bool = False) -> SegformerConfig:
    metadata = model_metadata()
    return SegformerConfig.from_pretrained(
        model_id,
        num_labels=metadata["num_labels"],
        id2label=metadata["id2label"],
        label2id=metadata["label2id"],
        semantic_loss_ignore_index=metadata["ignore_index"],
        local_files_only=local_files_only,
    )


def build_processor() -> SegformerImageProcessor:
    return SegformerImageProcessor(
        do_resize=True,
        size={"height": 512, "width": 512},
        do_rescale=True,
        do_normalize=True,
        image_mean=IMAGE_MEAN,
        image_std=IMAGE_STD,
        do_reduce_labels=False,
    )


def save_processor(processor: SegformerImageProcessor, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(directory)
