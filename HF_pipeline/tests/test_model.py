import torch
from transformers import SegformerConfig

from HF_pipeline.model import build_model_from_config, model_metadata


def test_model_metadata_declares_binary_segmentation():
    metadata = model_metadata()

    assert metadata["num_labels"] == 2
    assert metadata["id2label"] == {0: "background", 1: "foreground"}


def test_model_builds_two_channel_logits_from_config():
    config = SegformerConfig(num_labels=2, id2label={0: "background", 1: "foreground"})
    model = build_model_from_config(config)
    output = model(pixel_values=torch.zeros(1, 3, 64, 64))

    assert output.logits.shape[0] == 1
    assert output.logits.shape[1] == 2
