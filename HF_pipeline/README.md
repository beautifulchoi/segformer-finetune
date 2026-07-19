# Hugging Face CSV Pipeline

This directory provides a Hugging Face Transformers implementation of binary SegFormer fine-tuning for the repository's CSV dataset format. It is independent of the existing MMSegmentation workflow and uses an explicit `SegformerConfig`, `SegformerForSemanticSegmentation`, and `SegformerImageProcessor` with a two-class decoder: `0=background`, `1=foreground`.

## Dataset format

The manifest must contain these columns:

```csv
image,mask,split
images/example.jpg,masks/example.png,train
images/example2.jpg,masks/example2.png,val
```

Image and mask paths may be absolute or relative to the manifest directory. Masks may contain `0/1` or `0/255`; they are converted to class IDs `0/1`. Padding uses `255`, which is ignored by the Hugging Face cross-entropy loss.

## Training

Install the dependencies listed in the repository's `requirements_custom.txt` in the same environment as PyTorch, then run:

```bash
python -m HF_pipeline.train \
  --csv data/coco_binary/manifest.csv \
  --device cuda:0
```

The same command is available as a repository-root-safe wrapper:

```bash
HF_pipeline/scripts/train.sh
```

Override defaults with environment variables such as `PYTHON_BIN`, `CSV_PATH`, `OUTPUT_DIR`, `MODEL_ID`, and `DEVICE`, or append additional CLI arguments.

Defaults follow the original SegFormer-B0 configuration:

- 160,000 optimizer steps;
- AdamW with learning rate `6e-5`, betas `(0.9, 0.999)`, and weight decay `0.01`;
- decoder learning-rate multiplier `10`;
- linear warmup for 1,500 steps followed by polynomial decay with power `1.0`;
- batch size `2`, four workers, CUDA AMP, evaluation every 16,000 steps;
- `SegformerImageProcessor` resizing to `512x512`, rescaling by `1/255`, ImageNet normalization, and `do_reduce_labels=False` so background remains class `0`.

For a smoke run on a 15 GB GPU:

```bash
python -m HF_pipeline.train \
  --csv data/coco_binary/manifest.csv \
  --output-dir HF_pipeline/work_dirs/smoke \
  --device cuda:0 \
  --max-steps 2 \
  --batch-size 1 \
  --workers 0 \
  --eval-interval 1 \
  --checkpoint-interval 1
```

Use `--gradient-accumulation-steps 2` or `--batch-size 1` if a larger custom image set causes an out-of-memory error. The loop is native PyTorch because the target workflow is one GPU; Accelerate is not required.

The output directory contains `config.json`, `metrics.jsonl`, `best_model/`, `last_model/`, and periodic `checkpoints/`. Each model directory can be loaded directly by Transformers.

## Inference

Run inference on the held-out CSV split:

```bash
python -m HF_pipeline.inference \
  --checkpoint HF_pipeline/work_dirs/smoke/best_model \
  --csv data/coco_binary/manifest.csv \
  --split test \
  --output-dir HF_pipeline/outputs/test \
  --device cuda:0
```

The CSV inference command is also available as:

```bash
HF_pipeline/scripts/inference.sh
```

Its defaults can be changed with `PYTHON_BIN`, `CHECKPOINT`, `CSV_PATH`, `SPLIT`, `OUTPUT_DIR`, and `DEVICE`. Append `--image path/to/image.jpg` to use the single-image CLI path instead of the default CSV input.

Single images and directories are also supported:

```bash
python -m HF_pipeline.inference \
  --checkpoint HF_pipeline/work_dirs/smoke/best_model \
  --image path/to/image.jpg \
  --output-dir HF_pipeline/outputs/single \
  --device cuda:0
```

Inference writes a binary class-ID mask and a foreground overlay for each image. CSV inference also prints mean IoU and mean Dice.

## What was changed and added

- Added a CSV reader with image/mask validation and a PyTorch dataset returning Hugging Face-compatible `pixel_values` and `labels`.
- Added an explicit `SegformerConfig` model builder that replaces the ADE20K 150-class classifier with a binary head.
- Added direct `SegformerImageProcessor` preprocessing for paired images and masks in both training and inference.
- Added a single-GPU AMP training loop, polynomial scheduler, validation metrics, checkpoints, and resumable model artifacts.
- Added CSV, image, and directory inference with binary masks, overlays, and IoU/Dice reporting.
- Added unit tests for the dataset, processor image/mask geometry, binary metrics, and two-channel model contract.
