# SegFormer B0 CSV Binary Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a verifiable COCO-derived CSV binary-segmentation path to the official SegFormer repository and run one training epoch, validation, checkpointing, and inference.

**Architecture:** Keep the legacy MMSegmentation flow. Add a CSV dataset and binary-mask loader, a deterministic COCO128-seg preparation utility, an explicit HF-to-MMSeg conversion utility, one local config, and one root inference CLI. Use the existing CUDA Torch environment and never modify Torch packages.

**Tech Stack:** Python 3.9, PyTorch 2.8.0+cu128, legacy MMSegmentation API, mmcv-lite 1.7.x, Transformers, Pillow/OpenCV, NumPy, CSV/JSON.

## Global Constraints

- Preserve the original MMSegmentation-based architecture and training flow.
- Treat the CSV as the only source of image-mask pairing and train/val/test assignment.
- Do not install, upgrade, downgrade, or reinstall `torch`, `torchvision`, or `torchaudio`.
- Use only missing compatible dependencies; do not run the upstream requirements wholesale.
- Use `CUDA_VISIBLE_DEVICES=0` for the training run and record the actual visible device.
- Keep raw masks restricted to class IDs 0 and 1 and keep overlays at original image resolution.

## File map

- Create `tools/prepare_coco_csv.py`: download COCO128-seg, rasterize polygons, validate data, and create the immutable CSV manifest.
- Create `mmseg/datasets/csv_binary.py`: registered CSV dataset subclass with path and row validation.
- Create `mmseg/datasets/pipelines/csv_loading.py`: registered binary annotation loader.
- Modify `mmseg/datasets/__init__.py` and `mmseg/datasets/pipelines/__init__.py`: register the new components.
- Create `tools/convert_hf_segformer_to_mmseg.py`: verified HF-to-local checkpoint conversion and report.
- Create `local_configs/segformer_b0_binary_csv.py`: two-class, 512x512, one-epoch smoke configuration.
- Create `inference.py`: image/directory/CSV inference, raw masks, overlays, and IoU/Dice.
- Create `tests/test_csv_binary.py`: focused dataset and mask-loader behavior tests.
- Create `requirements_custom.txt`: only the missing lightweight dependencies used by the additions.

### Task 1: Bootstrap dependencies and deterministic dataset

**Files:**
- Create: `requirements_custom.txt`
- Create: `tools/prepare_coco_csv.py`
- Test: `tests/test_prepare_coco_csv.py`

- [ ] Write tests for polygon rasterization, split preservation, and binary-mask validation.
- [ ] Run the focused tests and confirm they fail because the utility is absent.
- [ ] Implement archive download, deterministic selection, polygon rasterization, CSV writing, and validation with `--force` protection.
- [ ] Install only `mmcv-lite`, `timm`, `terminaltables`, `addict`, and `yapf` in the existing CUDA environment, with Torch packages excluded.
- [ ] Run the tests, then run `prepare_coco_csv.py --max-images 30` and audit row counts, files, masks, and dimensions.

### Task 2: Add and register the CSV dataset

**Files:**
- Create: `mmseg/datasets/csv_binary.py`
- Create: `mmseg/datasets/pipelines/csv_loading.py`
- Modify: `mmseg/datasets/__init__.py`
- Modify: `mmseg/datasets/pipelines/__init__.py`
- Test: `tests/test_csv_binary.py`

- [ ] Write tests for split filtering, relative/absolute path resolution, duplicate/missing-row errors, and `{0,255}` to `{0,1}` normalization.
- [ ] Run the focused tests red.
- [ ] Implement the registered dataset and loader using the existing `CustomDataset` pipeline contract.
- [ ] Run the tests green and build one dataset object for each of train, val, and test from the local config.

### Task 3: Add the binary B0 config and HF conversion

**Files:**
- Create: `local_configs/segformer_b0_binary_csv.py`
- Create: `tools/convert_hf_segformer_to_mmseg.py`
- Test: `tests/test_model_config.py`

- [ ] Write a model-construction test that asserts two decode-head output channels and the expected B0 feature widths.
- [ ] Run it red before the config/converter exists.
- [ ] Implement the config, explicit HF key mapping, classifier skip, shape checks, equality verification, and JSON report.
- [ ] Run config/import tests, execute conversion, reload the produced checkpoint, and confirm backbone coverage and two-class logits.

### Task 4: Add inference and run the end-to-end pipeline

**Files:**
- Create: `inference.py`
- Test: `tests/test_inference_helpers.py`

- [ ] Write tests for collision-safe output names, binary mask encoding, original-resolution overlays, and IoU/Dice calculations.
- [ ] Run them red.
- [ ] Implement the CLI on top of existing MMSeg inference APIs with explicit device selection and CSV test filtering.
- [ ] Run syntax/import checks and helper tests.
- [ ] Train exactly one epoch with `CUDA_VISIBLE_DEVICES=0` and save the checkpoint under `work_dirs/segformer_b0_binary_smoke`.
- [ ] Evaluate the checkpoint on the CSV test split and record metrics.
- [ ] Run inference on at least three test images; inspect output shape and unique pixel values.

### Task 5: Final audit and report

**Files:**
- Create: `RUN_REPORT.md`

- [ ] Run the full focused test suite and syntax/import checks.
- [ ] Verify `nvidia-smi` and process logs show only GPU 0 visible/used.
- [ ] Review `git diff --stat` and changed-file scope inside `SegFormer`.
- [ ] Write exact commands, artifacts, conversion coverage, metrics, and any residual limitations in `RUN_REPORT.md`.
