# SegFormer B0 CSV Binary Segmentation

## Goal

Make the official SegFormer repository runnable end to end for binary semantic segmentation when no source CSV exists: download a small COCO-derived segmentation subset, materialize binary masks and a deterministic train/val/test CSV, load the CSV through MMSegmentation, convert the Hugging Face ADE-trained SegFormer-B0 checkpoint, train exactly one epoch on GPU 0, evaluate the test split, and run reusable inference that writes raw masks and overlays.

## Scope and assumptions

- The official SegFormer checkout is placed at `/home/prj/SegFormer`; unrelated repositories in `/home/prj` are out of scope.
- The dataset source is Ultralytics' `coco128-seg.zip`, a small COCO-derived subset with YOLO polygon annotations. The conversion creates binary foreground masks regardless of source object category.
- Thirty images are selected deterministically from the sorted archive contents using seed `20260719`, with 20 train, 5 validation, and 5 test rows. The generated CSV becomes the authoritative split manifest and is never reshuffled by training or inference.
- Images and masks are resized through the existing 512x512 MMSeg pipelines. CPU inference remains supported, but training uses `CUDA_VISIBLE_DEVICES=0` and the existing CUDA-capable environment because the default interpreter contains a CPU-only Torch build.
- The HF ADE-150 decoder classifier is skipped and reinitialized for two classes. Backbone tensors are mapped by explicit verified key rules and checked for shape/equality coverage before the checkpoint is written.

## Design

### Dataset preparation

`tools/prepare_coco_csv.py` downloads the archive only when absent, validates the archive layout, reads polygon labels, rasterizes foreground polygons into 0/1 PNG masks, writes the selected images and masks under `data/coco_binary/`, and writes `data/coco_binary/manifest.csv`. It validates image readability, mask values, dimensions, duplicate pairs, and all split values. Re-running without `--force` preserves the existing manifest and holdout assignment.

### MMSeg dataset

`mmseg/datasets/csv_binary.py` subclasses the repository's `CustomDataset`, resolves absolute paths or paths relative to the CSV directory/data root, filters by the requested split, validates every selected row, fixes classes to `('background', 'foreground')`, and reports split and camera-token counts. `mmseg/datasets/pipelines/csv_loading.py` provides a narrowly scoped annotation loader that accepts only mask values `{0, 1}` or `{0, 255}` and emits class IDs `{0, 1}`.

### Checkpoint conversion

`tools/convert_hf_segformer_to_mmseg.py` builds the local MMSeg B0 model from the binary config, downloads `nvidia/segformer-b0-finetuned-ade-512-512` with Transformers, maps HF encoder/decoder keys to local keys, skips the incompatible 150-class classifier, and writes an MMSeg checkpoint with a JSON conversion report. It fails if the mapped backbone coverage is incomplete or any mapped tensor shape differs, and it verifies tensor equality after serialization.

### Training and inference

`local_configs/segformer_b0_binary_csv.py` sets B0, two classes, the CSV dataset for train/val/test, 512x512 preprocessing, conservative augmentation, one epoch, fixed seed, one worker, and checkpoint/evaluation hooks. Root-level `inference.py` accepts an image, directory, or CSV, resolves test rows and masks, uses only the requested device, writes collision-safe raw `{0,1}` PNG masks and original-resolution overlays, and reports IoU/Dice where ground truth is available.

## Validation

The implementation is validated in this order: dataset preparation and manifest audit; dataset sample construction for all three splits; syntax/import checks; model construction and two-channel logits check; HF conversion report and checkpoint reload; one-GPU one-epoch training; explicit test evaluation; inference over at least three test images; output pixel/resolution checks; and a final review of the SegFormer-only git diff.
