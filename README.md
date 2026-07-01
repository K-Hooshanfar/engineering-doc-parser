# engineering-doc-parser

End-to-end pipeline for engineering document images: train a YOLO table detector, auto-rotate and crop title blocks, then extract structured fields (scale, title, drawing number, revision) with Qwen2.5-VL.

## Pipeline overview

```
Raw images + YOLO labels
        │
        ▼
   train.py ──► best.pt
        │
        ▼
    run.py ──► *.crop.png  (auto-rotation + table crop)
        │
        ▼
  extract.py ──► *.md  (Qwen2.5-VL JSON extraction)
```

## Project layout

```
.
├── train.py              # Dataset prep + YOLO training
├── run.py                # Crop tables from image(s) or a folder
├── extract.py            # Qwen2.5-VL field extraction on crops
├── Dockerfile            # GPU image (CUDA)
├── docker-compose.yml
├── configs/
│   ├── train.yaml
│   ├── run.yaml
│   ├── extract.yaml
│   └── prompts/document_extraction.txt
├── src/engineering_doc_parser/
│   ├── table_detector/   # YOLO dataset splitting
│   ├── table_cropper/    # Auto-rotation + YOLO crop
│   ├── extraction/       # Qwen2.5-VL extraction
│   ├── inference/        # Batch YOLO inference CLI (legacy)
│   ├── orientation/      # Legacy EAST + Tesseract orientation
│   └── rotation/         # EasyOCR-based rotation correction
├── scripts/
└── tests/
```

## Installation

**Local (development):**

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

**Docker (GPU, recommended for inference/training):**

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
docker build -t engineering-doc-parser .
```

## Quick start

### 1. Train

Edit `configs/train.yaml`, then:

```bash
python train.py --config configs/train.yaml
```

Or with CLI flags:

```bash
python train.py \
  --source-images data/raw/images \
  --source-labels data/raw/labels \
  --dest-base data/dataset \
  --model yolov8n.pt \
  --epochs 100 \
  --device 0
```

This will:

1. Split images/labels into `train/valid/test`
2. Write `data/dataset/data.yaml`
3. Train with Ultralytics and save `runs/detect/table_train/weights/best.pt`

Use `--prepare-only` to split the dataset without training, or `--skip-prepare` to train on an existing layout.

### 2. Crop

Edit `configs/run.yaml` (set `model_paths` to your `best.pt`), then:

```bash
python run.py --config configs/run.yaml
```

Single image or directory:

```bash
python run.py --input scan.png --output output/crops --model-path runs/detect/table_train/weights/best.pt
python run.py --input images/ --output output/crops --model-path weights/best.pt --device cuda:0
```

Output: `{filename}.crop.png` in the output folder.

The cropper tests all four rotations (0°/90°/180°/270°) and picks the best orientation using YOLO detection scores.

### 3. Extract fields (Qwen2.5-VL)

After cropping, run extraction on the crop folder:

```bash
python extract.py --config configs/extract.yaml
```

Or:

```bash
python extract.py --input output/crops
```

Output: `{filename}.md` next to each crop, containing JSON with `scale`, `title`, `drawing_number_cells`, and `revision`.

**Crop + extract in one step** — enable in `configs/run.yaml`:

```yaml
extraction:
  enabled: true
```

Or pass `--extract`:

```bash
python run.py --config configs/run.yaml --extract
```

To run extraction only on existing crops:

```bash
python run.py --extract-only --input output/crops
```

Edit the prompt in `configs/prompts/document_extraction.txt` without changing code.

## Configuration

| File | Used by | Purpose |
|------|---------|---------|
| `configs/train.yaml` | `train.py` | Dataset paths, split ratios, training hyperparameters |
| `configs/run.yaml` | `run.py` | Crop input/output, YOLO model, optional extraction block |
| `configs/extract.yaml` | `extract.py` | Qwen model, device, prompt path |
| `configs/prompts/document_extraction.txt` | `extract.py` | Extraction prompt |

## Docker

Mount your data under `./data/` and point config paths at `/data/...`.

```bash
# Train
docker run --rm --gpus all -v "%CD%\data:/data" engineering-doc-parser \
  train.py --config /data/train.yaml

# Crop
docker run --rm --gpus all -v "%CD%\data:/data" engineering-doc-parser \
  run.py --config /data/run.yaml

# Crop + extract
docker run --rm --gpus all -v "%CD%\data:/data" engineering-doc-parser \
  run.py --config /data/run.yaml --extract

# Extract only
docker run --rm --gpus all -v "%CD%\data:/data" engineering-doc-parser \
  extract.py --config /data/extract.yaml
```

**Docker Compose:**

```bash
docker compose run --rm train
docker compose run --rm crop
docker compose run --rm crop-and-extract
docker compose run --rm extract
```

Hugging Face model weights are cached under `HF_HOME` (`/data/hf_cache` when `./data` is mounted). The Qwen2.5-VL-7B model needs a GPU with sufficient VRAM (~16 GB+).

## Legacy modules

These are separate tools and are not wired into `train.py` / `run.py` / `extract.py`:

| Module | CLI | Method |
|--------|-----|--------|
| `orientation/` | `python -m engineering_doc_parser.orientation.rotation` | EAST + Tesseract OSD |
| `rotation/` | `python -m engineering_doc_parser.rotation.core` | EasyOCR + Tesseract OSD |
| `inference/` | `python src/.../inference/pipeline.py` | Batch YOLO labels + union crops |
| `table_cropper/` | `python -m engineering_doc_parser.table_cropper.cropper` | Low-level crop API used by `run.py` |

See each module's README for details.

## Testing

```bash
pytest --cov=engineering_doc_parser --cov-report=term-missing --cov-fail-under=85
```

## Typical workflow

```bash
# 1. Train detector
python train.py --config configs/train.yaml

# 2. Update model_paths in configs/run.yaml

# 3. Crop + extract
python run.py --config configs/run.yaml --extract
```

Cropped PNGs land in `output/crops/`; markdown results sit beside them as `*.md`.
