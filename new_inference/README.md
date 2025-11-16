# YOLO-Based Auto-Rotation Table Cropper

A sophisticated table detection and cropping system that automatically determines the correct document orientation by testing all four rotations (0°, 90°, 180°, 270°) and selecting the best one based on YOLO detection quality metrics.

## ✨ Features

- **Automatic Rotation Detection**: Tests all 4 cardinal rotations and selects the optimal orientation
- **Multi-Model Fallback**: Uses multiple YOLO models in sequence for robust detection
- **Intelligent Scoring**: Composite scoring system considers:
  - Union coverage of detected boxes
  - Average and top-K confidence scores
  - Fragmentation penalty (penalizes many tiny boxes)
  - Bottom position bias (prefers tables near bottom of page)
  - Geometry prior (prefers wide, horizontal title blocks)
- **Debug Visualization**: Optional rotation candidate grid showing all 4 rotations with metrics
- **Batch Processing**: Process entire directories with progress tracking
- **Flexible I/O**: Works with image bytes or file paths

## 📋 Overview

This tool is designed for processing scanned documents where the orientation may be unknown. It:

1. Loads an image (from bytes or file)
2. Tests all 4 rotations (0°, 90°, 180°, 270°)
3. Runs YOLO detection on each rotated version
4. Scores each rotation using a composite metric
5. Selects the best rotation and crops the detected table region
6. Falls back to additional models if the primary model fails

## 🔧 Requirements

### Python Dependencies

```bash
pip install ultralytics opencv-python pillow numpy tqdm
```

### YOLO Models

The script expects ONNX format YOLO models. Update `MODEL_PATHS` in `inference_yolo.py`:

```python
MODEL_PATHS: List[str] = [
    "/path/to/model1.onnx",
    "/path/to/model2.onnx",
    "/path/to/model3.onnx",
]
```

**Supported Formats:**
- ONNX models (`.onnx`)
- PyTorch models (`.pt`) - Ultralytics will handle conversion

## 🚀 Quick Start

### Command Line Usage

**Basic usage:**
```bash
python new_inference/inference_yolo.py \
  --in input_images \
  --out output_crops \
  --conf 0.25
```

**With all options:**
```bash
python new_inference/inference_yolo.py \
  --in /path/to/images \
  --out /path/to/output \
  --conf 0.3 \
  --device cuda:0 \
  --include-subdirs \
  --rotation-debug \
  --bottom-weight 0.3
```

### Python API Usage

**Process a single image from bytes:**
```python
from new_inference.inference_yolo import crop_tables_from_bytes_png

with open("document.png", "rb") as f:
    image_bytes = f.read()

cropped_png_bytes = crop_tables_from_bytes_png(
    image_bytes=image_bytes,
    conf_thresh=0.25,
    prefer_bottom=True,
    bottom_weight=0.3,
    verbose=True
)

with open("output_crop.png", "wb") as f:
    f.write(cropped_png_bytes)
```

**Process a folder:**
```python
from new_inference.inference_yolo import process_folder

process_folder(
    input_dir="input_images",
    output_dir="output_crops",
    conf_thresh=0.25,
    prefer_bottom=True,
    bottom_weight=0.3,
    verbose=True,
    include_subdirs=False,
    show_progress=True,
    rotation_debug=True
)
```

## 📖 CLI Reference

```text
--in, --input_dir          Input folder containing images (default: input_images)
--out, --output_dir        Output folder for cropped PNGs (default: output_crops)
--conf, --conf-thresh      Confidence threshold for detections (default: 0.25)
--device                   Device for inference: cpu or cuda:0 (default: cpu)
--no-bottom-bias           Disable preference for tables at bottom
--bottom-weight FLOAT      Weight for bottom position bias, 0-1 (default: 0.3)
--include-subdirs          Process files in subfolders recursively
--no-progress              Disable tqdm progress bar
--quiet                    Disable verbose logging
--rotation-debug           Save rotation candidates debug grid PNG per image
```

## 🎯 How It Works

### Rotation Testing

For each image, the system:

1. **Tests 4 rotations**: 0°, 90°, 180°, 270°
2. **Runs YOLO detection** on each rotated image
3. **Computes metrics** for each rotation:
   - `coverage`: Union area coverage of all detected boxes (0-1)
   - `topk_conf`: Average confidence of top-K largest boxes
   - `avg_conf`: Average confidence of all detections
   - `frag`: Fragmentation penalty (penalizes many small boxes)
   - `bottom`: Normalized Y-center position (0=top, 1=bottom)
   - `geom_score`: Geometry score based on largest box aspect ratio and position

4. **Scores each rotation** using weighted combination:
   ```
   score = coverage * 8.0
         + topk_conf * 3.0
         + avg_conf * 3.0
         - frag * 4.0
         + bottom * 0.2 (if prefer_bottom)
         + geom_score * 8.0
   ```

5. **Selects best rotation** (with tie-breaking favoring 0° if scores are close)

### Model Fallback

If the primary model finds no detections:
1. Uses the best rotation from the primary model
2. Tries additional models in sequence
3. Returns the first successful detection

### Output

- **Cropped image**: PNG format containing the union bounding box of all detections
- **Debug grid** (optional): Visual comparison of all 4 rotations with metrics overlay

## 📊 Scoring Components

### Coverage (`coverage_weight = 8.0`)
- Measures how much of the image is covered by detected boxes
- Higher coverage suggests better detection

### Confidence (`conf_weight = 3.0`, `largestk_weight = 3.0`)
- Average confidence of all detections
- Top-K confidence of largest boxes
- Lower weight to prevent over-reliance on confidence alone

### Fragmentation (`frag_weight = 4.0`)
- Penalizes many small, fragmented detections
- Encourages fewer, larger, more coherent detections

### Bottom Bias (`bottom_weight = 0.2`)
- Optional preference for detections near the bottom of the page
- Useful for title blocks and signature areas

### Geometry Score (`weight = 8.0`)
- Based on the largest detected box:
  - **Aspect ratio**: Prefers wide, horizontal boxes (title blocks)
  - **Position**: Prefers boxes lower on the page
- Strong influence (8.0x) to prioritize correct orientation

## 🖼️ Debug Visualization

Enable `--rotation-debug` to generate a debug grid showing all 4 rotation candidates:

```
image_name.rotation_debug.png
```

The grid displays:
- All 4 rotated versions side-by-side
- Overlaid bounding boxes on the largest detection
- Metrics for each rotation:
  - Score
  - Coverage
  - Geometry score
  - Aspect ratio
  - Number of boxes

## ⚙️ Configuration

### Model Paths

Edit the `MODEL_PATHS` list at the top of `inference_yolo.py`:

```python
MODEL_PATHS: List[str] = [
    "/path/to/primary_model.onnx",
    "/path/to/fallback_model1.onnx",
    "/path/to/fallback_model2.onnx",
]
```

### Device Selection

Set `DEVICE` or use `--device` flag:
- `"cpu"`: CPU inference (default)
- `"cuda:0"`: GPU inference (requires CUDA)

### Scoring Weights

Adjust weights in `find_best_rotation()` function:
- `coverage_weight`: Default 8.0
- `conf_weight`: Default 3.0
- `largestk_weight`: Default 3.0
- `frag_weight`: Default 4.0
- `bottom_weight`: Default 0.2 (CLI: `--bottom-weight`)

## 📁 Supported Formats

**Input:**
- `.tif`, `.tiff` (multi-page supported)
- `.png`
- `.jpg`, `.jpeg`

**Output:**
- `.png` (cropped table regions)

## 🔍 Example Output

```
============================================================
Processing: document_001.png
============================================================
🔄 Testing rotations with model: best_r_985_p_973_map_832.onnx
    0°: boxes=3, coverage=0.15, topK=0.82, avg_conf=0.78, frag=0.12, bottom=0.65, geom=0.45, aspect_main=2.1 => score=12.34
   90°: boxes=2, coverage=0.08, topK=0.65, avg_conf=0.62, frag=0.25, bottom=0.35, geom=0.20, aspect_main=0.8 => score=5.67
  180°: boxes=1, coverage=0.05, topK=0.55, avg_conf=0.55, frag=0.30, bottom=0.20, geom=0.10, aspect_main=0.5 => score=2.45
  270°: boxes=2, coverage=0.09, topK=0.70, avg_conf=0.65, frag=0.22, bottom=0.40, geom=0.25, aspect_main=0.9 => score=6.12
✅ Best rotation: 0° (score=12.34, geom=0.45)
✅ Using model: /code/Datasets/foolad/model_serve/best_r_985_p_973_map_832.onnx
✅ Saved: output_crops/document_001.crop.png
```

## 🐛 Troubleshooting

### No Detections Found

If all models fail to detect tables:
- Lower `--conf` threshold (try 0.15-0.20)
- Check that models are compatible with your table types
- Verify image quality and resolution
- Enable `--rotation-debug` to inspect detection attempts

### Wrong Rotation Selected

- Adjust scoring weights in code
- Increase `bottom_weight` if tables are typically at bottom
- Modify `geom_score` calculation if your tables have different aspect ratios
- Use `--rotation-debug` to see why a rotation was chosen

### Performance Issues

- Use `--device cuda:0` for GPU acceleration
- Reduce image resolution before processing
- Disable `--rotation-debug` for faster processing
- Use `--quiet` to reduce I/O overhead

## 📝 Notes

- The system prefers 0° rotation when scores are very close (within `tie_margin=0.5`)
- Multi-page TIFFs: Only the first page (index 0) is processed
- Model caching: Models are loaded once and cached for subsequent images
- The geometry score strongly influences selection (8.0x weight) to prioritize correct orientation


