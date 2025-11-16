# EasyOCR-Based Auto-Rotation Tool

An intelligent document orientation correction system that uses **EasyOCR** text detection and **Tesseract OSD** to automatically determine and correct document rotation. This tool tests all four cardinal rotations (0°, 90°, 180°, 270°) and selects the optimal orientation based on OCR quality metrics.

## ✨ Features

- **OCR-Based Rotation Detection**: Uses EasyOCR to detect and score text in all 4 rotations
- **Intelligent Scoring**: Composite scoring system considers:
  - Text detection confidence
  - Number of high-confidence detections
  - Horizontal text orientation preference
  - Word-like patterns (alphanumeric, all-caps)
  - Text length and readability
- **Tesseract OSD Integration**: Optional Tesseract OSD (Orientation and Script Detection) for additional hints
- **Contrast Enhancement**: Optional CLAHE (Contrast Limited Adaptive Histogram Equalization) for low-quality scans
- **Batch Processing**: Process entire directories with progress tracking
- **Multi-Language Support**: Supports multiple languages via EasyOCR
- **GPU Acceleration**: Optional GPU support for faster processing

## 📋 Overview

This tool is designed for processing scanned documents, technical drawings, and images where the orientation may be unknown. It:

1. Loads an image
2. Optionally enhances contrast (CLAHE) for better OCR results
3. Gets Tesseract OSD hint (optional)
4. Tests all 4 rotations (0°, 90°, 180°, 270°)
5. Runs EasyOCR on each rotated version
6. Scores each rotation using composite metrics
7. Selects the best rotation and saves the corrected image

## 🔧 Requirements

### System Dependencies

**Tesseract OCR** (required for OSD):
- **Debian/Ubuntu**: `sudo apt install tesseract-ocr libtesseract-dev`
- **macOS**: `brew install tesseract`
- **Windows**: Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)

### Python Dependencies

```bash
pip install easyocr pytesseract opencv-python-headless numpy
```

**Note**: For headless servers, use `opencv-python-headless` instead of `opencv-python`.

### GPU Support (Optional)

For GPU acceleration with EasyOCR:
- Install CUDA-enabled PyTorch
- Ensure CUDA is properly configured
- Use `--gpu` flag when running

## 🚀 Quick Start

### Command Line Usage

**Basic usage:**
```bash
python new_rotation/new_rotation.py \
  --in input_images \
  --out output_rotated
```

**With all options:**
```bash
python new_rotation/new_rotation.py \
  --in /path/to/images \
  --out /path/to/output \
  --lang en \
  --gpu \
  --osd-weight 3.0 \
  --use-osd-shortlist \
  --overwrite
```

### Python API Usage

**Process a single image:**
```python
from new_rotation.new_rotation import autorotate_and_save_improved

angle, ocr_result, output_path, meta = autorotate_and_save_improved(
    image_path="document.png",
    output_path="document_rotated.png",
    lang_list=['en'],
    gpu=False,
    overwrite=True,
    enhance_contrast=True
)

print(f"Rotated by {angle}°")
print(f"Saved to: {output_path}")
print(f"Scores: {meta['scores']}")
```

**Process a folder:**
```python
from new_rotation.new_rotation import autorotate_folder

results = autorotate_folder(
    input_dir="input_images",
    output_dir="output_rotated",
    overwrite=True,
    lang_list=['en'],
    gpu=True,
    osd_weight=3.0,
    use_osd_shortlist=False,
    enhance_contrast=True
)

for src, dst, angle, meta in results:
    print(f"{src} -> {dst} ({angle}°)")
```

## 📖 CLI Reference

```text
--in, --input_dir              Input folder containing images (default: debug_crops_3408)
--out, --output_dir            Output folder for rotated images (default: out_folder_debug_crops_3408_rotated)
--lang                         Language codes for EasyOCR, space-separated (default: ['en'])
                                Examples: --lang en fr de (for English, French, German)
--gpu                          Use GPU for EasyOCR (requires CUDA)
--osd-weight FLOAT             Weight for Tesseract OSD bias (default: 3.0)
--use-osd-shortlist            Use OSD to shortlist candidate angles (faster but may miss correct rotation)
--no-enhance-contrast           Disable contrast enhancement for low-quality images
--overwrite                     Overwrite existing output files
```

## 🎯 How It Works

### Rotation Testing Process

For each image, the system:

1. **Preprocessing** (optional):
   - Converts to grayscale
   - Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
   - Converts back to BGR for OCR

2. **Tesseract OSD** (optional):
   - Gets orientation hint from Tesseract
   - If `--use-osd-shortlist` is enabled and OSD confidence ≥ 15.0:
     - Tests only OSD angle and its 180° opposite
   - Otherwise: tests all 4 angles

3. **EasyOCR Detection**:
   - Runs EasyOCR on each rotated version
   - Uses permissive settings for difficult images:
     - `min_size=10` (detect smaller text)
     - `text_threshold=0.6` (lower threshold)
     - `low_text=0.3` (lower detection threshold)

4. **Scoring** (for each rotation):
   - Filters detections by confidence (≥0.45, with fallback to ≥0.3)
   - Calculates composite score:
     ```
     score = mean_conf * 80
           + num_detections * 15
           + total_chars * 1.5
           + horizontal_score * 30
           + word_score * 2
           + osd_bias (if applicable)
     ```

5. **OSD Bias** (if OSD available):
   - Adds weighted bonus to OSD-predicted angle
   - Adds smaller bonus to 180° opposite
   - Scale based on OSD confidence

6. **Fallback**:
   - If shortlisted and all scores poor (< 0), tests remaining angles

7. **Selection**:
   - Picks rotation with highest score
   - Saves rotated image with filename: `<stem>.rot<angle><suffix>`

### Scoring Components

#### Mean Confidence (`weight = 80`)
- Average confidence of all valid text detections
- Strongest factor in scoring

#### Number of Detections (`weight = 15`)
- Count of high-confidence text detections
- More detections suggest better orientation

#### Total Characters (`weight = 1.5`)
- Sum of characters in all detected text
- Rewards orientations with more readable text

#### Horizontal Score (`weight = 30`)
- Based on bounding box aspect ratios:
  - `aspect_ratio > 2.0`: Clearly horizontal → +conf × 3.0
  - `aspect_ratio > 1.2`: Somewhat horizontal → +conf × 1.5
  - `aspect_ratio < 0.5`: Vertical text → -conf × 2.0
- Most text should be horizontal when correctly oriented

#### Word Score (`weight = 2`)
- Bonus for word-like patterns:
  - Alphanumeric text: +conf × length × 0.5
  - All-caps words (≥3 chars): +conf × 2.0
- Rewards readable, meaningful text

#### OSD Bias (`weight = osd_weight × scale`)
- Modest bonus for Tesseract OSD prediction
- Scaled by OSD confidence (max 30.0)
- Helps when OCR is ambiguous

## 📁 Supported Formats

**Input:**
- `.png`
- `.jpg`, `.jpeg`
- `.tif`, `.tiff`
- `.bmp`
- `.webp`

**Output:**
- Same format as input
- Filename format: `<original_stem>.rot<angle><extension>`
  - Example: `document.png` → `document.rot90.png`

## 🔍 Example Output

```
============================================================
EasyOCR Auto-Rotation Tool
============================================================
Input dir:        input_images
Output dir:       output_rotated
Languages:        ['en']
GPU:              True
OSD weight:       3.0
OSD shortlist:    False
Enhance contrast: True
Overwrite:        True
============================================================

--- Processing document_001.png ---
  ✅ Saved: document_001.rot0.png  (angle=0°)
  OSD angle/conf: 0° / 12.5
  Scores: {0: 1250.5, 90: 450.2, 180: 320.1, 270: 480.3}

--- Processing document_002.png ---
  ✅ Saved: document_002.rot90.png  (angle=90°)
  OSD angle/conf: 90° / 18.2
  Scores: {0: 280.5, 90: 1890.2, 180: 150.1, 270: 320.8}

Done. Processed 2 image(s).

Summary:
  document_001.png -> document_001.rot0.png  (0°)
  document_002.png -> document_002.rot90.png  (90°)
```

## ⚙️ Configuration

### Language Selection

EasyOCR supports 80+ languages. Common examples:
- `en` - English
- `fr` - French
- `de` - German
- `es` - Spanish
- `ar` - Arabic
- `ch_sim` - Chinese (Simplified)
- `ja` - Japanese

**Multi-language:**
```bash
python new_rotation/new_rotation.py --lang en fr de --in input --out output
```

### OSD Shortlist Mode

When `--use-osd-shortlist` is enabled:
- Only tests 2 angles (OSD angle and its 180° opposite)
- **Faster** but may miss correct rotation if OSD is wrong
- Recommended for: high-quality documents where OSD is reliable
- Not recommended for: technical drawings, low-quality scans

### Contrast Enhancement

Contrast enhancement (CLAHE) is enabled by default:
- Improves OCR accuracy on low-quality scans
- Disable with `--no-enhance-contrast` if:
  - Images are already high quality
  - Processing speed is critical
  - Enhancement causes artifacts

### OSD Weight

The `--osd-weight` parameter controls how much Tesseract OSD influences scoring:
- **Low (1.0-2.0)**: OCR scoring dominates
- **Medium (3.0-5.0)**: Balanced (default: 3.0)
- **High (6.0+)**: OSD has strong influence

Adjust based on your use case:
- Technical drawings: Lower weight (OCR more reliable)
- Standard documents: Medium weight (balanced)
- When OSD is very reliable: Higher weight

## 🐛 Troubleshooting

### Poor Rotation Detection

**Symptoms**: Wrong rotation selected, or no clear winner

**Solutions**:
- Enable `--use-osd-shortlist` if OSD is reliable
- Increase `--osd-weight` if Tesseract OSD is accurate
- Ensure contrast enhancement is enabled (default)
- Check image quality (resolution, clarity)
- Try different language codes if text is non-English
- Verify EasyOCR can detect text in the images

### Slow Processing

**Solutions**:
- Use `--gpu` flag for GPU acceleration
- Enable `--use-osd-shortlist` to test fewer angles
- Disable contrast enhancement with `--no-enhance-contrast`
- Process smaller batches
- Use fewer languages in `--lang`

### No Text Detected

**Symptoms**: All rotations score poorly (< 0)

**Solutions**:
- Lower EasyOCR thresholds (modify code: `text_threshold`, `low_text`)
- Enable contrast enhancement
- Check image quality and resolution
- Verify language codes match document language
- Ensure images contain readable text (not just graphics)

### Tesseract OSD Errors

**Symptoms**: Warnings about Tesseract, OSD returns None

**Solutions**:
- Verify Tesseract is installed: `tesseract --version`
- Check Tesseract is in PATH
- Use `--use-osd-shortlist` only if OSD works reliably
- OSD is optional; tool works without it (just slower)

### GPU Not Working

**Symptoms**: `--gpu` flag has no effect or errors

**Solutions**:
- Verify CUDA is installed: `nvidia-smi`
- Install CUDA-enabled PyTorch
- Check EasyOCR GPU support: `python -c "import easyocr; print(easyocr.Reader(['en'], gpu=True))"`
- Fall back to CPU if GPU unavailable

## 📝 Notes

- **First Run**: EasyOCR downloads model weights on first use (one-time, ~100MB per language)
- **Model Caching**: EasyOCR Reader is created once per folder (not per image) for efficiency
- **File Naming**: Output files include rotation angle in filename (e.g., `.rot90.png`)
- **Overwrite**: By default, existing files are skipped; use `--overwrite` to replace
- **OSD Confidence**: Tesseract OSD confidence typically ranges 0-30; higher is better
- **Scoring**: Negative scores indicate poor text detection; positive scores indicate good detection


## 💡 Tips

1. **For Technical Drawings**: 
   - Use `--use-osd-shortlist=False` (test all angles)
   - Lower `--osd-weight` (OCR more reliable than OSD)
   - Enable contrast enhancement

2. **For Standard Documents**:
   - Use `--use-osd-shortlist=True` (faster)
   - Default `--osd-weight=3.0` (balanced)

3. **For Multi-Language Documents**:
   - Specify all languages: `--lang en fr de`
   - May be slower but more accurate

4. **For Batch Processing**:
   - Use `--gpu` for significant speedup
   - Process in smaller batches if memory constrained
   - Monitor disk space (outputs may be large)

