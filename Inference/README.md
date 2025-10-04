# YOLO Inference CLI for Documents & Images

This tool walks an input directory, renders PDF/TIFF pages to PNG, runs an Ultralytics YOLO model on each page/image, writes YOLO-format labels, and saves a **union crop** of all detections per item.

* Handles: **PDF**, **TIFF** (multi-page), **PNG/JPG/BMP**
* Outputs: PNG exports, YOLO labels (`.txt`), and cropped PNGs for detections
* Tunable via CLI flags (thresholds, device, output paths, DPI, recursion)

---

## ✨ Features

* PDF rasterization via **PyMuPDF (fitz)** with configurable DPI
* TIFF page selection and robust image loading
* Writes YOLO labels in normalized **YOLOv5/8** format
* Saves a single **union crop** around all detections per page/image
* Deterministic, progress-bar friendly, and lint/type-check friendly

---

## 📦 Requirements

Install Python packages (CPU example):

```bash
pip install ultralytics pillow pymupdf opencv-python numpy tqdm
# For servers/CI without GUI libs:
# pip install opencv-python-headless
```

> GPU: Ultralytics will use CUDA if available (`--device cuda:0`). Otherwise it falls back to CPU.

---

## 🚀 Quick Start

```bash
python Inference/inference_yolo.py \
  --input-dir /path/to/data \
  --model-path /path/to/best.pt \
  --output-sub crops \
  --labels-sub labels \
  --png-export-sub pngs \
  --conf-thresh 0.25 \
  --recursive 1
```

This will:

* Export all images/PDF pages found under `--input-dir` to `pngs/`
* Run YOLO
* Write YOLO label files to `labels/`
* Save union crops to `crops/`

All three output folders are created **inside** `--input-dir`.

---

## 🧭 CLI Reference

```text
--input-dir                       Directory containing input files. (default: /code/Datasets/foolad/Data_unzipped/Data/3408)
--output-sub                      Subdir (inside input-dir) for crops. (default: 3408_crop_2)
--model-path                      Path to YOLO model weights (.pt).

--extensions                      Glob patterns to scan (space-separated).
                                  (default: *.png *.jpg *.jpeg *.bmp *.tif *.tiff *.pdf)
--recursive [0|1]                 Recurse into subdirectories. (default: 0)

--conf-thresh FLOAT               Confidence threshold. (default: 0.25)
--device STR                      YOLO device, e.g. "cuda:0" or "cpu". (default: auto)
--decimals INT                    Decimals in YOLO labels. (default: 10)
--write-empty-label [0|1]         Write empty .txt when no detections. (default: 1)

--tiff-page INT                   Page index for multi-page TIFFs. (default: 0)
--pdf-dpi INT                     DPI for PDF rasterization. (default: 200)

--png-export-sub                  Subdir (inside input-dir) for exported PNGs. (default: png_export_2)
--png-compression INT             PNG compression (0..9). (default: 3)
--labels-sub                      Subdir (inside input-dir) for YOLO labels. (default: labels_2)

--keep-input-extension-in-names [0|1]
                                  If 1, keep original input extension in base name. (default: 0)
```

---

## 🗂️ Output Layout

Given `--input-dir /data`, `--output-sub crops`, `--labels-sub labels`, `--png-export-sub pngs`:

```
/data/
  pngs/      # Exported PNGs for every page/image processed
    doc_p000.png
    doc_p001.png
    photo.png
  labels/    # YOLO labels (normalized xc yc w h)
    doc_p000.txt
    doc_p001.txt
    photo.txt
  crops/     # Union crop per item when detections exist
    doc_p000_crop.png
    photo_crop.png
```

**YOLO label format** (per line):

```
<class_id> <xc> <yc> <w> <h>   # all normalized to [0,1]
```

---

## 📄 PDF & TIFF Notes

* **PDF**: Each page is rasterized at `--pdf-dpi` and processed independently.
* **TIFF**: Multi-page TIFFs select `--tiff-page` (0-based). If the TIFF has fewer pages, the last valid page is used.

---

## 🧪 Examples

**Process PDFs only, non-recursive:**

```bash
python Inference/inference_yolo.py \
  --input-dir /data \
  --extensions "*.pdf" \
  --model-path weights/best.pt
```

**Force CPU, higher threshold, keep file extensions in names:**

```bash
python Inference/inference_yolo.py \
  --input-dir /data \
  --device cpu \
  --conf-thresh 0.5 \
  --keep-input-extension-in-names 1 \
  --model-path weights/best.pt
```

**Deep directories, higher DPI for better small-text detection:**

```bash
python Inference/inference_yolo.py \
  --input-dir /big_corpus \
  --recursive 1 \
  --pdf-dpi 300 \
  --model-path weights/best.pt
```

---

## ⚙️ Performance Tips

* Increase `--pdf-dpi` (e.g., 300) to improve small-object detection on PDFs (at the cost of speed/memory).
* Use `--device cuda:0` on a GPU machine for large batches.
* If running in CI or headless servers, prefer `opencv-python-headless` to reduce dependency size.

---


