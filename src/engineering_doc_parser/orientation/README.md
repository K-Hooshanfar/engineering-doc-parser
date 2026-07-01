# Document Orientation Detector

A Python CLI that uses **Tesseract OSD** for primary orientation detection and an **EAST text-detector–based heuristic** as a fallback to auto-rotate scanned/photographed documents. Outputs are written to a chosen folder; rotated files keep the original name with a `_rotated` suffix. 

---

## ✨ What’s in this version

* **Two-stage orientation logic**

  1. Try **Tesseract OSD** (0/90/180/270).
  2. If OSD suggests 0°, use **EAST** boxes to compare horizontal vs. vertical text lines; if vertical dominates, rotate 90° CW.
     *(Note: there’s no extra “OCR ratio” upside-down check in this code.)* 
* **Single file or batch modes** (`--input-dir`, `--recursive`, `--exts`). 
* **Force a fixed angle** (`--force-angle`) or **disable OSD** (`--no-osd`). 
* **Configurable thresholds** (`--conf-thresh`, `--nms-thresh`, `--ratio-thresh`). 
* **Optionally save unchanged images** (`--save-unchanged`). 
* **Solid logging controls** (verbosity, log level/file). 

---

## 🔧 Prerequisites

* **Python**: 3.10 recommended
* **Tesseract** OCR installed and on your PATH

Install on Debian/Ubuntu:

```bash
sudo apt update
sudo apt install -y tesseract-ocr libtesseract-dev
```

Python packages:

```bash
pip install opencv-python numpy pytesseract
```

---

## 📂 Project layout

```plaintext
├── config.py                         # Holds EAST model path
├── rotation.py                       # CLI + orientation logic
└── weights/
    └── frozen_east_text_detection.pb # EAST model (.pb)
```

Create the weights folder and download the **EAST** model:

```bash
mkdir -p weights
gdown --id 10FgAQ31uSy_sjGl828ct1oVXeVUvpYe9
```

Then point `Config.EAST_TEXT_DETECTOR` at it (see below). 

---

## ⚙️ Configuration

`config.py` must define the EAST model location:

```python
class Config:
    EAST_TEXT_DETECTOR = "weights/frozen_east_text_detection.pb"
```

The CLI defaults to using this path unless you pass `-m/--east-model`. 

---

## 🚀 Usage

### Single image

```bash
python rotation.py \
  -i test_images/3.png \
  -m weights/frozen_east_text_detection.pb \
  -o result
```

### Non-recursive folder

```bash
python rotation.py \
  --input-dir scans/ \
  -m weights/frozen_east_text_detection.pb \
  -o result
```

### Recursive folder + custom extensions + save unchanged

```bash
python rotation.py \
  --input-dir scans/ --recursive \
  --exts png jpg jpeg tif tiff bmp webp \
  --save-unchanged \
  -m weights/frozen_east_text_detection.pb \
  -o result
```

### Logging examples

```bash
# short flag
python rotation.py -i doc.png -m weights/east.pb -o out -v

# explicit level + file
python rotation.py --input-dir scans -m weights/east.pb -o out \
  --log-level INFO --log-file run.log
```

All options above are defined by the CLI parser and shown in the header examples of `rotation.py`. 

---

## 🧠 How it decides the angle

1. **Tesseract OSD**: If it returns 90/180/270, that rotation is applied directly.
2. **EAST fallback**: If OSD returns 0, the code counts EAST boxes: “wide” vs “tall.” If **vertical > horizontal**, it rotates **90° clockwise**; otherwise leaves the image as is. 

---

## 🗃️ Output naming

* Rotated files: `<name>_rotated<ext>` in `--result-dir`.
* If `--save-unchanged` is used, even upright images are copied using the original name. 

---

## ✅ Testing

A `pytest` suite covers key paths (OSD parse, EAST fallback, batch/single flows, save naming). It mocks heavy dependencies for fast, deterministic runs.

Run tests (from the project root):

```bash
pytest -q
# or target the suite directly
pytest -q rotation/test/test_rotation.py
```

The test file stubs `config.Config` and `pytesseract.image_to_osd`, verifies `_rotated` naming, and exercises both single-image and directory flows. 

---

## 🔍 Troubleshooting

* **EAST model not found** → The program raises a `FileNotFoundError` if `-m/--east-model` doesn’t exist or `Config.EAST_TEXT_DETECTOR` is wrong. 
* **Image not found / unreadable** → You’ll see warnings and the file is skipped. 
* **Tesseract not installed** → OSD will fail; consider `--no-osd` to rely purely on EAST fallback, or install Tesseract. 

---

## 📝 Notes

* Accepted extensions for batch mode default to: `png jpg jpeg tif tiff bmp webp` (customizable via `--exts`). 
* You can **force** a specific rotation using `--force-angle {0,90,180,270}`; it overrides OSD and fallback. 
* Verbosity: `-v` sets DEBUG; or use `--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}` and optionally `--log-file`. 

---
