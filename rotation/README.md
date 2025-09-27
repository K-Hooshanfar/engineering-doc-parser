# 📄 Document Orientation Detector

A simple Python script that uses the EAST text detector and Tesseract OCR’s OSD to automatically detect and correct the rotation of scanned or photographed documents. The script saves the corrected image into a `result/` directory with the original filename plus a `_rotated` suffix.

---

## 🔧 Prerequisites

- **Python**: Tested with Python 3.10  
- **Ubuntu** (or any Debian-based Linux)

### Install system dependencies

```bash
sudo apt update
sudo apt install -y tesseract-ocr libtesseract-dev
````

### Install Python packages

```bash
pip install pytesseract opencv-python numpy
```

> Make sure you’re installing into your Python 3.10 environment (e.g. `pip3` or a venv).

## 📂 Project Structure

```plaintext
├── config.py                            # Configuration with EAST model path
├── rotation.py                          # Main orientation-correction script
└── weights/
    └── frozen_east_text_detection.pb    # Downloaded EAST model
```

## 📥 Download the EAST Model

Create the weights folder and place the pretrained EAST text detector there:

```bash
mkdir -p weights
```

**Download** the model: [Download](https://drive.google.com/file/d/10FgAQ31uSy_sjGl828ct1oVXeVUvpYe9/view?usp=drive_link)

*(Save it as `weights/frozen_east_text_detection.pb`.)*

## 🚀 How to Run

1. **Edit** the top of `rotation.py` if your paths differ:

   ```python
   IMAGE_PATH      = "test_images/1.png"
   EAST_MODEL_PATH = Config.EAST_TEXT_DETECTOR
   RESULT_DIR      = "result"
   ```
2. **Ensure** `config.py` contains:

   ```python
   class Config:
       EAST_TEXT_DETECTOR = 'weights/frozen_east_text_detection.pb'
   ```
3. **Run** the script:

   ```bash
   python3.10 rotation.py
   ```
4. **View** the result in the `result/` folder:

   ```
   └── test_images/1_rotated.png   ← the auto-rotated output
   ```

If the script finds the image already upright, it will print **No orientation needed** and won’t write any file.

## 🤖 How Orientation Is Determined

1. **Tesseract OSD**
   The script first asks Tesseract’s Orientation and Script Detection for an angle in `{0, 90, 180, 270}`. Any non-zero suggestion is immediately applied.
2. **EAST-based Heuristic**
   If OSD returns `0°`, the script falls back to:

   * Detect text boxes via the EAST model
   * Count “wide” (horizontal) vs. “tall” (vertical) boxes
   * If vertical boxes dominate, rotate 90° CW
3. **Upside-down Check**
   After that, an OCR-ratio sanity check compares the amount of readable text before/after a 180° flip—if flipping yields markedly better OCR results, a 180° rotation is applied.

---

