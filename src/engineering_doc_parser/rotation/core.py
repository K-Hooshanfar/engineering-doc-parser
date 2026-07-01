# --- deps (pip install these first) ---
# pip install easyocr pytesseract opencv-python-headless numpy

import argparse
import re
from pathlib import Path

import cv2
import easyocr
import numpy as np
import pytesseract


# ---------- utils ----------
def rotate_image(img, angle):
    if angle == 0:
        return img
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError("angle must be one of {0, 90, 180, 270}")


def parse_osd(osd_text):
    angle = None
    conf = None
    m = re.search(r"(Rotate|Orientation in degrees):\s*([0-9]+)", osd_text)
    if m:
        angle = int(m.group(2)) % 360
    m = re.search(r"Orientation confidence:\s*([0-9.]+)", osd_text)
    if m:
        conf = float(m.group(1))
    return angle, conf


def tesseract_osd_angle_conf(img):
    try:
        osd = pytesseract.image_to_osd(img)
        return parse_osd(osd)
    except Exception:
        return None, None


def get_bbox_dimensions(bbox):
    """Calculate width and height of bounding box"""
    points = np.array(bbox)
    width = np.linalg.norm(points[1] - points[0])
    height = np.linalg.norm(points[2] - points[1])
    return width, height


def calculate_orientation_score(result, conf_th=0.45):
    """
    Enhanced scoring that considers:
    - Text confidence
    - Number of high-confidence detections
    - Horizontal orientation preference (width > height for most text)
    - Text length
    - Word-like patterns
    """
    if not result:
        return -1e9

    # Filter for high-confidence, meaningful text
    valid_items = []
    low_conf_items = []

    for bbox, text, conf in result:
        if not text or not text.strip():
            continue
        text_clean = text.strip()
        if len(text_clean) < 2:
            continue

        if conf >= conf_th:
            valid_items.append((bbox, text_clean, conf))
        elif conf >= 0.3:  # Track lower confidence items
            low_conf_items.append((bbox, text_clean, conf))

    # If we have very few high-confidence items, include some low-conf ones
    if len(valid_items) < 3 and low_conf_items:
        valid_items.extend(low_conf_items[:5])

    if not valid_items:
        return -1e9

    # Calculate various metrics
    total_conf = sum(c for _, _, c in valid_items)
    mean_conf = total_conf / len(valid_items)
    num_detections = len(valid_items)
    total_chars = sum(len(t) for _, t, _ in valid_items)

    # Calculate horizontal preference score with more weight
    horizontal_score = 0
    word_score = 0

    for bbox, text, conf in valid_items:
        w, h = get_bbox_dimensions(bbox)
        aspect_ratio = w / (h + 1e-6)

        # Horizontal text detection
        if aspect_ratio > 2.0:  # Clearly horizontal
            horizontal_score += conf * 3.0
        elif aspect_ratio > 1.2:  # Somewhat horizontal
            horizontal_score += conf * 1.5
        elif aspect_ratio < 0.5:  # Vertical text
            horizontal_score -= conf * 2.0

        # Bonus for word-like patterns (letters, numbers, common words)
        if len(text) >= 3:
            # Check for readable words/numbers
            if text.replace(" ", "").isalnum():
                word_score += conf * len(text) * 0.5
            # Bonus for all-caps words (common in technical drawings)
            if text.isupper() and len(text) >= 3:
                word_score += conf * 2.0

    # Weight different factors with more emphasis on orientation
    score = (
        mean_conf * 80  # Mean confidence
        + num_detections * 15  # Number of detections (increased)
        + total_chars * 1.5  # Total characters
        + horizontal_score * 30  # Horizontal preference (increased)
        + word_score * 2  # Word-like bonus
    )

    return score


def predict_rotation_improved(
    image_path,
    lang_list=["en"],
    gpu=False,
    osd_weight=3.0,  # Further reduced
    use_osd_shortlist=False,
    enhance_contrast=True,  # New option for low-quality images
):
    """
    Improved rotation prediction with better scoring and preprocessing
    """
    reader = easyocr.Reader(lang_list, gpu=gpu)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Optional: Enhance contrast for low-quality scans
    if enhance_contrast:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        # Convert back to BGR for consistency
        img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    osd_angle, osd_conf = tesseract_osd_angle_conf(img)

    # For technical drawings, test all angles
    all_angles = [0, 90, 180, 270]

    if use_osd_shortlist and osd_angle is not None and (osd_conf or 0) >= 15.0:
        candidates = sorted({osd_angle, (osd_angle + 180) % 360})
    else:
        candidates = all_angles

    best = (None, float("-inf"), None, None)
    results = {}

    for a in candidates:
        rot = rotate_image(img, a)
        # Use more permissive EasyOCR settings for difficult images
        r = reader.readtext(
            rot,
            detail=1,
            paragraph=False,
            min_size=10,  # Detect smaller text
            text_threshold=0.6,  # Lower threshold
            low_text=0.3,  # Lower text detection threshold
        )
        s = calculate_orientation_score(r)

        # Add very modest bias toward OSD prediction if available
        if osd_angle is not None and osd_conf is not None and osd_conf > 3:
            scale = min(osd_conf / 30.0, 1.0)  # Even more conservative
            if a == osd_angle:
                s += osd_weight * scale
            elif a == (osd_angle + 180) % 360:
                s += 0.2 * osd_weight * scale

        results[a] = (s, r, rot)
        if s > best[1]:
            best = (a, s, r, rot)

    # Fallback: if shortlisted and all scores poor, try all angles
    if len(candidates) == 2 and best[1] < 0:
        for a in all_angles:
            if a in results:
                continue
            rot = rotate_image(img, a)
            r = reader.readtext(rot, detail=1, paragraph=False)
            s = calculate_orientation_score(r)
            results[a] = (s, r, rot)
            if s > best[1]:
                best = (a, s, r, rot)

    best_angle, _, best_result, best_img = best

    # Enhanced metadata for debugging
    meta = {
        "osd_angle": osd_angle,
        "osd_conf": osd_conf,
        "scores": {k: round(v[0], 2) for k, v in results.items()},
        "num_detections": {
            k: len([x for x in v[1] if x[2] >= 0.45]) for k, v in results.items()
        },
        "top_texts": {
            k: [x[1][:30] for x in v[1][:3] if x[2] >= 0.45] for k, v in results.items()
        },
    }
    return best_angle, best_result, best_img, meta


def autorotate_and_save_improved(
    image_path, output_path=None, lang_list=["en"], gpu=False, overwrite=False, **kwargs
):
    """Save auto-rotated image with improved detection"""
    angle, ocr_result, rotated_img, meta = predict_rotation_improved(
        image_path, lang_list=lang_list, gpu=gpu, **kwargs
    )

    if output_path is None:
        p = Path(image_path)
        output_path = p.with_name(f"{p.stem}.rot{angle}{p.suffix}")

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} exists. Pass overwrite=True to replace it."
        )

    if not cv2.imwrite(str(output_path), rotated_img):
        raise IOError(f"Failed to write image to: {output_path}")

    return angle, ocr_result, str(output_path), meta


# ---------- folder helper ----------
def autorotate_folder(
    input_dir,
    output_dir,
    overwrite=False,
    lang_list=["en"],
    gpu=False,
    osd_weight=3.0,
    use_osd_shortlist=False,
    enhance_contrast=True,
):
    """
    Run auto-rotation on all images in input_dir and save to output_dir.
    Filenames become: <stem>.rot<angle><suffix>
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # File extensions to process
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

    processed = []

    for img_path in sorted(input_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in exts:
            continue

        print(f"\n--- Processing {img_path.name} ---")

        try:
            angle, ocr_result, rotated_img, meta = predict_rotation_improved(
                str(img_path),
                lang_list=lang_list,
                gpu=gpu,
                osd_weight=osd_weight,
                use_osd_shortlist=use_osd_shortlist,
                enhance_contrast=enhance_contrast,
            )

            out_path = output_dir / f"{img_path.stem}.rot{angle}{img_path.suffix}"

            if out_path.exists() and not overwrite:
                print(f"  ⚠️ {out_path.name} already exists, skipping (overwrite=False)")
                continue

            if not cv2.imwrite(str(out_path), rotated_img):
                raise IOError(f"Failed to write image to: {out_path}")

            processed.append((str(img_path), str(out_path), angle, meta))

            print(f"  ✅ Saved: {out_path.name}  (angle={angle}°)")
            print(f"  OSD angle/conf: {meta['osd_angle']}° / {meta['osd_conf']}")
            print(f"  Scores: {meta['scores']}")

        except Exception as e:
            print(f"  ❌ Error processing {img_path.name}: {e}")

    print(f"\nDone. Processed {len(processed)} image(s).")
    return processed


# -------- CLI --------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="EasyOCR-based auto-rotation: tests all rotations and picks the best using OCR scoring."
    )
    ap.add_argument(
        "--in",
        dest="input_dir",
        required=False,
        default="debug_crops_3408",
        help="Input folder containing images.",
    )
    ap.add_argument(
        "--out",
        dest="output_dir",
        required=False,
        default="out_folder_debug_crops_3408_rotated",
        help="Output folder for rotated images.",
    )
    ap.add_argument(
        "--lang",
        nargs="+",
        default=["en"],
        help="Language codes for EasyOCR (default: ['en']).",
    )
    ap.add_argument(
        "--gpu", action="store_true", help="Use GPU for EasyOCR (requires CUDA)."
    )
    ap.add_argument(
        "--osd-weight",
        type=float,
        default=3.0,
        help="Weight for Tesseract OSD bias (default: 3.0).",
    )
    ap.add_argument(
        "--use-osd-shortlist",
        action="store_true",
        help="Use OSD to shortlist candidate angles (faster but may miss correct rotation).",
    )
    ap.add_argument(
        "--no-enhance-contrast",
        dest="enhance_contrast",
        action="store_false",
        help="Disable contrast enhancement for low-quality images.",
    )
    ap.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output files."
    )
    return ap.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("EasyOCR Auto-Rotation Tool")
    print("=" * 60)
    print(f"Input dir:        {args.input_dir}")
    print(f"Output dir:       {args.output_dir}")
    print(f"Languages:        {args.lang}")
    print(f"GPU:              {args.gpu}")
    print(f"OSD weight:       {args.osd_weight}")
    print(f"OSD shortlist:    {args.use_osd_shortlist}")
    print(f"Enhance contrast: {args.enhance_contrast}")
    print(f"Overwrite:        {args.overwrite}")
    print("=" * 60 + "\n")

    results = autorotate_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        lang_list=args.lang,
        gpu=args.gpu,
        osd_weight=args.osd_weight,
        use_osd_shortlist=args.use_osd_shortlist,
        enhance_contrast=args.enhance_contrast,
    )

    # Optional: quick summary
    print("\nSummary:")
    for src, dst, angle, meta in results:
        print(f"  {Path(src).name} -> {Path(dst).name}  ({angle}°)")


if __name__ == "__main__":
    main()
