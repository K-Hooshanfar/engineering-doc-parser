"""
YOLO inference CLI for documents and images (lint-cleaned).

This module:
  • Walks an input directory (optionally recursively)
  • Renders PDFs/TIFF pages to PNG
  • Runs an Ultralytics YOLO model on each page/image
  • Writes YOLO-format labels (.txt) and saves a union crop per item
  • Exposes a CLI via argparse with configurable thresholds and paths

Notes:
    - Requires: ultralytics, pillow, pymupdf (fitz), opencv-python, numpy, tqdm.
    - Maintains original public entry points: parse_args(), main().
    - Pylint: cv2 is a C-extension providing dynamic attributes; we suppress
      false-positive E1101 (no-member) for cv2 only.
"""

from __future__ import annotations

# Pylint/OpenCV: cv2 is a C-extension with dynamic members.
# This prevents false-positive E1101 for cv2.* symbols only.
# pylint: disable=no-member
import argparse
import glob
import os
from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple, TypeAlias, cast

import cv2  # type: ignore[attr-defined]
import fitz  # type: ignore[import-not-found]
import numpy as np
from PIL import Image as PILImage  # type: ignore[import-not-found]
from tqdm import tqdm  # type: ignore[import-untyped]
from ultralytics import YOLO  # type: ignore[import-untyped]

# =========================
# Argument parsing
# =========================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the YOLO inference pipeline.

    Returns:
        argparse.Namespace: Parsed flags including paths, thresholds, and toggles.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Export PDF pages/images to PNG, run YOLO, save YOLO labels and union crops."
        )
    )

    # Core paths
    parser.add_argument(
        "--input-dir",
        default="/code/Datasets/foolad/Data_unzipped/Data/3408",
        help="Directory containing input files (images/PDFs).",
    )
    parser.add_argument(
        "--output-sub",
        default="3408_crop_2",
        help="Subdirectory (inside input-dir) to save crops.",
    )
    parser.add_argument(
        "--model-path",
        default="/code/Datasets/foolad/table_detect_ocr/runs_3407_3408/exp1/weights/best.pt",
        help="Path to YOLO model weights.",
    )

    # File discovery
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.pdf"],
        help="Glob patterns of files to process (space-separated).",
    )
    parser.add_argument(
        "--recursive",
        type=int,
        choices=[0, 1],
        default=0,
        help="If 1, search recursively under input-dir.",
    )

    # Detection / thresholds
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.25,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device for YOLO (e.g., 'cuda:0' or 'cpu'). Default: auto",
    )
    parser.add_argument(
        "--decimals", type=int, default=10, help="Decimal places in YOLO txt labels."
    )
    parser.add_argument(
        "--write-empty-label",
        type=int,
        choices=[0, 1],
        default=1,
        help="If 1, write empty .txt when no detections.",
    )

    # TIFF & PDF rendering
    parser.add_argument(
        "--tiff-page",
        type=int,
        default=0,
        help="For multi-page TIFFs, which page index to read.",
    )
    parser.add_argument(
        "--pdf-dpi", type=int, default=200, help="DPI used when rasterizing PDF pages."
    )

    # Export layout
    parser.add_argument(
        "--png-export-sub",
        default="png_export_2",
        help="Subdirectory (inside input-dir) to save exported PNGs.",
    )
    parser.add_argument(
        "--png-compression",
        type=int,
        default=3,
        help="PNG compression level (0..9; lower=faster/larger).",
    )
    parser.add_argument(
        "--labels-sub",
        default="labels_2",
        help="Subdirectory (inside input-dir) to save YOLO label .txt files.",
    )

    # Naming
    parser.add_argument(
        "--keep-input-extension-in-names",
        type=int,
        choices=[0, 1],
        default=0,
        help="If 1, keep original input extension in base name.",
    )

    return parser.parse_args()


# =========================
# Helpers (TIFF & general)
# =========================


def _to_bgr8(arr: np.ndarray) -> np.ndarray:
    """Convert an arbitrary numpy array to a uint8 3-channel BGR image.

    Handles uint16, float, grayscale (2D), RGBA, and RGB inputs, normalizing and
    converting to BGR as needed for OpenCV.

    Args:
        arr (np.ndarray): Input image-like array.

    Returns:
        np.ndarray: BGR image of shape (H, W, 3) with dtype uint8.

    Raises:
        ValueError: If the input shape is unsupported.
    """
    if arr.dtype == np.uint16:
        arr = cv2.convertScaleAbs(arr, alpha=1.0 / 256.0)
    elif np.issubdtype(arr.dtype, np.floating):
        maxv = float(arr.max()) if arr.size else 1.0
        if maxv <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            mn, mx = float(arr.min()), float(arr.max())
            scale = 255.0 / (mx - mn) if mx > mn else 1.0
            arr = ((arr - mn) * scale).clip(0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = cv2.convertScaleAbs(arr)

    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3:
        if arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        else:
            arr = arr[:, :, :3]
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError("Unsupported image shape for conversion.")
    return arr


def load_image_any(path: str, page_index: int = 0) -> np.ndarray:
    """Load an image file (PNG/JPG/BMP/TIFF) into BGR uint8; handle multi-page TIFF.

    Args:
        path (str): Path to the image file.
        page_index (int): Page to read for multipage TIFFs.

    Returns:
        np.ndarray: BGR image array (H, W, 3), dtype uint8.

    Raises:
        FileNotFoundError: If OpenCV cannot load a non-TIFF image.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        # mypy: ensure the object is typed as PIL.Image.Image
        im = cast(PILImage.Image, PILImage.open(path))
        n_frames = int(getattr(im, "n_frames", 1))  # not all formats expose this
        if n_frames > 1:
            page_index = max(0, min(page_index, n_frames - 1))
            im.seek(page_index)
        if im.mode not in ("RGB", "RGBA", "L"):
            try:
                im = im.convert("RGB")
            except Exception:
                im = im.convert("L")
        arr = np.array(im)  # RGB/RGBA/L
        return _to_bgr8(arr)

    # Non-TIFF: use OpenCV. Keep behavior consistent with the rest of the pipeline.
    arr2 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr2 is None:
        raise FileNotFoundError(f"Could not load image from {path}")
    if arr2.ndim == 3 and arr2.shape[2] == 4:
        arr2 = cv2.cvtColor(arr2, cv2.COLOR_BGRA2BGR)
    if arr2.ndim == 3 and arr2.shape[2] == 3:
        # OpenCV loads BGR; _to_bgr8 expects RGB/RGBA/L input, so flip to RGB first
        arr2 = arr2[:, :, ::-1]
    return _to_bgr8(arr2)


# =========================
# YOLO label writer
# =========================


def save_yolo_labels(
    txt_path: str,
    boxes_xyxy: np.ndarray,
    classes: np.ndarray,
    img_w: int,
    img_h: int,
    digits: int = 6,
) -> None:
    """Write YOLO-format labels (.txt) from absolute XYXY detections.

    Each line has: ``class xc yc w h`` in normalized coordinates.

    Args:
        txt_path (str): Destination .txt path.
        boxes_xyxy (np.ndarray): Array of shape (N, 4) with absolute [x1,y1,x2,y2].
        classes (np.ndarray): Array of shape (N,) with integer class ids.
        img_w (int): Image width in pixels.
        img_h (int): Image height in pixels.
        digits (int): Decimal precision for floats.
    """
    lines = []
    for (x1, y1, x2, y2), cls in zip(boxes_xyxy, classes):
        x1 = np.clip(x1, 0, img_w - 1)
        x2 = np.clip(x2, 0, img_w - 1)
        y1 = np.clip(y1, 0, img_h - 1)
        y2 = np.clip(y2, 0, img_h - 1)
        if x2 <= x1 or y2 <= y1:
            continue
        xc = ((x1 + x2) / 2.0) / img_w
        yc = ((y1 + y2) / 2.0) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        xc, yc, w, h = [float(np.clip(v, 0.0, 1.0)) for v in (xc, yc, w, h)]
        lines.append(
            f"{int(cls)} {xc:.{digits}f} {yc:.{digits}f} {w:.{digits}f} {h:.{digits}f}"
        )
    os.makedirs(os.path.dirname(txt_path) or ".", exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =========================
# Naming control
# =========================


def make_name_from_input(
    basename: str, keep_ext: bool, page_index: Optional[int] = None
) -> str:
    """Build a base output name from an input filename.

    Optionally keeps the original extension and appends a page suffix for
    multi-page sources.

    Args:
        basename (str): Input filename (e.g., "photo.jpg").
        keep_ext (bool): If True, keep the original extension in the base name.
        page_index (Optional[int]): Page index to append as ``_p###``.

    Returns:
        str: Base name suitable for related outputs.
    """
    stem, _ext = os.path.splitext(basename)
    base_part = basename if keep_ext else stem
    if page_index is not None:
        return f"{base_part}_p{page_index:03d}"
    return base_part


# =========================
# Export PNG
# =========================


def export_png(
    image_bgr: np.ndarray, name_base: str, png_export_dir: str, png_compression: int
) -> str:
    """Write an image to PNG with the given base name in a target directory.

    Args:
        image_bgr (np.ndarray): Image array (BGR uint8).
        name_base (str): Base filename without extension.
        png_export_dir (str): Destination directory for PNGs.
        png_compression (int): OpenCV PNG compression level (0..9).

    Returns:
        str: Absolute path of the written PNG.

    Raises:
        RuntimeError: If the PNG cannot be written.
    """
    out_path = os.path.join(png_export_dir, f"{name_base}.png")
    ok = cv2.imwrite(
        out_path, image_bgr, [cv2.IMWRITE_PNG_COMPRESSION, int(png_compression)]
    )
    if not ok:
        raise RuntimeError(f"Failed to write PNG to {out_path}")
    return out_path


# =========================
# Detection helpers
# =========================

# Result typing that plays nicely with type-checkers
DetectOk = tuple[Literal[True], np.ndarray]
DetectErr = tuple[Literal[False], str]
DetectOutcome: TypeAlias = DetectOk | DetectErr


@dataclass(slots=True)
class DetectConfig:
    model: YOLO
    labels_dir: str
    conf_thresh: float
    digits: int
    write_empty_label: bool
    device: Optional[str] = None


def _predict(image_bgr: np.ndarray, cfg: DetectConfig):
    """Run the YOLO model with optional device override."""
    if cfg.device is None:
        return cfg.model(image_bgr)
    return cfg.model(image_bgr, device=cfg.device)


def _filtered_boxes(results, conf_thresh: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (boxes_xyxy, classes) filtered by confidence, or (0,0)-shaped arrays."""
    if not results or results[0].boxes is None or results[0].boxes.xyxy is None:
        return np.empty((0, 4)), np.empty((0,), dtype=int)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    clss = results[0].boxes.cls.cpu().numpy().astype(int)
    mask = confs >= conf_thresh
    fboxes = boxes[mask] if len(boxes) else np.empty((0, 4))
    fclss = clss[mask] if len(clss) else np.empty((0,), dtype=int)
    return fboxes, fclss


def process_image_like(
    image: np.ndarray,
    name_base: str,
    *,
    model: YOLO,
    labels_dir: str,
    conf_thresh: float,
    digits: int,
    write_empty_label: bool,
) -> DetectOutcome:
    """Run detection on an image, write labels, and compute the union crop.

    This retains the original signature for compatibility with any external
    code/tests while delegating to the refactored implementation.

    Returns:
        tuple[bool, np.ndarray|str]: ``(True, crop_ndarray)`` or ``(False, reason)``.
    """
    cfg = DetectConfig(
        model=model,
        labels_dir=labels_dir,
        conf_thresh=conf_thresh,
        digits=digits,
        write_empty_label=write_empty_label,
    )
    return _process_image_like_refactored(image, name_base, cfg)


def _process_image_like_refactored(
    image: np.ndarray,
    name_base: str,
    cfg: DetectConfig,
) -> DetectOutcome:
    height, width = image.shape[:2]
    results = _predict(image, cfg)
    txt_out = os.path.join(cfg.labels_dir, name_base + ".txt")

    boxes_xyxy, classes = _filtered_boxes(results, cfg.conf_thresh)

    if boxes_xyxy.shape[0] == 0:
        if cfg.write_empty_label:
            save_yolo_labels(
                txt_out,
                np.empty((0, 4)),
                np.empty((0,)),
                width,
                height,
                digits=cfg.digits,
            )
        return False, "no detections >= thresh"

    save_yolo_labels(txt_out, boxes_xyxy, classes, width, height, digits=cfg.digits)

    # Union crop
    x1 = int(np.min(boxes_xyxy[:, 0]))
    y1 = int(np.min(boxes_xyxy[:, 1]))
    x2 = int(np.max(boxes_xyxy[:, 2]))
    y2 = int(np.max(boxes_xyxy[:, 3]))
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, width), min(y2, height)
    if x2 <= x1 or y2 <= y1:
        return False, "empty crop after clamp"

    cropped = image[y1:y2, x1:x2]
    return True, cropped


# =========================
# Main runner
# =========================
def _ci_pattern(p: str) -> str:
    """Return a case-insensitive glob pattern by expanding letters to [aA]."""
    out = []
    in_class = False  # don't rewrite inside [...] classes
    for ch in p:
        if ch == "[":
            in_class = True
            out.append(ch)
        elif ch == "]":
            in_class = False
            out.append(ch)
        elif not in_class and ch.isalpha():
            out.append(f"[{ch.lower()}{ch.upper()}]")
        else:
            out.append(ch)
    return "".join(out)


def gather_paths(
    input_dir: str, extensions: Sequence[str], recursive: bool
) -> list[str]:
    """Collect file paths from a directory according to patterns (case-insensitive)."""
    paths: list[str] = []
    for patt in extensions:
        patt_ci = _ci_pattern(patt)
        if recursive:
            patt_ci = (
                patt_ci if patt_ci.startswith("**/") else os.path.join("**", patt_ci)
            )
            paths.extend(glob.glob(os.path.join(input_dir, patt_ci), recursive=True))
        else:
            paths.extend(glob.glob(os.path.join(input_dir, patt_ci)))
    # de-dup while preserving order
    return sorted(list(dict.fromkeys(paths)))


def _handle_pdf(
    path: str,
    base: str,
    name_keep_ext: bool,
    pdf_dpi: int,
    png_export_dir: str,
    png_compression: int,
    crops_dir: str,
    cfg: DetectConfig,
    progress_parent: tqdm,
) -> int:
    """Process a single PDF, returning number of saved crops."""
    saved = 0
    with fitz.open(path) as doc:
        page_count = doc.page_count
        mat = fitz.Matrix(pdf_dpi / 72.0, pdf_dpi / 72.0)
        for i in tqdm(
            range(page_count),
            desc=f"{base} pages",
            unit="page",
            leave=False,
            dynamic_ncols=True,
        ):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            name_base = make_name_from_input(base, name_keep_ext, page_index=i)
            exported_path = export_png(bgr, name_base, png_export_dir, png_compression)
            tqdm.write(f"🖼️  Exported PNG: {exported_path}")

            res = _process_image_like_refactored(bgr, name_base, cfg)
            if not res[0]:
                tqdm.write(f"❌ {base} page {i}: {res[1]}")
                continue

            out = cast(np.ndarray, res[1])
            out_path = os.path.join(crops_dir, f"{name_base}_crop.png")
            ok_write = cv2.imwrite(out_path, out, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            if ok_write:
                saved += 1
                tqdm.write(f"✅ {base} page {i}: {out_path}")
            else:
                tqdm.write(f"❌ {base} page {i}: failed to write {out_path}")
    return saved


def _handle_raster_image(
    path: str,
    base: str,
    tiff_page: int,
    name_keep_ext: bool,
    png_export_dir: str,
    png_compression: int,
    crops_dir: str,
    cfg: DetectConfig,
) -> int:
    """Process a single raster image/TIFF. Returns crops saved count."""
    img = load_image_any(path, page_index=tiff_page)

    name_base = make_name_from_input(base, name_keep_ext, page_index=None)
    exported_path = export_png(img, name_base, png_export_dir, png_compression)
    tqdm.write(f"🖼️  Exported PNG: {exported_path}")

    res = _process_image_like_refactored(img, name_base, cfg)
    if not res[0]:
        tqdm.write(f"❌ {base}: {res[1]}")
        return 0

    out = cast(np.ndarray, res[1])
    out_path = os.path.join(crops_dir, f"{name_base}_crop.png")
    ok_write = cv2.imwrite(out_path, out, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if ok_write:
        tqdm.write(f"✅ {base}: {out_path}")
        return 1

    tqdm.write(f"❌ {base}: failed to write {out_path}")
    return 0


def main() -> None:
    """Discover inputs, render pages, run YOLO, and emit outputs."""
    args = parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_sub = args.output_sub
    model_path = args.model_path
    extensions = tuple(args.extensions)

    conf_thresh = float(args.conf_thresh)
    device = args.device
    tiff_page = int(args.tiff_page)
    decimals = int(args.decimals)
    write_empty_label = bool(args.write_empty_label)
    pdf_dpi = int(args.pdf_dpi)

    png_export_sub = args.png_export_sub
    png_compression = int(args.png_compression)
    labels_sub = args.labels_sub

    keep_input_extension_in_names = bool(args.keep_input_extension_in_names)

    crops_dir = os.path.join(input_dir, output_sub)
    png_export_dir = os.path.join(input_dir, png_export_sub)
    labels_dir = os.path.join(input_dir, labels_sub)

    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(png_export_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Model
    model = YOLO(model_path)
    cfg = DetectConfig(
        model=model,
        labels_dir=labels_dir,
        conf_thresh=conf_thresh,
        digits=decimals,
        write_empty_label=write_empty_label,
        device=device,
    )

    # Gather files
    paths = gather_paths(input_dir, extensions, recursive=bool(args.recursive))

    total_files = len(paths)

    if total_files == 0:
        print(f"No files found in {input_dir} matching: {', '.join(extensions)})")
        return

    saved_crops = 0

    with tqdm(total=total_files, desc="Files", unit="file", dynamic_ncols=True) as pbar:
        for path in paths:
            base = os.path.basename(path)
            _stem, extn = os.path.splitext(base)
            extn_lower = extn.lower()

            try:
                if extn_lower == ".pdf":
                    saved_crops += _handle_pdf(
                        path=path,
                        base=base,
                        name_keep_ext=keep_input_extension_in_names,
                        pdf_dpi=pdf_dpi,
                        png_export_dir=png_export_dir,
                        png_compression=png_compression,
                        crops_dir=crops_dir,
                        cfg=cfg,
                        progress_parent=pbar,
                    )
                elif extn_lower in (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"):
                    saved_crops += _handle_raster_image(
                        path=path,
                        base=base,
                        tiff_page=tiff_page,
                        name_keep_ext=keep_input_extension_in_names,
                        png_export_dir=png_export_dir,
                        png_compression=png_compression,
                        crops_dir=crops_dir,
                        cfg=cfg,
                    )
                elif extn_lower == ".dwg":
                    tqdm.write(
                        (
                            f"ℹ️ Skipping DWG (no built-in renderer): {base}\n"
                            "   Convert to PDF/PNG first, then rerun. "
                        )
                    )
                else:
                    tqdm.write(f"⚠️ Unsupported extension: {base}")
            except (OSError, RuntimeError, ValueError) as e:
                tqdm.write(f"💥 Error processing {base}: {e}")
            finally:
                pbar.update(1)

    tqdm.write(f"Done. Processed {total_files} file(s); saved {saved_crops} crop(s).")


if __name__ == "__main__":
    main()
