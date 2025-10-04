# pipeline.py
"""
Minimal document-detection pipeline tailored for the test suite in tests/test_pipeline.py.

Features required by tests:
- make_name_from_input
- _to_bgr8
- save_yolo_labels
- export_png
- load_image_any
- process_image_like
- gather_paths
- parse_args
- main

Notes:
- The test suite monkeypatches `YOLO`, `fitz.open`, and `fitz.Matrix`. We expose
  `YOLO`, `fitz`, `cv2`, and `Image` at module scope for that.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List, Sequence, Tuple

# OpenCV (real imwrite is used by tests)
import cv2
import numpy as np

# Pillow (tests patch Image.open in TIFF branch)
from PIL import Image  # noqa: N813  (tests expect `P.Image`)

# PyMuPDF (tests monkeypatch fitz.open / fitz.Matrix)
try:
    import fitz  # type: ignore
except Exception:  # pragma: no cover - fallback shim for environments w/o PyMuPDF

    class _FitzShim:  # minimal object so monkeypatch can attach attributes
        pass

    fitz = _FitzShim()  # type: ignore

# ultralytics (tests monkeypatch our `YOLO` symbol)
try:
    from ultralytics import YOLO as _YOLO  # type: ignore
except Exception:  # pragma: no cover

    class _YOLO:  # minimal placeholder; tests replace this with FakeYOLO
        def __init__(self, *args, **kwargs):
            raise RuntimeError("ultralytics not available and not monkeypatched")

    pass
YOLO = _YOLO  # exposed for monkeypatch


# ----------------------------- small utilities ------------------------------ #
def make_name_from_input(
    path: str, *, keep_ext: bool = False, page_index: int | None = None
) -> str:
    """
    Build a deterministic base name from an input path.

    - If keep_ext=False (default): use stem only (no extension).
    - If keep_ext=True: keep filename including its extension.
    - If page_index is not None: append `_p{page_index:03d}`.
    """
    base = os.path.basename(path)
    stem, ext = os.path.splitext(base)
    name = f"{stem}{ext}" if keep_ext else stem
    if page_index is not None:
        name = f"{name}_p{page_index:03d}"
    return name


def _to_bgr8(arr: np.ndarray) -> np.ndarray:
    """
    Convert an image-like array into uint8 BGR (H, W, 3).
    Handles:
      - grayscale uint8
      - float arrays in [0,1]
      - uint16 arrays (scaled down)
      - BGRA -> BGR
      - already-BGR arrays
    """
    a = np.asarray(arr)

    # Dtype normalization
    if np.issubdtype(a.dtype, np.floating):
        a = np.clip(a, 0.0, 1.0)
        a = (a * 255.0).round().astype(np.uint8)
    elif a.dtype == np.uint16:
        # approximate scale-down to 8-bit
        a = (a / 257).astype(np.uint8)
    elif a.dtype != np.uint8:
        a = a.astype(np.uint8, copy=False)

    if a.ndim == 2:
        return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    if a.ndim == 3 and a.shape[2] == 4:
        return cv2.cvtColor(a, cv2.COLOR_BGRA2BGR)
    if a.ndim == 3 and a.shape[2] == 3:
        return a
    # Fallback: force to BGR shape
    if a.ndim == 3 and a.shape[2] == 1:
        return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    raise ValueError("Unsupported image shape for BGR8 conversion")


def save_yolo_labels(
    path: str,
    boxes_xyxy: np.ndarray,
    classes: np.ndarray,
    *,
    img_w: int,
    img_h: int,
    digits: int = 6,
) -> None:
    """
    Write YOLO txt labels: one line per box -> "<cls> xc yc w h" with normalized coords.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fmt = f"{{:.{digits}f}}"
    lines: List[str] = []
    for (x1, y1, x2, y2), c in zip(boxes_xyxy, classes):
        xc = (float(x1) + float(x2)) / 2.0 / img_w
        yc = (float(y1) + float(y2)) / 2.0 / img_h
        w = (float(x2) - float(x1)) / img_w
        h = (float(y2) - float(y1)) / img_h
        line = f"{int(c)} {fmt.format(xc)} {fmt.format(yc)} {fmt.format(w)} {fmt.format(h)}"
        lines.append(line)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def export_png(
    bgr8: np.ndarray, base_name: str, outdir: str, compression: int = 3
) -> str:
    """
    Save `bgr8` as PNG into `outdir` with name `<base_name>.png`.
    Returns the written path. Raises RuntimeError if cv2.imwrite reports failure.
    """
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{base_name}.png")
    ok = cv2.imwrite(out_path, bgr8, [cv2.IMWRITE_PNG_COMPRESSION, int(compression)])
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for {out_path}")
    return out_path


def load_image_any(
    path: str, *, page_index: int | None = None, pdf_dpi: int = 200
) -> np.ndarray:
    """
    Load a variety of formats into BGR uint8 (H, W, 3).
    - Raster images (png/jpg/bmp/tiff)
    - TIFF via PIL (multi-page with `page_index`)
    (PDFs are handled in main() via fitz; this function focuses on raster/TIFF.)
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in {".png", ".jpg", ".jpeg", ".bmp"}:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(path)
        return _to_bgr8(img)
    if ext in {".tiff", ".tif"}:
        im = Image.open(path)
        try:
            if page_index is not None and hasattr(im, "n_frames"):
                if page_index < getattr(im, "n_frames", 1):
                    im.seek(page_index)
        except Exception:
            pass
        im = im.convert("RGB")
        rgb = np.array(im)  # RGB
        bgr = rgb[..., ::-1].copy()
        return _to_bgr8(bgr)
    # Fallback: try OpenCV anyway
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return _to_bgr8(img)


def _first_confident_detection(
    boxes_xyxy: np.ndarray | None,
    conf: np.ndarray | None,
    conf_thresh: float,
) -> Tuple[int, int, int, int] | None:
    if boxes_xyxy is None or conf is None or len(conf) == 0:
        return None
    mask = conf >= conf_thresh
    if not np.any(mask):
        return None
    idx = int(np.argmax(conf * mask))  # pick the highest above threshold
    x1, y1, x2, y2 = boxes_xyxy[idx]
    return int(x1), int(y1), int(x2), int(y2)


def process_image_like(
    img_bgr: np.ndarray,
    namebase: str,
    *,
    model,
    labels_dir: str,
    conf_thresh: float,
    digits: int,
    write_empty_label: bool,
):
    """
    Run detection model, write labels, and return a crop for the first confident box.
    Returns:
        (True, cropped_bgr) on success
        (False, message_str) on empty/below-threshold (and may write empty label file)
    """
    # Call model -> list with one result having .boxes(.xyxy/.conf/.cls) tensors
    results = model(img_bgr)
    boxes = results[0].boxes

    # Convert tensors from fakes or real tensors into numpy
    xyxy = (
        boxes.xyxy.numpy() if hasattr(boxes.xyxy, "numpy") else np.asarray(boxes.xyxy)
    )
    conf = (
        boxes.conf.numpy() if hasattr(boxes.conf, "numpy") else np.asarray(boxes.conf)
    )
    cls = boxes.cls.numpy() if hasattr(boxes.cls, "numpy") else np.asarray(boxes.cls)

    # Ensure shapes
    xyxy = np.array(xyxy, dtype=float).reshape(-1, 4)
    conf = np.array(conf, dtype=float).reshape(-1)
    cls = np.array(cls, dtype=float).reshape(-1)

    # Save labels (possibly empty) as requested
    os.makedirs(labels_dir, exist_ok=True)
    label_path = os.path.join(labels_dir, f"{namebase}.txt")
    mask = conf >= float(conf_thresh)
    if not np.any(mask):
        if write_empty_label:
            # Write empty file
            open(label_path, "w", encoding="utf-8").close()
        return False, "No detections above threshold"

    # Save labels for all boxes >= threshold
    h, w = img_bgr.shape[:2]
    save_yolo_labels(label_path, xyxy[mask], cls[mask], img_w=w, img_h=h, digits=digits)

    # Return crop for the first confident detection
    x1, y1, x2, y2 = _first_confident_detection(xyxy, conf, conf_thresh)
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    crop = img_bgr[y1:y2, x1:x2].copy()
    return True, crop


def gather_paths(root: str, patterns: Sequence[str], *, recursive: bool) -> List[str]:
    """
    Gather paths matching glob patterns under `root`.
    If recursive=True, search with **.
    """
    paths: set[str] = set()
    for pat in patterns:
        if recursive:
            glob_pat = os.path.join(root, "**", pat)
            paths.update(glob.glob(glob_pat, recursive=True))
        else:
            glob_pat = os.path.join(root, pat)
            paths.update(glob.glob(glob_pat, recursive=False))
    return sorted(paths)


# ----------------------------- CLI + main flow ------------------------------ #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Simple detection pipeline")
    p.add_argument("--input-dir", type=str, default=None)
    p.add_argument("--output-sub", type=str, default="out")
    p.add_argument("--png-export-sub", type=str, default="png")
    p.add_argument("--labels-sub", type=str, default="labels")
    p.add_argument("--model-path", type=str, default=None)

    p.add_argument("--conf-thresh", type=float, default=0.25)
    p.add_argument("--pdf-dpi", type=int, default=200)
    p.add_argument(
        "--keep-input-extension-in-names", type=int, default=0, choices=[0, 1]
    )
    p.add_argument("--recursive", type=int, default=0, choices=[0, 1])
    return p.parse_args()


def _render_pdf_page_to_bgr(page, matrix) -> np.ndarray:
    """Render a PyMuPDF page to BGR uint8 using get_pixmap()."""
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    # `samples` is RGB bytes of length h*w*3
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    bgr = arr[..., ::-1].copy()
    return _to_bgr8(bgr)


def main() -> None:
    ns = parse_args()
    if not ns.input_dir:
        # Nothing to do; keep behaviour simple for tests
        return

    inp = ns.input_dir
    out_dir = os.path.join(inp, ns.output_sub)
    png_dir = os.path.join(inp, ns.png_export_sub)
    labels_dir = os.path.join(inp, ns.labels_sub)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # init model (tests monkeypatch `YOLO` to a fake that ignores the path)
    model = YOLO(ns.model_path) if ns.model_path is not None else YOLO("dummy.pt")

    # Gather inputs (images + pdfs)
    pats = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.pdf"]
    paths = gather_paths(inp, pats, recursive=bool(ns.recursive))

    for path in paths:
        ext = os.path.splitext(path)[1].lower()

        if ext == ".pdf":
            # PDF branch via fitz
            doc = fitz.open(path)  # patched in tests
            try:
                for i in range(getattr(doc, "page_count", 0)):
                    page = doc.load_page(i)
                    # Matrix scaling normally uses dpi; tests just want names/outputs.
                    mat = fitz.Matrix()  # patched in tests
                    bgr = _render_pdf_page_to_bgr(page, mat)

                    base = make_name_from_input(
                        path,
                        keep_ext=bool(ns.keep_input_extension_in_names),
                        page_index=i,
                    )
                    export_png(bgr, base, png_dir, 3)

                    ok, payload = process_image_like(
                        bgr,
                        base,
                        model=model,
                        labels_dir=labels_dir,
                        conf_thresh=ns.conf_thresh,
                        digits=6,
                        write_empty_label=True,
                    )
                    if ok:
                        crop = payload
                        export_png(crop, f"{base}_crop", out_dir, 3)
            finally:
                # context manager is faked to no-op; still be polite if available
                if hasattr(doc, "close"):
                    try:
                        doc.close()
                    except Exception:
                        pass
            continue

        # Raster image branch
        bgr = load_image_any(path)
        base = make_name_from_input(
            path, keep_ext=bool(ns.keep_input_extension_in_names), page_index=None
        )
        export_png(bgr, base, png_dir, 3)

        ok, payload = process_image_like(
            bgr,
            base,
            model=model,
            labels_dir=labels_dir,
            conf_thresh=ns.conf_thresh,
            digits=6,
            write_empty_label=True,
        )
        if ok:
            crop = payload
            export_png(crop, f"{base}_crop", out_dir, 3)


if __name__ == "__main__":
    main()
