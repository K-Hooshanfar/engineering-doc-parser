"""
Core rotation utilities for scanned documents.

Provides:
- EAST-based text box detection (`detect_east_boxes`)
- Tesseract OSD parsing (`detect_osd_rotation`)
- Simple EAST-box heuristic (`estimate_orientation`)
- Rotation orchestration (`rotate_if_needed`)
- IO helpers and CLI entry (`main`)

Testability:
- Functions that tests monkeypatch at the *package* level are looked up via the
  `rotation` package module (using `_pkg()`), not via local globals.
- `main()` delegates per-file work to `process_one_image()` and file discovery
  to `iter_image_files()`—both fetched from the package.
- `_call_process_one_image(...)` adapts calls to work with both the pytest fake
  (positional `img_path, net`) and the real implementation (keyword-only params).
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Protocol, Sequence, Tuple

import cv2
import numpy as np
import pytesseract  # type: ignore[import-untyped]  # noqa: F401 - used via _pkg() dynamic lookup for standalone execution


# ---------- Internal helpers ----------
def _pkg():
    """
    Return the `rotation` package module if available; otherwise fall back to the
    current module so direct script execution still works.
    """
    if __package__:
        mod = sys.modules.get(__package__)
        if mod is not None:
            return mod
    mod = sys.modules.get("engineering_doc_parser.orientation")
    if mod is not None:
        return mod
    return sys.modules[__name__]


def _call_process_one_image(pkg, image_path, net, **kwargs):
    """
    Call pkg.process_one_image in a way that works for both:
    - test monkeypatch: def fake(img_path, net, **kwargs)
    - real impl:        def process_one_image(*, image_path:..., net:..., ...)
    """
    fn = getattr(pkg, "process_one_image")
    try:
        return fn(image_path, net, **kwargs)  # pytest fake accepts positional first two
    except TypeError:
        return fn(image_path=image_path, net=net, **kwargs)  # real impl needs keywords


class DnnNet(Protocol):
    """Subset of the OpenCV DNN Net API used by our pipeline."""

    # Allow OpenCV-style camelCase in this Protocol only.
    # pylint: disable=invalid-name

    def setInput(self, blob: np.ndarray) -> None:
        """Set the input blob for the next forward pass.

        Args:
            blob: 4D array shaped (N, C, H, W) in BGR order as expected by
                OpenCV's DNN module.
        """

    def forward(self, outputNames: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Run a forward pass and return the requested outputs.

        Args:
            outputNames: Names of the output layers to fetch, in order.

        Returns:
            Tuple of NumPy arrays corresponding to the requested outputs, in the
            same order as `outputNames`.
        """


# ---------- Logging ----------
def setup_logging(
    *,
    verbose: bool = False,
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure logging handlers and levels.

    Parameters
    ----------
    verbose : bool
        If True, use DEBUG level by default.
    log_level : Optional[str]
        A string like "INFO", "WARNING", etc. Overrides `verbose` if provided.
    log_file : Optional[str]
        If provided, also log to this file (created if missing).
    """
    level = logging.DEBUG if verbose else logging.INFO
    if isinstance(log_level, str):
        level = getattr(logging, log_level.upper(), level)

    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        try:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_file))
        except Exception:
            pass

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )


# ---------- EAST detection ----------
def _decode_east_scores_geometry(
    scores: np.ndarray, geometry: np.ndarray, conf_thresh: float
) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
    """
    Decode EAST outputs to bounding boxes and confidences (simple implementation).

    Returns
    -------
    rects : List[Tuple[int, int, int, int]]  # (x1, y1, x2, y2)
    confidences : List[float]
    """
    (numRows, numCols) = scores.shape[2:4]
    rects: List[Tuple[int, int, int, int]] = []
    confidences: List[float] = []

    for y in range(numRows):
        scoresData = scores[0, 0, y]
        x0 = geometry[0, 0, y]
        x1 = geometry[0, 1, y]
        x2 = geometry[0, 2, y]
        x3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        for x in range(numCols):
            score = float(scoresData[x])
            if score < conf_thresh:
                continue
            offsetX, offsetY = (x * 4.0, y * 4.0)
            angle = float(anglesData[x])
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = float(x0[x] + x2[x])
            w = float(x1[x] + x3[x])

            endX = int(offsetX + (cos * x1[x]) + (sin * x2[x]))
            endY = int(offsetY - (sin * x1[x]) + (cos * x2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(score)

    return rects, confidences


def detect_east_boxes(
    image: np.ndarray,
    net: DnnNet,
    conf_thresh: float = 0.5,
    nms_thresh: float = 0.4,
) -> List[Tuple[int, int, int, int]]:
    """
    Detect text boxes with the EAST model.

    Notes
    -----
    Tests usually monkeypatch this function at the package level to avoid
    touching OpenCV DNN. Real calls expect `net` to be a cv2.dnn_Net.

    Returns
    -------
    List[(x1, y1, x2, y2)] scaled to original image size.
    """
    H, W = image.shape[:2]
    newW, newH = 320, 320
    rW, rH = W / newW, H / newH

    blob = cv2.dnn.blobFromImage(
        image, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False
    )
    net.setInput(blob)  # type: ignore[attr-defined]

    (scores, geometry) = net.forward(
        ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    )  # type: ignore[attr-defined]

    rects, confidences = _decode_east_scores_geometry(scores, geometry, conf_thresh)

    boxes_xywh = []
    for x1, y1, x2, y2 in rects:
        boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])
    idxs = cv2.dnn.NMSBoxes(boxes_xywh, confidences, conf_thresh, nms_thresh)

    boxes: List[Tuple[int, int, int, int]] = []
    if len(idxs) > 0:
        for i in idxs.flatten() if hasattr(idxs, "flatten") else idxs:
            x, y, w, h = boxes_xywh[i]
            startX, startY, endX, endY = int(x), int(y), int(x + w), int(y + h)
            boxes.append(
                (
                    int(startX * rW),
                    int(startY * rH),
                    int(endX * rW),
                    int(endY * rH),
                )
            )
    return boxes


# ---------- OSD / Orientation ----------
def detect_osd_rotation(image: np.ndarray) -> int:
    """
    Infer page rotation angle from Tesseract OSD output.

    Prefer the 'Rotate:' line over 'Orientation in degrees:'.
    Returns one of {0, 90, 180, 270}. Returns 0 on errors.
    """
    pkg = _pkg()
    try:
        osd = pkg.pytesseract.image_to_osd(image)  # use package-level pytesseract
    except Exception:
        return 0

    # Robust, simple line scan
    for line in str(osd).splitlines():
        if "rotate" in line.lower():
            m = re.search(r"(\d+)", line)
            if m:
                return int(m.group(1)) % 360

    # Last-ditch fallback: first integer anywhere
    m = re.search(r"(\d+)", str(osd))
    return int(m.group(1)) % 360 if m else 0


def estimate_orientation(
    boxes: Sequence[Tuple[int, int, int, int]],
    *,
    ratio_thresh: float = 1.2,
) -> Tuple[int, int]:
    """
    Count boxes clearly horizontal vs vertical; near-squares ignored.

    Parameters
    ----------
    ratio_thresh : float
        If w/h > ratio_thresh -> horizontal; if h/w > ratio_thresh -> vertical.

    Returns
    -------
    (horiz_count, vert_count)
    """
    horiz = 0
    vert = 0
    for x1, y1, x2, y2 in boxes:
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        if w <= 0 or h <= 0:
            continue
        if (w / h) > ratio_thresh:
            horiz += 1
        elif (h / w) > ratio_thresh:
            vert += 1
    return horiz, vert


def _rotate_by_angle(img: np.ndarray, angle: int) -> np.ndarray:
    """
    Rotate by a cardinal angle using cv2 rotate constants.

    Tests expect a 90° rotation to be clockwise.
    """
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def rotate_if_needed(
    img: np.ndarray,
    boxes: Sequence[Tuple[int, int, int, int]],
    *,
    use_osd: bool = True,
    force_angle: Optional[int] = None,
    ratio_thresh: float = 1.2,
) -> Tuple[np.ndarray, bool, str]:
    """
    Decide an angle (0/90/180/270) and rotate.

    Order:
    1) Honor `force_angle` if provided.
    2) If `use_osd`, prefer OSD angle.
    3) If OSD==0 and boxes exist, EAST fallback:
       more vertical than horizontal -> rotate 90° CW.

    Returns
    -------
    (out_img, changed, message)
    """
    if force_angle is not None:
        angle = int(force_angle) % 360
        angle = angle if angle in (0, 90, 180, 270) else 0
        out = _rotate_by_angle(img, angle)
        return out, angle != 0, f"Forced rotation {angle}°"

    angle = 0
    if use_osd:
        angle = detect_osd_rotation(img)

    used_fallback = False
    if angle == 0 and boxes:
        horiz, vert = estimate_orientation(boxes, ratio_thresh=ratio_thresh)
        if vert > horiz:
            angle = 90
            used_fallback = True

    out = _rotate_by_angle(img, angle)
    changed = angle != 0
    if not changed:
        return img, False, "No orientation needed"

    tag = " (EAST fallback)" if used_fallback else ""
    direction = "CW" if angle == 90 else "CCW" if angle == 270 else ""
    msg = f"Rotated {angle}° {direction}".strip() + tag
    return out, True, msg


# ---------- IO helpers ----------
def save_image(img: np.ndarray, output_path: str) -> None:
    """Save an image to disk, creating parent directories as needed."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, img)


def iter_image_files(
    root: str, exts: Sequence[str], recursive: bool = False
) -> Iterator[str]:
    """
    Yield image file paths under `root` whose extension is in `exts` (case-insensitive).
    """
    exts_low = {e.lower().lstrip(".") for e in exts}
    root_path = Path(root)
    if not root_path.exists():
        return
    it: Iterable[Path] = root_path.rglob("*") if recursive else root_path.glob("*")
    for p in it:
        if p.is_file() and p.suffix.lower().lstrip(".") in exts_low:
            yield str(p)


# ---------- CLI & main workflow ----------
def build_parser() -> argparse.ArgumentParser:
    """
    Build the CLI parser.

    Aliases:
    - image path: `-i`, `--image_path`, `--image-path`
    - input dir:  `-d`, `--input_dir`, `--input-dir`
    - east model: `-m`, `--east_model`, `--east-model`
    - output dir: `-o`, `--result_dir`, `--result-dir`
    """
    p = argparse.ArgumentParser(
        description="Detect & fix image rotation for document scans."
    )
    p.add_argument(
        "-i",
        "--image_path",
        "--image-path",
        dest="image_path",
        type=str,
        default=None,
        help="Single image path.",
    )
    p.add_argument(
        "-d",
        "--input_dir",
        "--input-dir",
        dest="input_dir",
        type=str,
        default=None,
        help="Directory of images.",
    )
    p.add_argument(
        "-m",
        "--east_model",
        "--east-model",
        dest="east_model",
        type=str,
        required=True,
        help="Path to EAST .pb model.",
    )
    p.add_argument(
        "-o",
        "--result_dir",
        "--result-dir",
        dest="result_dir",
        type=str,
        default="rotation/result",
        help="Where to write outputs (default: ./out).",
    )
    p.add_argument(
        "--conf_thresh", type=float, default=0.5, help="Confidence threshold for EAST."
    )
    p.add_argument(
        "--nms_thresh", type=float, default=0.4, help="NMS threshold for EAST."
    )
    p.add_argument(
        "--ratio_thresh",
        type=float,
        default=1.2,
        help="Aspect ratio threshold for EAST fallback.",
    )
    p.add_argument("--no_osd", action="store_true", help="Disable Tesseract OSD.")
    p.add_argument(
        "--force_angle",
        type=int,
        default=None,
        help="Force rotation angle (0,90,180,270).",
    )
    p.add_argument(
        "-r", "--recursive", action="store_true", help="Recurse into subdirectories."
    )
    p.add_argument(
        "--exts",
        nargs="+",
        default=["png", "jpg"],
        help="Extensions to scan (e.g., png jpg).",
    )
    p.add_argument(
        "--save_unchanged", action="store_true", help="Write even if unchanged."
    )
    p.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging."
    )
    p.add_argument(
        "--log_level",
        type=str,
        default=None,
        help="Explicit log level (INFO, DEBUG, ...).",
    )
    p.add_argument("--log_file", type=str, default=None, help="Optional log file path.")
    return p


def process_one_image(
    *,
    image_path: str,
    net: DnnNet,
    result_dir: str,
    conf_thresh: float,
    nms_thresh: float,
    ratio_thresh: float,
    use_osd: bool,
    force_angle: Optional[int],
    save_unchanged: bool,
    verbose: bool,
) -> None:
    """
    Load a single image, decide rotation, and write the result.

    Calls monkeypatchable functions through the package module (`rotation`), so tests
    can stub: `detect_east_boxes`, `rotate_if_needed`, `save_image`.
    """
    logging.info("Loading image: %s", image_path)
    img = cv2.imread(image_path)
    if img is None:
        logging.warning("Could not read image: %s", image_path)
        return

    pkg = _pkg()
    boxes = pkg.detect_east_boxes(
        img, net, conf_thresh=conf_thresh, nms_thresh=nms_thresh
    )
    out_img, changed, message = pkg.rotate_if_needed(
        img,
        boxes,
        use_osd=use_osd,
        force_angle=force_angle,
        ratio_thresh=ratio_thresh,
    )
    logging.info(message)

    src = Path(image_path)
    dst_dir = Path(result_dir)
    if changed:
        out_path = dst_dir / f"{src.stem}_rotated{src.suffix}"
        pkg.save_image(out_img, str(out_path))
    else:
        if save_unchanged:
            out_path = dst_dir / src.name
            pkg.save_image(out_img, str(out_path))


def main() -> None:
    """
    CLI entry point:
    - Validates EAST model path
    - Loads the EAST net once
    - Processes either a single image or walks a directory using `iter_image_files`
    - Delegates per-file work to `process_one_image`
    """
    args = build_parser().parse_args()
    setup_logging(
        verbose=args.verbose, log_level=args.log_level, log_file=args.log_file
    )

    if not os.path.isfile(args.east_model):
        raise FileNotFoundError(f"EAST model not found: {args.east_model}")
    logging.info("Loading EAST model: %s", args.east_model)

    net = cv2.dnn.readNet(args.east_model)  # tests stub this to a dummy object

    pkg = _pkg()

    # Directory mode
    if args.input_dir and os.path.isdir(args.input_dir):
        files = list(pkg.iter_image_files(args.input_dir, args.exts, args.recursive))
        if not files:
            logging.warning(
                "No images found in %s (exts=%s, recursive=%s).",
                args.input_dir,
                args.exts,
                args.recursive,
            )
            return
        for img_path in files:
            _call_process_one_image(
                pkg,
                img_path,
                net,
                result_dir=args.result_dir,
                conf_thresh=args.conf_thresh,
                nms_thresh=args.nms_thresh,
                ratio_thresh=args.ratio_thresh,
                use_osd=not args.no_osd,
                force_angle=args.force_angle,
                save_unchanged=args.save_unchanged,
                verbose=args.verbose,
            )
        return

    # Single-image mode
    if args.image_path:
        _call_process_one_image(
            pkg,
            args.image_path,
            net,
            result_dir=args.result_dir,
            conf_thresh=args.conf_thresh,
            nms_thresh=args.nms_thresh,
            ratio_thresh=args.ratio_thresh,
            use_osd=not args.no_osd,
            force_angle=args.force_angle,
            save_unchanged=args.save_unchanged,
            verbose=args.verbose,
        )
    else:
        logging.warning("No input specified (image_path or input_dir).")


if __name__ == "__main__":
    main()
