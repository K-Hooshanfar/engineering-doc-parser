"""
rotation.py

Detects document orientation using EAST text detector and Tesseract OSD,
and rotates the image if needed.

CLI examples:
  # single image (as before)
  python rotation.py -i test_images/3.png -m path/to/frozen_east_text_detection.pb -o result

  # whole folder (non-recursive)
  python rotation.py --input-dir scans/ -m path/to/frozen_east_text_detection.pb -o result

  # whole folder (recursive) with custom extensions & save unchanged too
  python rotation.py --input-dir scans/ --recursive --exts png jpg jpeg tif tiff \
      --save-unchanged -m path/to/frozen_east_text_detection.pb -o result

  # logging examples
  python rotation.py -i doc.png -m models/east.pb -o out -v
  python rotation.py --input-dir scans -m models/east.pb -o out --log-level INFO --log-file run.log
"""

import argparse
import logging
import os
import re
from typing import Iterable, List, Optional, Tuple, Protocol, Any, cast

import cv2
import numpy as np
import pytesseract  # type: ignore[import-not-found,import-untyped]

try:
    from .config import Config
except Exception:
    from rotation.config import Config


# --------------------------- Logging helpers ---------------------------------
def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure root logger with console (and optional file) handlers."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    # Clear existing handlers (important in notebooks/REPL)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(numeric_level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(numeric_level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)


log = logging.getLogger(__name__)
# ----------------------------------------------------------------------------


class DnnNet(Protocol):
    def setInput(self, blob: Any) -> None:
        """Set the input blob for the DNN before a forward pass."""
        ...

    def forward(self, out_names: Iterable[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Run a forward pass and return (scores, geometry) tensors for EAST.

        Args:
            out_names: Layer names to fetch (e.g., ["feature_fusion/Conv_7/Sigmoid",
                "feature_fusion/concat_3"]).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Scores map and geometry map.
        """
        ...


def detect_east_boxes(
    image: np.ndarray,
    net: DnnNet,
    conf_thresh: float = 0.5,
    nms_thresh: float = 0.4,
) -> List[Tuple[int, int, int, int]]:
    """Detect text boxes with EAST (vectorized)."""
    H, W = image.shape[:2]
    newW, newH = 320, 320
    rW, rH = W / newW, H / newH

    blob = cv2.dnn.blobFromImage(
        image, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False
    )
    net.setInput(blob)
    scores, geometry = net.forward(
        ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    )

    scores = scores[0, 0]
    geom = geometry[0]
    ys, xs = np.where(scores > conf_thresh)
    if len(xs) == 0:
        log.debug("EAST: no positions above conf_thresh=%.3f", conf_thresh)
        return []

    offset_x = xs * 4.0
    offset_y = ys * 4.0

    g0 = geom[0, ys, xs]
    g1 = geom[1, ys, xs]
    g2 = geom[2, ys, xs]
    g3 = geom[3, ys, xs]
    angles = geom[4, ys, xs]

    cos_a, sin_a = np.cos(angles), np.sin(angles)
    h = g0 + g2
    w = g1 + g3

    end_x = offset_x + cos_a * g1 + sin_a * g2
    end_y = offset_y - sin_a * g1 + cos_a * g2
    start_x = end_x - w
    start_y = end_y - h

    rects_resized = np.stack([start_x, start_y, end_x, end_y], axis=1)
    rects_int = rects_resized.round().astype(int).tolist()
    confidences = scores[ys, xs].tolist()

    indices = cv2.dnn.NMSBoxes(rects_int, confidences, conf_thresh, nms_thresh)
    idxs = (
        np.array(indices).flatten().astype(int)
        if len(indices)
        else np.array([], dtype=int)
    )
    if idxs.size == 0:
        log.debug(
            "EAST: NMS filtered all boxes (conf_thresh=%.2f, nms_thresh=%.2f)",
            conf_thresh,
            nms_thresh,
        )
        return []

    rects_resized[:, [0, 2]] *= rW
    rects_resized[:, [1, 3]] *= rH
    final_boxes = rects_resized[idxs].round().astype(int).tolist()
    log.debug("EAST: kept %d boxes after NMS", len(final_boxes))
    return final_boxes


def estimate_orientation(
    boxes: List[Tuple[int, int, int, int]], ratio_thresh: float = 1.2
) -> Tuple[int, int]:
    """Count horiz vs vert boxes."""
    horiz = 0
    vert = 0
    for x1, y1, x2, y2 in boxes:
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        if w / h > ratio_thresh:
            horiz += 1
        elif h / w > ratio_thresh:
            vert += 1
    log.debug(
        "Orientation counts: horiz=%d vert=%d (ratio_thresh=%.2f)",
        horiz,
        vert,
        ratio_thresh,
    )
    return horiz, vert


def detect_osd_rotation(image: np.ndarray) -> int:
    """Use Tesseract OSD to get rotation (0/90/180/270)."""
    osd = pytesseract.image_to_osd(image)
    match = re.search(r"Rotate: (\d+)", osd)
    angle = int(match.group(1)) if match else 0
    log.debug("OSD rotation suggestion: %d", angle)
    return angle


def rotate_if_needed(
    img: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    *,
    use_osd: bool = True,
    ratio_thresh: float = 1.2,
    force_angle: Optional[int] = None,
) -> Tuple[np.ndarray, bool, str]:
    """Decide rotation via forced angle → OSD → EAST heuristic."""
    if force_angle in {0, 90, 180, 270}:
        angle = force_angle
        log.debug("Force angle provided: %d", angle)
    else:
        angle = detect_osd_rotation(img) if use_osd else 0

    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), True, "Rotated 90° CW"
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180), True, "Rotated 180°"
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), True, "Rotated 90° CCW"
    else:
        horiz, vert = estimate_orientation(boxes, ratio_thresh=ratio_thresh)
        if vert > horiz:
            return (
                cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
                True,
                "Rotated 90° CW (EAST fallback)",
            )
        else:
            return img, False, "No orientation needed"


def save_image(img: np.ndarray, output_path: str) -> None:
    """Save with sensible defaults per extension."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ext = os.path.splitext(output_path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        params = [cv2.IMWRITE_JPEG_QUALITY, 100]
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 0]
    else:
        params = []
    ok = cv2.imwrite(output_path, img, params)
    if ok:
        log.info("Saved → %s", output_path)
    else:
        log.error("Failed to save → %s", output_path)


def build_parser() -> argparse.ArgumentParser:
    """CLI parser."""
    p = argparse.ArgumentParser(
        description="Detect and correct document orientation using EAST + Tesseract OSD."
    )
    # Single-image path (as before)
    p.add_argument(
        "-i",
        "--image-path",
        default="test_images/3.png",
        help="Path to a single input image.",
    )
    # NEW: directory input
    p.add_argument("--input-dir", help="Process all images in this directory.")
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when using --input-dir.",
    )
    p.add_argument(
        "--exts",
        nargs="+",
        default=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"],
        help="File extensions to include when using --input-dir (space-separated, no dots).",
    )

    p.add_argument(
        "-m",
        "--east-model",
        default=Config.EAST_TEXT_DETECTOR,
        help="Path to the frozen EAST detector .pb model.",
    )
    p.add_argument(
        "-o",
        "--result-dir",
        default="rotation/result",
        help="Directory to write outputs.",
    )
    p.add_argument(
        "--save-unchanged",
        action="store_true",
        help="Also save images that do not need rotation (keeps original filename).",
    )

    p.add_argument(
        "--conf-thresh",
        type=float,
        default=0.5,
        help="Confidence threshold for EAST detections.",
    )
    p.add_argument(
        "--nms-thresh",
        type=float,
        default=0.4,
        help="NMS IoU threshold for EAST boxes.",
    )
    p.add_argument(
        "--ratio-thresh",
        type=float,
        default=1.2,
        help="Aspect-ratio threshold for EAST fallback (horiz vs vert).",
    )
    p.add_argument(
        "--no-osd",
        action="store_true",
        help="Disable Tesseract OSD; rely only on EAST fallback.",
    )
    p.add_argument(
        "--force-angle",
        type=int,
        choices=[0, 90, 180, 270],
        help="Force a fixed rotation angle (overrides OSD and fallback).",
    )
    # Logging controls
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging (overridden by --log-level if provided).",
    )
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default INFO; DEBUG if -v).",
    )
    p.add_argument(
        "--log-file",
        help="Optional path to write logs to a file in addition to console.",
    )
    return p


def iter_image_files(
    root_dir: str, exts: Iterable[str], recursive: bool
) -> Iterable[str]:
    """Yield image file paths from a directory."""
    exts = tuple(f".{e.lower().lstrip('.')}" for e in exts)
    if recursive:
        for r, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(exts):
                    yield os.path.join(r, f)
    else:
        for f in sorted(os.listdir(root_dir)):
            p = os.path.join(root_dir, f)
            if os.path.isfile(p) and f.lower().endswith(exts):
                yield p


def process_one_image(
    image_path: str,
    net: DnnNet,
    *,
    result_dir: str,
    conf_thresh: float,
    nms_thresh: float,
    ratio_thresh: float,
    use_osd: bool,
    force_angle: Optional[int],
    save_unchanged: bool,
    verbose: bool,  # kept for backward compatibility; not used directly
) -> None:
    """Run pipeline for one image path."""
    if not os.path.isfile(image_path):
        log.warning("Skipping non-file: %s", image_path)
        return

    img = cv2.imread(image_path)
    if img is None:
        log.warning("Could not read image: %s", image_path)
        return

    boxes = detect_east_boxes(img, net, conf_thresh=conf_thresh, nms_thresh=nms_thresh)
    oriented_img, changed, decision = rotate_if_needed(
        img, boxes, use_osd=use_osd, ratio_thresh=ratio_thresh, force_angle=force_angle
    )

    base, ext = os.path.splitext(os.path.basename(image_path))
    if changed:
        out_name = f"{base}_rotated{ext}"
        out_path = os.path.join(result_dir, out_name)
        log.info("%s :: %s → %s", decision, image_path, out_path)
        save_image(oriented_img, out_path)
    else:
        log.info("%s :: %s", decision, image_path)
        if save_unchanged:
            out_path = os.path.join(result_dir, f"{base}{ext}")
            save_image(oriented_img, out_path)
            log.debug("Saved unchanged → %s", out_path)


def main() -> None:
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # Configure logging early
    chosen_level = args.log_level or ("DEBUG" if args.verbose else "INFO")
    setup_logging(level=chosen_level, log_file=args.log_file)
    log.debug("Logging initialized at %s", chosen_level)

    image_path: str = args.image_path
    input_dir: Optional[str] = args.input_dir
    east_model_path: str = args.east_model
    result_dir: str = args.result_dir
    conf_thresh: float = args.conf_thresh
    nms_thresh: float = args.nms_thresh
    ratio_thresh: float = args.ratio_thresh
    use_osd: bool = not args.no_osd
    force_angle: Optional[int] = args.force_angle
    recursive: bool = args.recursive
    exts: List[str] = args.exts
    save_unchanged: bool = args.save_unchanged

    if not os.path.isfile(east_model_path):
        log.error("EAST model not found: %s", east_model_path)
        raise FileNotFoundError(f"EAST model not found: {east_model_path}")
    os.makedirs(result_dir, exist_ok=True)

    if input_dir:
        if not os.path.isdir(input_dir):
            log.error("--input-dir is not a directory: %s", input_dir)
            raise NotADirectoryError(f"--input-dir is not a directory: {input_dir}")

        log.info("Loading EAST model: %s", east_model_path)
        net = cast(DnnNet, cv2.dnn.readNet(east_model_path))

        files = list(iter_image_files(input_dir, exts, recursive))
        if not files:
            log.warning(
                "No images found in %s (exts=%s, recursive=%s).",
                input_dir,
                exts,
                recursive,
            )
            return

        log.info("Found %d images. Writing to: %s", len(files), result_dir)
        for path in files:
            process_one_image(
                path,
                net,
                result_dir=result_dir,
                conf_thresh=conf_thresh,
                nms_thresh=nms_thresh,
                ratio_thresh=ratio_thresh,
                use_osd=use_osd,
                force_angle=force_angle,
                save_unchanged=save_unchanged,
                verbose=args.verbose,
            )
    else:
        if not os.path.isfile(image_path):
            log.error("Image not found: %s", image_path)
            raise FileNotFoundError(f"Image not found: {image_path}")

        log.info("Loading image: %s", image_path)
        log.info("Loading EAST model: %s", east_model_path)
        net = cast(DnnNet, cv2.dnn.readNet(east_model_path))

        process_one_image(
            image_path,
            net,
            result_dir=result_dir,
            conf_thresh=conf_thresh,
            nms_thresh=nms_thresh,
            ratio_thresh=ratio_thresh,
            use_osd=use_osd,
            force_angle=force_angle,
            save_unchanged=save_unchanged,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
