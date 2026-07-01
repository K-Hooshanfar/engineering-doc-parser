#!/usr/bin/env python3
"""
YOLO-based rotation detection: tries all 4 rotations and picks the best one.
Now with rotation_candidates_debug (metrics) + visual debug grid.
"""

import argparse
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from ultralytics import YOLO

# --------------------
# Config
# --------------------
MODEL_PATHS: List[str] = [
    "/code/Datasets/foolad/model_serve/best_r_985_p_973_map_832.onnx",
    "/code/Datasets/foolad/model_serve/best_r986_p962.onnx",
    "/code/Datasets/foolad/model_serve/best_2.onnx",
]
DEVICE = "cpu"
ALLOWED_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

_MODEL_CACHE: Dict[str, YOLO] = {}


# --------------------
# Image I/O utilities
# --------------------
def _to_bgr8(arr: np.ndarray) -> np.ndarray:
    """Normalize an array to uint8 3-channel BGR."""
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


def load_image_from_bytes(image_bytes: bytes, page_index: int = 0) -> np.ndarray:
    """Load image from bytes, returns uint8 BGR (H, W, 3)."""
    bio = io.BytesIO(image_bytes)
    try:
        im = Image.open(bio)
        if getattr(im, "n_frames", 1) > 1:
            page_index = max(0, min(page_index, im.n_frames - 1))
            im.seek(page_index)
        if im.mode not in ("RGB", "RGBA", "L"):
            try:
                im = im.convert("RGB")
            except Exception:
                im = im.convert("L")
        arr = np.array(im)
        return _to_bgr8(arr)
    except Exception:
        buf = np.frombuffer(image_bytes, dtype=np.uint8)
        arr = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError("Could not decode image bytes.")
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = arr[:, :, ::-1]
        return _to_bgr8(arr)


def encode_png_bytes(image_bgr: np.ndarray, compression: int = 3) -> bytes:
    ok, buf = cv2.imencode(
        ".png", image_bgr, [cv2.IMWRITE_PNG_COMPRESSION, compression]
    )
    if not ok:
        raise RuntimeError("Failed to encode PNG.")
    return buf.tobytes()


# --------------------
# Debug viz helpers
# --------------------
def _draw_box_with_label(
    img: np.ndarray, box, label: str, color=(0, 255, 0)
) -> np.ndarray:
    """Draw rectangle + text label on a copy of img."""
    x1, y1, x2, y2 = map(int, box)
    out = img.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    y0 = max(0, y1 - th - 6)
    cv2.rectangle(out, (x1, y0), (x1 + tw + 8, y0 + th + 6), (0, 0, 0), -1)
    cv2.putText(
        out,
        label,
        (x1 + 4, y0 + th + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def _make_grid(
    images: List[np.ndarray], cols: int = 2, pad: int = 6
) -> Optional[np.ndarray]:
    """Compose images into a grid; resizes each to the minimum height among them."""
    if not images:
        return None
    h = min(im.shape[0] for im in images)
    resized = [
        cv2.resize(im, (int(im.shape[1] * (h / im.shape[0])), h)) for im in images
    ]
    pad_col = np.full((h, pad, 3), 30, np.uint8)
    pad_row = np.full((pad, 1, 3), 30, np.uint8)
    rows = []
    for i in range(0, len(resized), cols):
        row_imgs = []
        for j in range(cols):
            idx = i + j
            if idx < len(resized):
                row_imgs.append(resized[idx])
                if j < cols - 1:
                    row_imgs.append(pad_col)
        if row_imgs:
            row = cv2.hconcat(row_imgs)
            rows.append(row)
            if i + cols < len(resized):
                rows.append(pad_row.repeat(row.shape[1], axis=1))
    return cv2.vconcat(rows) if rows else None


# --------------------
# Rotation utilities
# --------------------
def rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
    """Rotate image by 0, 90, 180, or 270 degrees."""
    angle = angle % 360
    if angle == 0:
        return img.copy()
    elif angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return img.copy()


# --------------------
# YOLO helpers
# --------------------
def _get_model(model_path: str) -> YOLO:
    if model_path not in _MODEL_CACHE:
        _MODEL_CACHE[model_path] = YOLO(model_path, task="detect", verbose=False)
    return _MODEL_CACHE[model_path]


def detect_with_model(
    model_path: str,
    image: np.ndarray,
    conf_thresh: float,
):
    """
    Run detection on image.

    Returns:
        success: bool
        cropped: Optional[np.ndarray]
        avg_conf: float
        num_boxes: int
        boxes: np.ndarray  (N, 4) in xyxy
        confs: np.ndarray  (N,)
        clss:  np.ndarray  (N,) int class ids or None if unavailable
    """
    h, w = image.shape[:2]
    try:
        model = _get_model(model_path)
        results = model(image, device=DEVICE, conf=conf_thresh, iou=0.6, verbose=False)
    except Exception:
        return False, None, 0.0, 0, np.empty((0, 4)), np.empty(0), None

    pred = results[0]
    if pred.boxes is None or pred.boxes.xyxy is None:
        return False, None, 0.0, 0, np.empty((0, 4)), np.empty(0), None

    xyxy = pred.boxes.xyxy
    confs = pred.boxes.conf
    clss = getattr(pred.boxes, "cls", None)

    if xyxy is None or confs is None:
        return False, None, 0.0, 0, np.empty((0, 4)), np.empty(0), None

    boxes = xyxy.cpu().numpy()
    confs = confs.cpu().numpy()
    clss = clss.cpu().numpy().astype(int) if clss is not None else None

    if boxes.size == 0:
        return False, None, 0.0, 0, boxes, confs, clss

    mask = confs >= conf_thresh
    boxes = boxes[mask] if mask.any() else np.empty((0, 4))
    confs = confs[mask] if mask.any() else np.empty(0)
    clss = clss[mask] if (clss is not None and mask.any()) else clss

    if boxes.shape[0] == 0:
        return False, None, 0.0, 0, boxes, confs, clss

    avg_conf = float(np.mean(confs))
    num_boxes = boxes.shape[0]

    min_x = int(np.clip(np.min(boxes[:, 0]), 0, w))
    min_y = int(np.clip(np.min(boxes[:, 1]), 0, h))
    max_x = int(np.clip(np.max(boxes[:, 2]), 0, w))
    max_y = int(np.clip(np.max(boxes[:, 3]), 0, h))

    cropped = None
    if max_x > min_x and max_y > min_y:
        cropped = image[min_y:max_y, min_x:max_x]

    return True, cropped, avg_conf, num_boxes, boxes, confs, clss


def _boxes_area_xyxy(boxes: np.ndarray) -> np.ndarray:
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    w = np.clip(boxes[:, 2] - boxes[:, 0], 0, None)
    h = np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    return w * h


def _union_coverage(
    boxes: np.ndarray, img_w: int, img_h: int, max_boxes: int = 1000
) -> float:
    """Approximate union area by rasterization (fast for typical doc sizes)."""
    if boxes.shape[0] == 0:
        return 0.0
    # Downscale mask to keep it cheap
    scale = max(1, int(max(img_w, img_h) / 1024))
    mask = np.zeros((img_h // scale + 1, img_w // scale + 1), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes[:max_boxes]:
        xs1, ys1 = int(x1 // scale), int(y1 // scale)
        xs2, ys2 = int(np.ceil(x2 / scale)), int(np.ceil(y2 / scale))
        mask[ys1:ys2, xs1:xs2] = 1
    cov = float(mask.mean())  # [0,1] in the downscaled space
    return cov


def _fragmentation_penalty(areas: np.ndarray) -> float:
    """
    Penalize many tiny boxes. Returns [0..1] where 1 means “very fragmented”.
    Uses coefficient of variation + small-area emphasis.
    """
    if areas.size == 0:
        return 0.0
    mean = areas.mean() + 1e-6
    cv = float(areas.std() / mean)  # high if sizes vary wildly
    small_ratio = float((areas < 0.002 * areas.max() + 1e-6).mean())
    raw = 0.5 * cv + 0.5 * small_ratio
    return np.clip(raw, 0.0, 1.0)


def _class_weights(clss: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Optional: give anchor classes a boost.
    Adjust IDs to match your model (examples shown).
    """
    if clss is None:
        return None
    weights = np.ones_like(clss, dtype=np.float32)
    # e.g., if class 0="title block", 1="table", 2="border"
    anchor_boost = {0: 1.3, 1: 1.2, 2: 1.1}
    for k, v in anchor_boost.items():
        weights[clss == k] *= v
    return weights


def compute_bottom_bias_score(boxes: np.ndarray, img_height: int) -> float:
    """
    Compute a score favoring detections near the bottom of the image.
    Returns average Y-center position normalized to [0, 1], where 1 = bottom.
    """
    if boxes.shape[0] == 0:
        return 0.0

    center_y = (boxes[:, 1] + boxes[:, 3]) / 2.0
    normalized_y = center_y / img_height
    return float(np.mean(normalized_y))


def find_best_rotation(
    image: np.ndarray,
    model_path: str,
    conf_thresh: float,
    prefer_bottom: bool = True,
    bottom_weight: float = 0.2,  # base weight for bottom bias
    conf_weight: float = 3.0,  # LOWER: confidence less dominant
    coverage_weight: float = 8.0,  # coverage still very important
    largestk_weight: float = 3.0,  # LOWER: top-K confidence less dominant
    frag_weight: float = 4.0,  # penalize fragmentation
    k_largest: int = 3,
    tie_margin: float = 0.5,  # prefer 0° if scores are very close
    verbose: bool = True,
    rotation_candidates_debug: Optional[List[Dict[str, float]]] = None,
    rotation_debug_images: Optional[List[np.ndarray]] = None,
) -> Tuple[int, np.ndarray, Optional[np.ndarray]]:
    """
    Try all 4 rotations and pick the best one according to a composite score.

    Score components:
      - union coverage of all boxes
      - avg / top-K confidence (down-weighted a bit)
      - fragmentation penalty
      - optional bottom bias
      - geometry prior based on the largest box:
          * prefers wide boxes (horizontal title block)
          * prefers boxes closer to the bottom

    rotation_candidates_debug (if provided) will receive one dict per angle:
        {
            "angle": 0/90/180/270,
            "score": ...,
            "coverage": ...,
            "topk_conf": ...,
            "avg_conf": ...,
            "frag": ...,
            "bottom": ...,
            "num_boxes": ...,
            "geom_score": ...,
            "aspect_main": ...,
            "center_y_norm": ...,
        }
    """

    def _log(msg: str):
        if verbose:
            tqdm.write(msg)

    angles = [0, 90, 180, 270]
    results = []  # (angle, score, rotated, cropped, num_boxes, avg_conf, geom_score)
    img_h, img_w = image.shape[:2]
    _log(f"🔄 Testing rotations with model: {Path(model_path).name}")

    for angle in angles:
        rotated = rotate_image(image, angle)
        success, cropped, avg_conf, num_boxes, boxes, confs, clss = detect_with_model(
            model_path, rotated, conf_thresh
        )

        if not (success and num_boxes > 0):
            results.append((angle, -1e9, rotated, None, 0, 0.0, 0.0))
            _log(f"  {angle:3d}°: No detections")

            if rotation_candidates_debug is not None:
                rotation_candidates_debug.append(
                    {
                        "angle": float(angle),
                        "score": float(-1e9),
                        "coverage": 0.0,
                        "topk_conf": 0.0,
                        "avg_conf": 0.0,
                        "frag": 0.0,
                        "bottom": 0.0,
                        "num_boxes": 0,
                        "geom_score": 0.0,
                        "aspect_main": 0.0,
                        "center_y_norm": 0.0,
                    }
                )
            if rotation_debug_images is not None:
                ann = rotated.copy()
                cv2.putText(
                    ann,
                    f"{angle}° | no detections",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                rotation_debug_images.append(ann)
            continue

        # Optional class weighting
        cweights = _class_weights(clss)
        if cweights is not None:
            confs_eff = confs * cweights
        else:
            confs_eff = confs

        areas = _boxes_area_xyxy(boxes)
        order = np.argsort(-areas)
        topk = order[: min(k_largest, areas.size)]

        # -------- base components --------
        coverage = _union_coverage(boxes, rotated.shape[1], rotated.shape[0])  # 0..1
        largestk_conf = float(confs_eff[topk].mean()) if topk.size else 0.0
        conf_component = float(np.mean(confs_eff))
        frag = _fragmentation_penalty(areas)
        bottom_score = compute_bottom_bias_score(boxes, rotated.shape[0])  # 0..1

        # -------- geometry prior from largest box --------
        img_h_i, img_w_i = rotated.shape[:2]

        largest_idx = int(np.argmax(areas))
        x1, y1, x2, y2 = boxes[largest_idx]
        bw = max(1.0, float(x2 - x1))
        bh = max(1.0, float(y2 - y1))

        aspect_main = bw / bh  # >1 = horizontal, <1 = vertical
        center_y = 0.5 * (y1 + y2)
        center_y_norm = center_y / img_h_i  # 0 top, 1 bottom

        # Map aspect_main into roughly [-1, 1]:
        #   horizontal (>=4:1) -> ~+1
        #   square (~1:1)      -> ~0
        #   tall (<=0.25:1)    -> ~-1
        aspect_score = float(np.tanh((aspect_main - 1.0) / 2.0))

        # We want: wide box (aspect_score high) AND low-ish on the page
        geom_score = 0.6 * aspect_score + 0.4 * center_y_norm
        # -----------------------------------------------

        # Final score (higher is better)
        score = (
            coverage * coverage_weight
            + largestk_conf * largestk_weight
            + conf_component * conf_weight
            - frag * frag_weight
            + (bottom_score * bottom_weight if prefer_bottom else 0.0)
            + 8.0 * geom_score  # strong geometry influence
        )

        _log(
            f"  {angle:3d}°: boxes={num_boxes}, "
            f"coverage={coverage:.3f}, topK={largestk_conf:.3f}, "
            f"avg_conf={conf_component:.3f}, frag={frag:.2f}, "
            f"bottom={bottom_score:.2f}, geom={geom_score:.2f}, "
            f"aspect_main={aspect_main:.2f} => score={score:.2f}"
        )

        results.append(
            (angle, score, rotated, cropped, num_boxes, conf_component, geom_score)
        )

        if rotation_candidates_debug is not None:
            rotation_candidates_debug.append(
                {
                    "angle": float(angle),
                    "score": float(score),
                    "coverage": float(coverage),
                    "topk_conf": float(largestk_conf),
                    "avg_conf": float(conf_component),
                    "frag": float(frag),
                    "bottom": float(bottom_score),
                    "num_boxes": int(num_boxes),
                    "geom_score": float(geom_score),
                    "aspect_main": float(aspect_main),
                    "center_y_norm": float(center_y_norm),
                }
            )

        if rotation_debug_images is not None:
            label = (
                f"{angle}° | s={score:.2f} | cov={coverage:.2f} | "
                f"geom={geom_score:.2f} | asp={aspect_main:.1f} | n={num_boxes}"
            )
            ann = _draw_box_with_label(rotated, (x1, y1, x2, y2), label)
            rotation_debug_images.append(ann)

    # Pick best; apply 0° hysteresis
    results.sort(key=lambda x: x[1], reverse=True)
    (
        best_angle,
        best_score,
        best_rotated,
        best_cropped,
        best_num,
        best_conf,
        best_geom,
    ) = results[0]

    # If 0° is close in score, prefer it (same as before)
    zero = next((r for r in results if r[0] == 0), None)
    if zero is not None and (best_angle != 0) and (best_score - zero[1] < tie_margin):
        best_angle, best_score, best_rotated, best_cropped = (
            zero[0],
            zero[1],
            zero[2],
            zero[3],
        )

    if best_score < -1e8:
        _log("⚠️  No detections at any rotation, using 0°")
        best_angle, best_rotated, best_cropped = 0, image.copy(), None
    else:
        _log(
            f"✅ Best rotation: {best_angle}° (score={best_score:.2f}, geom={best_geom:.2f})"
        )

    return best_angle, best_rotated, best_cropped


# --------------------
# Core API
# --------------------
def _resolve_model_paths(model_paths: Optional[List[str]]) -> List[str]:
    paths = model_paths if model_paths is not None else MODEL_PATHS
    if not paths:
        raise ValueError(
            "No model paths configured. Pass model_paths or set MODEL_PATHS."
        )
    return paths


def crop_tables_from_bytes(
    image_bytes: bytes,
    conf_thresh: float,
    prefer_bottom: bool = True,
    bottom_weight: float = 0.3,
    verbose: bool = True,
    rotation_candidates_debug: Optional[List[Dict[str, float]]] = None,
    return_rotation_debug: bool = False,
    save_rotation_debug_path: Optional[str] = None,
    model_paths: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Decode bytes, find best rotation using YOLO, then crop.

    Args:
        prefer_bottom: Prefer rotations where table is near bottom
        bottom_weight: Weight for bottom position (0-1)
        rotation_candidates_debug: optional list filled with per-rotation metrics
        return_rotation_debug: if True, also creates a debug grid image
        save_rotation_debug_path: where to save the debug grid (PNG) if enabled
    """

    def _log(msg: str):
        if verbose:
            tqdm.write(msg)

    image = load_image_from_bytes(image_bytes, page_index=0)
    paths = _resolve_model_paths(model_paths)

    debug_images: Optional[List[np.ndarray]] = [] if return_rotation_debug else None

    # Try all rotations with first model
    best_angle, rotated_image, cropped = find_best_rotation(
        image=image,
        model_path=paths[0],
        conf_thresh=conf_thresh,
        prefer_bottom=prefer_bottom,
        bottom_weight=bottom_weight,
        verbose=verbose,
        rotation_candidates_debug=rotation_candidates_debug,
        rotation_debug_images=debug_images,
    )

    # Save debug grid if requested
    if return_rotation_debug and debug_images:
        grid = _make_grid(debug_images)
        if grid is not None and save_rotation_debug_path is not None:
            cv2.imwrite(save_rotation_debug_path, grid)
            _log(f"🖼  Saved rotation debug grid: {save_rotation_debug_path}")

    if cropped is not None:
        _log(f"✅ Using model: {paths[0]}")
        return cropped

    # If first model failed, try others on the best rotation
    _log(f"⚠️  First model found no detections, trying other models at {best_angle}°...")
    for i, mp in enumerate(paths[1:], start=2):
        _log(f"[{i}/{len(paths)}] Trying model: {mp}")
        success, cropped, *_ = detect_with_model(mp, rotated_image, conf_thresh)
        if success and cropped is not None:
            _log(f"✅ Using model: {mp}")
            return cropped

    raise RuntimeError(
        f"No detections found by any model at any rotation. "
        f"Tried {len(paths)} models at 4 rotations each."
    )


def crop_tables_from_bytes_png(
    image_bytes: bytes,
    conf_thresh: float,
    prefer_bottom: bool = True,
    bottom_weight: float = 0.3,
    verbose: bool = True,
    rotation_candidates_debug: Optional[List[Dict[str, float]]] = None,
    return_rotation_debug: bool = False,
    save_rotation_debug_path: Optional[str] = None,
    model_paths: Optional[List[str]] = None,
) -> bytes:
    cropped_bgr = crop_tables_from_bytes(
        image_bytes=image_bytes,
        conf_thresh=conf_thresh,
        prefer_bottom=prefer_bottom,
        bottom_weight=bottom_weight,
        verbose=verbose,
        rotation_candidates_debug=rotation_candidates_debug,
        return_rotation_debug=return_rotation_debug,
        save_rotation_debug_path=save_rotation_debug_path,
        model_paths=model_paths,
    )
    return encode_png_bytes(cropped_bgr)


# --------------------
# Batch processing
# --------------------
def collect_files(input_dir: Path, include_subdirs: bool) -> List[Path]:
    if include_subdirs:
        all_paths = list(input_dir.rglob("*"))
        return [
            p for p in all_paths if p.is_file() and p.suffix.lower() in ALLOWED_EXTS
        ]
    else:
        return [
            p
            for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in ALLOWED_EXTS
        ]


def _process_one_file(
    image_path: Path,
    output_dir: Path,
    *,
    conf_thresh: float,
    prefer_bottom: bool,
    bottom_weight: float,
    verbose: bool,
    rotation_debug: bool,
    model_paths: Optional[List[str]],
) -> None:
    if verbose:
        tqdm.write(f"\n{'='*60}")
        tqdm.write(f"Processing: {image_path.name}")
        tqdm.write("=" * 60)

    image_bytes = image_path.read_bytes()

    debug_path = None
    if rotation_debug:
        debug_path = str(output_dir / f"{image_path.stem}.rotation_debug.png")

    cropped_png = crop_tables_from_bytes_png(
        image_bytes=image_bytes,
        conf_thresh=conf_thresh,
        prefer_bottom=prefer_bottom,
        bottom_weight=bottom_weight,
        verbose=verbose,
        rotation_candidates_debug=None,
        return_rotation_debug=rotation_debug,
        save_rotation_debug_path=debug_path,
        model_paths=model_paths,
    )

    out_path = output_dir / f"{image_path.stem}.crop.png"
    with open(out_path, "wb") as f:
        f.write(cropped_png)

    if verbose:
        tqdm.write(f"✅ Saved: {out_path}")


def process_folder(
    input_dir: str,
    output_dir: str,
    conf_thresh: float = 0.25,
    prefer_bottom: bool = True,
    bottom_weight: float = 0.3,
    verbose: bool = True,
    include_subdirs: bool = False,
    show_progress: bool = True,
    rotation_debug: bool = False,
    model_paths: Optional[List[str]] = None,
) -> Tuple[int, int]:
    inp = Path(input_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not inp.exists() or not inp.is_dir():
        raise ValueError(f"Input directory not found: {input_dir}")

    files = collect_files(inp, include_subdirs)
    total = len(files)

    saved = 0
    failed = 0

    iterator = files
    if show_progress:
        iterator = tqdm(files, desc="Processing images", unit="img")

    for p in iterator:
        try:
            _process_one_file(
                p,
                out,
                conf_thresh=conf_thresh,
                prefer_bottom=prefer_bottom,
                bottom_weight=bottom_weight,
                verbose=verbose,
                rotation_debug=rotation_debug,
                model_paths=model_paths,
            )
            saved += 1
        except RuntimeError as e:
            failed += 1
            tqdm.write(f"❌ Failed: {p.name} | {e}")
        except Exception as e:
            failed += 1
            tqdm.write(f"❌ Error: {p.name} | {e}")

        if show_progress:
            iterator.set_postfix(saved=saved, failed=failed, total=total)

    summary = (
        f"\n{'='*60}\n"
        f"SUMMARY\n"
        f"{'='*60}\n"
        f"Total files:  {total}\n"
        f"Saved crops:  {saved}\n"
        f"Failed:       {failed}\n"
        f"{'='*60}\n"
    )
    tqdm.write(summary)
    return saved, failed


def process_path(
    input_path: str,
    output_dir: str,
    conf_thresh: float = 0.25,
    prefer_bottom: bool = True,
    bottom_weight: float = 0.3,
    verbose: bool = True,
    include_subdirs: bool = False,
    show_progress: bool = True,
    rotation_debug: bool = False,
    model_paths: Optional[List[str]] = None,
    device: Optional[str] = None,
) -> Tuple[int, int]:
    """Process a single image file or every image in a directory."""
    global DEVICE
    if device is not None:
        DEVICE = device

    inp = Path(input_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        raise ValueError(f"Input path not found: {input_path}")

    if inp.is_file():
        if inp.suffix.lower() not in ALLOWED_EXTS:
            raise ValueError(
                f"Unsupported file type: {inp.suffix}. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTS))}"
            )
        _process_one_file(
            inp,
            out,
            conf_thresh=conf_thresh,
            prefer_bottom=prefer_bottom,
            bottom_weight=bottom_weight,
            verbose=verbose,
            rotation_debug=rotation_debug,
            model_paths=model_paths,
        )
        return 1, 0

    return process_folder(
        input_dir=str(inp),
        output_dir=str(out),
        conf_thresh=conf_thresh,
        prefer_bottom=prefer_bottom,
        bottom_weight=bottom_weight,
        verbose=verbose,
        include_subdirs=include_subdirs,
        show_progress=show_progress,
        rotation_debug=rotation_debug,
        model_paths=model_paths,
    )


# --------------------
# CLI
# --------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="YOLO-based auto-rotation: tries all rotations and picks the best."
    )
    ap.add_argument(
        "--in",
        dest="input_path",
        required=False,
        default="input_images",
        help="Input image file or folder.",
    )
    ap.add_argument(
        "--out",
        dest="output_dir",
        required=False,
        default="output_crops",
        help="Output folder for cropped PNGs.",
    )
    ap.add_argument(
        "--conf",
        dest="conf_thresh",
        type=float,
        default=0.25,
        help="Confidence threshold for detections.",
    )
    ap.add_argument(
        "--device", default=DEVICE, help="Device for inference (cpu or cuda:0)."
    )
    ap.add_argument(
        "--no-bottom-bias",
        dest="prefer_bottom",
        action="store_false",
        help="Disable preference for tables at bottom.",
    )
    ap.add_argument(
        "--bottom-weight",
        type=float,
        default=0.3,
        help="Weight for bottom position bias (0-1).",
    )
    ap.add_argument(
        "--include-subdirs",
        action="store_true",
        help="Process files in subfolders recursively.",
    )
    ap.add_argument(
        "--no-progress", action="store_true", help="Disable tqdm progress bar."
    )
    ap.add_argument("--quiet", action="store_true", help="Disable verbose logging.")
    ap.add_argument(
        "--rotation-debug",
        action="store_true",
        help="Save a rotation candidates debug grid PNG per image.",
    )
    ap.add_argument(
        "--model-path",
        dest="model_paths",
        nargs="+",
        default=None,
        help="YOLO model path(s) (.pt or .onnx). First model is used for rotation search.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    global DEVICE
    DEVICE = args.device

    tqdm.write("=" * 60)
    tqdm.write("YOLO Auto-Rotation Table Cropper")
    tqdm.write("=" * 60)
    tqdm.write(f"Input:          {args.input_path}")
    tqdm.write(f"Output dir:      {args.output_dir}")
    tqdm.write(f"Confidence:      {args.conf_thresh}")
    tqdm.write(f"Device:          {DEVICE}")
    tqdm.write(f"Bottom bias:     {args.prefer_bottom}")
    tqdm.write(f"Bottom weight:   {args.bottom_weight}")
    tqdm.write(f"Include subdirs: {args.include_subdirs}")
    tqdm.write(f"Verbose:         {not args.quiet}")
    tqdm.write(f"Rotation debug:  {args.rotation_debug}")
    if args.model_paths:
        tqdm.write(f"Model paths:     {args.model_paths}")
    tqdm.write("=" * 60 + "\n")

    process_path(
        input_path=args.input_path,
        output_dir=args.output_dir,
        conf_thresh=args.conf_thresh,
        prefer_bottom=args.prefer_bottom,
        bottom_weight=args.bottom_weight,
        verbose=not args.quiet,
        include_subdirs=args.include_subdirs,
        show_progress=not args.no_progress,
        rotation_debug=args.rotation_debug,
        model_paths=args.model_paths,
        device=args.device,
    )


if __name__ == "__main__":
    main()
