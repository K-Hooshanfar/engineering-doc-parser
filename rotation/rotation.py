"""
rotation.py

Detects document orientation using EAST text detector and Tesseract OSD,
and rotates the image if needed.
"""

import os
import re
from typing import List, Tuple
import cv2
import numpy as np
import pytesseract

from config import Config

# Paths
IMAGE_PATH: str = "test_images/3.png"
EAST_MODEL_PATH: str = Config.EAST_TEXT_DETECTOR
RESULT_DIR: str = "result"


def detect_east_boxes(
        image: np.ndarray,
        net: cv2.dnn_Net,
        conf_thresh: float = 0.5,
        nms_thresh: float = 0.4
) -> List[Tuple[int, int, int, int]]:
    """
    Detects text bounding boxes in an image using the EAST model (vectorized).

    This version function Runs the model, then uses NumPy
    to find all positions above the confidence threshold in one go,
    compute the rotated box corners with array ops, and finally applies
    NMS in a single pass.

    Args:
        image (np.ndarray): Input image in which to detect text.
        net (cv2.dnn_Net): Pre-loaded EAST text detection model.
        conf_thresh (float): Confidence threshold to filter weak detections.
        nms_thresh (float): Non-maximum suppression threshold.

    Returns:
        List[Tuple[int, int, int, int]]: A list of bounding boxes
        (startX, startY, endX, endY) in the original image’s coordinate space.
    """
    H, W = image.shape[:2]
    newW, newH = 320, 320
    rW, rH = W / newW, H / newH

    # 1) Prepare blob & run
    blob = cv2.dnn.blobFromImage(
        image, 1.0, (newW, newH),
        (123.68, 116.78, 103.94),
        swapRB=True, crop=False
    )
    net.setInput(blob)
    scores, geometry = net.forward([
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
    ])

    # 2) Vectorized extraction of all positions above conf_thresh
    scores = scores[0, 0]  # shape (rows, cols)
    geom = geometry[0]  # shape (5, rows, cols)
    ys, xs = np.where(scores > conf_thresh)  # all y,x where we have enough confidence
    if len(xs) == 0:
        return []

    # offsets in the resized (320×320) space
    offset_x = xs * 4.0
    offset_y = ys * 4.0

    # geometry components at those positions
    g0 = geom[0, ys, xs]
    g1 = geom[1, ys, xs]
    g2 = geom[2, ys, xs]
    g3 = geom[3, ys, xs]
    angles = geom[4, ys, xs]

    # compute rectangle corners in resized space
    cos_a, sin_a = np.cos(angles), np.sin(angles)
    h = g0 + g2
    w = g1 + g3

    end_x = offset_x + cos_a * g1 + sin_a * g2
    end_y = offset_y - sin_a * g1 + cos_a * g2
    start_x = end_x - w
    start_y = end_y - h

    # stack into an array of [startX, startY, endX, endY]
    rects_resized = np.stack([start_x, start_y, end_x, end_y], axis=1)
    rects_int = rects_resized.round().astype(int).tolist()
    confidences = scores[ys, xs].tolist()

    # 3) Apply NMS (on the resized coords)
    indices = cv2.dnn.NMSBoxes(rects_int, confidences, conf_thresh, nms_thresh)
    idxs = np.array(indices).flatten().astype(int) if len(indices) else np.array([], dtype=int)
    if idxs.size == 0:
        return []

    # 4) Scale chosen boxes back to original image size
    rects_resized[:, [0, 2]] *= rW
    rects_resized[:, [1, 3]] *= rH
    final_boxes = rects_resized[idxs].round().astype(int).tolist()

    return final_boxes


def estimate_orientation(
        boxes: List[Tuple[int, int, int, int]],
        ratio_thresh: float = 1.2
) -> Tuple[int, int]:
    """
    Estimates the number of horizontal and vertical text boxes to infer document orientation.

    Args:
        boxes (List[Tuple[int, int, int, int]]): List of bounding boxes.
        ratio_thresh (float): Threshold ratio to classify a box as horizontal or vertical.

    Returns:
        Tuple[int, int]: Counts of horizontal and vertical boxes.
    """
    horiz = 0
    vert = 0
    for (x1, y1, x2, y2) in boxes:
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        if w / h > ratio_thresh:
            horiz += 1
        elif h / w > ratio_thresh:
            vert += 1
        else:
            # neither clearly horizontal nor vertical
            continue
    return horiz, vert


def detect_osd_rotation(image: np.ndarray) -> int:
    """
    Uses Tesseract's Orientation and Script Detection (OSD) to detect rotation angle.

    Args:
        image (np.ndarray): Input image.

    Returns:
        int: Rotation angle in degrees (0, 90, 180, or 270).
    """
    osd = pytesseract.image_to_osd(image)
    match = re.search(r'Rotate: (\d+)', osd)
    if match:
        return int(match.group(1))
    else:
        return 0


def rotate_if_needed(
        img: np.ndarray,
        boxes: List[Tuple[int, int, int, int]]
) -> Tuple[np.ndarray, bool, str]:
    """
    Determines if the image needs rotation and applies the best rotation.

    The logic is:
      1. If Tesseract OSD suggests 90, 180, or 270°, apply it.
      2. Otherwise (including 0°), fall back to an EAST-based heuristic:
         count vertical vs. horizontal boxes—if vertical dominate, rotate 90° CW.

    Args:
        img (np.ndarray): Original image.
        boxes (List[Tuple[int, int, int, int]]): Text bounding boxes.

    Returns:
        Tuple[np.ndarray, bool, str]:
          - Rotated (or original) image
          - Whether a rotation was applied
          - Decision message
    """
    angle = detect_osd_rotation(img)

    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), True, "Rotated 90° CW"
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180), True, "Rotated 180°"
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), True, "Rotated 90° CCW"
    else:
        # angle == 0 (or unexpected), fall back to EAST heuristic
        horiz, vert = estimate_orientation(boxes)
        if vert > horiz:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), True, "Rotated 90° CW (EAST fallback)"
        else:
            return img, False, "No orientation needed"


def save_image(img: np.ndarray, output_path: str) -> None:
    """
    Saves an image to disk, preserving quality based on file extension.

    Args:
        img (np.ndarray): Image to save.
        output_path (str): Destination file path.
    """
    ext = os.path.splitext(output_path)[1].lower()
    if ext in ['.jpg', '.jpeg']:
        params = [cv2.IMWRITE_JPEG_QUALITY, 100]
    elif ext == '.png':
        params = [cv2.IMWRITE_PNG_COMPRESSION, 0]
    else:
        # preserve default params for other formats
        params = []
    cv2.imwrite(output_path, img, params)


def main() -> None:
    """
    Main function to detect and correct document orientation.
    """
    if not os.path.isfile(IMAGE_PATH):
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")
    if not os.path.isfile(EAST_MODEL_PATH):
        raise FileNotFoundError(f"EAST model not found: {EAST_MODEL_PATH}")

    os.makedirs(RESULT_DIR, exist_ok=True)

    img = cv2.imread(IMAGE_PATH)
    net = cv2.dnn.readNet(EAST_MODEL_PATH)

    boxes = detect_east_boxes(img, net)
    oriented_img, changed, decision = rotate_if_needed(img, boxes)

    if changed:
        base, ext = os.path.splitext(os.path.basename(IMAGE_PATH))
        out_name = f"{base}_rotated{ext}"
        out_path = os.path.join(RESULT_DIR, out_name)
        save_image(oriented_img, out_path)
        print(decision)
        print(f"Saved rotated → {out_path}")
    else:
        print(decision)


if __name__ == "__main__":
    main()
