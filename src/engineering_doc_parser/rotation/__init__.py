"""EasyOCR-based document rotation correction."""

from engineering_doc_parser.rotation.core import (
    autorotate_and_save_improved,
    autorotate_folder,
    calculate_orientation_score,
    get_bbox_dimensions,
    parse_args,
    parse_osd,
    predict_rotation_improved,
    rotate_image,
    tesseract_osd_angle_conf,
)

__all__ = [
    "rotate_image",
    "parse_osd",
    "tesseract_osd_angle_conf",
    "get_bbox_dimensions",
    "calculate_orientation_score",
    "predict_rotation_improved",
    "autorotate_and_save_improved",
    "autorotate_folder",
    "parse_args",
]
