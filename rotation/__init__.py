# rotation/__init__.py

# 1) All imports at the very top
import argparse as _argparse
import os as _os

import cv2 as _cv2
import pytesseract as _pytesseract  # type: ignore[import-not-found,import-untyped]

from .config import Config
from .rotation import (
    build_parser,
    detect_east_boxes,
    detect_osd_rotation,
    estimate_orientation,
    iter_image_files,
    main,
    process_one_image,
    rotate_if_needed,
    save_image,
    setup_logging,
)

# 2) Re-export for tests to monkeypatch at package level
argparse = _argparse
os = _os
cv2 = _cv2
pytesseract = _pytesseract

__all__ = [
    "setup_logging",
    "detect_east_boxes",
    "estimate_orientation",
    "detect_osd_rotation",
    "rotate_if_needed",
    "save_image",
    "build_parser",
    "iter_image_files",
    "process_one_image",
    "main",
    "argparse",
    "os",
    "cv2",
    "pytesseract",
    "Config",
]
