"""Public package interface for the `rotation` toolkit.

This module:
- Re-exports selected functions from :mod:`rotation.rotation` so tests can
  monkeypatch them at the *package* level (e.g., `rotation.detect_east_boxes`).
- Re-exports third-party modules (`argparse`, `os`, `cv2`, `pytesseract`) under
  the same names for test stubbing.
- Exposes :class:`Config` for configuration defaults/convenience.
"""

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
