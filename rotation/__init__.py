"""Public API for the `rotation` package.

This module re-exports the core functions from `.rotation` so that
`import rotation` exposes a flat, user-friendly API (and so tests that
import `rotation` hit the real implementation).
"""

from .rotation import (
    setup_logging,
    detect_east_boxes,
    estimate_orientation,
    detect_osd_rotation,
    rotate_if_needed,
    save_image,
    build_parser,
    iter_image_files,
    process_one_image,
    main,
)

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
]
