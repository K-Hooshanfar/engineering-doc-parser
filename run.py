#!/usr/bin/env python3
"""Run table detection + auto-rotation cropping on an image or folder.

Usage:
  python run.py --config configs/run.yaml
  python run.py --input scan.png --output out/ --model-path weights/best.pt
  python run.py --input images/ --output crops/ --model-path weights/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from engineering_doc_parser.table_cropper.cropper import process_path  # noqa: E402


def _run_extraction(cfg: Dict[str, Any], crop_dir: str) -> None:
    from engineering_doc_parser.extraction.qwen_vl import (
        extract_from_directory,
        extraction_config_from_dict,
    )

    extraction_raw = cfg.get("extraction", {})
    if cfg.get("extract") is True:
        extraction_raw = {**extraction_raw, "enabled": True}

    extraction_cfg = extraction_config_from_dict(extraction_raw)
    print()
    print("=" * 60)
    print("Qwen2.5-VL extraction (post-crop)")
    print("=" * 60)
    print(f"  Crops dir:  {crop_dir}")
    print(f"  Model:      {extraction_cfg.model_id}")
    print(f"  Device:     {extraction_cfg.device}")
    print("=" * 60)
    print()

    saved, failed, _ = extract_from_directory(crop_dir, config=extraction_cfg)
    if failed:
        raise SystemExit(
            f"Extraction finished with {failed} failure(s), {saved} markdown file(s) saved."
        )


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Crop tables from images using a trained YOLO model."
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config file (see configs/run.yaml).",
    )
    p.add_argument(
        "--input",
        "-i",
        type=Path,
        default=None,
        help="Input image file or directory.",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output directory for cropped PNGs.",
    )
    p.add_argument(
        "--model-path",
        dest="model_paths",
        nargs="+",
        default=None,
        help="YOLO model path(s) (.pt or .onnx).",
    )
    p.add_argument("--conf", type=float, default=None, help="Detection confidence.")
    p.add_argument(
        "--device", default=None, help="Inference device, e.g. cpu or cuda:0."
    )
    p.add_argument(
        "--no-bottom-bias",
        dest="prefer_bottom",
        action="store_false",
        default=None,
        help="Disable preference for tables near the bottom.",
    )
    p.add_argument("--bottom-weight", type=float, default=None)
    p.add_argument(
        "--include-subdirs",
        action="store_true",
        default=None,
        help="Process images in subfolders (directory input only).",
    )
    p.add_argument(
        "--rotation-debug",
        action="store_true",
        default=None,
        help="Save rotation candidate debug grids.",
    )
    p.add_argument("--quiet", action="store_true", help="Less logging.")
    p.add_argument("--no-progress", action="store_true", help="Disable progress bar.")
    p.add_argument(
        "--extract",
        action="store_true",
        default=None,
        help="Run Qwen2.5-VL extraction on crops after cropping.",
    )
    p.add_argument(
        "--extract-only",
        action="store_true",
        help="Skip cropping; only run Qwen extraction on --input (crop folder).",
    )
    return p


def merge_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if args.config:
        cfg = _load_yaml(args.config)

    if args.input is not None:
        cfg["input"] = str(args.input)
    if args.output is not None:
        cfg["output"] = str(args.output)
    if args.model_paths is not None:
        cfg["model_paths"] = args.model_paths
    if args.conf is not None:
        cfg["conf"] = args.conf
    if args.device is not None:
        cfg["device"] = args.device
    if args.prefer_bottom is not None:
        cfg["prefer_bottom"] = args.prefer_bottom
    if args.bottom_weight is not None:
        cfg["bottom_weight"] = args.bottom_weight
    if args.include_subdirs is not None:
        cfg["include_subdirs"] = args.include_subdirs
    if args.rotation_debug is not None:
        cfg["rotation_debug"] = args.rotation_debug
    if args.quiet:
        cfg["verbose"] = False
    if args.no_progress:
        cfg["show_progress"] = False
    if args.extract is not None:
        extraction = cfg.setdefault("extraction", {})
        extraction["enabled"] = args.extract
    if args.extract_only:
        cfg["extract_only"] = True

    return cfg


def _model_paths_from_cfg(cfg: Dict[str, Any]) -> Optional[List[str]]:
    paths = cfg.get("model_paths") or cfg.get("model_path")
    if paths is None:
        return None
    if isinstance(paths, str):
        return [paths]
    return [str(p) for p in paths]


def main() -> None:
    args = build_parser().parse_args()
    cfg = merge_config(args)

    input_path = cfg.get("input")
    output_dir = cfg.get("output")
    model_paths = _model_paths_from_cfg(cfg)
    extraction = cfg.get("extraction", {})
    extract_enabled = bool(extraction.get("enabled", False))
    extract_only = bool(cfg.get("extract_only", False))

    if extract_only:
        if not input_path:
            raise SystemExit(
                "--extract-only requires --input (directory of crop images)."
            )
        _run_extraction(cfg, input_path)
        return

    if not input_path or not output_dir:
        raise SystemExit(
            "Missing --input and --output. Provide them on the CLI or in --config configs/run.yaml."
        )
    if not model_paths:
        raise SystemExit(
            "Missing model path. Set model_paths in configs/run.yaml or pass --model-path."
        )

    print("=" * 60)
    print("Table cropper")
    print("=" * 60)
    print(f"  Input:       {input_path}")
    print(f"  Output:      {output_dir}")
    print(f"  Models:      {model_paths}")
    print(f"  Confidence:  {cfg.get('conf', 0.25)}")
    print(f"  Device:      {cfg.get('device', 'cpu')}")
    print(f"  Extract:     {extract_enabled}")
    print("=" * 60)
    print()

    saved, failed = process_path(
        input_path=input_path,
        output_dir=output_dir,
        conf_thresh=float(cfg.get("conf", 0.25)),
        prefer_bottom=bool(cfg.get("prefer_bottom", True)),
        bottom_weight=float(cfg.get("bottom_weight", 0.3)),
        verbose=bool(cfg.get("verbose", True)),
        include_subdirs=bool(cfg.get("include_subdirs", False)),
        show_progress=bool(cfg.get("show_progress", True)),
        rotation_debug=bool(cfg.get("rotation_debug", False)),
        model_paths=model_paths,
        device=cfg.get("device"),
    )

    if failed:
        raise SystemExit(
            f"Cropping finished with {failed} failure(s), {saved} crop(s) saved."
        )

    if extract_enabled:
        _run_extraction(cfg, output_dir)


if __name__ == "__main__":
    main()
