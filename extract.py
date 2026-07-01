#!/usr/bin/env python3
"""Run Qwen2.5-VL field extraction on cropped document images.

Usage:
  python extract.py --config configs/extract.yaml
  python extract.py --input output/crops
  python extract.py --input output/crops --image scan.crop.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from engineering_doc_parser.extraction.qwen_vl import (  # noqa: E402
    QwenVLExtractor,
    extract_from_directory,
    extraction_config_from_dict,
)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract document fields from crop images using Qwen2.5-VL."
    )
    p.add_argument("--config", type=Path, default=None, help="YAML config file.")
    p.add_argument(
        "--input",
        "-i",
        type=Path,
        default=None,
        help="Directory of crop images, or a single image file.",
    )
    p.add_argument(
        "--model-id",
        default=None,
        help="Hugging Face model id (default: Qwen/Qwen2.5-VL-7B-Instruct).",
    )
    p.add_argument("--device", default=None, help="Device, e.g. cuda or cpu.")
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument(
        "--prompt-path",
        type=Path,
        default=None,
        help="Path to extraction prompt text file.",
    )
    p.add_argument(
        "--glob-pattern",
        default=None,
        help="Glob for images in a directory (default: *.crop.png).",
    )
    p.add_argument(
        "--flash-attention-2",
        action="store_true",
        default=None,
        help="Enable flash_attention_2 (requires compatible GPU/build).",
    )
    p.add_argument("--quiet", action="store_true", help="Less logging.")
    return p


def merge_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if args.config:
        cfg = _load_yaml(args.config)

    if args.input is not None:
        cfg["input"] = str(args.input)
    if args.model_id is not None:
        cfg["model_id"] = args.model_id
    if args.device is not None:
        cfg["device"] = args.device
    if args.max_new_tokens is not None:
        cfg["max_new_tokens"] = args.max_new_tokens
    if args.prompt_path is not None:
        cfg["prompt_path"] = str(args.prompt_path)
    if args.glob_pattern is not None:
        cfg["glob_pattern"] = args.glob_pattern
    if args.flash_attention_2:
        cfg["flash_attention_2"] = True
    if args.quiet:
        cfg["verbose"] = False

    return cfg


def main() -> None:
    args = build_parser().parse_args()
    cfg = merge_config(args)

    input_path = cfg.get("input")
    if not input_path:
        raise SystemExit(
            "Missing --input. Provide a crop image or directory, or use --config configs/extract.yaml."
        )

    path = Path(input_path)
    extraction_cfg = extraction_config_from_dict(cfg)

    print("=" * 60)
    print("Qwen2.5-VL document extraction")
    print("=" * 60)
    print(f"  Input:     {path}")
    print(f"  Model:     {extraction_cfg.model_id}")
    print(f"  Device:    {extraction_cfg.device}")
    print(f"  Prompt:    {extraction_cfg.prompt_path or 'default'}")
    print("=" * 60)
    print()

    if path.is_file():
        extractor = QwenVLExtractor(extraction_cfg)
        extractor.load()
        md_path = extractor.extract_and_save(path)
        print(f"Saved: {md_path}")
        return

    saved, failed, _ = extract_from_directory(path, config=extraction_cfg)
    if failed:
        raise SystemExit(
            f"Finished with {failed} failure(s), {saved} markdown file(s) saved."
        )


if __name__ == "__main__":
    main()
