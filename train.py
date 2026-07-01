#!/usr/bin/env python3
"""End-to-end YOLO table-detector training: dataset prep + Ultralytics train.

Usage:
  python train.py --config configs/train.yaml
  python train.py --source-images data/images --source-labels data/labels --dest-base dataset/out
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from engineering_doc_parser.table_detector.dataset import (  # noqa: E402
    SplitConfig,
    prepare_dataset,
)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_data_yaml(
    dest_base: Path,
    output_path: Path,
    *,
    nc: int = 1,
    names: Optional[list] = None,
) -> Path:
    """Write an Ultralytics data YAML pointing at a prepared dataset."""
    dest_base = dest_base.resolve()
    data = {
        "train": str(dest_base / "train" / "images"),
        "val": str(dest_base / "valid" / "images"),
        "test": str(dest_base / "test" / "images"),
        "nc": nc,
        "names": names or ["table"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    return output_path


def run_prepare(cfg: Dict[str, Any]) -> tuple[int, int, int]:
    dataset = cfg["dataset"]
    source_images = Path(dataset["source_images"])
    source_labels = Path(dataset["source_labels"])
    dest_base = Path(dataset["dest_base"])

    split = dataset.get("split", {})
    split_cfg = SplitConfig(
        train=float(split.get("train", 0.7)),
        val=float(split.get("val", 0.2)),
        test=float(split.get("test", 0.1)),
    )
    seed = int(dataset.get("seed", 42))

    print("=" * 60)
    print("Step 1/2: Preparing dataset splits")
    print("=" * 60)
    print(f"  Source images: {source_images}")
    print(f"  Source labels: {source_labels}")
    print(f"  Destination:   {dest_base}")
    print(
        f"  Split:         train={split_cfg.train}, val={split_cfg.val}, test={split_cfg.test}"
    )
    print()

    counts = prepare_dataset(
        source_images=source_images,
        source_labels=source_labels,
        dest_base=dest_base,
        cfg=split_cfg,
        seed=seed,
    )
    print(f"Dataset ready: train={counts[0]}, val={counts[1]}, test={counts[2]}")
    return counts


def run_train(cfg: Dict[str, Any]) -> Path:
    dataset = cfg["dataset"]
    training = cfg.get("training", {})
    dest_base = Path(dataset["dest_base"])

    data_yaml = Path(dataset.get("data_yaml", dest_base / "data.yaml"))
    if not data_yaml.exists():
        write_data_yaml(
            dest_base,
            data_yaml,
            nc=int(dataset.get("nc", 1)),
            names=dataset.get("names", ["table"]),
        )
        print(f"Wrote data config: {data_yaml}")

    print()
    print("=" * 60)
    print("Step 2/2: Training YOLO model")
    print("=" * 60)
    print(f"  Data:    {data_yaml}")
    print(f"  Model:   {training.get('model', 'yolov8n.pt')}")
    print(f"  Epochs:  {training.get('epochs', 100)}")
    print(f"  Batch:   {training.get('batch', 8)}")
    print(f"  Device:  {training.get('device', 0)}")
    print()

    model = YOLO(training.get("model", "yolov8n.pt"))
    model.train(
        data=str(data_yaml),
        epochs=int(training.get("epochs", 100)),
        batch=int(training.get("batch", 8)),
        imgsz=int(training.get("imgsz", 640)),
        device=training.get("device", 0),
        workers=int(training.get("workers", 0)),
        project=training.get("project", "runs/detect"),
        name=training.get("name", "table_train"),
        patience=int(training.get("patience", 50)),
        save=True,
    )

    best_pt = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print()
    print("=" * 60)
    print("Training complete")
    print("=" * 60)
    print(f"  Best weights: {best_pt}")
    print(f"  Run folder:   {model.trainer.save_dir}")
    print()
    print("Use this model in run.py:")
    print(f"  model_paths: [{best_pt}]")
    return best_pt


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Prepare a YOLO dataset and train a table detector."
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config file (see configs/train.yaml).",
    )
    p.add_argument(
        "--source-images",
        type=Path,
        default=None,
        help="Folder with source images (overrides config).",
    )
    p.add_argument(
        "--source-labels",
        type=Path,
        default=None,
        help="Folder with YOLO .txt labels (overrides config).",
    )
    p.add_argument(
        "--dest-base",
        type=Path,
        default=None,
        help="Output dataset root (overrides config).",
    )
    p.add_argument("--train-split", type=float, default=None, help="Train ratio.")
    p.add_argument("--val-split", type=float, default=None, help="Val ratio.")
    p.add_argument("--test-split", type=float, default=None, help="Test ratio.")
    p.add_argument("--seed", type=int, default=None, help="Random seed.")
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Base model for training, e.g. yolov8n.pt or path/to/best.pt.",
    )
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--device", default=None, help="Training device, e.g. 0 or cpu.")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--project", type=str, default=None)
    p.add_argument("--name", type=str, default=None)
    p.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip dataset preparation (use existing dest-base layout).",
    )
    p.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only prepare the dataset, do not train.",
    )
    return p


def merge_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {"dataset": {}, "training": {}}
    if args.config:
        cfg = _load_yaml(args.config)

    dataset = cfg.setdefault("dataset", {})
    training = cfg.setdefault("training", {})

    if args.source_images:
        dataset["source_images"] = str(args.source_images)
    if args.source_labels:
        dataset["source_labels"] = str(args.source_labels)
    if args.dest_base:
        dataset["dest_base"] = str(args.dest_base)
    if (
        args.train_split is not None
        or args.val_split is not None
        or args.test_split is not None
    ):
        split = dataset.setdefault("split", {})
        if args.train_split is not None:
            split["train"] = args.train_split
        if args.val_split is not None:
            split["val"] = args.val_split
        if args.test_split is not None:
            split["test"] = args.test_split
    if args.seed is not None:
        dataset["seed"] = args.seed

    if args.model:
        training["model"] = args.model
    if args.epochs is not None:
        training["epochs"] = args.epochs
    if args.batch is not None:
        training["batch"] = args.batch
    if args.imgsz is not None:
        training["imgsz"] = args.imgsz
    if args.device is not None:
        training["device"] = args.device
    if args.workers is not None:
        training["workers"] = args.workers
    if args.project:
        training["project"] = args.project
    if args.name:
        training["name"] = args.name

    return cfg


def main() -> None:
    args = build_parser().parse_args()
    cfg = merge_config(args)

    dataset = cfg.get("dataset", {})
    if not args.skip_prepare:
        required = ("source_images", "source_labels", "dest_base")
        missing = [k for k in required if k not in dataset]
        if missing:
            raise SystemExit(
                "Missing dataset settings: "
                + ", ".join(missing)
                + ". Provide --config configs/train.yaml or the CLI flags."
            )
        run_prepare(cfg)
        write_data_yaml(
            Path(dataset["dest_base"]),
            Path(dataset.get("data_yaml", Path(dataset["dest_base"]) / "data.yaml")),
            nc=int(dataset.get("nc", 1)),
            names=dataset.get("names", ["table"]),
        )
    elif "dest_base" not in dataset:
        raise SystemExit("--skip-prepare requires dest_base in config.")

    if args.prepare_only:
        print("Prepare-only mode: skipping training.")
        return

    run_train(cfg)


if __name__ == "__main__":
    main()
