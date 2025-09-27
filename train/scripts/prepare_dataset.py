from __future__ import annotations

import argparse
from pathlib import Path

from train.focr_table_detector.dataset import SplitConfig, prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset splits.")
    parser.add_argument(
        "--source-images",
        type=Path,
        required=True,
        help="Path to source images directory.",
    )
    parser.add_argument(
        "--source-labels",
        type=Path,
        required=True,
        help="Path to source labels directory.",
    )
    parser.add_argument(
        "--dest-base",
        type=Path,
        required=True,
        help="Destination base (e.g., dataset_3408_1/dataset)",
    )
    parser.add_argument(
        "--train", type=float, default=0.7, help="Train split ratio (default: 0.7)"
    )
    parser.add_argument(
        "--val", type=float, default=0.2, help="Validation split ratio (default: 0.2)"
    )
    parser.add_argument(
        "--test", type=float, default=0.1, help="Test split ratio (default: 0.1)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    args = parser.parse_args()
    cfg = SplitConfig(train=args.train, val=args.val, test=args.test)

    n_train, n_val, n_test = prepare_dataset(
        source_images=args.source_images,
        source_labels=args.source_labels,
        dest_base=args.dest_base,
        cfg=cfg,
        seed=args.seed,
    )
    print(
        f"Dataset split completed successfully! train={n_train}, val={n_val}, test={n_test}"
    )


if __name__ == "__main__":
    main()
