from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pytest

from train.focr_table_detector.dataset import SplitConfig, prepare_dataset


def make_dummy_yolo(tmp: Path, n: int) -> Tuple[Path, Path]:
    """Create a minimal YOLO-style dataset tree for tests.

    Builds:
        tmp/
          images/ img_0.jpg, ..., img_{n-1}.jpg
          labels/ img_1.txt, img_2.txt, ... (about 2/3 have labels)

    Notes:
        - Image bytes are dummy placeholders; only filesystem IO is exercised.
        - Every 3rd image (i % 3 == 0) intentionally lacks a label to trigger
          code paths that warn/handle missing annotations.

    Args:
        tmp: Temporary root directory to create the dataset in.
        n: Number of image files to generate.

    Returns:
        A tuple (images_dir, labels_dir) pointing to the generated folders.
    """
    images = tmp / "images"
    labels = tmp / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    # create n image/label pairs (some without labels)
    for i in range(n):
        (images / f"img_{i}.jpg").write_bytes(
            b"fake-jpeg"
        )  # not a real image; good enough for IO test
        if i % 3 != 0:  # leave some without labels to exercise warning path
            (labels / f"img_{i}.txt").write_text(
                "0 0.5 0.5 0.5 0.5\n", encoding="utf-8"
            )
    return images, labels


def test_prepare_dataset_copies_files(tmp_path: Path) -> None:
    """Copies files into split folders and returns counts summing to total.

    Asserts:
        - Sum of returned (n_train, n_val, n_test) equals number of inputs.
        - `train/valid/test` directories (with images/labels subfolders) exist.
        - With a fixed seed, the split is roughly ordered: train ≥ val ≥ test.
    """
    src_images, src_labels = make_dummy_yolo(tmp_path / "src", 20)
    dest = tmp_path / "out"

    n_train, n_val, n_test = prepare_dataset(
        source_images=src_images,
        source_labels=src_labels,
        dest_base=dest,
        cfg=SplitConfig(train=0.7, val=0.2, test=0.1),
        seed=42,
    )

    assert n_train + n_val + n_test == 20
    # Check folders exist
    for split in ("train", "valid", "test"):
        assert (dest / split / "images").exists()
        assert (dest / split / "labels").exists()

    # Rough distribution checks (deterministic with seed=42)
    assert n_train >= n_val >= n_test


@pytest.mark.parametrize("ratios", [(0.7, 0.2, 0.1), (7, 2, 1)])
def test_split_normalization(tmp_path: Path, ratios) -> None:
    """Accepts both proportions and weights for split ratios.

    Parameterized Cases:
        - (0.7, 0.2, 0.1): direct proportions
        - (7, 2, 1): weights that normalize to the same proportions

    Asserts:
        The sum of (n_train, n_val, n_test) equals the number of inputs.
    """
    src_images, src_labels = make_dummy_yolo(tmp_path / "src", 10)
    dest = tmp_path / "out"
    train, val, test = ratios
    n_train, n_val, n_test = prepare_dataset(
        source_images=src_images,
        source_labels=src_labels,
        dest_base=dest,
        cfg=SplitConfig(train=train, val=val, test=test),
        seed=0,
    )
    assert n_train + n_val + n_test == 10
