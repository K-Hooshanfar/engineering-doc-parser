from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.7
    val: float = 0.2
    test: float = 0.1

    def normalized(self) -> "SplitConfig":
        total = self.train + self.val + self.test
        if total <= 0:
            raise ValueError("Split ratios must be positive.")
        return SplitConfig(self.train / total, self.val / total, self.test / total)


def list_images(source_images: Path) -> List[Path]:
    """List image files under a directory (jpg/jpeg/png).

    Args:
        source_images: Directory containing images.

    Returns:
        List of image file paths.
    """
    exts = {".jpg", ".jpeg", ".png"}
    return [
        p for p in source_images.iterdir() if p.suffix.lower() in exts and p.is_file()
    ]


def split_indices(
    n_total: int, cfg: SplitConfig, seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """Compute randomized indices for train/val/test splits.

    Args:
        n_total: Total number of items.
        cfg: Split configuration (ratios).
        seed: RNG seed for reproducibility.

    Returns:
        Tuple of (train_idx, val_idx, test_idx).
    """
    if n_total < 1:
        return [], [], []

    cfg = cfg.normalized()
    rng = random.Random(seed)
    indices = list(range(n_total))
    rng.shuffle(indices)

    n_train = int(n_total * cfg.train)
    n_val = int(n_total * cfg.val)
    n_test = n_total - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return train_idx, val_idx, test_idx


def copy_pair(
    image: Path, src_labels: Path, dst_images: Path, dst_labels: Path
) -> None:
    """Copy an image and its matching YOLO label (if present)."""
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    shutil.copy2(image, dst_images / image.name)

    label = src_labels / (image.stem + ".txt")
    if label.exists():
        shutil.copy2(label, dst_labels / label.name)
    else:
        print(f"Warning: Label file {label.name} does not exist.")


def materialize_split(
    images: Iterable[Path],
    src_labels: Path,
    dest_base: Path,
    split_name: str,
) -> None:
    """Copy a split's images/labels into the YOLO folder layout."""
    dst_img = dest_base / split_name / "images"
    dst_lbl = dest_base / split_name / "labels"
    for img in images:
        copy_pair(img, src_labels, dst_img, dst_lbl)


def prepare_dataset(
    source_images: Path,
    source_labels: Path,
    dest_base: Path,
    cfg: SplitConfig = SplitConfig(),
    seed: int = 42,
) -> Tuple[int, int, int]:
    """Prepare YOLO dataset folders and copy images/labels.

    Mirrors the user's original script but with typing + structure.

    Args:
        source_images: Directory with input images.
        source_labels: Directory with YOLO .txt labels.
        dest_base: Root of the `dataset/{train,val,test}/{images,labels}` layout.
        cfg: Split ratios (train/val/test).
        seed: RNG seed.

    Returns:
        Tuple of counts for (n_train, n_val, n_test).
    """
    imgs = list_images(source_images)
    n_total = len(imgs)
    train_idx, val_idx, test_idx = split_indices(n_total, cfg, seed)

    # Pick images by indices
    train_imgs = [imgs[i] for i in train_idx]
    val_imgs = [imgs[i] for i in val_idx]
    test_imgs = [imgs[i] for i in test_idx]

    materialize_split(train_imgs, source_labels, dest_base, "train")
    materialize_split(val_imgs, source_labels, dest_base, "valid")
    materialize_split(test_imgs, source_labels, dest_base, "test")

    return len(train_imgs), len(val_imgs), len(test_imgs)
