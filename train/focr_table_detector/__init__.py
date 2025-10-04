"""Dataset utilities for the FOCR table detector.

Exposes:
- SplitConfig: configuration for dataset locations/splits.
- prepare_dataset: builds/validates splits and returns paths/metadata used by training.
"""

from train.focr_table_detector.dataset import SplitConfig, prepare_dataset

__all__ = ["SplitConfig", "prepare_dataset", "dataset"]
