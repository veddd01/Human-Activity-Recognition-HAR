"""
Data loading and preprocessing for the UCI HAR dataset.

This module provides two public functions:

* :func:`download_and_extract_data` — downloads the raw ZIP from the UCI
  repository and extracts it into ``data/``.
* :func:`load_data` — reads the pre-split train/test files and returns
  NumPy-friendly DataFrames ready for model training.

All paths are resolved relative to the project root so the module works
regardless of the caller's working directory.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import requests

from src.utils import get_project_root

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_URL: str = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)

# Resolved once — safe to cache at module level.
_PROJECT_ROOT: Path = get_project_root()
DATA_DIR: Path = _PROJECT_ROOT / "data"
EXTRACT_DIR: Path = DATA_DIR / "UCI_HAR_Dataset"
DATASET_DIR: Path = EXTRACT_DIR / "UCI HAR Dataset"

# Type alias for the return value of load_data()
LoadDataResult = Tuple[
    pd.DataFrame,   # X_train
    pd.Series,       # y_train
    pd.DataFrame,   # X_test
    pd.Series,       # y_test
    pd.DataFrame,   # activity_labels
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_and_extract_data(data_dir: str | Path | None = None) -> Path:
    """Download and extract the UCI HAR dataset.

    If the dataset is already present on disk the download is skipped.

    Parameters
    ----------
    data_dir : str | Path | None, optional
        Override for the ``data/`` directory.  Defaults to
        ``<project_root>/data``.

    Returns
    -------
    Path
        Path to the extracted dataset root
        (``data/UCI_HAR_Dataset/UCI HAR Dataset``).
    """
    data_path = Path(data_dir) if data_dir else DATA_DIR
    data_path.mkdir(parents=True, exist_ok=True)

    zip_path = data_path / "human_activity_recognition.zip"
    extract_path = data_path / "UCI_HAR_Dataset"

    if extract_path.exists():
        logger.info("Dataset already exists at %s — skipping download.", extract_path)
        return extract_path / "UCI HAR Dataset"

    # ---- Download outer zip --------------------------------------------------
    logger.info("Downloading dataset from %s …", DATASET_URL)
    response = requests.get(DATASET_URL, stream=True, timeout=120)
    response.raise_for_status()

    total_bytes = 0
    with open(zip_path, "wb") as fh:
        for chunk in response.iter_content(chunk_size=8192):
            fh.write(chunk)
            total_bytes += len(chunk)
    logger.info("Downloaded %.2f MB.", total_bytes / (1024 * 1024))

    # ---- Extract outer zip ---------------------------------------------------
    logger.info("Extracting outer zip to %s …", data_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_path)

    # ---- Handle inner zip (UCI HAR Dataset.zip) if present -------------------
    inner_zip = data_path / "UCI HAR Dataset.zip"
    if inner_zip.exists():
        logger.info("Extracting inner zip …")
        with zipfile.ZipFile(inner_zip, "r") as zf:
            zf.extractall(extract_path)
        inner_zip.unlink()

    final_path = extract_path / "UCI HAR Dataset"
    logger.info("Dataset ready at %s", final_path)
    return final_path


def load_data(data_dir: str | Path | None = None) -> LoadDataResult:
    """Load the UCI HAR train/test split and return DataFrames.

    Parameters
    ----------
    data_dir : str | Path | None, optional
        Path to the ``UCI HAR Dataset`` directory.  Defaults to
        ``data/UCI_HAR_Dataset/UCI HAR Dataset`` under the project root.

    Returns
    -------
    X_train : pd.DataFrame
        Training features (7352 × 561).
    y_train : pd.Series
        Training labels (0-indexed, 0–5).
    X_test : pd.DataFrame
        Test features (2947 × 561).
    y_test : pd.Series
        Test labels (0-indexed, 0–5).
    activity_labels : pd.DataFrame
        DataFrame with columns ``label_id``, ``activity_name``, ``label``.
    """
    data_path = Path(data_dir) if data_dir else DATASET_DIR
    logger.info("Loading data from %s", data_path)

    # ---- Feature names -------------------------------------------------------
    features_file = data_path / "features.txt"
    features_df = pd.read_csv(
        features_file, sep=r"\s+", header=None, usecols=[1], engine="python",
    )
    feature_names: list[str] = features_df[1].tolist()

    # De-duplicate feature names (e.g. repeated bandsEnergy entries)
    name_series = pd.Series(feature_names)
    is_dup = name_series.duplicated(keep=False)
    name_series[is_dup] = (
        name_series[is_dup]
        + "_"
        + name_series.groupby(name_series).cumcount().astype(str)
    )
    feature_names = name_series.tolist()
    logger.debug("Loaded %d feature names (%d deduplicated).", len(feature_names), is_dup.sum())

    # ---- Activity labels -----------------------------------------------------
    activity_labels = pd.read_csv(
        data_path / "activity_labels.txt",
        sep=r"\s+",
        header=None,
        names=["label_id", "activity_name"],
        engine="python",
    )
    activity_labels["label"] = activity_labels["label_id"] - 1

    # ---- Train set -----------------------------------------------------------
    train_dir = data_path / "train"
    X_train = pd.read_csv(
        train_dir / "X_train.txt", sep=r"\s+", header=None,
        names=feature_names, engine="python",
    )
    y_train = pd.read_csv(
        train_dir / "y_train.txt", sep=r"\s+", header=None,
        names=["label_id"], engine="python",
    )

    # ---- Test set ------------------------------------------------------------
    test_dir = data_path / "test"
    X_test = pd.read_csv(
        test_dir / "X_test.txt", sep=r"\s+", header=None,
        names=feature_names, engine="python",
    )
    y_test = pd.read_csv(
        test_dir / "y_test.txt", sep=r"\s+", header=None,
        names=["label_id"], engine="python",
    )

    # 0-index labels (1-6 → 0-5)
    y_train_labels = y_train["label_id"] - 1
    y_test_labels = y_test["label_id"] - 1

    logger.info(
        "Data loaded — X_train: %s, X_test: %s, classes: %d",
        X_train.shape, X_test.shape, len(activity_labels),
    )

    return X_train, y_train_labels, X_test, y_test_labels, activity_labels


def load_raw_signals(data_dir: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load raw 9-channel inertial signals for train and test sets.

    Returns
    -------
    X_train_signals : np.ndarray
        Shape (7352, 128, 9)
    X_test_signals : np.ndarray
        Shape (2947, 128, 9)
    """
    data_path = Path(data_dir) if data_dir else DATASET_DIR

    signal_files = [
        "body_acc_x_", "body_acc_y_", "body_acc_z_",
        "body_gyro_x_", "body_gyro_y_", "body_gyro_z_",
        "total_acc_x_", "total_acc_y_", "total_acc_z_",
    ]

    def _load_split(split: str) -> np.ndarray:
        split_dir = data_path / split / "Inertial Signals"
        signals = []
        for stem in signal_files:
            file_path = split_dir / f"{stem}{split}.txt"
            df = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")
            signals.append(df.values)
        # Stack along channel axis -> (N, 128, 9)
        return np.transpose(np.array(signals), (1, 2, 0))

    X_train_raw = _load_split("train")
    X_test_raw = _load_split("test")

    logger.info("Raw signals loaded — Train shape: %s, Test shape: %s", X_train_raw.shape, X_test_raw.shape)
    return X_train_raw, X_test_raw



# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.utils import setup_logging

    setup_logging()
    download_and_extract_data()
    X_tr, y_tr, X_te, y_te, labels = load_data()
    print(f"Train: {X_tr.shape}, Test: {X_te.shape}")
    print(f"Activities:\n{labels}")
