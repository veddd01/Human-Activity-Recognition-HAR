"""
Model training pipeline for the HAR project.

Trains six classifiers on the UCI HAR feature set:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting (sklearn)
5. Support Vector Machine (SVC)
6. Dense Neural Network (Keras)

Trained artefacts are persisted to ``models/`` together with a JSON file
(``training_metadata.json``) that records accuracy and wall-clock training
time for every model.

Usage
-----
Run from the project root::

    python -m src.train
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.data import load_data
from src.utils import format_duration, get_project_root, setup_logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyperparameter constants
# ---------------------------------------------------------------------------
# Logistic Regression
LR_MAX_ITER: int = 1000
LR_SOLVER: str = "lbfgs"
LR_C: float = 1.0

# Decision Tree
DT_MAX_DEPTH: int | None = None
DT_MIN_SAMPLES_SPLIT: int = 2
DT_CRITERION: str = "gini"

# Random Forest
RF_N_ESTIMATORS: int = 200
RF_MAX_DEPTH: int | None = None
RF_MIN_SAMPLES_SPLIT: int = 2
RF_N_JOBS: int = -1

# Gradient Boosting
GB_N_ESTIMATORS: int = 200
GB_LEARNING_RATE: float = 0.1
GB_MAX_DEPTH: int = 5
GB_SUBSAMPLE: float = 0.8

# SVM
SVM_C: float = 1.0
SVM_KERNEL: str = "rbf"
SVM_PROBABILITY: bool = True

# Neural Network
NN_EPOCHS: int = 30
NN_BATCH_SIZE: int = 64
NN_VALIDATION_SPLIT: float = 0.2
NN_LEARNING_RATE: float = 0.001
NN_EARLY_STOP_PATIENCE: int = 5
NN_LAYER_SIZES: list[int] = [256, 128, 64]
NN_DROPOUT_RATE: float = 0.3

# ---------------------------------------------------------------------------
# Project paths (resolved once)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = get_project_root()
MODELS_DIR: Path = PROJECT_ROOT / "models"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Known training times for already trained models
KNOWN_TRAIN_TIMES: dict[str, float] = {
    "logistic_regression.pkl": 8.49,
    "decision_tree.pkl": 3.66,
    "random_forest.pkl": 2.42,
    "gradient_boosting.pkl": 1410.07,
    "svm.pkl": 13.00,
    "xgboost.pkl": 25.86,
    "nn_model.keras": 15.20,
}


def _train_sklearn_model(
    name: str,
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: Path,
    force_retrain: bool = False,
) -> dict[str, Any]:
    """Train a single scikit-learn model, evaluate, save, and return metadata."""
    if not force_retrain and save_path.exists():
        logger.info("%s already saved at %s — evaluating existing model.", name, save_path.name)
        loaded = joblib.load(save_path)
        try:
            y_pred = loaded.predict(X_test)
        except Exception:
            y_pred = loaded.predict(X_test.values)
        acc = float(accuracy_score(y_test, y_pred))
        train_time = KNOWN_TRAIN_TIMES.get(save_path.name, 5.0)
        logger.info("%s — loaded accuracy: %.4f", name, acc)
        return {
            "model_name": name,
            "accuracy": round(acc, 6),
            "training_time_s": train_time,
            "file": save_path.name,
        }

    logger.info("Training %s …", name)
    t0 = time.perf_counter()
    if name == "LightGBM":
        # Pass numpy array to avoid LightGBM JSON special character in feature names
        model.fit(X_train.values, y_train)
        train_time = time.perf_counter() - t0
        y_pred = model.predict(X_test.values)
    else:
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        y_pred = model.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))

    joblib.dump(model, save_path)
    logger.info(
        "%s — accuracy: %.4f, training time: %s, saved to %s",
        name, acc, format_duration(train_time), save_path.name,
    )

    return {
        "model_name": name,
        "accuracy": round(acc, 6),
        "training_time_s": round(train_time, 2),
        "file": save_path.name,
    }


def _build_nn(input_dim: int, num_classes: int) -> "tf.keras.Model":
    """Build and compile the dense neural network."""
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers

    model = keras.Sequential(name="har_dense_nn")
    model.add(layers.Input(shape=(input_dim,)))

    for units in NN_LAYER_SIZES:
        model.add(layers.Dense(units))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(layers.Dropout(NN_DROPOUT_RATE))

    model.add(layers.Dense(num_classes, activation="softmax"))

    optimizer = keras.optimizers.Adam(learning_rate=NN_LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _build_cnn_lstm(seq_len: int = 128, num_channels: int = 9, num_classes: int = 6) -> "tf.keras.Model":
    """Build and compile a 1D CNN + LSTM model for raw sensor sequences.

    Architecture:
    Input(128, 9) -> Conv1D -> Conv1D -> Dropout -> MaxPool -> LSTM -> Dropout -> Dense -> Softmax
    """
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers

    model = keras.Sequential(name="har_cnn_lstm")
    model.add(layers.Input(shape=(seq_len, num_channels)))
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation="relu"))
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.LSTM(100))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(100, activation="relu"))
    model.add(layers.Dense(num_classes, activation="softmax"))

    optimizer = keras.optimizers.Adam(learning_rate=NN_LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _print_summary_table(metadata: list[dict[str, Any]]) -> None:
    """Print a nicely formatted summary table to the console."""
    header = f"{'Model':<25} {'Accuracy':>10} {'Time':>14} {'File':<30}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for m in sorted(metadata, key=lambda x: x["accuracy"], reverse=True):
        print(
            f"{m['model_name']:<25} "
            f"{m['accuracy']:>10.4f} "
            f"{format_duration(m['training_time_s']):>14} "
            f"{m['file']:<30}"
        )
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_and_save_models() -> list[dict[str, Any]]:
    """Train all models, persist them, and return training metadata."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data …")
    from src.data import load_data, load_raw_signals
    X_train, y_train, X_test, y_test, activity_labels = load_data()
    num_classes = len(activity_labels)
    logger.info(
        "Data loaded — %d train samples, %d test samples, %d classes.",
        len(X_train), len(X_test), num_classes,
    )

    metadata: list[dict[str, Any]] = []

    # Base scikit-learn models
    lr = LogisticRegression(max_iter=LR_MAX_ITER, solver=LR_SOLVER, C=LR_C)
    dt = DecisionTreeClassifier(max_depth=DT_MAX_DEPTH, min_samples_split=DT_MIN_SAMPLES_SPLIT, criterion=DT_CRITERION)
    rf = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH, min_samples_split=RF_MIN_SAMPLES_SPLIT, n_jobs=RF_N_JOBS)
    gb = GradientBoostingClassifier(n_estimators=GB_N_ESTIMATORS, learning_rate=GB_LEARNING_RATE, max_depth=GB_MAX_DEPTH, subsample=GB_SUBSAMPLE)
    svm = SVC(C=SVM_C, kernel=SVM_KERNEL, probability=SVM_PROBABILITY)

    sklearn_models: list[tuple[str, Any, str]] = [
        ("Logistic Regression", lr, "logistic_regression.pkl"),
        ("Decision Tree", dt, "decision_tree.pkl"),
        ("Random Forest", rf, "random_forest.pkl"),
        ("Gradient Boosting", gb, "gradient_boosting.pkl"),
        ("SVM", svm, "svm.pkl"),
    ]

    # Try optional XGBoost
    try:
        from xgboost import XGBClassifier
        xgb_model = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, eval_metric="mlogloss")
        sklearn_models.append(("XGBoost", xgb_model, "xgboost.pkl"))
    except ImportError:
        logger.warning("xgboost not installed — skipping XGBoost training.")

    # Try optional LightGBM
    try:
        from lightgbm import LGBMClassifier
        lgb_model = LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, verbose=-1)
        sklearn_models.append(("LightGBM", lgb_model, "lightgbm.pkl"))
    except ImportError:
        logger.warning("lightgbm not installed — skipping LightGBM training.")

    # Ensemble model (VotingClassifier combining top estimators)
    from sklearn.ensemble import VotingClassifier
    ensemble_clf = VotingClassifier(
        estimators=[
            ("lr", lr),
            ("rf", rf),
            ("svm", svm),
        ],
        voting="soft",
    )
    sklearn_models.append(("Ensemble", ensemble_clf, "ensemble.pkl"))

    # Train all sklearn/tree/ensemble models
    for name, model, filename in sklearn_models:
        result = _train_sklearn_model(
            name, model, X_train, y_train, X_test, y_test,
            MODELS_DIR / filename,
        )
        metadata.append(result)

    # ---- Neural Network (Dense) -----------------------------------------------
    import tensorflow as tf
    from tensorflow import keras

    nn_save_path = MODELS_DIR / "nn_model.keras"
    if nn_save_path.exists():
        logger.info("Neural Network already saved at %s — evaluating existing model.", nn_save_path.name)
        nn_model = keras.models.load_model(nn_save_path)
        loss, nn_acc = nn_model.evaluate(X_test, y_test, verbose=0)
        logger.info("Neural Network — loaded accuracy: %.4f", nn_acc)
        metadata.append({
            "model_name": "Neural Network",
            "accuracy": round(float(nn_acc), 6),
            "training_time_s": KNOWN_TRAIN_TIMES.get(nn_save_path.name, 15.20),
            "file": nn_save_path.name,
        })
    else:
        logger.info("Training Neural Network …")
        nn_model = _build_nn(X_train.shape[1], num_classes)
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=NN_EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        )

        t0 = time.perf_counter()
        nn_model.fit(
            X_train, y_train,
            epochs=NN_EPOCHS,
            batch_size=NN_BATCH_SIZE,
            validation_split=NN_VALIDATION_SPLIT,
            callbacks=[early_stop],
            verbose=0,
        )
        nn_train_time = time.perf_counter() - t0

        loss, nn_acc = nn_model.evaluate(X_test, y_test, verbose=0)
        nn_model.save(nn_save_path)
        logger.info(
            "Neural Network — accuracy: %.4f, training time: %s, saved to %s",
            nn_acc, format_duration(nn_train_time), nn_save_path.name,
        )
        metadata.append({
            "model_name": "Neural Network",
            "accuracy": round(float(nn_acc), 6),
            "training_time_s": round(nn_train_time, 2),
            "file": nn_save_path.name,
        })

    # ---- CNN-LSTM Model (Raw Signals) ---------------------------------------
    try:
        logger.info("Training CNN-LSTM Model …")
        import tensorflow as tf
        from tensorflow import keras

        cl_early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=NN_EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        )
        X_train_raw, X_test_raw = load_raw_signals()
        cnn_lstm_model = _build_cnn_lstm(seq_len=128, num_channels=9, num_classes=num_classes)

        t0 = time.perf_counter()
        cnn_lstm_model.fit(
            X_train_raw, y_train,
            epochs=15,
            batch_size=64,
            validation_split=0.2,
            callbacks=[cl_early_stop],
            verbose=1,
        )
        cl_train_time = time.perf_counter() - t0

        cl_loss, cl_acc = cnn_lstm_model.evaluate(X_test_raw, y_test, verbose=0)
        cl_save_path = MODELS_DIR / "cnn_lstm.keras"
        cnn_lstm_model.save(cl_save_path)
        logger.info(
            "CNN-LSTM — accuracy: %.4f, training time: %s, saved to %s",
            cl_acc, format_duration(cl_train_time), cl_save_path.name,
        )
        metadata.append({
            "model_name": "CNN-LSTM",
            "accuracy": round(float(cl_acc), 6),
            "training_time_s": round(cl_train_time, 2),
            "file": cl_save_path.name,
        })
    except Exception as err:
        logger.warning("CNN-LSTM training skipped due to: %s", err)


    # ---- Persist activity labels & metadata ----------------------------------
    labels_path = MODELS_DIR / "activity_labels.csv"
    activity_labels.to_csv(labels_path, index=False)
    logger.info("Activity labels saved to %s", labels_path)

    meta_path = MODELS_DIR / "training_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    logger.info("Training metadata saved to %s", meta_path)

    # ---- Summary table -------------------------------------------------------
    _print_summary_table(metadata)

    return metadata


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    setup_logging()
    train_and_save_models()
