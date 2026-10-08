"""
Model evaluation pipeline for the HAR project.

Loads every trained model from ``models/``, runs it against the held-out
test set, and produces:

* Per-model confusion matrix plots (improved styling with modern colormap)
* Per-model classification reports saved as CSV
* Feature importance plot for tree-based models
* Accuracy comparison bar chart
* Inference speed measurement (time per sample)
* ``evaluation_results.json`` with all metrics

Usage
-----
Run from the project root::

    python -m src.evaluate
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.data import load_data
from src.utils import format_duration, get_project_root, setup_logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = get_project_root()
MODELS_DIR: Path = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# Sklearn models to evaluate (display name → filename)
# ---------------------------------------------------------------------------
SKLEARN_MODELS: list[tuple[str, str]] = [
    ("Logistic Regression", "logistic_regression.pkl"),
    ("Decision Tree", "decision_tree.pkl"),
    ("Random Forest", "random_forest.pkl"),
    ("Gradient Boosting", "gradient_boosting.pkl"),
    ("SVM", "svm.pkl"),
    ("XGBoost", "xgboost.pkl"),
    ("LightGBM", "lightgbm.pkl"),
    ("Ensemble", "ensemble.pkl"),
]

# Tree-based models that support feature_importances_
TREE_BASED_MODELS: set[str] = {"Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost", "LightGBM"}


# Matplotlib style
COLORMAP: str = "viridis"
BAR_COLORS: list[str] = [
    "#4c72b0", "#55a868", "#c44e52", "#8172b3", "#ccb974", "#64b5cd",
]


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    title: str,
    save_path: Path,
) -> None:
    """Save a styled confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(9, 7))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=COLORMAP,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor="white",
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.tick_params(axis="both", labelsize=9)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", save_path.name)


def _plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    model_name: str,
    save_path: Path,
    top_n: int = 20,
) -> None:
    """Save a horizontal bar chart of the top-N most important features."""
    indices = np.argsort(importances)[-top_n:]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(top_n), top_importances, color="#4c72b0", edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features, fontsize=9)
    ax.set_xlabel("Feature Importance", fontsize=11)
    ax.set_title(f"{model_name} — Top {top_n} Features", fontsize=13, fontweight="bold")
    ax.invert_yaxis()  # most important on top
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Feature importance plot saved to %s", save_path.name)


def _plot_accuracy_comparison(
    results: dict[str, float],
    save_path: Path,
) -> None:
    """Save a bar chart comparing model accuracies."""
    names = list(results.keys())
    accs = list(results.values())
    colors = BAR_COLORS[: len(names)]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(names, accs, color=colors, edgecolor="white", width=0.6)

    # Annotate bars
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{acc:.4f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
    ax.set_ylim(max(0, min(accs) - 0.05), 1.0)
    ax.tick_params(axis="x", labelsize=10, rotation=15)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Accuracy comparison chart saved to %s", save_path.name)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def _measure_inference_speed(
    model: Any,
    X_test: pd.DataFrame | np.ndarray,
    is_keras: bool = False,
) -> float:
    """Return average inference time *per sample* in seconds."""
    n = len(X_test)
    if is_keras:
        # Warm-up pass
        model.predict(X_test[:1], verbose=0)
        t0 = time.perf_counter()
        model.predict(X_test, verbose=0)
        elapsed = time.perf_counter() - t0
    else:
        # Warm-up pass
        model.predict(X_test.iloc[:1] if isinstance(X_test, pd.DataFrame) else X_test[:1])
        t0 = time.perf_counter()
        model.predict(X_test)
        elapsed = time.perf_counter() - t0
    return elapsed / n


def evaluate_models() -> list[dict[str, Any]]:
    """Evaluate all trained models and persist results.

    Returns
    -------
    list[dict]
        One dict per model with keys ``model_name``, ``accuracy``,
        ``f1_macro``, ``inference_time_per_sample_ms``, etc.
    """
    logger.info("Loading test data …")
    X_train, _, X_test, y_test, activity_labels = load_data()
    labels: list[str] = activity_labels["activity_name"].tolist()
    feature_names: list[str] = list(X_test.columns)

    accuracies: dict[str, float] = {}
    all_results: list[dict[str, Any]] = []

    # ---- Scikit-learn models -------------------------------------------------
    for model_name, filename in SKLEARN_MODELS:
        model_path = MODELS_DIR / filename
        if not model_path.exists():
            logger.warning("Model file %s not found — skipping.", model_path)
            continue

        logger.info("Evaluating %s …", model_name)
        model = joblib.load(model_path)
        X_eval = X_test.values if model_name == "LightGBM" else X_test
        y_pred = model.predict(X_eval)

        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="macro"))
        accuracies[model_name] = acc

        # Inference speed
        time_per_sample = _measure_inference_speed(model, X_eval)

        logger.info(
            "%s — accuracy: %.4f, F1 (macro): %.4f, %.4f ms/sample",
            model_name, acc, f1, time_per_sample * 1000,
        )

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        _plot_confusion_matrix(
            cm, labels,
            f"{model_name} — Confusion Matrix",
            MODELS_DIR / f"{filename.replace('.pkl', '')}_cm.png",
        )

        # Classification report → CSV
        report_dict = classification_report(
            y_test, y_pred, target_names=labels, output_dict=True,
        )
        report_df = pd.DataFrame(report_dict).transpose()
        report_csv = MODELS_DIR / f"{filename.replace('.pkl', '')}_report.csv"
        report_df.to_csv(report_csv)
        logger.info("Classification report saved to %s", report_csv.name)

        # Feature importance (tree-based only)
        if model_name in TREE_BASED_MODELS and hasattr(model, "feature_importances_"):
            _plot_feature_importance(
                model.feature_importances_,
                feature_names,
                model_name,
                MODELS_DIR / f"{filename.replace('.pkl', '')}_feature_importance.png",
            )

        all_results.append({
            "model_name": model_name,
            "accuracy": round(acc, 6),
            "f1_macro": round(f1, 6),
            "inference_time_per_sample_ms": round(time_per_sample * 1000, 4),
            "file": filename,
        })

    # ---- Neural Network ------------------------------------------------------
    nn_path = MODELS_DIR / "nn_model.keras"
    if nn_path.exists():
        logger.info("Evaluating Neural Network …")
        import tensorflow as tf
        from tensorflow import keras

        nn_model = keras.models.load_model(nn_path)
        loss, nn_acc = nn_model.evaluate(X_test, y_test, verbose=0)
        nn_acc = float(nn_acc)

        y_pred_proba = nn_model.predict(X_test, verbose=0)
        y_pred_nn = np.argmax(y_pred_proba, axis=1)
        f1_nn = float(f1_score(y_test, y_pred_nn, average="macro"))

        time_per_sample = _measure_inference_speed(nn_model, X_test, is_keras=True)

        accuracies["Neural Network"] = nn_acc
        logger.info(
            "Neural Network — accuracy: %.4f, F1 (macro): %.4f, %.4f ms/sample",
            nn_acc, f1_nn, time_per_sample * 1000,
        )

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_nn)
        _plot_confusion_matrix(
            cm, labels,
            "Neural Network — Confusion Matrix",
            MODELS_DIR / "nn_model_cm.png",
        )

        # Classification report
        report_dict = classification_report(
            y_test, y_pred_nn, target_names=labels, output_dict=True,
        )
        report_df = pd.DataFrame(report_dict).transpose()
        report_df.to_csv(MODELS_DIR / "nn_model_report.csv")

        all_results.append({
            "model_name": "Neural Network",
            "accuracy": round(nn_acc, 6),
            "f1_macro": round(f1_nn, 6),
            "inference_time_per_sample_ms": round(time_per_sample * 1000, 4),
            "file": "nn_model.keras",
        })
    else:
        logger.warning("Neural Network model not found — skipping.")

    # ---- CNN-LSTM Model (Raw Signals) ----------------------------------------
    cl_path = MODELS_DIR / "cnn_lstm.keras"
    if cl_path.exists():
        try:
            logger.info("Evaluating CNN-LSTM Model …")
            from src.data import load_raw_signals
            _, X_test_raw = load_raw_signals()
            import tensorflow as tf
            from tensorflow import keras

            cl_model = keras.models.load_model(cl_path)
            loss, cl_acc = cl_model.evaluate(X_test_raw, y_test, verbose=0)
            cl_acc = float(cl_acc)

            y_pred_proba = cl_model.predict(X_test_raw, verbose=0)
            y_pred_cl = np.argmax(y_pred_proba, axis=1)
            f1_cl = float(f1_score(y_test, y_pred_cl, average="macro"))

            time_per_sample = _measure_inference_speed(cl_model, X_test_raw, is_keras=True)

            accuracies["CNN-LSTM"] = cl_acc
            logger.info(
                "CNN-LSTM — accuracy: %.4f, F1 (macro): %.4f, %.4f ms/sample",
                cl_acc, f1_cl, time_per_sample * 1000,
            )

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred_cl)
            _plot_confusion_matrix(
                cm, labels,
                "CNN-LSTM — Confusion Matrix",
                MODELS_DIR / "cnn_lstm_cm.png",
            )

            # Classification report
            report_dict = classification_report(
                y_test, y_pred_cl, target_names=labels, output_dict=True,
            )
            report_df = pd.DataFrame(report_dict).transpose()
            report_df.to_csv(MODELS_DIR / "cnn_lstm_report.csv")

            all_results.append({
                "model_name": "CNN-LSTM",
                "accuracy": round(cl_acc, 6),
                "f1_macro": round(f1_cl, 6),
                "inference_time_per_sample_ms": round(time_per_sample * 1000, 4),
                "file": "cnn_lstm.keras",
            })
        except Exception as err:
            logger.warning("CNN-LSTM evaluation failed: %s", err)


    # ---- Accuracy comparison chart ------------------------------------------
    if accuracies:
        _plot_accuracy_comparison(accuracies, MODELS_DIR / "accuracy_comparison.png")

    # ---- Save evaluation results JSON ----------------------------------------
    eval_path = MODELS_DIR / "evaluation_results.json"
    with open(eval_path, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2)
    logger.info("Evaluation results saved to %s", eval_path)

    # ---- Print summary -------------------------------------------------------
    header = (
        f"{'Model':<25} {'Accuracy':>10} {'F1 (macro)':>12} "
        f"{'ms/sample':>12}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in sorted(all_results, key=lambda x: x["accuracy"], reverse=True):
        print(
            f"{r['model_name']:<25} "
            f"{r['accuracy']:>10.4f} "
            f"{r['f1_macro']:>12.4f} "
            f"{r['inference_time_per_sample_ms']:>12.4f}"
        )
    print(f"{sep}\n")

    return all_results


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    setup_logging()
    evaluate_models()
