"""
Startup validation for the HAR project.

Verifies environment configuration, database connectivity, model availability,
and dependency compatibility on application startup. Provides clear, actionable
error messages rather than allowing silent failures.
"""

from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Models required for full functionality
REQUIRED_MODELS: dict[str, str] = {
    "logistic_regression.pkl": "Logistic Regression",
    "random_forest.pkl": "Random Forest",
    "svm.pkl": "SVM",
    "xgboost.pkl": "XGBoost",
    "lightgbm.pkl": "LightGBM",
    "ensemble.pkl": "Ensemble",
    "nn_model.keras": "Neural Network",
    "cnn_lstm.keras": "CNN-LSTM",
}

# Optional models (not required for basic functionality)
OPTIONAL_MODELS: dict[str, str] = {
    "decision_tree.pkl": "Decision Tree",
    "gradient_boosting.pkl": "Gradient Boosting",
}


def validate_environment() -> list[str]:
    """Check environment variable configuration.

    Returns a list of warning messages. Critical issues raise exceptions.
    """
    warnings: list[str] = []

    jwt_secret = os.getenv("HAR_JWT_SECRET", "har-dev-secret-change-in-production-2026")
    if jwt_secret == "har-dev-secret-change-in-production-2026":
        warnings.append(
            "HAR_JWT_SECRET is not set — using insecure default. "
            "Set HAR_JWT_SECRET for production deployments."
        )

    return warnings


def validate_models(models_dir: Path) -> tuple[list[str], list[str], list[str]]:
    """Check model file availability.

    Returns:
        (available, missing_required, missing_optional)
    """
    available: list[str] = []
    missing_required: list[str] = []
    missing_optional: list[str] = []

    for filename, display_name in REQUIRED_MODELS.items():
        path = models_dir / filename
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            available.append(f"{display_name} ({filename}, {size_mb:.1f} MB)")
        else:
            missing_required.append(f"{display_name} ({filename})")

    for filename, display_name in OPTIONAL_MODELS.items():
        path = models_dir / filename
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            available.append(f"{display_name} ({filename}, {size_mb:.1f} MB)")
        else:
            missing_optional.append(f"{display_name} ({filename})")

    return available, missing_required, missing_optional


def validate_database() -> list[str]:
    """Check database connectivity.

    Returns a list of warning messages.
    """
    warnings: list[str] = []
    try:
        from src.database import engine
        with engine.connect() as conn:
            conn.execute(engine.dialect.statement_compiler(engine.dialect, None).__class__.__module__  # noqa
                         if False else  # noqa
                         __import__("sqlalchemy").text("SELECT 1"))
        logger.info("Database connection verified.")
    except Exception as e:
        warnings.append(f"Database connection failed: {e}")
    return warnings


def validate_dependencies() -> list[str]:
    """Check critical Python package availability.

    Returns a list of warning messages for missing packages.
    """
    warnings: list[str] = []

    critical_packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("sklearn", "scikit-learn"),
        ("joblib", "joblib"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("sqlalchemy", "sqlalchemy"),
        ("jwt", "pyjwt"),
        ("bcrypt", "bcrypt"),
        ("pydantic", "pydantic"),
    ]

    optional_packages = [
        ("tensorflow", "tensorflow"),
        ("xgboost", "xgboost"),
        ("lightgbm", "lightgbm"),
        ("streamlit", "streamlit"),
    ]

    for import_name, pip_name in critical_packages:
        try:
            importlib.import_module(import_name)
        except ImportError:
            warnings.append(f"Missing critical dependency: {pip_name} (pip install {pip_name})")

    for import_name, pip_name in optional_packages:
        try:
            importlib.import_module(import_name)
        except ImportError:
            logger.info("Optional dependency not available: %s", pip_name)

    return warnings


def run_startup_validation(models_dir: Path) -> None:
    """Run all startup validations and log results.

    Logs warnings for non-critical issues. Raises RuntimeError for
    critical failures that would prevent the application from functioning.
    """
    logger.info("Running startup validation...")

    # 1. Check dependencies
    dep_warnings = validate_dependencies()
    for w in dep_warnings:
        logger.warning("⚠️  %s", w)
    if any("critical" in w.lower() for w in dep_warnings):
        raise RuntimeError(
            "Critical dependencies missing. Install requirements: pip install -r requirements.txt"
        )

    # 2. Check environment
    env_warnings = validate_environment()
    for w in env_warnings:
        logger.warning("⚠️  %s", w)

    # 3. Check models
    if not models_dir.exists():
        raise RuntimeError(
            f"Models directory not found: {models_dir}. "
            f"Train models first: python -m src.train"
        )

    available, missing_required, missing_optional = validate_models(models_dir)

    for m in available:
        logger.info("  ✓ %s", m)

    for m in missing_optional:
        logger.info("  ○ %s (optional, not found)", m)

    if missing_required:
        missing_list = ", ".join(missing_required)
        logger.error(
            "❌ Missing required model files: %s. "
            "Train models with: python -m src.train",
            missing_list,
        )
        raise RuntimeError(
            f"Missing required model files: {missing_list}. "
            f"Train models with: python -m src.train"
        )

    logger.info("Startup validation complete — %d models available.", len(available))
