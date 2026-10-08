"""
FastAPI backend for the HAR (Human Activity Recognition) project.

Provides REST and WebSocket endpoints for real-time model inference,
persistent database storage, JWT authentication, prediction history,
activity tracking, health metrics, and mobile sensor ingestion.

Endpoints
---------
- ``GET  /``                    — Health check / welcome message.
- ``GET  /models``              — List available models with training metadata.
- ``POST /predict``             — Single-sample prediction.
- ``POST /predict/batch``       — Batch prediction (multiple samples).
- ``WS   /ws/predict``          — WebSocket endpoint for live predictions streaming.
- ``POST /sensor/ingest``       — Sensor data ingestion & feature extraction.
- ``GET  /sensor``              — Mobile HTML5 real-time sensor streamer client.
- ``GET  /history``             — Persistent prediction history records.
- ``GET  /history/stats``       — Prediction history statistics.
- ``GET  /history/export``      — Export history as CSV/JSON.
- ``GET  /activity/summary``    — Daily/Weekly activity summary.
- ``GET  /health-metrics``      — Steps, calories, distance, active time snapshot.
- ``POST /auth/register``       — Register a new user account with hashed password.
- ``POST /auth/login``          — User authentication and JWT token generation.
- ``GET  /auth/profile``        — Authenticated user profile.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Header, Query, Request, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from src.activity_tracker import ActivityTracker
from src.auth import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    authenticate_websocket_token,
    create_access_token,
    get_current_user,
    get_optional_user,
    hash_password,
    verify_password,
)
from src.database import (
    SessionLocal,
    User,
    add_prediction,
    add_user,
    get_db,
    get_latest_health_snapshot,
    get_prediction_stats,
    get_predictions,
    get_user_by_username,
    init_db,
    save_health_snapshot,
)
from src.health_metrics import HealthMetrics
from src.prediction_history import PredictionHistory
from src.prediction_smoother import PredictionSmoother
from src.sensor_simulator import extract_features_from_window, generate_sensor_window

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project paths & instances
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
STATIC_DIR = PROJECT_ROOT / "static"


class UserStateManager:
    """Per-user state isolation for prediction smoothing, history, tracking, and health.

    Each user (identified by ``user_id``) gets independent instances.
    Anonymous users (``user_id=None``) share a single fallback set.
    """

    def __init__(self) -> None:
        self._smoothers: dict[int | None, PredictionSmoother] = {}
        self._histories: dict[int | None, PredictionHistory] = {}
        self._trackers: dict[int | None, ActivityTracker] = {}
        self._health: dict[int | None, HealthMetrics] = {}

    def get_smoother(self, user_id: int | None) -> PredictionSmoother:
        if user_id not in self._smoothers:
            self._smoothers[user_id] = PredictionSmoother(window_size=5)
        return self._smoothers[user_id]

    def get_history(self, user_id: int | None) -> PredictionHistory:
        if user_id not in self._histories:
            self._histories[user_id] = PredictionHistory()
        return self._histories[user_id]

    def get_tracker(self, user_id: int | None) -> ActivityTracker:
        if user_id not in self._trackers:
            self._trackers[user_id] = ActivityTracker()
        return self._trackers[user_id]

    def get_health(self, user_id: int | None, weight_kg: float = 70.0) -> HealthMetrics:
        if user_id not in self._health:
            self._health[user_id] = HealthMetrics(weight_kg=weight_kg)
        return self._health[user_id]


user_state = UserStateManager()

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Request body for single-sample prediction."""
    model_name: str = Field(..., description="Name of the model.", examples=["logistic_regression"])
    features: list[float] = Field(..., min_length=561, max_length=561, description="Feature vector — 561 floats.")
    use_smoothing: bool = Field(False, description="Whether to apply rolling window smoothing.")


class PredictResponse(BaseModel):
    """Response body for a single prediction."""
    model_name: str
    predicted_label: int
    activity_name: str
    confidence: float | None = None
    inference_time_ms: float
    is_smoothed: bool = False
    smoothed_label: int | None = None
    smoothed_activity: str | None = None
    smoothed_confidence: float | None = None


class BatchPredictRequest(BaseModel):
    """Request body for batch prediction."""
    model_name: str = Field(..., examples=["random_forest"])
    samples: list[list[float]] = Field(..., description="List of 561-float feature vectors.")

    @field_validator("samples")
    @classmethod
    def validate_sample_lengths(cls, v: list[list[float]]) -> list[list[float]]:
        """Ensure every inner sample has exactly 561 features."""
        errors: list[str] = []
        for i, sample in enumerate(v):
            if len(sample) != 561:
                errors.append(f"Sample {i}: expected 561 features, got {len(sample)}")
        if errors:
            raise ValueError("; ".join(errors))
        return v


class BatchPredictResponse(BaseModel):
    """Response body for batch prediction."""
    model_name: str
    predictions: list[PredictResponse]
    total_inference_time_ms: float


class SensorIngestRequest(BaseModel):
    """Raw sensor readings for feature extraction and prediction."""
    activity_hint: str = Field("WALKING", description="Simulated activity hint.")
    weight_kg: float = Field(70.0, description="User body weight in kg.")


class SequencePredictRequest(BaseModel):
    """Request body for CNN-LSTM sequence prediction."""
    model_name: str = Field("cnn_lstm", description="Model name (must support sequence input).")
    sequence: list[list[float]] = Field(
        ...,
        description="Temporal sequence of shape (128, 9): 128 timesteps × 9 sensor channels.",
    )
    use_smoothing: bool = Field(False, description="Whether to apply rolling window smoothing.")


class ModelInfo(BaseModel):
    """Metadata for a single loaded model."""
    name: str
    file: str
    accuracy: float | None = None
    training_time_s: float | None = None
    type: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: int
    message: str


# Backward-compatible alias for AuthTokenResponse
AuthTokenResponse = TokenResponse


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
_MODELS: dict[str, Any] = {}
_MODEL_META: dict[str, dict[str, Any]] = {}
_ACTIVITY_LABELS: dict[int, str] = {}

_SKLEARN_FILES: dict[str, str] = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "svm": "SVM",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "ensemble": "Ensemble",
}


def _load_models() -> None:
    """Load all trained models and metadata into memory."""
    global _ACTIVITY_LABELS

    labels_path = MODELS_DIR / "activity_labels.csv"
    if labels_path.exists():
        labels_df = pd.read_csv(labels_path)
        _ACTIVITY_LABELS = dict(zip(labels_df["label"].astype(int), labels_df["activity_name"]))
        logger.info("Loaded %d activity labels.", len(_ACTIVITY_LABELS))

    train_meta: dict[str, dict[str, Any]] = {}
    meta_path = MODELS_DIR / "training_metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as fh:
                for entry in json.load(fh):
                    key = entry["file"].replace(".pkl", "").replace(".keras", "")
                    train_meta[key] = entry
        except Exception:
            logger.warning("Could not parse %s", meta_path)

    for stem, display in _SKLEARN_FILES.items():
        pkl_path = MODELS_DIR / f"{stem}.pkl"
        if pkl_path.exists():
            try:
                _MODELS[stem] = joblib.load(pkl_path)
                meta = train_meta.get(stem, {})
                _MODEL_META[stem] = {
                    "name": display,
                    "file": pkl_path.name,
                    "accuracy": meta.get("accuracy"),
                    "training_time_s": meta.get("training_time_s"),
                    "type": "sklearn",
                }
                logger.info("Loaded model: %s", display)
            except Exception:
                logger.exception("Failed to load %s", pkl_path)

    # Keras Dense model
    nn_path = MODELS_DIR / "nn_model.keras"
    if nn_path.exists():
        try:
            from tensorflow import keras
            _MODELS["nn_model"] = keras.models.load_model(nn_path)
            meta = train_meta.get("nn_model", {})
            _MODEL_META["nn_model"] = {
                "name": "Neural Network",
                "file": nn_path.name,
                "accuracy": meta.get("accuracy"),
                "training_time_s": meta.get("training_time_s"),
                "type": "keras",
            }
            logger.info("Loaded Keras model: Neural Network")
        except Exception:
            logger.exception("Failed to load Keras model.")

    # Keras CNN-LSTM model
    cl_path = MODELS_DIR / "cnn_lstm.keras"
    if cl_path.exists():
        try:
            from tensorflow import keras
            _MODELS["cnn_lstm"] = keras.models.load_model(cl_path)
            meta = train_meta.get("cnn_lstm", {})
            _MODEL_META["cnn_lstm"] = {
                "name": "CNN-LSTM",
                "file": cl_path.name,
                "accuracy": meta.get("accuracy"),
                "training_time_s": meta.get("training_time_s"),
                "type": "keras",
            }
            logger.info("Loaded Keras model: CNN-LSTM")
        except Exception:
            logger.exception("Failed to load CNN-LSTM model.")

    logger.info("Total models loaded: %d", len(_MODELS))


# ---------------------------------------------------------------------------
# Lifespan Context Manager
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan: initializes DB, seeds default demo users, loads models."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s")
    
    from src.auth import SECRET_KEY
    if SECRET_KEY == "har-dev-secret-change-in-production-2026":
        logger.warning(
            "⚠️  Using default JWT secret. Set HAR_JWT_SECRET environment variable for production."
        )

    # Run startup validation (checks models, deps, environment)
    try:
        from src.startup_validation import run_startup_validation
        run_startup_validation(MODELS_DIR)
    except RuntimeError as e:
        logger.error("Startup validation failed: %s", e)
        raise  # Fail fast — do not start with missing critical requirements
    except Exception as e:
        logger.warning("Startup validation warning: %s", e)

    logger.info("Starting up HAR API backend…")
    init_db()

    # Seed default demo users in SQLite if not present
    if os.getenv("HAR_SEED_DEMO_USERS", "false").lower() in ("true", "1", "yes"):
        with SessionLocal() as db:
            admin_user = get_user_by_username(db, "admin")
            if not admin_user:
                add_user(db, username="admin", hashed_password=hash_password("password"))
                logger.info("Created default 'admin' user.")
            demo_user = get_user_by_username(db, "user")
            if not demo_user:
                add_user(db, username="user", hashed_password=hash_password("password"))
                logger.info("Created default 'user' account.")

    _load_models()
    yield
    logger.info("Shutting down HAR API backend…")


# ---------------------------------------------------------------------------
# FastAPI app instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title="HAR Prediction API",
    description="REST & WebSocket API for Human Activity Recognition with SQLite persistence and JWT auth.",
    version="2.1.0",
    lifespan=lifespan,
)

_cors_origins_raw = os.getenv("HAR_CORS_ORIGINS", "")
_har_env = os.getenv("HAR_ENVIRONMENT", "development").lower()

if _cors_origins_raw:
    _cors_origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]
elif _har_env == "production":
    logger.warning(
        "⚠️  HAR_CORS_ORIGINS is not set in production — defaulting to no origins allowed. "
        "Set HAR_CORS_ORIGINS to your frontend domain(s)."
    )
    _cors_origins = []
else:
    _cors_origins = ["*"]

_cors_is_wildcard = _cors_origins == ["*"]
if _cors_is_wildcard and _har_env == "production":
    logger.warning(
        "⚠️  Wildcard CORS ('*') is active in production. "
        "Set HAR_CORS_ORIGINS to restrict allowed origins."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=(not _cors_is_wildcard),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional Prometheus metrics
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app)
except Exception:
    logger.debug("Prometheus instrumentator not installed or disabled.")

# Application-level rate limiting (slowapi)
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded

    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    logger.info("Rate limiting enabled via slowapi.")
except ImportError:
    limiter = None
    logger.info("slowapi not installed — rate limiting disabled. Install with: pip install slowapi")


def rate_limit(limit_string: str):
    """Apply rate limiting in normal environments, but disable it for tests."""
    rate_limits_enabled = os.getenv("HAR_RATE_LIMIT_ENABLED", "true").lower() not in {
        "0", "false", "no", "off"
    }
    if limiter is not None and rate_limits_enabled:
        return limiter.limit(limit_string)

    def noop(func):
        return func

    return noop


# ---------------------------------------------------------------------------
# Core Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def root() -> HealthResponse:
    return HealthResponse(
        status="ok",
        models_loaded=len(_MODELS),
        message="HAR Prediction API v2.1 (Persistent DB + JWT Auth) is running. Visit /docs or /sensor.",
    )


@app.get("/models", response_model=list[ModelInfo], tags=["Models"])
async def list_models() -> list[ModelInfo]:
    return [
        ModelInfo(
            name=meta["name"],
            file=meta["file"],
            accuracy=meta.get("accuracy"),
            training_time_s=meta.get("training_time_s"),
            type=meta["type"],
        )
        for meta in _MODEL_META.values()
    ]


@app.get("/sensor", tags=["Real-Time Sensor"])
async def sensor_page() -> FileResponse:
    """Serve the mobile-friendly HTML5 real-time sensor streamer page."""
    sensor_html = STATIC_DIR / "sensor.html"
    if not sensor_html.exists():
        raise HTTPException(status_code=404, detail="sensor.html not found.")
    return FileResponse(sensor_html)


def _predict_single(
    model_key: str,
    features: list[float],
    use_smoothing: bool = False,
    user_id: int | None = None,
) -> PredictResponse:
    if model_key not in _MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_key}' not found. Available models: {list(_MODELS.keys())}",
        )

    model = _MODELS[model_key]
    meta = _MODEL_META[model_key]
    X = np.array(features).reshape(1, -1)
    if hasattr(model, "feature_names_in_"):
        X = pd.DataFrame(X, columns=model.feature_names_in_)

    t0 = time.perf_counter()

    if meta["type"] == "keras":
        input_shape = model.input_shape
        if len(input_shape) == 3:  # CNN-LSTM expects (None, 128, 9)
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_key}' requires sequence input (128×9). "
                f"Use POST /predict/sequence instead of /predict.",
            )
        else:
            proba = model.predict(X, verbose=0)
        label = int(np.argmax(proba, axis=1)[0])
        confidence = float(np.max(proba))
    elif hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        label = int(np.argmax(proba, axis=1)[0])
        confidence = float(np.max(proba))
    else:
        label = int(model.predict(X)[0])
        confidence = 1.0

    elapsed_ms = (time.perf_counter() - t0) * 1000
    activity = _ACTIVITY_LABELS.get(label, f"UNKNOWN_{label}")

    sm_label, sm_act, sm_conf = None, None, None
    if use_smoothing:
        user_smoother = user_state.get_smoother(user_id)
        sm_res = user_smoother.add(label, confidence or 1.0)
        sm_label = sm_res.smoothed_label
        sm_act = _ACTIVITY_LABELS.get(sm_label, f"UNKNOWN_{sm_label}")
        sm_conf = sm_res.smoothed_confidence

    # 1. Update user-scoped in-memory trackers
    user_history = user_state.get_history(user_id)
    user_tracker = user_state.get_tracker(user_id)
    user_health = user_state.get_health(user_id)

    user_history.add(
        model_name=model_key,
        predicted_label=label,
        activity_name=activity,
        confidence=confidence,
        is_smoothed=use_smoothing,
        smoothed_label=sm_label,
        smoothed_activity=sm_act,
        smoothed_confidence=sm_conf,
    )
    user_tracker.update(sm_act or activity)
    user_health.update(sm_act or activity)

    # 2. Persist to SQLite database
    try:
        with SessionLocal() as db_session:
            add_prediction(
                db_session,
                user_id=user_id,
                model_name=model_key,
                predicted_label=label,
                activity_name=activity,
                confidence=confidence,
                is_smoothed=use_smoothing,
                smoothed_label=sm_label,
                smoothed_activity=sm_act,
                smoothed_confidence=sm_conf,
                inference_time_ms=elapsed_ms,
            )
            snap = user_health.get_snapshot()
            save_health_snapshot(
                db_session,
                user_id=user_id,
                total_steps=snap.total_steps,
                calories_burned=snap.calories_burned,
                distance_m=snap.distance_m,
                active_time_s=snap.active_time_s,
                sedentary_time_s=snap.sedentary_time_s,
            )
    except Exception as db_err:
        logger.warning("Database persistence failed: %s", db_err)

    return PredictResponse(
        model_name=model_key,
        predicted_label=label,
        activity_name=activity,
        confidence=round(confidence, 6) if confidence is not None else None,
        inference_time_ms=round(elapsed_ms, 4),
        is_smoothed=use_smoothing,
        smoothed_label=sm_label,
        smoothed_activity=sm_act,
        smoothed_confidence=sm_conf,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
@rate_limit("30/minute")
async def predict(
    request: Request,
    payload: PredictRequest,
    current_user: User | None = Depends(get_optional_user),
) -> PredictResponse:
    uid = current_user.id if current_user else None
    return _predict_single(payload.model_name, payload.features, payload.use_smoothing, user_id=uid)


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
@rate_limit("10/minute")
async def predict_batch(
    request: Request,
    payload: BatchPredictRequest,
    current_user: User | None = Depends(get_optional_user),
) -> BatchPredictResponse:
    if not payload.samples:
        raise HTTPException(status_code=400, detail="samples list is empty.")

    uid = current_user.id if current_user else None
    t0 = time.perf_counter()
    predictions = [_predict_single(payload.model_name, sample, user_id=uid) for sample in payload.samples]
    total_ms = (time.perf_counter() - t0) * 1000

    return BatchPredictResponse(
        model_name=payload.model_name,
        predictions=predictions,
        total_inference_time_ms=round(total_ms, 4),
    )


@app.post("/predict/sequence", response_model=PredictResponse, tags=["Prediction"])
async def predict_sequence(
    request: SequencePredictRequest,
    current_user: User | None = Depends(get_optional_user),
) -> PredictResponse:
    """Predict from a temporal sensor sequence (128 timesteps × 9 channels) using CNN-LSTM."""
    model_key = request.model_name
    if model_key not in _MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_key}' not found. Available models: {list(_MODELS.keys())}",
        )
    meta = _MODEL_META.get(model_key, {})
    if meta.get("type") != "keras" or model_key != "cnn_lstm":
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_key}' does not support sequence input. Use /predict for flat-feature models.",
        )

    seq = np.array(request.sequence, dtype=np.float32)
    if seq.shape != (128, 9):
        raise HTTPException(
            status_code=422,
            detail=f"Sequence must have shape (128, 9), got {seq.shape}. "
            f"Expected 128 timesteps × 9 channels (body_acc_xyz, body_gyro_xyz, total_acc_xyz).",
        )

    model = _MODELS[model_key]
    X = seq.reshape(1, 128, 9)

    t0 = time.perf_counter()
    proba = model.predict(X, verbose=0)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    label = int(np.argmax(proba, axis=1)[0])
    confidence = float(np.max(proba))
    activity = _ACTIVITY_LABELS.get(label, f"UNKNOWN_{label}")

    sm_label, sm_act, sm_conf = None, None, None
    if request.use_smoothing:
        uid = current_user.id if current_user else None
        user_smoother = user_state.get_smoother(uid)
        sm_res = user_smoother.add(label, confidence)
        sm_label = sm_res.smoothed_label
        sm_act = _ACTIVITY_LABELS.get(sm_label, f"UNKNOWN_{sm_label}")
        sm_conf = sm_res.smoothed_confidence

    uid = current_user.id if current_user else None
    try:
        with SessionLocal() as db_session:
            add_prediction(
                db_session,
                user_id=uid,
                model_name=model_key,
                predicted_label=label,
                activity_name=activity,
                confidence=confidence,
                is_smoothed=request.use_smoothing,
                smoothed_label=sm_label,
                smoothed_activity=sm_act,
                smoothed_confidence=sm_conf,
                inference_time_ms=elapsed_ms,
            )
    except Exception as db_err:
        logger.warning("Database persistence failed: %s", db_err)

    return PredictResponse(
        model_name=model_key,
        predicted_label=label,
        activity_name=activity,
        confidence=round(confidence, 6),
        inference_time_ms=round(elapsed_ms, 4),
        is_smoothed=request.use_smoothing,
        smoothed_label=sm_label,
        smoothed_activity=sm_act,
        smoothed_confidence=sm_conf,
    )


# ---------------------------------------------------------------------------
# Real-Time Sensor Ingestion Endpoint
# ---------------------------------------------------------------------------

@app.post("/sensor/ingest", response_model=PredictResponse, tags=["Real-Time Sensor"])
async def ingest_sensor(
    request: SensorIngestRequest,
    current_user: User | None = Depends(get_optional_user),
) -> PredictResponse:
    """Ingest simulated sensor window, extract 561 features, and predict."""
    window = generate_sensor_window(request.activity_hint)
    features = extract_features_from_window(window).tolist()
    uid = current_user.id if current_user else None
    user_health = user_state.get_health(uid, weight_kg=request.weight_kg)
    user_health.weight_kg = request.weight_kg

    model_key = "logistic_regression" if "logistic_regression" in _MODELS else list(_MODELS.keys())[0]
    return _predict_single(model_key, features, use_smoothing=True, user_id=uid)


# ---------------------------------------------------------------------------
# WebSocket Endpoint for Live Prediction Streaming
# ---------------------------------------------------------------------------

@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket, token: str = Query(default="")) -> None:
    """Stream live predictions over WebSocket.

    Requires a JWT token passed as query parameter: ``ws://host/ws/predict?token=<jwt>``.

    Supported JSON payloads:
    1. Simulation:
       {"action": "simulate", "activity_hint": "WALKING", "model_name": "logistic_regression"}
    2. Raw 9-channel sensor window (from static/sensor.html):
       {"action": "raw_sensor", "channels": {...}, "model_name": "cnn_lstm", "use_smoothing": true}
    3. UCI 561-features vector:
       {"features": [561 floats], "model_name": "logistic_regression", "use_smoothing": true}
    """
    # Authenticate the WebSocket connection
    ws_payload = authenticate_websocket_token(token)
    if ws_payload is None:
        await websocket.close(code=4001, reason="Authentication required. Pass ?token=<jwt>.")
        return

    ws_user_id: int | None = None
    try:
        with SessionLocal() as db_session:
            from src.database import get_user_by_username
            user = get_user_by_username(db_session, ws_payload.get("sub", ""))
            if user:
                ws_user_id = user.id
    except Exception:
        pass  # Proceed with None user_id

    await websocket.accept()
    logger.info("WebSocket client connected (user_id=%s).", ws_user_id)
    try:
        while True:
            data_text = await websocket.receive_text()
            data = json.loads(data_text)

            if data.get("action") == "simulate":
                act_hint = data.get("activity_hint", "WALKING")
                model_name = data.get("model_name", "logistic_regression")
                if model_name not in _MODELS:
                    model_name = list(_MODELS.keys())[0]

                win = generate_sensor_window(act_hint)
                feats = extract_features_from_window(win).tolist()
                res = _predict_single(model_name, feats, use_smoothing=True, user_id=ws_user_id)
                await websocket.send_json(res.model_dump() if hasattr(res, "model_dump") else res.dict())

            elif data.get("action") == "raw_sensor" and "channels" in data:
                ch = data["channels"]
                m_name = data.get("model_name", "logistic_regression")
                if m_name not in _MODELS:
                    m_name = list(_MODELS.keys())[0]

                if m_name == "cnn_lstm" and "cnn_lstm" in _MODELS:
                    # Feed raw 9-channel sequence (1, 128, 9) directly to CNN-LSTM
                    channel_order = [
                        "body_acc_x", "body_acc_y", "body_acc_z",
                        "body_gyro_x", "body_gyro_y", "body_gyro_z",
                        "total_acc_x", "total_acc_y", "total_acc_z",
                    ]
                    raw_matrix = np.column_stack([np.array(ch[k])[:128] for k in channel_order])
                    raw_tensor = raw_matrix.reshape(1, 128, 9)

                    t0 = time.perf_counter()
                    proba = _MODELS["cnn_lstm"].predict(raw_tensor, verbose=0)
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    lbl = int(np.argmax(proba, axis=1)[0])
                    cnf = float(np.max(proba))
                    act = _ACTIVITY_LABELS.get(lbl, f"UNKNOWN_{lbl}")

                    ws_smoother = user_state.get_smoother(ws_user_id)
                    sm_res = ws_smoother.add(lbl, cnf)
                    sm_act = _ACTIVITY_LABELS.get(sm_res.smoothed_label, f"UNKNOWN_{sm_res.smoothed_label}")
                    res = PredictResponse(
                        model_name="cnn_lstm",
                        predicted_label=lbl,
                        activity_name=act,
                        confidence=round(cnf, 6),
                        inference_time_ms=round(elapsed_ms, 4),
                        is_smoothed=True,
                        smoothed_label=sm_res.smoothed_label,
                        smoothed_activity=sm_act,
                        smoothed_confidence=sm_res.smoothed_confidence,
                    )
                else:
                    # 561-feature extraction from raw window
                    from src.sensor_simulator import SensorWindow
                    win = SensorWindow(
                        activity=data.get("activity_hint", "WALKING"),
                        channels={k: np.array(v)[:128] for k, v in ch.items()},
                        timestamp=time.time(),
                    )
                    feats = extract_features_from_window(win).tolist()
                    res = _predict_single(m_name, feats, use_smoothing=data.get("use_smoothing", True), user_id=ws_user_id)

                await websocket.send_json(res.model_dump() if hasattr(res, "model_dump") else res.dict())

            elif "features" in data:
                m_name = data.get("model_name", "logistic_regression")
                if m_name not in _MODELS:
                    m_name = list(_MODELS.keys())[0]
                res = _predict_single(m_name, data["features"], data.get("use_smoothing", True), user_id=ws_user_id)
                await websocket.send_json(res.model_dump() if hasattr(res, "model_dump") else res.dict())

            else:
                await websocket.send_json({"error": "Invalid request payload."})
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.exception("WebSocket error: %s", e)
        await websocket.close()


# ---------------------------------------------------------------------------
# History & Analytics Endpoints
# ---------------------------------------------------------------------------

@app.get("/history", tags=["History & Analytics"])
async def get_history(
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_optional_user),
) -> list[dict[str, Any]]:
    """Return recent predictions from SQLite database (falling back to memory)."""
    uid = current_user.id if current_user else None
    db_records = get_predictions(db, limit=limit, user_id=uid)
    if db_records:
        return db_records
    user_history = user_state.get_history(uid)
    return user_history.get_recent(limit)


@app.get("/history/stats", tags=["History & Analytics"])
async def get_history_stats(
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_optional_user),
) -> dict[str, Any]:
    """Return aggregate statistics across sessions and database records."""
    uid = current_user.id if current_user else None
    db_stats = get_prediction_stats(db, user_id=uid)
    user_history = user_state.get_history(uid)
    mem_stats = user_history.to_dict()
    mem_stats["db_records_total"] = db_stats.get("total_predictions", 0)
    mem_stats["db_per_model"] = db_stats.get("per_model", {})
    mem_stats["db_per_activity"] = db_stats.get("per_activity", {})
    return mem_stats


@app.get("/history/export", tags=["History & Analytics"])
async def export_history(
    format: str = "csv",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Export prediction history from database. Requires authentication."""
    import csv
    import io

    uid = current_user.id
    db_records = get_predictions(db, limit=10000, user_id=uid)

    if db_records:
        if format.lower() == "json":
            return db_records
        # CSV export from database records
        if db_records:
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=db_records[0].keys())
            writer.writeheader()
            writer.writerows(db_records)
            csv_content = output.getvalue()
            from fastapi.responses import Response
            return Response(
                content=csv_content,
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=har_history.csv"},
            )

    # Fallback to in-memory history
    user_history = user_state.get_history(uid)
    if format.lower() == "json":
        return user_history.export_json()
    return user_history.export_csv()


@app.get("/activity/summary", tags=["History & Analytics"])
async def get_activity_summary(
    date: str | None = None,
    current_user: User | None = Depends(get_optional_user),
) -> dict[str, Any]:
    from dataclasses import asdict
    uid = current_user.id if current_user else None
    user_tracker = user_state.get_tracker(uid)
    return asdict(user_tracker.get_daily_summary(date))


@app.get("/health-metrics", tags=["Health Metrics"])
async def get_health_metrics(
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_optional_user),
) -> dict[str, Any]:
    from dataclasses import asdict
    uid = current_user.id if current_user else None
    latest_db_snap = get_latest_health_snapshot(db, user_id=uid)
    if latest_db_snap:
        return latest_db_snap
    user_health = user_state.get_health(uid)
    return asdict(user_health.get_snapshot())


# ---------------------------------------------------------------------------
# Production Authentication Endpoints (JWT + bcrypt)
# ---------------------------------------------------------------------------

@app.post("/auth/register", response_model=TokenResponse, tags=["User Auth"])
@rate_limit("3/minute")
async def register(request: Request, payload: RegisterRequest, db: Session = Depends(get_db)) -> TokenResponse:
    """Register a new user account with bcrypt password hashing."""
    existing = get_user_by_username(db, payload.username)
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already registered.")

    hashed = hash_password(payload.password)
    user = add_user(db, username=payload.username, hashed_password=hashed)
    token = create_access_token({"sub": user.username, "user_id": user.id, "role": user.role})
    return TokenResponse(access_token=token, token_type="bearer", username=user.username, user_id=user.id)


@app.post("/auth/login", response_model=TokenResponse, tags=["User Auth"])
@rate_limit("5/minute")
async def login(request: Request, payload: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    """Authenticate with username & password, returns JWT bearer token."""
    user = get_user_by_username(db, payload.username)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password.")

    if not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password.")

    token = create_access_token({"sub": user.username, "user_id": user.id, "role": user.role})
    return TokenResponse(access_token=token, token_type="bearer", username=user.username, user_id=user.id)


@app.get("/auth/profile", tags=["User Auth"])
async def profile(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Get authenticated user profile."""
    uid = current_user.id
    user_history = user_state.get_history(uid)
    return {
        "user_id": current_user.id,
        "username": current_user.username,
        "role": current_user.role,
        "created_at": current_user.created_at.isoformat() if current_user.created_at else None,
        "total_predictions": len(current_user.predictions) if current_user.predictions else user_history.count,
    }
