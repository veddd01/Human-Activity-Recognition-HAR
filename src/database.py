"""
SQLAlchemy database models and helpers for the HAR project.

Provides persistent storage for prediction history, activity sessions,
health metric snapshots, and user accounts using SQLite.
"""

from __future__ import annotations

import datetime
import logging
import os
from pathlib import Path
from typing import Any, Generator

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
    func,
)
from sqlalchemy.orm import (
    Session,
    declarative_base,
    relationship,
    sessionmaker,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engine & session
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{_PROJECT_ROOT / 'data' / 'har_data.db'}")
# Ensure the data directory exists for SQLite
Path(DATABASE_URL.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------

class User(Base):
    """Registered user account."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    hashed_password = Column(String(256), nullable=False)
    role = Column(String(32), default="user")
    created_at = Column(
        DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc)
    )

    predictions = relationship("PredictionRecord", back_populates="user")
    sessions = relationship("ActivitySession", back_populates="user")
    health_snapshots = relationship("HealthMetricSnapshot", back_populates="user")


class PredictionRecord(Base):
    """Single prediction event persisted to the database."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    model_name = Column(String(64), nullable=False)
    predicted_label = Column(Integer, nullable=False)
    activity_name = Column(String(64), nullable=False)
    confidence = Column(Float, nullable=True)
    is_smoothed = Column(Boolean, default=False)
    smoothed_label = Column(Integer, nullable=True)
    smoothed_activity = Column(String(64), nullable=True)
    smoothed_confidence = Column(Float, nullable=True)
    inference_time_ms = Column(Float, nullable=True)
    created_at = Column(
        DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc)
    )

    user = relationship("User", back_populates="predictions")


class ActivitySession(Base):
    """Contiguous period spent in a single activity."""

    __tablename__ = "activity_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    activity_name = Column(String(64), nullable=False)
    started_at = Column(
        DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    ended_at = Column(DateTime, nullable=True)
    duration_s = Column(Float, nullable=True)

    user = relationship("User", back_populates="sessions")


class HealthMetricSnapshot(Base):
    """Point-in-time snapshot of cumulative health metrics."""

    __tablename__ = "health_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    total_steps = Column(Integer, default=0)
    calories_burned = Column(Float, default=0.0)
    distance_m = Column(Float, default=0.0)
    active_time_s = Column(Float, default=0.0)
    sedentary_time_s = Column(Float, default=0.0)
    created_at = Column(
        DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc)
    )

    user = relationship("User", back_populates="health_snapshots")


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create all tables if they do not exist."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created / verified.")


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency — yields a database session and closes after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Helper CRUD functions
# ---------------------------------------------------------------------------

def add_prediction(db: Session, **kwargs: Any) -> PredictionRecord:
    """Insert a prediction record and return it."""
    record = PredictionRecord(**kwargs)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_predictions(db: Session, limit: int = 50, user_id: int | None = None) -> list[dict[str, Any]]:
    """Return the *limit* most recent predictions as dicts, optionally filtered by user."""
    query = db.query(PredictionRecord).order_by(PredictionRecord.created_at.desc())
    if user_id is not None:
        query = query.filter(PredictionRecord.user_id == user_id)
    rows = query.limit(limit).all()
    return [
        {
            "id": r.id,
            "model_name": r.model_name,
            "predicted_label": r.predicted_label,
            "activity_name": r.activity_name,
            "confidence": r.confidence,
            "is_smoothed": r.is_smoothed,
            "smoothed_label": r.smoothed_label,
            "smoothed_activity": r.smoothed_activity,
            "smoothed_confidence": r.smoothed_confidence,
            "inference_time_ms": r.inference_time_ms,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


def get_prediction_stats(db: Session, user_id: int | None = None) -> dict[str, Any]:
    """Aggregate prediction statistics, optionally scoped to a user."""
    base_query = db.query(PredictionRecord)
    if user_id is not None:
        base_query = base_query.filter(PredictionRecord.user_id == user_id)
    
    total = base_query.with_entities(func.count(PredictionRecord.id)).scalar() or 0
    per_model = dict(
        base_query.with_entities(PredictionRecord.model_name, func.count(PredictionRecord.id))
        .group_by(PredictionRecord.model_name)
        .all()
    )
    per_activity = dict(
        base_query.with_entities(PredictionRecord.activity_name, func.count(PredictionRecord.id))
        .group_by(PredictionRecord.activity_name)
        .all()
    )
    return {"total_predictions": total, "per_model": per_model, "per_activity": per_activity}


# ---- User helpers ---------------------------------------------------------

def add_user(db: Session, username: str, hashed_password: str) -> User:
    """Create a new user and return the ORM instance."""
    user = User(username=username, hashed_password=hashed_password)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user_by_username(db: Session, username: str) -> User | None:
    """Look up a user by username; returns ``None`` if not found."""
    return db.query(User).filter(User.username == username).first()


# ---- Health snapshot helpers -----------------------------------------------

def save_health_snapshot(db: Session, **kwargs: Any) -> HealthMetricSnapshot:
    """Persist a health metric snapshot."""
    snap = HealthMetricSnapshot(**kwargs)
    db.add(snap)
    db.commit()
    db.refresh(snap)
    return snap


def get_latest_health_snapshot(db: Session, user_id: int | None = None) -> dict[str, Any] | None:
    """Return the most recent health snapshot as a dict, optionally scoped to a user."""
    query = db.query(HealthMetricSnapshot).order_by(HealthMetricSnapshot.created_at.desc())
    if user_id is not None:
        query = query.filter(HealthMetricSnapshot.user_id == user_id)
    row = query.first()
    if row is None:
        return None
    return {
        "total_steps": row.total_steps,
        "calories_burned": row.calories_burned,
        "distance_m": row.distance_m,
        "active_time_s": row.active_time_s,
        "sedentary_time_s": row.sedentary_time_s,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }
