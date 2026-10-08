"""
Prediction history storage for the HAR project.

Records timestamped predictions with optional session management,
statistics, and export capabilities.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """A single timestamped prediction."""

    timestamp: str
    model_name: str
    predicted_label: int
    activity_name: str
    confidence: float | None
    is_smoothed: bool = False
    smoothed_label: int | None = None
    smoothed_activity: str | None = None
    smoothed_confidence: float | None = None


@dataclass
class SessionStats:
    """Statistics for a prediction session."""

    session_id: str
    start_time: str
    end_time: str | None
    total_predictions: int
    activity_counts: dict[str, int]
    activity_durations_s: dict[str, float]
    average_confidence: float
    transitions: int


class PredictionHistory:
    """In-memory prediction history with session management.

    Parameters
    ----------
    max_records : int
        Maximum number of records to keep per session.  Oldest are
        dropped when the limit is reached.  Default 10_000.
    """

    def __init__(self, max_records: int = 10_000) -> None:
        self._max_records = max_records
        self._records: list[PredictionRecord] = []
        self._session_id: str = str(uuid.uuid4())[:8]
        self._session_start: str = datetime.now(timezone.utc).isoformat()
        self._session_end: str | None = None
        self._prediction_interval_s: float = 2.0  # assumed interval between preds

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def records(self) -> list[PredictionRecord]:
        return list(self._records)

    @property
    def count(self) -> int:
        return len(self._records)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        model_name: str,
        predicted_label: int,
        activity_name: str,
        confidence: float | None = None,
        is_smoothed: bool = False,
        smoothed_label: int | None = None,
        smoothed_activity: str | None = None,
        smoothed_confidence: float | None = None,
    ) -> PredictionRecord:
        """Record a new prediction."""
        record = PredictionRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_name=model_name,
            predicted_label=predicted_label,
            activity_name=activity_name,
            confidence=confidence,
            is_smoothed=is_smoothed,
            smoothed_label=smoothed_label,
            smoothed_activity=smoothed_activity,
            smoothed_confidence=smoothed_confidence,
        )
        self._records.append(record)

        # Trim if over max
        if len(self._records) > self._max_records:
            self._records = self._records[-self._max_records:]

        return record

    def get_recent(self, n: int = 50) -> list[dict[str, Any]]:
        """Return the *n* most recent records as dicts."""
        return [asdict(r) for r in self._records[-n:]]

    def get_statistics(self) -> SessionStats:
        """Compute session statistics."""
        if not self._records:
            return SessionStats(
                session_id=self._session_id,
                start_time=self._session_start,
                end_time=self._session_end,
                total_predictions=0,
                activity_counts={},
                activity_durations_s={},
                average_confidence=0.0,
                transitions=0,
            )

        # Activity counts
        counts: dict[str, int] = {}
        for r in self._records:
            name = r.smoothed_activity or r.activity_name
            counts[name] = counts.get(name, 0) + 1

        # Durations (approximate: count × interval)
        durations = {
            name: round(cnt * self._prediction_interval_s, 1)
            for name, cnt in counts.items()
        }

        # Average confidence
        confs = [
            r.smoothed_confidence if r.smoothed_confidence is not None else r.confidence
            for r in self._records
            if (r.smoothed_confidence is not None or r.confidence is not None)
        ]
        avg_conf = float(sum(confs) / len(confs)) if confs else 0.0

        # Transitions
        transitions = 0
        for i in range(1, len(self._records)):
            prev = self._records[i - 1].smoothed_activity or self._records[i - 1].activity_name
            curr = self._records[i].smoothed_activity or self._records[i].activity_name
            if prev != curr:
                transitions += 1

        return SessionStats(
            session_id=self._session_id,
            start_time=self._session_start,
            end_time=self._session_end or datetime.now(timezone.utc).isoformat(),
            total_predictions=len(self._records),
            activity_counts=counts,
            activity_durations_s=durations,
            average_confidence=round(avg_conf, 4),
            transitions=transitions,
        )

    def export_csv(self) -> str:
        """Export all records as a CSV string."""
        if not self._records:
            return ""
        buf = io.StringIO()
        fieldnames = list(asdict(self._records[0]).keys())
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for r in self._records:
            writer.writerow(asdict(r))
        return buf.getvalue()

    def export_json(self) -> str:
        """Export all records as a JSON string."""
        return json.dumps([asdict(r) for r in self._records], indent=2)

    def new_session(self) -> str:
        """Start a new session, clearing all records."""
        self._records.clear()
        self._session_id = str(uuid.uuid4())[:8]
        self._session_start = datetime.now(timezone.utc).isoformat()
        self._session_end = None
        return self._session_id

    def end_session(self) -> SessionStats:
        """End the current session and return final statistics."""
        self._session_end = datetime.now(timezone.utc).isoformat()
        return self.get_statistics()

    def to_dict(self) -> dict[str, Any]:
        """Serialise state for API responses."""
        stats = self.get_statistics()
        return {
            "session_id": self._session_id,
            "total_predictions": len(self._records),
            "statistics": asdict(stats),
        }
