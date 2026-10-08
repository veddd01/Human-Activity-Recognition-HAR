"""
Activity tracking for the HAR project.

Tracks per-activity durations, detects transitions, and provides
daily and weekly summary statistics from a stream of predictions.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ActivityTransition:
    """A detected transition between two activities."""

    timestamp: str
    from_activity: str
    to_activity: str
    duration_in_previous_s: float


@dataclass
class DailySummary:
    """Summary of activity for a single day."""

    date: str
    total_predictions: int
    active_time_s: float
    sedentary_time_s: float
    activity_durations_s: dict[str, float]
    activity_percentages: dict[str, float]
    transitions: int
    most_common_activity: str


class ActivityTracker:
    """Tracks activity durations and transitions over time.

    Parameters
    ----------
    prediction_interval_s : float
        Expected interval between predictions in seconds.
    """

    ACTIVE_ACTIVITIES = {"WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"}
    SEDENTARY_ACTIVITIES = {"SITTING", "STANDING", "LAYING"}

    def __init__(self, prediction_interval_s: float = 2.0) -> None:
        self._interval = prediction_interval_s
        self._current_activity: str | None = None
        self._current_start: datetime | None = None
        self._transitions: list[ActivityTransition] = []

        # Daily tracking
        self._daily_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._daily_transitions: dict[str, int] = defaultdict(int)
        self._total_predictions = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, activity_name: str) -> ActivityTransition | None:
        """Process a new activity prediction.

        Returns an ``ActivityTransition`` if the activity changed,
        otherwise ``None``.
        """
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")
        self._daily_counts[today][activity_name] += 1
        self._total_predictions += 1

        transition = None

        if self._current_activity is None:
            # First prediction
            self._current_activity = activity_name
            self._current_start = now
        elif activity_name != self._current_activity:
            # Transition detected
            duration = (now - self._current_start).total_seconds() if self._current_start else 0
            transition = ActivityTransition(
                timestamp=now.isoformat(),
                from_activity=self._current_activity,
                to_activity=activity_name,
                duration_in_previous_s=round(duration, 1),
            )
            self._transitions.append(transition)
            self._daily_transitions[today] += 1

            self._current_activity = activity_name
            self._current_start = now

        return transition

    def get_daily_summary(self, date: str | None = None) -> DailySummary:
        """Get activity summary for a specific date (YYYY-MM-DD).

        Defaults to today.
        """
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        counts = dict(self._daily_counts.get(date, {}))
        total = sum(counts.values())

        if total == 0:
            return DailySummary(
                date=date,
                total_predictions=0,
                active_time_s=0,
                sedentary_time_s=0,
                activity_durations_s={},
                activity_percentages={},
                transitions=0,
                most_common_activity="N/A",
            )

        durations = {
            name: round(cnt * self._interval, 1)
            for name, cnt in counts.items()
        }
        percentages = {
            name: round(cnt / total * 100, 1)
            for name, cnt in counts.items()
        }

        active_time = sum(
            dur for name, dur in durations.items()
            if name in self.ACTIVE_ACTIVITIES
        )
        sedentary_time = sum(
            dur for name, dur in durations.items()
            if name in self.SEDENTARY_ACTIVITIES
        )
        most_common = max(counts, key=counts.get)  # type: ignore[arg-type]

        return DailySummary(
            date=date,
            total_predictions=total,
            active_time_s=round(active_time, 1),
            sedentary_time_s=round(sedentary_time, 1),
            activity_durations_s=durations,
            activity_percentages=percentages,
            transitions=self._daily_transitions.get(date, 0),
            most_common_activity=most_common,
        )

    def get_weekly_summary(self) -> list[DailySummary]:
        """Get summaries for the last 7 days."""
        today = datetime.now(timezone.utc).date()
        summaries = []
        for i in range(7):
            date = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            summaries.append(self.get_daily_summary(date))
        return summaries

    def get_recent_transitions(self, n: int = 20) -> list[dict[str, Any]]:
        """Return the *n* most recent transitions."""
        from dataclasses import asdict
        return [asdict(t) for t in self._transitions[-n:]]

    @property
    def current_activity(self) -> str | None:
        return self._current_activity

    @property
    def total_transitions(self) -> int:
        return len(self._transitions)

    def reset(self) -> None:
        """Reset all tracking state."""
        self._current_activity = None
        self._current_start = None
        self._transitions.clear()
        self._daily_counts.clear()
        self._daily_transitions.clear()
        self._total_predictions = 0
