"""
Health metrics estimation for the HAR project.

Provides step counting, calorie estimation, active time calculation,
and distance estimation based on predicted activities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MET values for each activity (Metabolic Equivalent of Task)
# Source: Compendium of Physical Activities
# ---------------------------------------------------------------------------
ACTIVITY_MET: dict[str, float] = {
    "WALKING": 3.5,
    "WALKING_UPSTAIRS": 8.0,
    "WALKING_DOWNSTAIRS": 3.5,
    "SITTING": 1.3,
    "STANDING": 1.8,
    "LAYING": 1.0,
}

# Average steps per second by activity
ACTIVITY_STEPS_PER_SECOND: dict[str, float] = {
    "WALKING": 1.8,           # ~108 steps/min
    "WALKING_UPSTAIRS": 1.2,  # ~72 steps/min (slower)
    "WALKING_DOWNSTAIRS": 1.5, # ~90 steps/min
    "SITTING": 0.0,
    "STANDING": 0.0,
    "LAYING": 0.0,
}

# Average speed in m/s by activity
ACTIVITY_SPEED_MS: dict[str, float] = {
    "WALKING": 1.4,            # ~5 km/h
    "WALKING_UPSTAIRS": 0.5,   # ~1.8 km/h
    "WALKING_DOWNSTAIRS": 0.8, # ~2.9 km/h
    "SITTING": 0.0,
    "STANDING": 0.0,
    "LAYING": 0.0,
}


@dataclass
class HealthSnapshot:
    """Current health metrics snapshot."""

    total_steps: int
    calories_burned: float
    distance_m: float
    distance_km: float
    active_time_s: float
    active_time_min: float
    sedentary_time_s: float
    sedentary_time_min: float
    activity_calories: dict[str, float]
    activity_steps: dict[str, int]


class HealthMetrics:
    """Estimates health metrics from activity predictions.

    Parameters
    ----------
    weight_kg : float
        User body weight in kilograms (used for calorie estimation).
        Default 70.0.
    prediction_interval_s : float
        Expected interval between predictions in seconds.
    """

    ACTIVE_ACTIVITIES = {"WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"}

    def __init__(
        self,
        weight_kg: float = 70.0,
        prediction_interval_s: float = 2.0,
    ) -> None:
        self._weight_kg = weight_kg
        self._interval = prediction_interval_s

        # Accumulators
        self._steps: dict[str, float] = {}
        self._calories: dict[str, float] = {}
        self._distance: dict[str, float] = {}
        self._active_time: float = 0.0
        self._sedentary_time: float = 0.0
        self._total_predictions: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def weight_kg(self) -> float:
        return self._weight_kg

    @weight_kg.setter
    def weight_kg(self, value: float) -> None:
        if value <= 0:
            raise ValueError("weight_kg must be positive")
        self._weight_kg = value

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, activity_name: str) -> None:
        """Process a new activity prediction and update metrics.

        Parameters
        ----------
        activity_name : str
            Predicted activity name (e.g. ``"WALKING"``).
        """
        self._total_predictions += 1

        # Steps
        steps_rate = ACTIVITY_STEPS_PER_SECOND.get(activity_name, 0.0)
        steps_added = steps_rate * self._interval
        self._steps[activity_name] = self._steps.get(activity_name, 0.0) + steps_added

        # Calories: cal/s = MET × weight_kg × 3.5 / (200 × 60)
        met = ACTIVITY_MET.get(activity_name, 1.0)
        cal_per_second = met * self._weight_kg * 3.5 / 12000
        cal_added = cal_per_second * self._interval
        self._calories[activity_name] = self._calories.get(activity_name, 0.0) + cal_added

        # Distance
        speed = ACTIVITY_SPEED_MS.get(activity_name, 0.0)
        dist_added = speed * self._interval
        self._distance[activity_name] = self._distance.get(activity_name, 0.0) + dist_added

        # Active vs sedentary
        if activity_name in self.ACTIVE_ACTIVITIES:
            self._active_time += self._interval
        else:
            self._sedentary_time += self._interval

    def get_snapshot(self) -> HealthSnapshot:
        """Return current health metrics."""
        total_steps = int(sum(self._steps.values()))
        total_cal = round(sum(self._calories.values()), 1)
        total_dist = round(sum(self._distance.values()), 1)

        return HealthSnapshot(
            total_steps=total_steps,
            calories_burned=total_cal,
            distance_m=total_dist,
            distance_km=round(total_dist / 1000, 2),
            active_time_s=round(self._active_time, 1),
            active_time_min=round(self._active_time / 60, 1),
            sedentary_time_s=round(self._sedentary_time, 1),
            sedentary_time_min=round(self._sedentary_time / 60, 1),
            activity_calories={k: round(v, 2) for k, v in self._calories.items()},
            activity_steps={k: int(v) for k, v in self._steps.items()},
        )

    def reset(self) -> None:
        """Reset all metrics."""
        self._steps.clear()
        self._calories.clear()
        self._distance.clear()
        self._active_time = 0.0
        self._sedentary_time = 0.0
        self._total_predictions = 0
