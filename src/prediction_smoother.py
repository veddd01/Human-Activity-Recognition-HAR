"""
Prediction smoothing for the HAR project.

Reduces noisy predictions by applying rolling window majority voting
and confidence averaging across a configurable window of recent predictions.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SmoothedPrediction:
    """Result of a smoothed prediction."""

    raw_label: int
    smoothed_label: int
    raw_confidence: float
    smoothed_confidence: float
    window_size: int
    votes: dict[int, int]


class PredictionSmoother:
    """Rolling window smoother for activity predictions.

    Maintains a fixed-size window of recent predictions and applies
    majority voting and confidence averaging to reduce noise.

    Parameters
    ----------
    window_size : int
        Number of recent predictions to keep in the rolling window.
        Default is 5.
    """

    def __init__(self, window_size: int = 5) -> None:
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self._window_size = window_size
        self._labels: deque[int] = deque(maxlen=window_size)
        self._confidences: deque[float] = deque(maxlen=window_size)
        # Per-label confidence accumulator for weighted voting
        self._label_confidences: deque[tuple[int, float]] = deque(maxlen=window_size)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def window_size(self) -> int:
        return self._window_size

    @window_size.setter
    def window_size(self, value: int) -> None:
        if value < 1:
            raise ValueError("window_size must be >= 1")
        self._window_size = value
        # Resize deques
        new_labels: deque[int] = deque(self._labels, maxlen=value)
        new_confs: deque[float] = deque(self._confidences, maxlen=value)
        new_lc: deque[tuple[int, float]] = deque(self._label_confidences, maxlen=value)
        self._labels = new_labels
        self._confidences = new_confs
        self._label_confidences = new_lc

    @property
    def current_window(self) -> list[int]:
        return list(self._labels)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, label: int, confidence: float = 1.0) -> SmoothedPrediction:
        """Add a new raw prediction and return the smoothed result.

        Parameters
        ----------
        label : int
            Raw predicted label index (0-based).
        confidence : float
            Prediction confidence in [0, 1].  Default 1.0.

        Returns
        -------
        SmoothedPrediction
        """
        self._labels.append(label)
        self._confidences.append(confidence)
        self._label_confidences.append((label, confidence))

        # --- majority voting ---
        vote_counts: dict[int, int] = {}
        for lbl in self._labels:
            vote_counts[lbl] = vote_counts.get(lbl, 0) + 1

        smoothed_label = max(vote_counts, key=lambda k: vote_counts[k])

        # --- confidence averaging for the winning label ---
        winning_confs = [c for l, c in self._label_confidences if l == smoothed_label]
        smoothed_conf = float(np.mean(winning_confs)) if winning_confs else confidence

        return SmoothedPrediction(
            raw_label=label,
            smoothed_label=smoothed_label,
            raw_confidence=confidence,
            smoothed_confidence=round(smoothed_conf, 6),
            window_size=len(self._labels),
            votes=vote_counts,
        )

    def reset(self) -> None:
        """Clear the rolling window."""
        self._labels.clear()
        self._confidences.clear()
        self._label_confidences.clear()

    def to_dict(self) -> dict[str, Any]:
        """Serialise current state for API responses."""
        return {
            "window_size": self._window_size,
            "current_length": len(self._labels),
            "labels_in_window": list(self._labels),
            "confidences_in_window": [round(c, 4) for c in self._confidences],
        }
