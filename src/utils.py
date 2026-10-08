"""
Shared utility functions for the HAR project.

Provides common helpers used across data loading, training, evaluation,
and the API layer — including project root resolution, activity emoji
mapping, and duration formatting.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Activity ↔ emoji mapping
# ---------------------------------------------------------------------------
ACTIVITY_EMOJIS: dict[str, str] = {
    "WALKING": "🚶",
    "WALKING_UPSTAIRS": "⬆️",
    "WALKING_DOWNSTAIRS": "⬇️",
    "SITTING": "🪑",
    "STANDING": "🧍",
    "LAYING": "🛏️",
}


def get_project_root() -> Path:
    """Return the absolute path to the HAR project root directory.

    The project root is defined as the parent of the ``src/`` package
    directory (i.e. the directory that contains ``src/``, ``data/``,
    ``models/``, etc.).

    Returns
    -------
    Path
        Absolute path to the project root.
    """
    # This file lives at <project_root>/src/utils.py
    return Path(__file__).resolve().parent.parent


def get_activity_emoji(activity_name: str) -> str:
    """Map an activity label to its corresponding emoji.

    Parameters
    ----------
    activity_name : str
        The activity name in uppercase (e.g. ``"WALKING"``).

    Returns
    -------
    str
        A single emoji character, or ``"❓"`` if the activity is unknown.
    """
    return ACTIVITY_EMOJIS.get(activity_name.upper(), "❓")


def format_duration(seconds: float) -> str:
    """Format a duration in seconds into a human-readable string.

    Parameters
    ----------
    seconds : float
        Duration in seconds.

    Returns
    -------
    str
        Formatted string, e.g. ``"2m 35.12s"`` or ``"0.84s"``.
    """
    if seconds < 0:
        return "N/A"
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    remaining = seconds % 60
    if minutes < 60:
        return f"{minutes}m {remaining:.2f}s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h {mins}m {remaining:.2f}s"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging with a consistent format.

    Parameters
    ----------
    level : int, optional
        Logging level (default ``logging.INFO``).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
