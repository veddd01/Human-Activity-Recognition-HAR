"""
Sensor data simulator for the HAR project.

Generates realistic accelerometer and gyroscope data streams that mimic
smartphone sensor readings for each of the six HAR activities.  Used for
demo/testing when real hardware is unavailable.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-activity signal profiles (mean, std for each axis)
# These approximate real UCI HAR sensor characteristics.
# ---------------------------------------------------------------------------
ACTIVITY_PROFILES: dict[str, dict[str, tuple[float, float]]] = {
    "WALKING": {
        "body_acc_x": (0.28, 0.15),  "body_acc_y": (-0.02, 0.12),  "body_acc_z": (-0.10, 0.10),
        "body_gyro_x": (-0.02, 0.25), "body_gyro_y": (0.05, 0.18),  "body_gyro_z": (0.01, 0.15),
        "total_acc_x": (1.02, 0.18),  "total_acc_y": (-0.03, 0.12), "total_acc_z": (-0.05, 0.12),
    },
    "WALKING_UPSTAIRS": {
        "body_acc_x": (0.25, 0.18),  "body_acc_y": (0.05, 0.14),   "body_acc_z": (-0.15, 0.12),
        "body_gyro_x": (0.10, 0.30), "body_gyro_y": (-0.05, 0.22), "body_gyro_z": (0.03, 0.18),
        "total_acc_x": (1.05, 0.20), "total_acc_y": (0.05, 0.14),  "total_acc_z": (-0.10, 0.14),
    },
    "WALKING_DOWNSTAIRS": {
        "body_acc_x": (0.30, 0.20),  "body_acc_y": (-0.03, 0.15),  "body_acc_z": (-0.08, 0.14),
        "body_gyro_x": (-0.05, 0.28), "body_gyro_y": (0.08, 0.20), "body_gyro_z": (-0.02, 0.16),
        "total_acc_x": (1.08, 0.22), "total_acc_y": (-0.02, 0.15), "total_acc_z": (-0.03, 0.15),
    },
    "SITTING": {
        "body_acc_x": (0.00, 0.02),  "body_acc_y": (0.00, 0.02),   "body_acc_z": (0.00, 0.02),
        "body_gyro_x": (0.00, 0.01), "body_gyro_y": (0.00, 0.01),  "body_gyro_z": (0.00, 0.01),
        "total_acc_x": (0.98, 0.02), "total_acc_y": (0.00, 0.02),  "total_acc_z": (0.00, 0.02),
    },
    "STANDING": {
        "body_acc_x": (0.00, 0.01),  "body_acc_y": (0.00, 0.01),   "body_acc_z": (0.00, 0.01),
        "body_gyro_x": (0.00, 0.005), "body_gyro_y": (0.00, 0.005), "body_gyro_z": (0.00, 0.005),
        "total_acc_x": (0.99, 0.01), "total_acc_y": (0.00, 0.01),  "total_acc_z": (0.00, 0.01),
    },
    "LAYING": {
        "body_acc_x": (0.00, 0.015), "body_acc_y": (0.00, 0.015),  "body_acc_z": (0.00, 0.015),
        "body_gyro_x": (0.00, 0.008), "body_gyro_y": (0.00, 0.008), "body_gyro_z": (0.00, 0.008),
        "total_acc_x": (0.20, 0.02), "total_acc_y": (0.00, 0.02),  "total_acc_z": (-0.95, 0.02),
    },
}

CHANNEL_NAMES = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]

# UCI HAR: 50 Hz sampling, 2.56-second windows = 128 samples
WINDOW_SIZE = 128
SAMPLING_RATE = 50  # Hz


@dataclass
class SensorWindow:
    """A single window of raw sensor data."""

    activity: str
    channels: dict[str, np.ndarray]   # channel_name → array[128]
    timestamp: float


def generate_sensor_window(
    activity: str,
    rng: np.random.Generator | None = None,
) -> SensorWindow:
    """Generate a single 128-sample sensor window for the given activity.

    Parameters
    ----------
    activity : str
        Activity name (must be a key in ``ACTIVITY_PROFILES``).
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    SensorWindow
    """
    if rng is None:
        rng = np.random.default_rng()

    profile = ACTIVITY_PROFILES.get(activity, ACTIVITY_PROFILES["STANDING"])
    channels: dict[str, np.ndarray] = {}

    for ch_name in CHANNEL_NAMES:
        mean, std = profile[ch_name]
        signal = rng.normal(mean, std, WINDOW_SIZE)

        # Add periodic component for walking activities
        if activity.startswith("WALKING"):
            freq = 1.8 if activity == "WALKING" else (1.4 if "UPSTAIRS" in activity else 2.0)
            t = np.arange(WINDOW_SIZE) / SAMPLING_RATE
            amplitude = std * 0.6
            signal += amplitude * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))

        channels[ch_name] = signal.astype(np.float32)

    return SensorWindow(
        activity=activity,
        channels=channels,
        timestamp=time.time(),
    )


def extract_features_from_window(window: SensorWindow) -> np.ndarray:
    """Extract a 561-dimensional feature vector from a sensor window.

    This is a simplified feature extraction that produces 561 features
    matching the UCI HAR dataset dimensionality.  It computes time-domain
    and frequency-domain statistics for each of the 9 sensor channels.

    Parameters
    ----------
    window : SensorWindow

    Returns
    -------
    np.ndarray
        Feature vector of shape (561,).
    """
    features: list[float] = []

    all_signals = [window.channels[ch] for ch in CHANNEL_NAMES]

    for signal in all_signals:
        # Time-domain features (per channel)
        features.append(float(np.mean(signal)))
        features.append(float(np.std(signal)))
        features.append(float(np.median(signal)))
        features.append(float(np.max(signal)))
        features.append(float(np.min(signal)))
        features.append(float(np.max(signal) - np.min(signal)))  # range
        features.append(float(np.mean(np.abs(signal - np.mean(signal)))))  # MAD
        sma = float(np.sum(np.abs(signal)) / len(signal))
        features.append(sma)
        features.append(float(np.sum(signal ** 2) / len(signal)))  # energy
        q75, q25 = np.percentile(signal, [75, 25])
        features.append(float(q75 - q25))  # IQR

        # Frequency-domain features (per channel)
        fft = np.fft.fft(signal)
        fft_mag = np.abs(fft[:WINDOW_SIZE // 2])
        features.append(float(np.mean(fft_mag)))
        features.append(float(np.std(fft_mag)))
        features.append(float(np.max(fft_mag)))
        features.append(float(np.argmax(fft_mag)))  # dominant freq index
        features.append(float(np.sum(fft_mag ** 2) / len(fft_mag)))  # spectral energy

        # Entropy
        fft_norm = fft_mag / (np.sum(fft_mag) + 1e-10)
        entropy = -np.sum(fft_norm * np.log2(fft_norm + 1e-10))
        features.append(float(entropy))

        # Band energies (4 bands)
        band_size = len(fft_mag) // 4
        for b in range(4):
            band = fft_mag[b * band_size: (b + 1) * band_size]
            features.append(float(np.sum(band ** 2) / len(band)))

    # Cross-channel features (acceleration magnitudes, correlations)
    acc_x = all_signals[0]
    acc_y = all_signals[1]
    acc_z = all_signals[2]
    acc_mag = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)

    gyro_x = all_signals[3]
    gyro_y = all_signals[4]
    gyro_z = all_signals[5]
    gyro_mag = np.sqrt(gyro_x ** 2 + gyro_y ** 2 + gyro_z ** 2)

    for mag_signal in [acc_mag, gyro_mag]:
        features.append(float(np.mean(mag_signal)))
        features.append(float(np.std(mag_signal)))
        features.append(float(np.max(mag_signal)))
        features.append(float(np.min(mag_signal)))
        features.append(float(np.median(mag_signal)))

    # Correlations
    for a, b in [(acc_x, acc_y), (acc_x, acc_z), (acc_y, acc_z),
                  (gyro_x, gyro_y), (gyro_x, gyro_z), (gyro_y, gyro_z)]:
        corr = float(np.corrcoef(a, b)[0, 1])
        features.append(corr if not np.isnan(corr) else 0.0)

    # Jerk features (derivative)
    for signal in all_signals[:6]:
        jerk = np.diff(signal) * SAMPLING_RATE
        features.append(float(np.mean(jerk)))
        features.append(float(np.std(jerk)))
        features.append(float(np.max(np.abs(jerk))))

    # Pad or truncate to exactly 561
    if len(features) < 561:
        features.extend([0.0] * (561 - len(features)))
    elif len(features) > 561:
        features = features[:561]

    return np.array(features, dtype=np.float32)


def sensor_stream(
    activities: list[str] | None = None,
    interval_s: float = 2.0,
    max_windows: int | None = None,
    seed: int | None = None,
) -> Generator[tuple[SensorWindow, np.ndarray], None, None]:
    """Generate a continuous stream of sensor windows and feature vectors.

    Parameters
    ----------
    activities : list[str], optional
        Activities to cycle through.  Defaults to all six.
    interval_s : float
        Pause between windows in seconds.  Default 2.0.
    max_windows : int, optional
        Stop after this many windows.  ``None`` = infinite.
    seed : int, optional
        Random seed for reproducibility.

    Yields
    ------
    (SensorWindow, np.ndarray)
        Sensor window and its 561-dimensional feature vector.
    """
    if activities is None:
        activities = list(ACTIVITY_PROFILES.keys())

    rng = np.random.default_rng(seed)
    count = 0

    while max_windows is None or count < max_windows:
        activity = activities[count % len(activities)]
        window = generate_sensor_window(activity, rng)
        features = extract_features_from_window(window)
        yield window, features
        count += 1
        if interval_s > 0:
            time.sleep(interval_s)
