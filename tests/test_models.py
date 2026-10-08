"""
Unit tests for sensor simulation, preprocessing, smoothing, tracker, and health metrics.
"""

import unittest
import numpy as np
from src.sensor_simulator import (
    generate_sensor_window,
    extract_features_from_window,
    CHANNEL_NAMES,
    ACTIVITY_PROFILES,
    WINDOW_SIZE,
)
from src.prediction_smoother import PredictionSmoother
from src.prediction_history import PredictionHistory
from src.activity_tracker import ActivityTracker
from src.health_metrics import HealthMetrics


class TestSensorSimulator(unittest.TestCase):
    """Tests for sensor window generation and feature extraction."""

    def test_generate_window_shape(self):
        window = generate_sensor_window("WALKING")
        self.assertEqual(window.activity, "WALKING")
        self.assertEqual(len(window.channels), 9)
        for ch_name in CHANNEL_NAMES:
            self.assertIn(ch_name, window.channels)
            self.assertEqual(window.channels[ch_name].shape, (WINDOW_SIZE,))

    def test_generate_window_all_activities(self):
        for activity in ACTIVITY_PROFILES:
            window = generate_sensor_window(activity)
            self.assertEqual(window.activity, activity)
            self.assertEqual(len(window.channels), 9)

    def test_generate_window_unknown_activity_fallback(self):
        window = generate_sensor_window("UNKNOWN_ACTIVITY")
        self.assertEqual(window.activity, "UNKNOWN_ACTIVITY")
        # Falls back to STANDING profile
        self.assertEqual(len(window.channels), 9)

    def test_feature_extraction_shape(self):
        window = generate_sensor_window("WALKING")
        features = extract_features_from_window(window)
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape, (561,))

    def test_feature_extraction_no_nan(self):
        window = generate_sensor_window("SITTING")
        features = extract_features_from_window(window)
        self.assertFalse(np.any(np.isnan(features)))

    def test_feature_extraction_dtype(self):
        window = generate_sensor_window("LAYING")
        features = extract_features_from_window(window)
        self.assertEqual(features.dtype, np.float32)

    def test_reproducibility_with_seed(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        w1 = generate_sensor_window("WALKING", rng=rng1)
        w2 = generate_sensor_window("WALKING", rng=rng2)
        for ch in CHANNEL_NAMES:
            np.testing.assert_array_equal(w1.channels[ch], w2.channels[ch])

    def test_window_has_timestamp(self):
        window = generate_sensor_window("WALKING")
        self.assertIsInstance(window.timestamp, float)
        self.assertGreater(window.timestamp, 0)


class TestPredictionSmoother(unittest.TestCase):
    """Tests for rolling-window prediction smoothing."""

    def test_single_prediction(self):
        smoother = PredictionSmoother(window_size=3)
        res = smoother.add(label=0, confidence=0.9)
        self.assertEqual(res.smoothed_label, 0)
        self.assertEqual(res.raw_label, 0)

    def test_majority_voting(self):
        smoother = PredictionSmoother(window_size=3)
        smoother.add(label=0, confidence=0.9)
        smoother.add(label=1, confidence=0.8)
        res = smoother.add(label=1, confidence=0.95)
        self.assertEqual(res.smoothed_label, 1)
        self.assertEqual(res.window_size, 3)

    def test_confidence_averaging(self):
        smoother = PredictionSmoother(window_size=3)
        smoother.add(label=0, confidence=0.8)
        smoother.add(label=0, confidence=0.9)
        res = smoother.add(label=0, confidence=1.0)
        self.assertAlmostEqual(res.smoothed_confidence, 0.9, places=4)

    def test_window_overflow(self):
        smoother = PredictionSmoother(window_size=3)
        smoother.add(0, 0.9)
        smoother.add(0, 0.9)
        smoother.add(0, 0.9)
        # Now window is full. Adding label=1 should still be majority 0
        res = smoother.add(1, 0.95)
        self.assertEqual(res.smoothed_label, 0)

    def test_invalid_window_size(self):
        with self.assertRaises(ValueError):
            PredictionSmoother(window_size=0)

    def test_reset(self):
        smoother = PredictionSmoother(window_size=3)
        smoother.add(0, 0.9)
        smoother.reset()
        self.assertEqual(len(smoother.current_window), 0)


class TestPredictionHistory(unittest.TestCase):
    """Tests for in-memory prediction history."""

    def test_add_and_count(self):
        history = PredictionHistory()
        history.add("logistic_regression", 0, "WALKING", 0.95)
        history.add("logistic_regression", 1, "SITTING", 0.85)
        self.assertEqual(history.count, 2)

    def test_statistics(self):
        history = PredictionHistory()
        history.add("logistic_regression", 0, "WALKING", 0.95)
        history.add("logistic_regression", 1, "SITTING", 0.85)
        stats = history.get_statistics()
        self.assertEqual(stats.total_predictions, 2)
        self.assertEqual(stats.transitions, 1)

    def test_max_records(self):
        history = PredictionHistory(max_records=5)
        for i in range(10):
            history.add("model", i % 6, f"ACT_{i % 6}", 0.9)
        self.assertEqual(history.count, 5)

    def test_export_csv(self):
        history = PredictionHistory()
        history.add("logistic_regression", 0, "WALKING", 0.95)
        csv = history.export_csv()
        self.assertIn("model_name", csv)
        self.assertIn("WALKING", csv)

    def test_export_json(self):
        history = PredictionHistory()
        history.add("logistic_regression", 0, "WALKING", 0.95)
        json_str = history.export_json()
        self.assertIn("WALKING", json_str)

    def test_new_session_clears(self):
        history = PredictionHistory()
        history.add("model", 0, "WALKING", 0.9)
        old_id = history.session_id
        new_id = history.new_session()
        self.assertNotEqual(old_id, new_id)
        self.assertEqual(history.count, 0)


class TestActivityTracker(unittest.TestCase):
    """Tests for activity tracking and transition detection."""

    def test_first_update_no_transition(self):
        tracker = ActivityTracker(prediction_interval_s=2.0)
        result = tracker.update("WALKING")
        self.assertIsNone(result)

    def test_same_activity_no_transition(self):
        tracker = ActivityTracker(prediction_interval_s=2.0)
        tracker.update("WALKING")
        result = tracker.update("WALKING")
        self.assertIsNone(result)

    def test_transition_detected(self):
        tracker = ActivityTracker(prediction_interval_s=2.0)
        tracker.update("WALKING")
        result = tracker.update("SITTING")
        self.assertIsNotNone(result)
        self.assertEqual(result.from_activity, "WALKING")
        self.assertEqual(result.to_activity, "SITTING")

    def test_daily_summary(self):
        tracker = ActivityTracker(prediction_interval_s=2.0)
        tracker.update("WALKING")
        tracker.update("WALKING")
        tracker.update("SITTING")
        summary = tracker.get_daily_summary()
        self.assertGreater(summary.total_predictions, 0)

    def test_reset(self):
        tracker = ActivityTracker()
        tracker.update("WALKING")
        tracker.reset()
        self.assertIsNone(tracker.current_activity)
        self.assertEqual(tracker.total_transitions, 0)


class TestHealthMetrics(unittest.TestCase):
    """Tests for health metric estimation."""

    def test_walking_steps(self):
        hm = HealthMetrics(weight_kg=70.0, prediction_interval_s=2.0)
        for _ in range(10):
            hm.update("WALKING")
        snap = hm.get_snapshot()
        self.assertGreater(snap.total_steps, 0)
        self.assertGreater(snap.calories_burned, 0.0)
        self.assertEqual(snap.active_time_s, 20.0)

    def test_sedentary_no_steps(self):
        hm = HealthMetrics(weight_kg=70.0, prediction_interval_s=2.0)
        for _ in range(10):
            hm.update("SITTING")
        snap = hm.get_snapshot()
        self.assertEqual(snap.total_steps, 0)
        self.assertEqual(snap.sedentary_time_s, 20.0)

    def test_distance_walking(self):
        hm = HealthMetrics(weight_kg=70.0, prediction_interval_s=2.0)
        for _ in range(10):
            hm.update("WALKING")
        snap = hm.get_snapshot()
        self.assertGreater(snap.distance_m, 0)

    def test_reset(self):
        hm = HealthMetrics(weight_kg=70.0)
        hm.update("WALKING")
        hm.reset()
        snap = hm.get_snapshot()
        self.assertEqual(snap.total_steps, 0)

    def test_weight_validation(self):
        hm = HealthMetrics(weight_kg=70.0)
        with self.assertRaises(ValueError):
            hm.weight_kg = -10


if __name__ == "__main__":
    unittest.main()
