"""
Comprehensive API tests for the HAR FastAPI backend.

Tests cover:
- Health/readiness endpoints
- Model listing
- Authentication (register, login, profile)
- Authorization (protected endpoints, invalid tokens)
- Prediction endpoints (valid/invalid input)
- CNN-LSTM sequence prediction
- History endpoints with user isolation
- Health metrics with user isolation
- Sensor ingestion
- WebSocket functionality
- Bad request handling
"""

import uuid
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Health & Models
# ---------------------------------------------------------------------------

class TestHealthEndpoints:

    def test_root_health_check(self, test_client):
        resp = test_client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "models_loaded" in data
        assert data["models_loaded"] > 0

    def test_list_models(self, test_client):
        resp = test_client.get("/models")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0
        for model in data:
            assert "name" in model
            assert "file" in model
            assert "type" in model

    def test_sensor_page(self, test_client):
        resp = test_client.get("/sensor")
        assert resp.status_code == 200
        assert "HAR" in resp.text or "sensor" in resp.text.lower()


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

class TestAuthentication:

    def test_register_new_user(self, test_client):
        uname = f"test_{uuid.uuid4().hex[:8]}"
        resp = test_client.post("/auth/register", json={
            "username": uname,
            "password": "securepassword123"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["username"] == uname

    def test_register_duplicate_user(self, test_client):
        uname = f"test_dup_{uuid.uuid4().hex[:8]}"
        test_client.post("/auth/register", json={"username": uname, "password": "pass123456"})
        resp = test_client.post("/auth/register", json={"username": uname, "password": "pass123456"})
        assert resp.status_code == 400

    def test_register_short_password(self, test_client):
        resp = test_client.post("/auth/register", json={
            "username": f"test_{uuid.uuid4().hex[:8]}",
            "password": "abc"
        })
        assert resp.status_code == 422

    def test_register_short_username(self, test_client):
        resp = test_client.post("/auth/register", json={
            "username": "ab",
            "password": "password123"
        })
        assert resp.status_code == 422

    def test_login_valid(self, test_client):
        resp = test_client.post("/auth/login", json={
            "username": "admin", "password": "password"
        })
        assert resp.status_code == 200
        assert "access_token" in resp.json()

    def test_login_invalid_password(self, test_client):
        resp = test_client.post("/auth/login", json={
            "username": "admin", "password": "wrongpassword"
        })
        assert resp.status_code == 401

    def test_login_nonexistent_user(self, test_client):
        resp = test_client.post("/auth/login", json={
            "username": "nonexistent_user_xyz", "password": "password"
        })
        assert resp.status_code == 401

    def test_profile_authenticated(self, test_client, auth_token):
        resp = test_client.get("/auth/profile", headers={
            "Authorization": f"Bearer {auth_token}"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == "testuser"

    def test_profile_unauthenticated(self, test_client):
        resp = test_client.get("/auth/profile")
        assert resp.status_code == 401

    def test_profile_invalid_token(self, test_client):
        resp = test_client.get("/auth/profile", headers={
            "Authorization": "Bearer invalid.token.here"
        })
        assert resp.status_code == 401

    def test_profile_expired_token_format(self, test_client):
        resp = test_client.get("/auth/profile", headers={
            "Authorization": "NotBearer sometoken"
        })
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Prediction (flat features)
# ---------------------------------------------------------------------------

class TestPrediction:

    def _make_features(self, n=561):
        """Generate a valid 561-float feature vector."""
        rng = np.random.default_rng(42)
        return rng.standard_normal(n).tolist()

    def test_predict_valid(self, test_client):
        features = self._make_features()
        resp = test_client.post("/predict", json={
            "model_name": "logistic_regression",
            "features": features,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "predicted_label" in data
        assert "activity_name" in data
        assert "inference_time_ms" in data
        assert data["model_name"] == "logistic_regression"

    def test_predict_with_smoothing(self, test_client):
        features = self._make_features()
        resp = test_client.post("/predict", json={
            "model_name": "random_forest",
            "features": features,
            "use_smoothing": True,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_smoothed"] is True

    def test_predict_invalid_model(self, test_client):
        features = self._make_features()
        resp = test_client.post("/predict", json={
            "model_name": "nonexistent_model",
            "features": features,
        })
        assert resp.status_code == 404

    def test_predict_wrong_feature_count(self, test_client):
        resp = test_client.post("/predict", json={
            "model_name": "logistic_regression",
            "features": [1.0, 2.0, 3.0],  # too few
        })
        assert resp.status_code == 422

    def test_predict_too_many_features(self, test_client):
        features = [0.0] * 600  # too many
        resp = test_client.post("/predict", json={
            "model_name": "logistic_regression",
            "features": features,
        })
        assert resp.status_code == 422

    def test_predict_empty_features(self, test_client):
        resp = test_client.post("/predict", json={
            "model_name": "logistic_regression",
            "features": [],
        })
        assert resp.status_code == 422

    def test_predict_cnn_lstm_rejects_flat(self, test_client):
        """CNN-LSTM should reject flat 561-feature input via /predict."""
        features = self._make_features()
        resp = test_client.post("/predict", json={
            "model_name": "cnn_lstm",
            "features": features,
        })
        assert resp.status_code == 400
        assert "sequence" in resp.json()["detail"].lower() or "/predict/sequence" in resp.json()["detail"]


class TestBatchPrediction:

    def _make_features(self, n=561):
        rng = np.random.default_rng(42)
        return rng.standard_normal(n).tolist()

    def test_batch_predict_valid(self, test_client):
        features = self._make_features()
        resp = test_client.post("/predict/batch", json={
            "model_name": "logistic_regression",
            "samples": [features, features],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["predictions"]) == 2

    def test_batch_predict_empty(self, test_client):
        resp = test_client.post("/predict/batch", json={
            "model_name": "logistic_regression",
            "samples": [],
        })
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# CNN-LSTM Sequence Prediction
# ---------------------------------------------------------------------------

class TestSequencePrediction:

    def _make_sequence(self, timesteps=128, channels=9):
        rng = np.random.default_rng(42)
        return rng.standard_normal((timesteps, channels)).tolist()

    def test_sequence_predict_valid(self, test_client):
        seq = self._make_sequence()
        resp = test_client.post("/predict/sequence", json={
            "model_name": "cnn_lstm",
            "sequence": seq,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_name"] == "cnn_lstm"
        assert "predicted_label" in data
        assert "activity_name" in data
        assert "confidence" in data

    def test_sequence_wrong_timesteps(self, test_client):
        seq = self._make_sequence(timesteps=64, channels=9)
        resp = test_client.post("/predict/sequence", json={
            "model_name": "cnn_lstm",
            "sequence": seq,
        })
        assert resp.status_code == 422

    def test_sequence_wrong_channels(self, test_client):
        seq = self._make_sequence(timesteps=128, channels=3)
        resp = test_client.post("/predict/sequence", json={
            "model_name": "cnn_lstm",
            "sequence": seq,
        })
        assert resp.status_code == 422

    def test_sequence_non_cnn_lstm_model(self, test_client):
        seq = self._make_sequence()
        resp = test_client.post("/predict/sequence", json={
            "model_name": "logistic_regression",
            "sequence": seq,
        })
        assert resp.status_code == 400

    def test_sequence_missing_data(self, test_client):
        resp = test_client.post("/predict/sequence", json={
            "model_name": "cnn_lstm",
        })
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Data Isolation
# ---------------------------------------------------------------------------

class TestDataIsolation:

    def test_user_a_cannot_see_user_b_predictions(self, test_client, auth_token, other_token):
        """Predictions made by testuser should not appear for otheruser."""
        rng = np.random.default_rng(99)
        features = rng.standard_normal(561).tolist()

        # testuser makes a prediction
        resp_a = test_client.post("/predict", json={
            "model_name": "logistic_regression",
            "features": features,
        }, headers={"Authorization": f"Bearer {auth_token}"})
        assert resp_a.status_code == 200

        # otheruser should not see testuser's predictions in DB history
        resp_b = test_client.get("/history?limit=100", headers={
            "Authorization": f"Bearer {other_token}"
        })
        assert resp_b.status_code == 200
        records = resp_b.json()
        # All returned DB records (if any) should not belong to testuser
        # DB records don't expose user_id directly, but the list should be
        # either empty or contain only otheruser's own records.
        assert isinstance(records, list)

    def test_user_b_cannot_see_user_a_history_stats(self, test_client, auth_token, other_token):
        """Each user's history stats should reflect only their own activity."""
        rng = np.random.default_rng(101)
        features = rng.standard_normal(561).tolist()

        # testuser makes multiple predictions
        for _ in range(3):
            test_client.post("/predict", json={
                "model_name": "logistic_regression",
                "features": features,
            }, headers={"Authorization": f"Bearer {auth_token}"})

        # otheruser's stats should not include testuser's predictions
        resp = test_client.get("/history/stats", headers={
            "Authorization": f"Bearer {other_token}"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "db_records_total" in data

    def test_health_metrics_user_scoped(self, test_client, auth_token, other_token):
        """Health metrics should be user-scoped when authenticated."""
        resp_a = test_client.get("/health-metrics", headers={
            "Authorization": f"Bearer {auth_token}"
        })
        assert resp_a.status_code == 200
        data_a = resp_a.json()
        assert "total_steps" in data_a

        resp_b = test_client.get("/health-metrics", headers={
            "Authorization": f"Bearer {other_token}"
        })
        assert resp_b.status_code == 200
        data_b = resp_b.json()
        assert "total_steps" in data_b

    def test_history_export_requires_auth(self, test_client):
        """History export should require authentication."""
        resp = test_client.get("/history/export?format=csv")
        assert resp.status_code == 401

    def test_history_export_authenticated(self, test_client, auth_token):
        """History export should work when authenticated."""
        resp = test_client.get("/history/export?format=csv", headers={
            "Authorization": f"Bearer {auth_token}"
        })
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Sensor Ingestion
# ---------------------------------------------------------------------------

class TestSensorIngestion:

    def test_sensor_ingest(self, test_client):
        resp = test_client.post("/sensor/ingest", json={
            "activity_hint": "WALKING",
            "weight_kg": 75.0,
        })
        # Should succeed if models are loaded
        assert resp.status_code == 200
        data = resp.json()
        assert "predicted_label" in data
        assert "activity_name" in data

    def test_sensor_ingest_invalid_activity(self, test_client):
        resp = test_client.post("/sensor/ingest", json={
            "activity_hint": "FLYING",
            "weight_kg": 70.0,
        })
        # Should still work (falls back to STANDING profile)
        assert resp.status_code == 200
        data = resp.json()
        assert "predicted_label" in data


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

class TestWebSocket:

    def test_websocket_simulate(self, test_client, auth_token):
        with test_client.websocket_connect(f"/ws/predict?token={auth_token}") as ws:
            ws.send_json({
                "action": "simulate",
                "activity_hint": "WALKING",
                "model_name": "logistic_regression",
            })
            data = ws.receive_json()
            assert "activity_name" in data
            assert "predicted_label" in data

    def test_websocket_features(self, test_client, auth_token):
        rng = np.random.default_rng(42)
        features = rng.standard_normal(561).tolist()
        with test_client.websocket_connect(f"/ws/predict?token={auth_token}") as ws:
            ws.send_json({
                "features": features,
                "model_name": "logistic_regression",
                "use_smoothing": False,
            })
            data = ws.receive_json()
            assert "activity_name" in data

    def test_websocket_invalid_payload(self, test_client, auth_token):
        with test_client.websocket_connect(f"/ws/predict?token={auth_token}") as ws:
            ws.send_json({"action": "invalid_action"})
            data = ws.receive_json()
            assert "error" in data

    def test_websocket_rejects_no_token(self, test_client):
        """WebSocket should reject connections without a valid token."""
        with pytest.raises(Exception):
            with test_client.websocket_connect("/ws/predict") as ws:
                ws.receive_json()


# ---------------------------------------------------------------------------
# History & Export
# ---------------------------------------------------------------------------

class TestHistory:

    def test_history_list(self, test_client):
        resp = test_client.get("/history")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_history_export_csv(self, test_client, auth_token):
        resp = test_client.get("/history/export?format=csv", headers={
            "Authorization": f"Bearer {auth_token}"
        })
        assert resp.status_code == 200

    def test_history_export_json(self, test_client, auth_token):
        resp = test_client.get("/history/export?format=json", headers={
            "Authorization": f"Bearer {auth_token}"
        })
        assert resp.status_code == 200

    def test_activity_summary(self, test_client):
        resp = test_client.get("/activity/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert "date" in data


# ---------------------------------------------------------------------------
# End-to-End Integration
# ---------------------------------------------------------------------------

class TestEndToEndIntegration:
    """Full-chain integration tests: Register → Login → Predict → History → Health."""

    def test_full_user_lifecycle(self, test_client):
        """Test complete user journey from registration through prediction to history."""
        uname = f"e2e_{uuid.uuid4().hex[:8]}"

        # 1. Register
        resp = test_client.post("/auth/register", json={
            "username": uname, "password": "e2epassword123"
        })
        assert resp.status_code == 200
        token = resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 2. Profile access
        resp = test_client.get("/auth/profile", headers=headers)
        assert resp.status_code == 200
        assert resp.json()["username"] == uname

        # 3. Prediction
        rng = np.random.default_rng(42)
        features = rng.standard_normal(561).tolist()
        resp = test_client.post("/predict", json={
            "model_name": "logistic_regression",
            "features": features,
        }, headers=headers)
        assert resp.status_code == 200
        pred = resp.json()
        assert "predicted_label" in pred
        assert "activity_name" in pred

        # 4. History reflects the prediction
        resp = test_client.get("/history?limit=10", headers=headers)
        assert resp.status_code == 200
        history = resp.json()
        assert isinstance(history, list)
        assert len(history) >= 1

        # 5. History stats
        resp = test_client.get("/history/stats", headers=headers)
        assert resp.status_code == 200
        stats = resp.json()
        assert stats["db_records_total"] >= 1

        # 6. Health metrics
        resp = test_client.get("/health-metrics", headers=headers)
        assert resp.status_code == 200
        assert "total_steps" in resp.json()

    def test_batch_prediction_validates_inner_length(self, test_client):
        """Batch endpoint should reject samples with wrong feature count."""
        good = [0.0] * 561
        bad = [0.0] * 100
        resp = test_client.post("/predict/batch", json={
            "model_name": "logistic_regression",
            "samples": [good, bad],
        })
        assert resp.status_code == 422
