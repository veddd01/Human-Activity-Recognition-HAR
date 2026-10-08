"""
Pytest configuration and shared fixtures for the HAR test suite.
"""

import os
import pytest

# Use a test-only JWT secret and in-memory database
os.environ.setdefault("HAR_JWT_SECRET", "test-secret-not-for-production-min32bytes")
os.environ.setdefault("HAR_SEED_DEMO_USERS", "true")

from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def test_client():
    """Create a TestClient for the FastAPI app with models loaded."""
    from api import app, _load_models
    from src.database import init_db, SessionLocal, get_user_by_username, add_user
    from src.auth import hash_password

    init_db()

    # Seed test users
    with SessionLocal() as db:
        if not get_user_by_username(db, "admin"):
            add_user(db, username="admin", hashed_password=hash_password("password"))
        if not get_user_by_username(db, "testuser"):
            add_user(db, username="testuser", hashed_password=hash_password("testpass123"))
        if not get_user_by_username(db, "otheruser"):
            add_user(db, username="otheruser", hashed_password=hash_password("otherpass123"))

    _load_models()

    client = TestClient(app)
    return client


@pytest.fixture(scope="session")
def auth_token(test_client):
    """Get a JWT token for testuser."""
    resp = test_client.post("/auth/login", json={"username": "testuser", "password": "testpass123"})
    assert resp.status_code == 200
    return resp.json()["access_token"]


@pytest.fixture(scope="session")
def other_token(test_client):
    """Get a JWT token for otheruser."""
    resp = test_client.post("/auth/login", json={"username": "otheruser", "password": "otherpass123"})
    assert resp.status_code == 200
    return resp.json()["access_token"]


@pytest.fixture(scope="session")
def admin_token(test_client):
    """Get a JWT token for admin."""
    resp = test_client.post("/auth/login", json={"username": "admin", "password": "password"})
    assert resp.status_code == 200
    return resp.json()["access_token"]
