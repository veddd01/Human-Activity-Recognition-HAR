"""
Authentication and JWT token management for the HAR project.

Provides password hashing via bcrypt, JWT access token creation/verification,
and FastAPI dependencies for user registration and protected endpoints.
"""

from __future__ import annotations

import datetime
import os
from datetime import timedelta
from typing import Any

import bcrypt
import jwt
from fastapi import Depends, Header, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.database import User, get_db, get_user_by_username

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_INSECURE_DEFAULTS = frozenset({
    "har-dev-secret-change-in-production-2026",
    "test-dev-secret-for-local-development-only",
    "test-secret-not-for-production",
    "test-secret-not-for-production-min32bytes",
    "CHANGE_ME_GENERATE_A_STRONG_SECRET",
})

_ENVIRONMENT = os.getenv("HAR_ENVIRONMENT", "development").lower()
_jwt_secret_raw = os.getenv("HAR_JWT_SECRET", "")

if _ENVIRONMENT == "production":
    if not _jwt_secret_raw:
        raise RuntimeError(
            "FATAL: HAR_JWT_SECRET is not set. A cryptographically random secret is "
            "required in production. Generate one with:\n"
            "  python -c \"import secrets; print(secrets.token_urlsafe(64))\""
        )
    if _jwt_secret_raw in _INSECURE_DEFAULTS:
        raise RuntimeError(
            "FATAL: HAR_JWT_SECRET is set to a known insecure default. "
            "Generate a strong secret for production."
        )
    if len(_jwt_secret_raw) < 32:
        raise RuntimeError(
            "FATAL: HAR_JWT_SECRET is too short (minimum 32 characters for production). "
            "Generate a strong secret with:\n"
            "  python -c \"import secrets; print(secrets.token_urlsafe(64))\""
        )

SECRET_KEY: str = _jwt_secret_raw or "har-dev-secret-change-in-production-2026"
ALGORITHM: str = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours default


# ---------------------------------------------------------------------------
# Pydantic Request / Response Schemas
# ---------------------------------------------------------------------------
class RegisterRequest(BaseModel):
    """User registration request body."""
    username: str = Field(..., min_length=3, max_length=64, description="Unique username")
    password: str = Field(..., min_length=6, description="Password (at least 6 characters)")


class LoginRequest(BaseModel):
    """User login request body."""
    username: str = Field(..., description="Registered username")
    password: str = Field(..., description="User password")


class TokenResponse(BaseModel):
    """JWT bearer token response."""
    access_token: str
    token_type: str = "bearer"
    username: str
    user_id: int


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------
def hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against its bcrypt hash."""
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except Exception:
        return False


# ---------------------------------------------------------------------------
# JWT tokens
# ---------------------------------------------------------------------------
def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    """Generate a signed JWT access token."""
    to_encode = data.copy()
    expire = datetime.datetime.now(datetime.timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any] | None:
    """Decode and validate a JWT access token. Returns None if invalid or expired."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.PyJWTError:
        return None


# ---------------------------------------------------------------------------
# FastAPI Dependency
# ---------------------------------------------------------------------------
def get_current_user(
    authorization: str | None = Header(None),
    db: Session = Depends(get_db),
) -> User:
    """FastAPI dependency to extract and authenticate the current user from Authorization header."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not authorization:
        raise credentials_exception

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise credentials_exception

    token = parts[1]
    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception

    username: str = payload.get("sub")
    if username is None:
        raise credentials_exception

    user = get_user_by_username(db, username=username)
    if user is None:
        raise credentials_exception

    return user


def get_optional_user(
    authorization: str | None = Header(None),
    db: Session = Depends(get_db),
) -> User | None:
    """Extract user if valid token is provided, otherwise return None without raising error."""
    if not authorization:
        return None
    try:
        return get_current_user(authorization, db)
    except HTTPException:
        return None


def authenticate_websocket_token(token: str) -> dict[str, Any] | None:
    """Authenticate a WebSocket connection using a JWT token.

    Typically the token is passed as a query parameter: ``ws://host/ws/predict?token=<jwt>``.

    Returns the decoded payload if the token is valid, otherwise ``None``.
    """
    if not token:
        return None
    payload = decode_access_token(token)
    if payload is None:
        return None
    if payload.get("sub") is None:
        return None
    return payload
