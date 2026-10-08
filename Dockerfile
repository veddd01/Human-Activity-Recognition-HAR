# ============================================================================
# HAR Project — Multi-service Docker Image
# ============================================================================
# This single image supports both the Streamlit UI and the FastAPI backend.
# The default CMD launches Streamlit; override with docker-compose for the
# API service.
# ============================================================================

FROM python:3.10-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ---- Install OS-level dependencies ----
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# ---- Create non-root user ----
RUN useradd --create-home --shell /bin/bash appuser

# ---- Python dependencies ----
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- Copy application code ----
COPY . .

# ---- Set ownership ----
RUN chown -R appuser:appuser /app
USER appuser

# ---- Expose ports: Streamlit (8501) and FastAPI (8000) ----
EXPOSE 8501 8000

# ---- Health check ----
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=30s \
    CMD curl -f http://localhost:8000/ || curl -f http://localhost:8501/_stcore/health || exit 1

# ---- Default: run Streamlit ----
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
