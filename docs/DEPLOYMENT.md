# HAR System — Deployment Guide

This guide covers deploying the Human Activity Recognition system using Docker Compose, both locally and on cloud infrastructure (e.g., AWS EC2).

---

## Prerequisites

1. **Docker** and **Docker Compose** installed.
2. Pre-trained model files in the `models/` directory.
3. A `.env` file (copy from `.env.example` and configure).

---

## Local Deployment (Docker Compose)

### Step 1: Configure Environment

```bash
cp .env.example .env
# Edit .env:
#   HAR_JWT_SECRET=<generate-a-strong-secret>
#   HAR_SEED_DEMO_USERS=true   # optional, for testing
```

Generate a secret:
```bash
python -c "import secrets; print(secrets.token_urlsafe(64))"
```

### Step 2: Development Mode

```bash
docker-compose up -d --build
```

Access:
- **Streamlit Dashboard**: `http://localhost:8501`
- **FastAPI Swagger UI**: `http://localhost:8000/docs`
- **WebSocket**: `ws://localhost:8000/ws/predict`

### Step 3: Production Mode (with Nginx)

```bash
docker-compose -f docker-compose.prod.yml up -d --build
```

Access:
- **Dashboard**: `http://localhost/`
- **API**: `http://localhost/api/docs`
- **WebSocket**: `ws://localhost/ws/predict`

---

## Cloud Deployment (AWS EC2)

### Step 1: Provision EC2 Instance

- Launch an EC2 Instance (`t3.medium` recommended, Ubuntu 22.04 LTS).
- Associate an **Elastic IP**.
- Configure Security Group inbound rules:
  - `HTTP (80)` from Anywhere
  - `HTTPS (443)` from Anywhere
  - `SSH (22)` from your IP

### Step 2: Install Docker

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y docker.io docker-compose git
sudo systemctl enable --now docker
sudo usermod -aG docker ubuntu
```

### Step 3: Clone & Configure

```bash
git clone https://github.com/<your-username>/HAR.git
cd HAR
cp .env.example .env
# Edit .env with a strong HAR_JWT_SECRET
```

### Step 4: Launch Production Stack

```bash
docker-compose -f docker-compose.prod.yml up -d --build
```

### Step 5: HTTPS (Optional)

For HTTPS with Let's Encrypt, install Certbot on the host and configure SSL:

```bash
sudo apt install -y certbot
sudo certbot certonly --standalone -d yourdomain.com
```

Then update `deploy/nginx.conf` to add an SSL server block with the generated certificates, and mount `/etc/letsencrypt` into the Nginx container via `docker-compose.prod.yml`.

---

## Verification

| Resource | URL |
|----------|-----|
| Dashboard | `http://<host>/` |
| API Docs | `http://<host>/api/docs` |
| WebSocket | `ws://<host>/ws/predict` |
| Sensor Client | `http://<host>/api/sensor` |

---

## Troubleshooting

- **Models not found**: Ensure `models/` directory contains trained `.pkl` and `.keras` files.
- **JWT errors**: Verify `HAR_JWT_SECRET` is set in `.env`.
- **Database issues**: Check `DATABASE_URL` in `.env` or use the default SQLite path.
- **Streamlit connection error**: Ensure the API container is running and healthy.
