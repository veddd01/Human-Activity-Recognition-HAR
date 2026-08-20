# Human Activity Recognition (HAR)

A portfolio-grade Human Activity Recognition system built with **Python, Scikit-Learn, TensorFlow, FastAPI, Streamlit, WebSockets, and Plotly**. The project combines classical machine learning and deep learning models with a real-time prediction architecture, sensor simulation, prediction smoothing, activity tracking, and health analytics.

> **Project status:** Advanced portfolio/research project. The core ML/API/dashboard system is implemented, while some production features such as direct smartphone sensor integration, persistent authentication/database infrastructure, and additional testing remain future work.

---

## ✨ Features

### 🤖 Machine Learning & Deep Learning

- Logistic Regression
- Decision Tree
- Support Vector Machine (SVM)
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- Soft-Voting Ensemble (LR + RF + SVM)
- Dense Neural Network with Keras
- CNN-LSTM sequence model

### 📡 Real-Time Prediction Architecture

- Tri-axial accelerometer and gyroscope sensor simulation
- 50 Hz sensor windows with 128 samples
- UCI HAR-style 561-feature extraction pipeline
- FastAPI REST API for prediction requests
- WebSocket endpoint for continuous prediction streams
- Rolling-window prediction smoothing to reduce jitter
- Confidence scores for predictions

### 📊 Prediction & Activity Analytics

- Prediction history and session tracking
- Activity transition detection
- CSV/JSON prediction export
- Activity distribution analytics
- Step-count estimation
- MET-based calorie estimation
- Distance estimation
- Active vs. sedentary time tracking

### 🎨 Interactive Dashboard

Built with **Streamlit + Plotly**, including:

- Dashboard overview
- Live monitoring
- Prediction interface
- Model comparison
- Analytics
- Health metrics
- About/project information
- Dark/light theme support
- Live sensor visualizations
- Confusion matrices and model evaluation visualizations

### 🛠️ DevOps & Deployment

- Docker and Docker Compose
- Nginx reverse proxy configuration
- Prometheus metrics instrumentation
- Grafana dashboard template
- GitHub Actions CI/CD configuration
- AWS deployment documentation

---

## 🏗️ Architecture

```text
┌──────────────────────────────┐
│ Sensor Data / Simulation     │
│ Accelerometer + Gyroscope    │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Sensor Processing             │
│ Windowing + Feature Extract.  │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ ML / DL Models                │
│ sklearn + XGBoost + LightGBM  │
│ TensorFlow / CNN-LSTM         │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ FastAPI Backend               │
│ REST API + WebSocket          │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Streamlit Dashboard           │
│ Predictions + Analytics       │
└──────────────────────────────┘
```

---

## 📁 Project Structure

```text
Human-Activity-Recognition-HAR/
├── api.py
├── app.py
├── requirements.txt
├── docker-compose.prod.yml
├── deploy/
│   └── nginx.conf
├── docs/
│   └── DEPLOYMENT.md
├── src/
│   ├── data.py
│   ├── train.py
│   ├── evaluate.py
│   └── ...
├── models/
├── data/
├── notebooks/
├── tests/
└── .github/
    └── workflows/
        └── ci.yml
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/veddd01/Human-Activity-Recognition-HAR.git
cd Human-Activity-Recognition-HAR
```

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare data and train models

```bash
python src/data.py
python src/train.py
python src/evaluate.py
```

### 5. Start the FastAPI backend

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Start the Streamlit dashboard

Open a second terminal with the virtual environment activated:

```bash
streamlit run app.py
```

The dashboard will normally be available at `http://localhost:8501` and the API at `http://localhost:8000`.

---

## 🔌 API

When the FastAPI backend is running:

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`
- **WebSocket:** `ws://localhost:8000/ws/predict`
- **Prometheus metrics:** `http://localhost:8000/metrics`

The API supports prediction requests, batch prediction, sensor-window ingestion, health checks, and continuous WebSocket prediction streams.

---

## 🐳 Docker

For the production-style container setup:

```bash
docker-compose -f docker-compose.prod.yml up -d --build
```

To stop the services:

```bash
docker-compose -f docker-compose.prod.yml down
```

See `docs/DEPLOYMENT.md` for deployment-related configuration and AWS guidance.

---

## 📈 Model Evaluation

The project supports comparing multiple models using common classification metrics and visualizations. Model comparison can include accuracy, precision, recall, F1 score, confusion matrices, and inference-related measurements depending on the available evaluation outputs.

The original UCI HAR-based implementation reported approximately **94% accuracy** for its earlier model configuration. Results can vary depending on the model, training configuration, and dataset preparation.

---

## ⚠️ Current Limitations

This repository is suitable as a portfolio/research project, but it should not be described as a fully production-hardened fitness platform yet.

Current areas for further development include:

- Direct real smartphone sensor integration
- A production-grade CNN-LSTM raw-signal inference pipeline
- Persistent database storage for user/activity history
- Proper production authentication and authorization
- Expanded automated unit/API/WebSocket test coverage
- Additional production hardening such as rate limiting, robust reconnect handling, and structured logging

These limitations are intentionally documented so the repository reflects the current implementation accurately.

---

## 🔮 Future Improvements

- Android/iOS sensor client for live phone data
- Persistent PostgreSQL/SQLite storage
- JWT-based authentication and user accounts
- More comprehensive automated testing
- Improved CNN-LSTM sequence handling from raw inertial windows
- Production monitoring and alerting
- Improved deployment automation
- Personalized activity and fitness analytics

---

## 🧰 Tech Stack

**Languages & Frameworks**

- Python
- FastAPI
- Streamlit
- TensorFlow / Keras
- Scikit-Learn

**ML Libraries**

- XGBoost
- LightGBM
- Joblib
- NumPy
- Pandas

**Visualization**

- Plotly
- Matplotlib
- Seaborn

**Infrastructure**

- Docker
- Docker Compose
- Nginx
- Prometheus
- Grafana
- GitHub Actions

---

## 📄 License

Add your preferred open-source license before distributing the project publicly. If no license is included, the repository remains under the default copyright terms and others generally do not receive permission to reuse or redistribute the code.

---

## 👤 Author

**Vedant**

GitHub: [@veddd01](https://github.com/veddd01)

---

⭐ If you find this project useful, consider giving the repository a star.