# HoneyChain ML - Hive Health Diagnostic Microservice

> **Independent, stateless Python HTTP inference microservice for the HoneyChain honey provenance and apiculture monitoring ecosystem.**

---

## 🐝 Overview

HoneyChain ML evaluates honeybee colony health, swarming risk, robbing activity, and seasonal baseline anomalies from IoT telemetry. It loads 26 pre-trained machine learning artifacts into memory once at startup and performs low-latency inference on demand.

The service operates on a **6-Tier Continuous Inference Hierarchy** (`T1` through `T48`) where more historical data enables higher-tier models:
- **`T1` (1 hour)**: Seasonal activity baseline deviation and temperature anomaly (Isolation Forest).
- **`T6` (6 hours)**: Fast weight loss detection (robbing / swarming alert).
- **`T12` (12 hours)**: Medium-term weight trend dynamics.
- **`T24` (24 hours)**: Full circadian cycle and diurnal foraging rhythms (Trained XGBoost classifier active).
- **`T36` (36 hours)**: Multi-day sustained weight decline.
- **`T48` (48 hours)**: Long-term starvation and colony depletion patterns (Trained XGBoost classifier active).

---

## 🏛️ Microservice Architecture

```text
┌────────────────────────────────────────────────────────┐
│                   HoneyChain Frontend                  │
└───────────────────────────┬────────────────────────────┘
                            │ Public HTTPS / REST
                            ▼
┌────────────────────────────────────────────────────────┐
│                   Express API Backend                  │
│       MongoDB • Ethereum Sepolia • JWT Auth/RBAC       │
└───────────────────────────┬────────────────────────────┘
                            │ Server-to-Server HTTPS
                            │ Headers: X-ML-API-Key
                            ▼
┌────────────────────────────────────────────────────────┐
│               HoneyChain ML Microservice               │
│            0.0.0.0:${PORT:-5001} (Python)              │
│       Isolation Forests • XGBoost Classifiers          │
└────────────────────────────────────────────────────────┘
```

---

## 📡 API Specification

### 1. Health Probe
```http
GET /health
```
Returns service status, version, model loading verification, and available inference tiers.

**Response (`200 OK`):**
```json
{
  "status": "ok",
  "service": "honeychain-hive-health-ml",
  "version": "1.0.0",
  "modelLoaded": true,
  "tiers": ["T1", "T6", "T12", "T24", "T36", "T48"]
}
```

---

### 2. Predict Hive Health
```http
POST /predict
Content-Type: application/json
X-ML-API-Key: <optional-configured-key>
```

**Request Body:**
```json
{
  "hiveId": "HIVE-KV-201",
  "readings": [
    {
      "timestamp": "2026-05-10 12:00:00",
      "temperature": 24.5,
      "humidity": 62.0,
      "weight": 35.8,
      "flow": 45
    }
  ]
}
```

*Note: `timestamp` should represent local time at the apiary (`YYYY-MM-DD HH:mm:ss`) for solar daylight alignment. `flow` is signed net flow (`entering - leaving`).*

**Response (`200 OK`):**
```json
{
  "hiveId": "HIVE-KV-201",
  "timestamp": "2026-05-10T12:00:00",
  "status": "OK",
  "tier": "T24",
  "hoursAvailable": 25,
  "hoursObserved": 25,
  "healthScore": 88.4,
  "stressRisk": "LOW",
  "stressProbability": 0.082,
  "abnormalityRisk": 22.1,
  "detectionScope": [
    "colony activity anomaly",
    "fast weight drop (robbing / swarm)",
    "sustained weight decline (starvation)"
  ],
  "drivers": {
    "activityDeviation": -0.42,
    "temperatureDeviation": 0.15,
    "netFlow": 65,
    "weightTrend": 0.08,
    "weightDrop": -0.02
  },
  "recommendation": "Colony within normal range. No action needed.",
  "stressBasis": "classifier"
}
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `5001` | Service listen port (automatically provided by Render/Heroku/Railway) |
| `HOST` | `0.0.0.0` | Host IP binding (listens on all interfaces) |
| `ML_API_KEY` | *(None)* | Optional shared secret. If configured, requires incoming `X-ML-API-Key` header on `POST /predict`. Open access if omitted. |

---

## 🛠️ Local Development & Testing

### 1. Prerequisites
- Python 3.10+ installed
- Pip and virtualenv

### 2. Setup
```bash
# 1. Clone or navigate to the ML repository
cd HoneyChain_ML

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Run automated tests
python3 tests/test_service.py
```

### 3. Run Locally
```bash
python3 service.py
```
Server starts on `http://0.0.0.0:5001`. You can test with curl:
```bash
curl http://localhost:5001/health
```

---

## 🚀 Independent Deployment Guide (e.g. Render)

To deploy this service independently on [Render](https://render.com) (or any Python container platform):

1. **Push to GitHub**:
   ```bash
   cd HoneyChain_ML
   git init
   git add .
   git commit -m "feat: initial HoneyChain ML microservice"
   git branch -M main
   git remote add origin https://github.com/<YOUR_USER>/HoneyChain_ML.git
   git push -u origin main
   ```

2. **Create New Web Service on Render**:
   - Environment: **Python 3**
   - Build Command:
     ```bash
     pip install -r requirements.txt
     ```
   - Start Command:
     ```bash
     python service.py
     ```
   - Health Check Path: `/health`

3. **Set Environment Variables on Render**:
   - `ML_API_KEY`: *(Generate a secure random string, e.g. `openssl rand -hex 24`)*
   - `PYTHON_VERSION`: `3.11.8` (or `3.10+`)

4. **Connect to Express Backend**:
   In your deployed Express backend (Render service):
   - `ML_SERVICE_URL`: `https://honeychain-ml.onrender.com` (your Render ML URL)
   - `ML_API_KEY`: *(The same matching key configured on Render)*

---

## 📦 Model Artifacts Reference

Located in `./models/`:
- `iso_T*.pkl` (6 files): IsolationForest models for anomaly detection across tiers.
- `xgb_T*.pkl` (5 files): XGBoost classifiers trained on multi-hour time windows (`T6`-`T48`).
- `scaler_T*.pkl` (6 files): StandardScalers for numerical feature normalization.
- `quantiles_T*.csv` (6 files): Calibrated anomaly score quantile thresholds.
- `seasonal_baseline.csv`: Diurnal baseline metrics indexed by day-of-year and hour.
- `feature_config.json`: Feature definitions and tier mapping thresholds.
- `predict.py`: Core inference engine.
