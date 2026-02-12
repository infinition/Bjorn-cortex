<div align="center">
  <img src="https://github.com/user-attachments/assets/2d2ed3e7-e159-42a5-9e99-22074c10581b"
       alt="icon"
       width="140"
       height="125" />
</div>

# Bjorn Cortex
Neural command center for Bjorn units.

This service receives anonymized training data from clients, trains a model, and serves the latest model to connected Bjorn devices.


<img width="896" height="965" alt="image" src="https://github.com/user-attachments/assets/b49609e8-dfcf-4bfe-986c-c681da06fbdd" />

## Features
- FastAPI backend + dashboard.
- Upload endpoint for `.csv.gz` training files.
- Training pipeline with TensorFlow/Keras.
- Live status via WebSocket.
- JWT + MFA (TOTP) support.

## Requirements
- Python 3.10+ (3.8+ may work depending on TensorFlow build).
- Linux/Windows/macOS.

Install dependencies:

```bash
pip install fastapi uvicorn python-multipart jinja2 websockets tensorflow pandas numpy scikit-learn pyotp qrcode python-jose[cryptography] passlib[bcrypt] slowapi
```

## Run (local)
From this folder:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

Before first run, create local configs:

```bash
cp security_config.example.json security_config.json
cp server_config.example.json server_config.json
```

Dashboard:
- `http://localhost:8000`

## Docker
From this folder:

```bash
docker compose up -d --build
```

## Security Before GitHub Push
This project currently uses local JSON configs (`security_config.json`, `server_config.json`) that can contain secrets.

These files are now ignored by `.gitignore`.

Recommended publish flow:
1. Keep real local configs untracked.
2. Commit only sanitized examples (`security_config.example.json`, `server_config.example.json`) if needed.
3. Rotate any secret previously exposed (JWT secret, TOTP secret, API keys, admin password hash).

## Runtime Data
The following are generated at runtime and are ignored:
- `data/*.csv.gz`
- `models/*`
- `training_history.json`
- `__pycache__/`

## Project Layout
```text
bjorn_ai_server/
├── data/                  # uploaded training files (runtime)
├── models/                # trained models (runtime)
├── templates/             # HTML templates
├── server.py              # FastAPI app
├── trainer.py             # training logic
├── Dockerfile
├── docker-compose.yml
├── .gitignore
└── README.md
```

## Core Endpoints
- `GET /` dashboard
- `POST /upload` training file upload
- `POST /train/start` start training
- `GET /model/latest` latest model metadata
- `GET /model/download/{filename}` model download
- `GET /stats` server stats
- `WS /ws/logs` realtime logs/training events

## Notes
- For internet exposure, put this service behind a reverse proxy with HTTPS.
- Do not deploy with default secrets.
