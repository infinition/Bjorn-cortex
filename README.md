<div align="center">
  <img src="https://github.com/user-attachments/assets/2d2ed3e7-e159-42a5-9e99-22074c10581b"
       alt="icon"
       width="140"
       height="125" />
</div>


# Bjorn Cortex
Neural command center for Bjorn units.

## Vision
Bjorn Cortex is the brain of a Swarm AI: each Bjorn explores its environment, learns from real missions, and sends anonymized feedback to the Cortex.

The Cortex aggregates this collective field experience to continuously improve navigation, recognition, and exploitation strategies, then redistributes a stronger model to the whole fleet.

Goal: build an ultra-efficient, community-driven AI brain for makers, tinkerers, and ethical hackers working in authorized and defensive security contexts.

Roadmap note: Bjorn Cortex is designed to be deployed on a VPS in future production-oriented versions. Local server and local AI model usage may also be integrated directly into future Bjorn releases. The current GitHub version is primarily intended for local experimentation, validation, and iterative testing.

This service receives anonymized training data from clients, trains a model, and serves the latest model to connected Bjorn devices.
<div align="center">

<img width="896" height="965" alt="image" src="https://github.com/user-attachments/assets/b49609e8-dfcf-4bfe-986c-c681da06fbdd" />
</div>

## Features
- FastAPI backend and dashboard UI.
- Upload endpoint for `.csv.gz` training files.
- Training pipeline with TensorFlow/Keras.
- Live status via WebSocket.
- JWT authentication + MFA (TOTP).

## Requirements
- Python 3.10+ (3.8+ may work depending on TensorFlow build).
- Linux, Windows, or macOS.

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

Base URL:
- `http://localhost:8000`

## First Login / Setup Flow
1. Open `http://localhost:8000/setup`.
2. Scan TOTP QR code with your authenticator app.
3. Open `http://localhost:8000/login`.
4. Authenticate (password + TOTP).
5. Access dashboard on `http://localhost:8000/`.

## Docker
From this folder:

```bash
docker compose up -d --build
```

## Security Before GitHub Push
This project uses local JSON configs (`security_config.json`, `server_config.json`) that can contain secrets.

These files are ignored by `.gitignore`.

Recommended publish flow:
1. Keep real local configs untracked.
2. Commit only sanitized examples (`security_config.example.json`, `server_config.example.json`).
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
|-- data/                  # uploaded training files (runtime)
|-- models/                # trained models (runtime)
|-- templates/             # HTML templates
|-- server.py              # FastAPI app
|-- trainer.py             # training logic
|-- Dockerfile
|-- docker-compose.yml
|-- .gitignore
`-- README.md
```

## Core Endpoints
- `GET /setup` initial setup page
- `GET /setup/qr` TOTP provisioning QR
- `GET /setup/secret` raw TOTP secret
- `GET /login` login page
- `POST /token` issues JWT token (OAuth2 password flow)
- `GET /` dashboard
- `GET /history` activity and training history
- `GET /config` read runtime config
- `POST /config/update` update runtime config
- `POST /upload` upload training data (`.csv.gz`)
- `POST /train/start` trigger training
- `GET /model/latest` latest model metadata
- `GET /model/download/{filename}` model artifact download
- `GET /stats` server stats
- `WS /ws/logs` realtime logs/training events

## Notes
- For internet exposure, place the app behind a reverse proxy with HTTPS.
- Do not deploy with default secrets.
