<div align="center">
  <img src="https://github.com/user-attachments/assets/2d2ed3e7-e159-42a5-9e99-22074c10581b" alt="Bjorn Cortex" width="140" height="125" />
</div>

# Bjorn Cortex

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white) [![Release](https://img.shields.io/github/v/release/infinition/Bjorn-cortex?style=flat)](https://github.com/infinition/Bjorn-cortex/releases) [![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=flat&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/infinition)

A centralized training server for the Bjorn swarm. Each Bjorn device explores its environment and sends anonymized feedback to the Cortex. The Cortex aggregates field experience across the fleet, trains a model, and redistributes it to all connected devices.

The current version targets local experimentation and iterative testing. Production deployment on a VPS is a future milestone.

---

## How it works

1. Bjorn devices send anonymized `.csv.gz` training files to the Cortex upload endpoint.
2. The Cortex trains a TensorFlow/Keras model on the aggregated data.
3. The latest model is served back to connected devices on request.
4. Live training status is streamed via WebSocket.

---

## Features

- FastAPI backend and dashboard UI.
- Upload endpoint for `.csv.gz` training data.
- Training pipeline with TensorFlow/Keras.
- Live status via WebSocket.
- JWT authentication with MFA (TOTP).

---

## Requirements

- Python 3.10+
- TensorFlow-compatible environment (GPU optional)

```bash
pip install -r Cortex/requirements.txt
```

---

## Running

```bash
cd Cortex
python server.py
```

Access the dashboard on `http://localhost:8000`.

---

## Docker

```bash
docker compose up -d --build
```

---

## Security

Local configs (`security_config.json`, `server_config.json`) can contain secrets and are git-ignored. Commit only the sanitized `.example` variants. Rotate any secret previously exposed before pushing.

Do not expose this service publicly without HTTPS and a reverse proxy.

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/setup` | Initial setup page |
| GET | `/setup/qr` | TOTP provisioning QR |
| POST | `/token` | Issue JWT (OAuth2 password flow) |
| GET | `/` | Dashboard |
| POST | `/upload` | Upload training data |
| POST | `/train/start` | Trigger training |
| GET | `/model/latest` | Latest model metadata |
| GET | `/model/download/{filename}` | Download model artifact |
| WS | `/ws/logs` | Real-time training events |

---

## Star History

<a href="https://www.star-history.com/?repos=infinition%2FBjorn-cortex&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=infinition/Bjorn-cortex&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=infinition/Bjorn-cortex&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=infinition/Bjorn-cortex&type=date&legend=top-left" />
 </picture>
</a>

---

## License

MIT.
