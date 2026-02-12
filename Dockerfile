# Bjorn Cortex - VPS Deployment Dockerfile
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    libffi-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add security dependencies (since they might not be in the main requirements yet)
RUN pip install --no-cache-dir \
    python-jose[cryptography] \
    passlib[bcrypt] \
    python-multipart \
    pyotp \
    qrcode \
    slowapi

# Copy application code
COPY bjorn_ai_server/ /app/

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/icons

# Expose FastAPI port
EXPOSE 8000

# Start command
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
