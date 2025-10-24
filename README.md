# cp-swiss-german-asr
Comparative ASR for Swiss‑German dialects: reproducible Docker pipeline, baseline Whisper inference, fine‑tuned Whisper‑medium &amp; wav2vec2 models, per‑canton error analysis, Streamlit dashboard, and scripts for data preparation, training, and evaluation. Ready for replication on modest hardware.

## Setup with Docker

1. **Prerequisites**
    - Docker and Docker Compose installed
    - Git to clone the repository

2. **Quick Start**
    ```bash
    docker compose up
    ```
    This will:
    - Build the Python 3.11 environment
    - Install required dependencies (PyTorch, Whisper, FastAPI)
    - Start the API server on port 8000

3. **API Endpoints**
    - Health check: `GET /health`
    - Model loading: `GET /load-model`
    - Root endpoint: `GET /`

4. **Development**
    The Docker setup includes:
    - Volume mounts for `./data` and `./src`
    - Automatic reload with uvicorn
    - Non-root user for security
    - Health checks every 30s