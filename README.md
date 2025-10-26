# cp-swiss-german-asr
Comparative ASR for Swiss‑German dialects: reproducible Docker pipeline, baseline Whisper inference, fine‑tuned Whisper‑medium & wav2vec2 models, per‑canton error analysis, Streamlit dashboard, and scripts for data preparation, training, and evaluation. Ready for replication on modest hardware.

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

## Data Pipeline

### Download Dataset
1. Download FHNW's All Swiss German Dialects Test Corpus from [here](https://cs.technik.fhnw.ch/i4ds-datasets).
2. Extract to `data/raw/fhnw-swiss-german-corpus/`

### Prepare Data Splits
```bash
docker compose run --rm api python scripts/prepare_scripts.py
```

This will create:
- `data/metadata/train.tsv` (70% of data)
- `data/metadata/val.tsv` (15% of data)
- `data/metadata/test.tsv` (15% of data)

Each TSV contains: `client_id`, `path`, `sentence`, `up_votes`, `down_votes`, `age`, `gender`, `accent`, `audio_path`

### Verify Data Loading
```bash
docker compose run --rm api python -c "import pandas as pd
train = pd.read_csv('/app/data/metadata/train.tsv', sep='\t')
print(f'Loaded {len(train)} training samples')
print(train.head())"
```