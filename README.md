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
1. Download FHNW's All Swiss German Dialects Test Corpus [here](https://cs.technik.fhnw.ch/i4ds-datasets).
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

## Evaluation API

Run ASR model evaluation on Swiss German test set.

### Example Request

For Whisper small model evaluation on 10 samples:
```bash
curl -X POST http://localhost:8000/api/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "whisper", 
    "model": "small",                                
    "limit": 10
  }'
```

For wav2vec2 model evaluation on full test set:
```bash
curl -X POST http://localhost:8000/api/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "wav2vec2",
    "model": "facebook/wav2vec2-large-xlsr-53-german",
    "limit": 100
  }'
```

### Response Fields
- `model`: Name of the evaluated ASR model
- `total_samples`: Number of audio samples processed
- `overall_wer`: Word Error Rate across all dialects (percentage, 0-100)
- `overall_cer`: Character Error Rate across all dialects (percentage, 0-100)
- `overall_bleu`: BLEU score across all dialects (0-100, higher is better)
- `per_dialect_wer`: Dictionary mapping each dialect/canton to its WER (percentage, 0-100)
- `per_dialect_cer`: Dictionary mapping each dialect/canton to its CER (percentage, 0-100)
- `per_dialect_bleu`: Dictionary mapping each dialect/canton to its BLEU score (0-100, higher is better)

**Note:** WER and CER are returned as percentages (0-100), where lower values indicate better accuracy. BLEU is also scaled to 0-100, where higher values indicate better translation quality.


**Tip:** Use the `limit` parameter for quick testing with a subset of the test set.

## Running Model Evaluation

Batch evaluation script for comparing multiple ASR models on the Swiss German test set.

### Usage

**Evaluate multiple models (default: whisper-base and wav2vec2-german):**
```bash
docker compose run --rm api python scripts/evaluate_models.py
```

**Evaluate a single model:**
```bash
docker compose run --rm api python scripts/evaluate_models.py --models whisper-base
```

**Evaluate specific models:**
```bash
docker compose run --rm api python scripts/evaluate_models.py --models whisper-small whisper-medium wav2vec2-german
```

**Quick test with sample limit:**
```bash
docker compose run --rm api python scripts/evaluate_models.py --limit 10
```

### Available Models

- **Whisper:** `whisper-tiny`, `whisper-base`, `whisper-small`, `whisper-medium`, `whisper-large`, `whisper-large-v2`, `whisper-large-v3`
- **Wav2Vec2:** `wav2vec2-german`, `wav2vec2-base`, `wav2vec2-large`

### Output

Results are saved to `results/metrics/YYYYMMDD_HHMMSS/` with:
- `{model}_results.json` - Full metrics including per-sample predictions
- `{model}_results.csv` - Summary metrics table

Example output structure:
```
results/metrics/20251105_110723/
├── whisper-medium_results.json
├── whisper-medium_results.csv
├── wav2vec2-german_results.json
└── wav2vec2-german_results.csv
```

The script prints a summary table comparing WER, CER, BLEU scores, and sample counts across all evaluated models.

## Testing

Run the test suite to verify metrics calculations (WER, CER, BLEU) and evaluation logic.

### Running All Tests

```bash
docker compose run --rm api python -m pytest tests/ -v
```

### Running Specific Test Classes

**Test WER calculations:**
```bash
docker compose run --rm api python -m pytest tests/test_evaluation.py::TestCalculateWER -v
```

**Test CER calculations:**
```bash
docker compose run --rm api python -m pytest tests/test_evaluation.py::TestCalculateCER -v
```

**Test batch metrics:**
```bash
docker compose run --rm api python -m pytest tests/test_evaluation.py::TestBatchWER -v
```

### Running Individual Tests

```bash
docker compose run --rm api python -m pytest tests/test_evaluation.py::TestCalculateWER::test_calculate_wer_exact_match -v
```

### Coverage Report

Generate a test coverage report:
```bash
docker compose run --rm api python -m pytest tests/ --cov=src/evaluation --cov-report=term-missing
```

### What Is Tested

The test suite validates:
- **WER (Word Error Rate):** Exact matches, partial matches, empty strings, case sensitivity
- **CER (Character Error Rate):** Character-level edit distance calculations
- **BLEU Score:** Translation quality metrics for ASR outputs
- **Batch Processing:** Aggregated metrics across multiple samples
- **Edge Cases:** Empty inputs, mismatched lengths, complete mismatches

### Expected Output

All tests should pass:
```
======================== test session starts ========================
collected 64 items

tests/test_evaluation.py::TestCalculateWER::test_calculate_wer_exact_match PASSED
tests/test_evaluation.py::TestCalculateWER::test_calculate_wer_complete_mismatch PASSED
tests/test_evaluation.py::TestCalculateWER::test_calculate_wer_partial_match PASSED
...
tests/test_evaluation.py::TestSwissGermanRealistic::test_bleu_partial_swiss_german_match PASSED

======================== 64 passed in 4.42s ========================
```