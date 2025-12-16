# Swiss German ASR Evaluation Platform

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-required-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Comparative ASR evaluation for Swiss German dialects:** Reproducible Docker pipeline with baseline Whisper inference, fine-tuned Wav2Vec2 models, per-canton error analysis, and interactive Streamlit dashboard.

---

## Table of Contents

- [Quick Start](#quick-start)
- [What You'll See](#what-youll-see)
- [System Components](#system-components)
- [Running Evaluations](#running-evaluations)
- [Error Analysis](#error-analysis)
- [Testing Suite](#testing-suite)
- [Data Pipeline](#data-pipeline)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## Quick Start

### Prerequisites (5 minutes)

- **Docker Desktop** installed ([download](https://www.docker.com/products/docker-desktop))
- **Git** for cloning the repository
- **10GB** free disk space
- **8GB** RAM available

### One-Command Setup (2 minutes)

```bash
# Clone repository
git clone https://github.com/Till1983/cp-swiss-german-asr.git
cd cp-swiss-german-asr

# Start dashboard (downloads dependencies on first run)
docker compose up dashboard
```

**Wait for:** `Streamlit app started` message, then open your browser.

### Access the Application

**📊 Dashboard:** http://localhost:8501
- Interactive visualizations of evaluation results
- Multi-model comparison (Whisper, Wav2Vec2)
- Per-dialect performance breakdown
- Error analysis with word-level alignment

**📚 API Documentation:** http://localhost:8000/docs
- Swagger UI for FastAPI endpoints
- Test evaluation endpoints interactively

**⚠️ First Start:** Initial setup downloads model weights (~2GB). Subsequent starts take <30 seconds.

---

## What You'll See

The dashboard displays **pre-computed evaluation results** from multiple ASR models across Swiss German dialects:

### Key Metrics
- **WER** (Word Error Rate): Lower is better, measures word-level accuracy
- **CER** (Character Error Rate): Lower is better, measures character-level accuracy
- **BLEU**: Higher is better (0-100), measures translation quality

### Visualizations
1. **Model Comparison:** Side-by-side WER/CER/BLEU across all models
2. **Dialect Analysis:** Per-canton performance breakdown (BE, ZH, AG, etc.)
3. **Error Patterns:** Confusion matrices and systematic error identification
4. **Sample Inspection:** Word-level alignment visualization for error analysis

### Data Source
All results are loaded from the `results/metrics/` directory - **no computation happens during dashboard viewing**. This ensures fast, reproducible access to evaluation data.

---

## System Components

### Dashboard (Streamlit)
Interactive web interface for exploring evaluation results.

```bash
docker compose up dashboard
```

Access at http://localhost:8501

**Features:**
- Model performance comparison
- Dialect breakdown analysis
- Interactive Plotly charts
- Error sample viewer with audio playback
- Downloadable CSV exports

### API (FastAPI)

RESTful API for programmatic model evaluation and result browsing.

```bash
docker compose up api
```

**Base URL:** http://localhost:8000  
**Interactive Documentation:** http://localhost:8000/docs (Swagger UI with try-it-out)

#### Core Endpoints

**Model Evaluation:**
- `POST /api/evaluate` - Run evaluation on test set
  ```bash
  curl -X POST http://localhost:8000/api/evaluate \
    -H "Content-Type: application/json" \
    -d '{"model_type": "whisper", "model": "small", "limit": 10}'
  ```
  
**Result Browsing:**
- `GET /api/results` - List all saved evaluation results
- `GET /api/results/{model}` - Get latest results for specific model
- `GET /api/results/{model}?timestamp=YYYYMMDD_HHMMSS` - Get results from specific run
- `GET /api/results/{model}/{dialect}` - Filter results by dialect

**Model Cache Management:**
- `GET /api/cache/info` - Check loaded models and cache status
- `POST /api/cache/clear` - Free GPU/CPU memory (clear all cached models)
- `GET /api/models` - List all registered ASR models

**System:**
- `GET /health` - API health check
- `GET /load-model` - Warm up default Whisper base model

#### Cache Management

The API caches loaded models to avoid repeated downloads. Use cache endpoints when:
- **Before switching models:** `POST /api/cache/clear` to free GPU memory
- **Debugging memory issues:** Check `GET /api/cache/info` for loaded models
- **After extended idle:** Models remain cached until manually cleared

**Example workflow:**
```bash
# Check what's cached
curl http://localhost:8000/api/cache/info

# Clear before loading large model
curl -X POST http://localhost:8000/api/cache/clear

# Run evaluation
curl -X POST http://localhost:8000/api/evaluate \
  -H "Content-Type: application/json" \
  -d '{"model_type": "whisper", "model": "large-v3", "limit": 100}'
```

For detailed request/response schemas and interactive testing, use the Swagger documentation at `/docs`.

### Testing Suite
Comprehensive unit, integration, and end-to-end tests.

```bash
# Run all tests
docker compose run --rm test

# Run specific test categories
docker compose run --rm test-unit          # Fast unit tests
docker compose run --rm test-integration   # Integration tests
docker compose run --rm test-e2e          # End-to-end tests
docker compose run --rm test-coverage     # With coverage report
```

For detailed explanation of the testing framework, see [TESTING.md](docs/TESTING.md).

---

## Running Evaluations

### Batch Model Evaluation

Evaluate multiple ASR models on the Swiss German test set:

```bash
# Evaluate default models (whisper-base, wav2vec2-german)
docker compose run --rm api python scripts/evaluate_models.py

# Evaluate specific models
docker compose run --rm api python scripts/evaluate_models.py \
  --models whisper-small whisper-medium wav2vec2-german

# Quick test with sample limit
docker compose run --rm api python scripts/evaluate_models.py --limit 10
```

### Available Models

**Whisper:** `whisper-tiny`, `whisper-base`, `whisper-small`, `whisper-medium`, `whisper-large`, `whisper-large-v2`, `whisper-large-v3`

**Wav2Vec2:** `wav2vec2-german`, `wav2vec2-base`, `wav2vec2-large`

### Output Structure

Results are saved to `results/metrics/YYYYMMDD_HHMMSS/`:

```
results/metrics/20241209_143022/
├── whisper-medium_results.json    # Full metrics + per-sample predictions
├── whisper-medium_results.csv     # Summary metrics table
├── wav2vec2-german_results.json
└── wav2vec2-german_results.csv
```

### API Evaluation

Alternatively, use the FastAPI endpoint programmatically. See the [API section](#api-fastapi) for complete endpoint documentation.

**Quick example:**

```bash
curl -X POST http://localhost:8000/api/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "whisper",
    "model": "small",
    "limit": 10
  }'
```

**Response structure:**
```json
{
  "model": "small",
  "total_samples": 10,
  "failed_samples": 0,
  "overall_wer": 28.5,
  "overall_cer": 12.3,
  "overall_bleu": 68.7,
  "per_dialect_wer": {"BE": 25.0, "ZH": 30.0},
  "per_dialect_cer": {"BE": 10.0, "ZH": 13.0},
  "per_dialect_bleu": {"BE": 72.0, "ZH": 67.0}
}
```

**Note:** WER and CER are percentages (0-100, lower is better). BLEU is 0-100 (higher is better).

---

## Error Analysis

Analyze error patterns and identify worst-performing samples:

```bash
docker compose run --rm api python scripts/analyze_errors.py \
  --input_dir results/metrics/20241209_143022 \
  --output_dir results/error_analysis \
  --top_percent 0.10
```

**Generates:**
- `worst_samples.json` - Top 10% highest-WER samples
- `error_statistics.json` - Confusion patterns and error type distributions
- Per-dialect error breakdowns

**View in Dashboard:**
Navigate to the "Sample Predictions" tab to interactively explore:
- Word-level alignment (correct/substitution/insertion/deletion)
- Error type distributions (pie charts)
- Confusion pairs (most common word errors)

---

## Data Pipeline

### Download Dataset

1. Download FHNW's All Swiss German Dialects Test Corpus [here](https://cs.technik.fhnw.ch/i4ds-datasets)
2. Extract to `data/raw/fhnw-swiss-german-corpus/`

### Prepare Data Splits

```bash
docker compose run --rm api python scripts/prepare_scripts.py
```

Creates train/val/test splits:
- `data/metadata/train.tsv` (70%)
- `data/metadata/val.tsv` (15%)
- `data/metadata/test.tsv` (15%)

**TSV Format:** `client_id | path | sentence | up_votes | down_votes | age | gender | accent | audio_path`

### Verify Data Loading

```bash
docker compose run --rm api python -c "
import pandas as pd
train = pd.read_csv('/app/data/metadata/train.tsv', sep='\t')
print(f'Loaded {len(train)} training samples')
print(train.head())
"
```

---

## Troubleshooting

### Dashboard Won't Start

**Issue:** Docker container fails to start

**Solutions:**
- Verify Docker Desktop is running: `docker --version`
- Check port availability: `lsof -i :8501` (kill process if occupied)
- Review logs: `docker compose logs dashboard`

### Port Conflict (8501 in use)

**Issue:** Port 8501 already in use

**Solutions:**

**Option 1:** Change port in `docker-compose.yml`:
```yaml
dashboard:
  ports:
    - "8502:8501"  # Change 8501 to 8502
```

**Option 2:** Use alternative port via command line:
```bash
docker compose run --rm -p 8502:8501 dashboard
```

Then access at http://localhost:8502

### Slow First Start

**Issue:** Initial startup takes 5+ minutes

**Explanation:** First run downloads:
- Model weights (~2GB) from HuggingFace
- Python dependencies
- Dataset processing

**Solution:** Wait for completion. Subsequent starts are <30 seconds (cached).

### No Data in Dashboard

**Issue:** Dashboard shows "No evaluation results found"

**Solutions:**
- Verify `results/metrics/` directory contains evaluation files
- Run evaluation: `docker compose run --rm api python scripts/evaluate_models.py`
- Check file permissions: `ls -la results/metrics/`

### Memory Issues

**Issue:** Docker reports out of memory

**Solutions:**
- Increase Docker memory limit (Docker Desktop → Settings → Resources → Memory)
- Recommend: 8GB minimum, 16GB optimal
- Reduce concurrent models being evaluated

### API Returns 404

**Issue:** API endpoints return 404 Not Found

**Solutions:**
- Ensure API service is running: `docker compose up api`
- Verify URL: http://localhost:8000 (not 8501)
- Check API logs: `docker compose logs api`

### Browser Compatibility

**Issue:** Dashboard doesn't display correctly

**Solutions:**
- Use modern browser (Chrome, Firefox, Edge, Safari)
- Clear browser cache
- Disable browser extensions
- Try incognito/private mode

---

## Citation

If you use this work, please cite:

```bibtex
@thesis{ermold2025swissgerman,
  author = {Ermold, Till},
  title = {Comparative Evaluation of ASR Models for Swiss German Dialects},
  school = {CODE University of Applied Sciences Berlin},
  year = {2025},
  type = {Bachelor's Thesis}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Contact

**Author:** Till Ermold  
**Email:** till.ermold@code.berlin  
**Institution:** CODE University of Applied Sciences Berlin

---

**Last Updated:** December 16, 2025  
**Version:** 3.0 (Add Core API Endpoints)