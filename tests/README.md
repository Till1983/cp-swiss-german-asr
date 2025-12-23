# Test Suite for Swiss German ASR Project

This directory contains the comprehensive test suite for the Swiss German ASR (Automatic Speech Recognition) evaluation project.

## Test Structure

```
tests/
├── conftest.py           # Root fixtures (Docker-aware path resolution)
├── pytest.ini            # Pytest configuration (in project root)
│
├── fixtures/             # Test fixture data
│   ├── audio/            # Sample audio files
│   │   ├── sample_be_1.wav
│   │   ├── sample_zh_2.wav
│   │   └── sample_vs_3.wav
│   └── data/
│       ├── mock_swiss_german.tsv
│       └── mock_results.json
│
├── unit/                 # Fast unit tests (no external dependencies)
│   ├── conftest.py
│   ├── data/
│   │   ├── test_loader.py
│   │   ├── test_preprocessor.py
│   │   ├── test_collator.py
│   │   └── test_splitter.py
│   ├── evaluation/
│   │   ├── test_metrics.py
│   │   ├── test_error_analyzer.py
│   │   └── test_evaluator.py
│   ├── models/
│   │   ├── test_wav2vec2_model.py
│   │   └── test_mms_model.py
│   ├── utils/
│   │   ├── test_audio_utils.py
│   │   ├── test_file_utils.py
│   │   ├── test_checkpoint_manager.py
│   │   └── test_logging_config.py
│   └── backend/
│       └── test_pydantic_models.py
│
├── integration/          # Integration tests (may require data volumes)
│   ├── conftest.py
│   ├── test_data_pipeline.py
│   ├── test_model_evaluation.py
│   └── test_backend_endpoints.py
│
└── e2e/                  # End-to-end tests (full workflows)
    ├── conftest.py
    ├── test_evaluation_workflow.py
    └── test_api_workflow.py
```

## Running Tests

### Using Docker Compose (Recommended)

```bash
# Run all tests
docker compose run --rm test

# Run only unit tests (fast, no external dependencies)
docker compose run --rm test-unit

# Run integration tests
docker compose run --rm test-integration

# Run end-to-end tests
docker compose run --rm test-e2e

# Run tests with coverage report
docker compose run --rm test-coverage
```

### Running Tests Locally

```bash
# Install dependencies first
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run only unit tests
pytest tests/unit/ -v -m unit

# Run integration tests
pytest tests/integration/ -v -m integration

# Run e2e tests
pytest tests/e2e/ -v -m e2e

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Test Markers

Tests are organised with pytest markers:

- `@pytest.mark.unit` - Fast unit tests with no external dependencies
- `@pytest.mark.integration` - Tests requiring data volumes or external services
- `@pytest.mark.e2e` - End-to-end workflow tests
- `@pytest.mark.slow` - Tests taking more than 10 seconds
- `@pytest.mark.gpu` - Tests requiring GPU (skip on CPU-only systems)

### Running by Marker

```bash
# Run only unit tests
pytest -m unit

# Run integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Run everything except GPU tests
pytest -m "not gpu"
```

## Test Categories

### Unit Tests (`tests/unit/`)

Fast, isolated tests with mocked dependencies:

- **Data Module**: Tests for data loading, preprocessing, collation, and splitting
- **Evaluation Module**: Tests for metrics (WER, CER, BLEU), error analysis, and evaluator
- **Models Module**: Tests for Wav2Vec2 and MMS model wrappers
- **Utils Module**: Tests for audio utilities, file handling, checkpoint management, logging
- **Backend Module**: Tests for Pydantic models and request/response validation

### Integration Tests (`tests/integration/`)

Tests that verify component interactions:

- **Data Pipeline**: End-to-end data loading and preprocessing
- **Model Evaluation**: Evaluator integration with metrics and models
- **Backend Endpoints**: FastAPI endpoint integration tests

### End-to-End Tests (`tests/e2e/`)

Full workflow tests:

- **Evaluation Workflow**: Complete evaluation from data to results
- **API Workflow**: Full API request/response cycles

## Fixtures

### Root Fixtures (`tests/conftest.py`)

- `app_root`: Project root directory (Docker-aware)
- `is_docker`: Check if running in Docker
- `temp_dir`: Auto-cleaned temporary directory
- `sample_audio_path`: Path to sample audio file
- `mock_swiss_german_data`: Path to mock TSV dataset
- `sample_audio_array`: Generated sine wave for testing

### Unit Test Fixtures (`tests/unit/conftest.py`)

- `sample_swiss_german_texts`: Sample text pairs
- `mock_wer_inputs`: Reference/hypothesis pairs for WER testing
- `mock_metadata_dict`: Mock metadata as list of dicts
- `mock_evaluation_results`: Sample evaluation results
- `constant_audio`, `silent_audio`: Edge case audio arrays

## Adding New Tests

1. Create test file in appropriate directory (`unit/`, `integration/`, or `e2e/`)
2. Add appropriate marker (`@pytest.mark.unit`, etc.)
3. Use fixtures from `conftest.py` files
4. Follow naming convention: `test_<module>_<function>.py`
5. Group related tests in classes prefixed with `Test`

### Example Test

```python
import pytest
from src.module import function_to_test

class TestFunctionToTest:
    @pytest.mark.unit
    def test_basic_functionality(self):
        """Test basic function behaviour."""
        result = function_to_test("input")
        assert result == "expected"

    @pytest.mark.unit
    def test_edge_case(self, temp_dir):
        """Test edge case with temp directory fixture."""
        # Use fixture
        output_file = temp_dir / "output.txt"
        function_to_test(output=str(output_file))
        assert output_file.exists()
```

## Coverage Goals

- Unit tests: ~150-200 tests
- Integration tests: ~20-30 tests
- E2E tests: ~5-10 tests
- Target coverage: >80% for `src/` modules

## Troubleshooting

### Tests Skipped Due to Missing Fixtures

Some tests require fixture audio files. Generate them if missing:

```python
# In tests/fixtures/audio/
import numpy as np
from scipy.io import wavfile

sample_rate = 16000
duration = 1.0
t = np.linspace(0, duration, int(sample_rate * duration))
audio = (np.sin(2 * np.pi * 440.0 * t) * 32767).astype(np.int16)

for name in ['sample_be_1.wav', 'sample_zh_2.wav', 'sample_vs_3.wav']:
    wavfile.write(name, sample_rate, audio)
```

### Tests Skipped Due to Missing Data

Integration tests may require the FHNW corpus. Run with data volume:

```bash
docker compose run --rm test-integration
```

### Import Errors

Ensure `PYTHONPATH=/app` is set (automatic in Docker).
