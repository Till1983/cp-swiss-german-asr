# Testing Guide

> Comprehensive documentation for the test suite, coverage philosophy, and contributing new tests to the Swiss German ASR project.

---

## Table of Contents

- [Test Structure](#test-structure)
- [Coverage Philosophy](#coverage-philosophy)
- [Running Tests](#running-tests)
- [Current Coverage Status](#current-coverage-status)
- [Known Coverage Gaps](#known-coverage-gaps)
- [Adding New Tests](#adding-new-tests)
- [Troubleshooting Tests](#troubleshooting-tests)

---

## Test Structure

The test suite is organized into three layers, each serving distinct validation purposes:

### Unit Tests (`tests/unit/`)

**Purpose:** Validate individual functions and classes in isolation.

**Characteristics:**
- Fast execution (<1ms per test)
- No external dependencies (mocked)
- No file I/O or network calls
- Tests single responsibility per test case

**Structure:**
```
tests/unit/
├── backend/           # API models and validation
├── data_tests/        # Data loading, preprocessing, collation
├── evaluation/        # Metrics, error analysis, evaluators
├── model_tests/       # Model wrappers (Whisper, Wav2Vec2, MMS)
└── utils/             # Helper functions, logging, file I/O
```

**Example:** Testing WER calculation:

```python
def test_calculate_wer_exact_match():
    """Perfect match should yield 0% WER."""
    wer = calculate_wer("hello world", "hello world")
    assert wer == 0.0
```

**Best Practices:**

- One logical assertion per test
- Descriptive test names: `test_<function>_<condition>_<expected_outcome>`
- Use fixtures for repeated setup
- Mock external dependencies (file systems, models, APIs)

### Integration Tests (`tests/integration/`)

**Purpose:** Validate interactions between multiple components.

**Characteristics:**

- Medium execution time (100ms-1s per test)
- Real dependencies (processors, file systems)
- Tests data flow through multiple layers
- Validates integration boundaries

**Structure:**

```
tests/integration/
├── test_backend_endpoints.py    # API endpoint behavior
├── test_data_pipeline.py         # Data loading → preprocessing → batching
└── test_model_evaluation.py      # Model → evaluator → metrics
```

**Example:** Data pipeline integration:

```python
def test_load_and_preprocess_audio():
    """Test audio loading flows into preprocessing correctly."""
    audio_path = "fixtures/audio/test_sample.wav"
    audio, sr = load_audio(audio_path)
    preprocessor = AudioPreprocessor(target_sample_rate=16000)
    processed, final_sr = preprocessor.preprocess(audio, sr)
    assert final_sr == 16000
    assert processed.mean() < 0.1  # Normalized
```

**Best Practices:**

- Use real fixtures (audio files, TSV metadata)
- Test boundary conditions (empty datasets, malformed files)
- Validate end-to-end data transformations
- Keep execution time reasonable (<5s total)

### End-to-End Tests (`tests/e2e/`)

**Purpose:** Validate complete user workflows from start to finish.

**Characteristics:**

- Slowest execution (1-10s per test)
- Tests full system behavior
- Validates user-facing APIs
- Uses production-like configurations

**Structure:**

```
tests/e2e/
├── test_api_workflow.py          # API evaluation endpoint flows
└── test_evaluation_workflow.py   # Complete evaluation pipeline
```

**Example:** Complete API evaluation flow:

```python
def test_complete_api_evaluation_flow():
    """Test full evaluation from API request to results."""
    response = client.post("/api/evaluate", json={
        "model_type": "whisper",
        "model": "base",
        "limit": 5
    })
    assert response.status_code == 200
    data = response.json()
    assert "overall_wer" in data
    assert data["total_samples"] == 5
```

**Best Practices:**

- Mock heavyweight operations (GPU inference) with realistic outputs
- Test error handling and validation
- Verify response schemas
- Ensure idempotency where applicable

---

## Coverage Philosophy

### Target Coverage

**Overall Target:** 99%+ (excluding intentional gaps)

**Per-Module Targets:**

- **Critical paths:** 100% (metrics, evaluators, model wrappers)
- **Business logic:** 100% (data loading, preprocessing, API endpoints)
- **Utilities:** 95%+ (logging, file I/O, checkpoint management)
- **Integration seams:** Validated via integration tests

### Risk-Based Prioritization

Coverage targets are based on failure impact:

1. **High Risk (100% required):**
    - Metric calculations (WER, CER, BLEU)
    - Model transcription logic
    - Error analysis algorithms
    - API validation and error handling

2. **Medium Risk (95%+ required):**
    - Data loading and preprocessing
    - File I/O utilities
    - Configuration management

3. **Lower Risk (Integration-tested acceptable):**
    - Heavy external dependencies (HuggingFace processors)
    - Visualization code (dashboard rendering)

### Why Not 100% Everywhere?

Chasing absolute 100% coverage can lead to:
- **Brittle tests:** Over-mocking complex external libraries
- **Low-value assertions:** Testing framework behavior instead of our logic
- **Maintenance burden:** Tests break with minor refactors
- **False confidence:** High coverage ≠ good tests

**Our approach:** Focus on meaningful coverage of our code paths, with integration tests validating external dependencies.

---

## Running Tests

### Quick Reference

```bash
# All tests (unit + integration + e2e)
docker compose run --rm test

# Test categories
docker compose run --rm test-unit          # Fast unit tests only
docker compose run --rm test-integration   # Integration tests only
docker compose run --rm test-e2e          # End-to-end tests only

# Coverage report
docker compose run --rm test-coverage     # Generates htmlcov/index.html

# Specific test file
docker compose run --rm test pytest tests/unit/evaluation/test_metrics.py

# Specific test function
docker compose run --rm test pytest tests/unit/evaluation/test_metrics.py::test_calculate_wer_exact_match

# Verbose output with print statements
docker compose run --rm test pytest -v -s tests/unit/
```

### Coverage Reports

After running `test-coverage`, view the HTML report:

```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

**Report shows:**

- Line coverage per module
- Missed lines highlighted in red
- Branch coverage for conditionals
- Execution counts per line

### Continuous Integration

GitHub Actions runs the full test suite on every push:

```yaml
- name: Run tests with coverage
  run: docker compose run --rm test-coverage

- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v3
```

**Coverage thresholds enforced:**

- Overall: 95%
- Per-file: 90% (with documented exceptions)

---

## Current Coverage Status

Coverage is generated automatically and should not be hand-maintained in the repo. To see up-to-date metrics and missed lines:

```bash
docker compose run --rm test-coverage
open htmlcov/index.html  # macOS
```

Targets and philosophy:

- Overall target: 97%+ (100% on critical paths)
- Business logic (data loading, preprocessing, API): 100%
- Utilities (logging, file I/O): 95%+
- Heavy external integrations validated via integration/e2e tests

Recent improvements (Dec 2025):

- Added comprehensive unit tests for `src/frontend/utils/data_loader.py` and `src/training/trainer.py`
- Total tests: 463; overall coverage ~97%
- HTML coverage report generated at `htmlcov/index.html`

Note: Coverage figures are expected to change as the codebase evolves; rely on the generated report rather than static documentation.

---

## Known Coverage Gaps

### Collator Module (68% Coverage)

**File:** `src/data/collator.py`

**Current Coverage:** 68% (8 lines missed)

**Rationale:**

The `AudioDataCollatorCTC` class wraps HuggingFace processor methods that are difficult to mock accurately:

1. **processor.pad()** — Internal padding logic depends on:
   - Tokenizer configuration (pad token, special tokens)
   - Feature extractor settings (sampling rate, normalization)
   - Model-specific defaults (Whisper vs. Wav2Vec2 vs. MMS)

2. **as_target_processor()** — Context manager for label processing:
   - Switches processor mode internally
   - Requires real tokenizer state
   - Mocking requires re-implementing HuggingFace internals

3. **Dynamic padding behavior:**
   - Depends on actual input tensor shapes
   - Attention mask generation logic varies by model
   - Requires realistic audio/label pairs to validate

**Why Not Mock It?**

Attempting full mocking would require:
```python
# This defeats the purpose of unit testing our code
mock_processor.pad.return_value = {
    "input_values": torch.randn(2, 100),  # Guessing shapes
    "attention_mask": torch.ones(2, 100)  # Guessing mask logic
}
# ^ False confidence: we're not testing real processor behavior
```

**How It's Validated:**

✅ **Integration tests** exercise the collator with real processors:

- `test_data_pipeline.py::test_load_and_preprocess_audio`
- `test_model_evaluation.py::test_wav2vec2_evaluator_flow`
- `test_model_evaluation.py::test_mms_evaluator_flow`

✅ **E2E tests** validate end-to-end data flow through collator:

- `test_evaluation_workflow.py::test_full_workflow_mocked`

✅ **Unit tests** cover testable parts:

- Initialization and configuration
- `get_processor_for_model()` function
- Error handling for unsupported models

**Conclusion:** The 68% coverage is acceptable because:

1. Critical logic (processor selection, configuration) is covered
2. Integration tests validate real-world behavior
3. Adding unit tests would require brittle, high-fidelity mocks
4. Risk is low: HuggingFace processors are well-tested upstream

### Frontend Data Loader (contextual message paths)

Some branches in `src/frontend/utils/data_loader.py` are primarily Streamlit UI messaging (warnings/info when scanning directories or skipping invalid models). These are low-risk pathways and are exercised indirectly; full unit coverage would require asserting Streamlit side-effects. We prioritize testing data correctness and error handling over UI messages.

### Trainer Checkpoint Naming

`src/training/trainer.py` includes custom checkpoint naming in `_save_checkpoint`. Full verification requires the real HuggingFace `Trainer` save flow. Unit tests cover logging, argument handling, and metrics persistence; the save path logic is validated in integration runs.

### Future Improvements

If mocking becomes easier (e.g., HuggingFace provides test utilities), we can:

- Add unit tests for padding edge cases
- Test label masking logic independently
- Validate attention mask generation

**For now:** Document the gap, validate via integration tests, revisit periodically.

---

## Adding New Tests

### Step 1: Choose Test Type

**Add unit test if:**

- Testing a pure function (no I/O, no dependencies)
- Validating single class method in isolation
- Execution time < 1ms

**Add integration test if:**
- Testing interaction between 2+ components
- Requires real file system or processors
- Execution time 100ms-1s

**Add e2e test if:**
- Testing complete user workflow
- Involves API endpoints or CLI commands
- Execution time 1-10s

### Step 2: Create Test File

Follow the naming convention:

```
tests/
├── unit/
│   └── module_name/
│       └── test_feature.py
├── integration/
│   └── test_feature_integration.py
└── e2e/
    └── test_feature_workflow.py
```

### Step 3: Write Test

Use this template:

```python
"""Unit tests for [feature description]."""
import pytest
from unittest.mock import Mock, patch
from src.module.feature import FunctionToTest


class TestFeature:
    """Test suite for Feature class."""

    @pytest.fixture
    def sample_input(self):
        """Create sample input data."""
        return {"key": "value"}

    @pytest.mark.unit
    def test_function_expected_behavior(self, sample_input):
        """Test function handles expected input correctly."""
        result = FunctionToTest(sample_input)
        
        assert result is not None
        assert result["output"] == "expected"

    @pytest.mark.unit
    def test_function_edge_case(self):
        """Test function handles empty input gracefully."""
        result = FunctionToTest({})
        
        assert result == {}

    @pytest.mark.unit
    def test_function_error_handling(self):
        """Test function raises ValueError for invalid input."""
        with pytest.raises(ValueError, match="Invalid input"):
            FunctionToTest(None)
```

### Step 4: Run Test

```bash
# Run your new test
docker compose run --rm test pytest tests/unit/module_name/test_feature.py -v

# Verify coverage increased
docker compose run --rm test-coverage
```

### Step 5: Update Coverage Targets

If adding critical tests, update coverage thresholds:

```ini
# pytest.ini
[tool:pytest]
addopts = 
    --cov=src 
    --cov-report=html 
    --cov-fail-under=99
```

### Best Practices Checklist

- [ ] Test name describes behavior: `test_<function>_<condition>_<outcome>`
- [ ] One logical assertion per test
- [ ] Uses fixtures for setup/teardown
- [ ] Mocks external dependencies
- [ ] Tests both success and error paths
- [ ] Includes docstring explaining purpose
- [ ] Marked with appropriate decorator (`@pytest.mark.unit`)
- [ ] Execution time < 1s (for unit tests)

---

## Troubleshooting Tests

### Test Failures

**Issue:** Test fails locally but passes in CI

**Solutions:**
- Check file path assumptions (absolute vs relative)
- Verify Docker volume mounts: `docker compose config`
- Ensure test data fixtures are committed to Git
- Check for timezone or locale dependencies

**Issue:** Flaky test (passes sometimes, fails other times)

**Solutions:**
- Look for race conditions (threading, async)
- Check for random number generation without seeded fixtures
- Verify cleanup in teardown (temp files, mock state)
- Use `pytest-repeat` to reproduce: `pytest --count=10 test_file.py`

### Coverage Issues

**Issue:** Coverage report shows missed lines but test runs them

**Solutions:**
- Check if code is in `except` block (may not execute)
- Verify all branches tested (if/else, try/except)
- Look for early returns or short-circuits
- Use `coverage debug trace` for detailed execution

**Issue:** Coverage report missing files

**Solutions:**
- Ensure `__init__.py` exists in all packages
- Check `pytest.ini` source paths: `--cov=src`
- Verify files aren't in `.coveragerc` exclude list
- Rebuild Docker image: `docker compose build test`

### Fixture Issues

**Issue:** Fixture not found

**Solutions:**
- Check fixture is defined in `conftest.py`
- Verify fixture scope (function/module/session)
- Ensure test imports fixture correctly
- Check fixture dependencies (fixtures can use other fixtures)

**Issue:** Fixture cleanup not running

**Solutions:**

- Use `yield` instead of `return` for teardown:

```python
@pytest.fixture
def temp_file():
        f = open("temp.txt", "w")
        yield f
        f.close()  # Cleanup runs after test
```

- Check for exceptions in test (may skip cleanup)
- Verify fixture scope matches test needs

---

## References

- **Pytest Documentation:** <https://docs.pytest.org/>
- **Coverage.py:** <https://coverage.readthedocs.io/>
- **Testing Best Practices:** [Martin Fowler - Test Pyramid](https://martinfowler.com/articles/practical-test-pyramid.html)
- **Mocking Guide:** <https://docs.python.org/3/library/unittest.mock.html>

---

**Last Updated:** December 11, 2025  
**Maintainer:** Till Ermold (<till.ermold@code.berlin>)
