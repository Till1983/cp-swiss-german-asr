# Claude Code: COMPREHENSIVE Test Suite Implementation

## CRITICAL INSTRUCTION
**DO NOT CREATE PLACEHOLDER TESTS.** You must write COMPLETE, WORKING tests based on the actual implementation in the codebase. This means:
- Read the actual source code for each module
- Understand what functions/classes exist and what they do
- Write comprehensive tests that actually test the functionality
- Include edge cases, error handling, and boundary conditions
- Use proper mocking for external dependencies

The test suite should be **production-ready**, not a skeleton to fill in later.

---

## Context
You are setting up a comprehensive test suite for a Swiss German ASR (Automatic Speech Recognition) evaluation project. The project uses Docker for all execution (no virtual environments), with tests run via `docker compose run --rm test`.

## Current State
- **Existing test files**:
  - `tests/test_evaluation.py` - Comprehensive unit tests for metrics (91 tests, ~500 lines)
  - `tests/test_error_analyzer.py` - Unit tests for error analysis
- **Project Architecture**: FastAPI backend + Streamlit frontend + evaluation pipeline
- **Tech Stack**: Python 3.11, PyTorch, transformers, pytest, Docker, librosa, jiwer, pandas

## Source Code Modules to Test

### Data Module (`src/data/`)
1. **loader.py**:
   - `load_swiss_german_metadata(filepath: str) -> pd.DataFrame` - Load TSV metadata
   - `load_audio(audio_path: str, sample_rate: int = 16000) -> np.ndarray` - Load audio with librosa

2. **preprocessor.py**:
   - `AudioPreprocessor` class with:
     - `__init__(target_sample_rate: int = 16000)`
     - `normalize_audio(audio: np.ndarray) -> np.ndarray` - Zero mean, unit variance
     - `resample_audio(audio: np.ndarray, orig_sr: int) -> np.ndarray` - Resample with librosa
     - `preprocess(audio: np.ndarray, orig_sr: int) -> Tuple[np.ndarray, int]` - Full pipeline

3. **collator.py**:
   - `AudioDataCollatorCTC` - Custom collator for CTC models (inspect actual implementation)

4. **splitter.py**:
   - Data splitting logic (inspect actual implementation)

### Evaluation Module (`src/evaluation/`)
1. **metrics.py** (ALREADY TESTED - tests/test_evaluation.py has 91 tests):
   - `calculate_wer()`, `calculate_cer()`, `calculate_bleu_score()`
   - `batch_wer()`, `batch_cer()`, `batch_bleu()`
   - `_normalize_text()`, `_filter_empty_references()`

2. **error_analyzer.py** (ALREADY TESTED - tests/test_error_analyzer.py exists):
   - `ErrorAnalyzer` class with alignment, categorization, confusion pairs

3. **evaluator.py**:
   - `ASREvaluator` class with:
     - `__init__(model_type: str, model_name: str)`
     - `load_model()` - Load ASR model
     - `transcribe(audio_path: str) -> str` - Single file transcription
     - `evaluate_dataset(metadata_path: str, audio_base_path, limit) -> Dict` - Batch evaluation

### Utils Module (`src/utils/`)
1. **audio_utils.py**:
   - `validate_audio_file(path: str) -> bool` - Check file exists and has valid extension
   - `get_audio_duration(path: str) -> Optional[float]` - Get duration with librosa

2. **file_utils.py**:
   - File handling utilities (inspect actual implementation)

3. **checkpoint_manager.py**:
   - Checkpoint management (inspect actual implementation)

4. **logging_config.py**:
   - Logging configuration (inspect actual implementation)

### Backend Module (`src/backend/`)
1. **models.py**:
   - Pydantic models: `EvaluateRequest`, `EvaluateResponse`

2. **endpoints.py**:
   - FastAPI router with `/evaluate` endpoint

### Models Module (`src/models/`)
1. **wav2vec2_model.py**:
   - Wav2Vec2 model wrapper (inspect for testable methods)

2. **mms_model.py**:
   - MMS model wrapper (inspect for testable methods)

---

## Your Mission

Implement a **hybrid test organization approach** with **COMPLETE, PRODUCTION-READY TESTS** that:
1. Create the full directory structure
2. Move existing tests to proper locations
3. Write comprehensive tests for ALL modules (not placeholders)
4. Set up Docker-friendly execution with multiple test services
5. Create comprehensive fixtures and configuration

---

## Phase 1: Create Directory Structure

```
tests/
├── __init__.py
├── conftest.py
├── pytest.ini
├── README.md
│
├── fixtures/
│   ├── __init__.py
│   ├── audio/
│   │   ├── sample_be_1.wav
│   │   ├── sample_zh_2.wav
│   │   └── sample_vs_3.wav
│   └── data/
│       ├── mock_swiss_german.tsv
│       └── mock_results.json
│
├── unit/
│   ├── __init__.py
│   ├── conftest.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── test_loader.py              # COMPLETE TESTS
│   │   ├── test_preprocessor.py        # COMPLETE TESTS
│   │   ├── test_collator.py            # COMPLETE TESTS
│   │   └── test_splitter.py            # COMPLETE TESTS
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── test_metrics.py             # MOVE FROM tests/test_evaluation.py
│   │   ├── test_error_analyzer.py      # MOVE FROM tests/test_error_analyzer.py
│   │   └── test_evaluator.py           # COMPLETE TESTS (mock models)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── test_wav2vec2_model.py      # COMPLETE TESTS (mock transformers)
│   │   └── test_mms_model.py           # COMPLETE TESTS (mock transformers)
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── test_audio_utils.py         # COMPLETE TESTS
│   │   ├── test_file_utils.py          # COMPLETE TESTS
│   │   ├── test_checkpoint_manager.py  # COMPLETE TESTS
│   │   └── test_logging_config.py      # COMPLETE TESTS
│   └── backend/
│       ├── __init__.py
│       └── test_pydantic_models.py     # COMPLETE TESTS
│
├── integration/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_data_pipeline.py           # COMPLETE TESTS
│   ├── test_model_evaluation.py        # COMPLETE TESTS
│   ├── test_backend_endpoints.py       # COMPLETE TESTS
│   └── test_training_pipeline.py       # COMPLETE TESTS (if applicable)
│
└── e2e/
    ├── __init__.py
    ├── conftest.py
    ├── test_evaluation_workflow.py     # COMPLETE TESTS
    └── test_api_workflow.py            # COMPLETE TESTS
```

---

## Phase 2: Configuration Files

### 1. `pytest.ini` (in project root)
```ini
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]

markers =
    unit: Fast unit tests (no external dependencies, no model loading)
    integration: Integration tests (requires data volumes, may load models)
    e2e: End-to-end tests (full system workflows)
    gpu: Tests requiring GPU (skip on CPU-only containers)
    slow: Tests taking >10 seconds

addopts = 
    --verbose
    --strict-markers
    --tb=short
    -ra
    --disable-warnings

pythonpath = .
norecursedirs = .git .venv __pycache__ data models results fixtures
```

### 2. `tests/conftest.py` (COMPLETE IMPLEMENTATION)
```python
"""Root test fixtures - Docker-aware path resolution and common fixtures."""
import pytest
from pathlib import Path
from typing import Generator
import tempfile
import numpy as np

# Docker-aware path resolution
IS_DOCKER = Path("/.dockerenv").exists()
APP_ROOT = Path("/app") if IS_DOCKER else Path(__file__).parent.parent

@pytest.fixture(scope="session")
def app_root() -> Path:
    """Project root directory - works in Docker and locally."""
    return APP_ROOT

@pytest.fixture(scope="session")
def is_docker() -> bool:
    """Check if tests are running in Docker container."""
    return IS_DOCKER

@pytest.fixture
def tests_root(app_root: Path) -> Path:
    """Tests directory root."""
    return app_root / "tests"

@pytest.fixture
def fixtures_root(tests_root: Path) -> Path:
    """Test fixtures directory."""
    return tests_root / "fixtures"

@pytest.fixture
def src_root(app_root: Path) -> Path:
    """Source code root."""
    return app_root / "src"

@pytest.fixture
def data_root(app_root: Path) -> Path:
    """Data directory - may not exist in unit tests."""
    return app_root / "data"

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Temporary directory that's automatically cleaned up."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def temp_results_dir(temp_dir: Path) -> Path:
    """Temporary results directory for test outputs."""
    results = temp_dir / "results"
    results.mkdir(parents=True)
    return results

@pytest.fixture
def temp_models_dir(temp_dir: Path) -> Path:
    """Temporary models directory for test model files."""
    models = temp_dir / "models"
    models.mkdir(parents=True)
    return models

@pytest.fixture
def sample_audio_path(fixtures_root: Path) -> Path:
    """Path to sample Bern dialect audio file."""
    audio_path = fixtures_root / "audio" / "sample_be_1.wav"
    if not audio_path.exists():
        pytest.skip(f"Sample audio not found: {audio_path}")
    return audio_path

@pytest.fixture
def mock_swiss_german_data(fixtures_root: Path) -> Path:
    """Path to mock Swiss German TSV dataset."""
    data_path = fixtures_root / "data" / "mock_swiss_german.tsv"
    if not data_path.exists():
        pytest.skip(f"Mock data not found: {data_path}")
    return data_path

@pytest.fixture
def fhnw_corpus_root(data_root: Path) -> Path:
    """FHNW Swiss German corpus - only available if volume mounted."""
    corpus_path = data_root / "raw" / "fhnw-swiss-german-corpus"
    if not corpus_path.exists():
        pytest.skip("FHNW corpus not mounted - run with data volume")
    return corpus_path

@pytest.fixture
def metadata_dir(data_root: Path) -> Path:
    """Metadata directory with train/val/test splits."""
    metadata = data_root / "metadata"
    if not metadata.exists():
        pytest.skip("Metadata directory not found - generate with prepare_scripts.py")
    return metadata

@pytest.fixture
def sample_audio_array() -> np.ndarray:
    """Generate a simple sine wave audio array for testing."""
    # 1 second of 440Hz sine wave at 16kHz sample rate
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    return audio
```

### 3. `tests/unit/conftest.py`
```python
"""Unit test fixtures - fast, no external dependencies."""
import pytest
from typing import Dict, List
import numpy as np

@pytest.fixture
def sample_swiss_german_texts() -> List[Dict[str, str]]:
    """Sample Swiss German sentences with Standard German translations."""
    return [
        {"swiss_german": "Grüezi mitenand", "standard_german": "Hallo zusammen", "dialect": "ZH"},
        {"swiss_german": "Härzlich willkomme z Bärn", "standard_german": "Herzlich willkommen in Bern", "dialect": "BE"},
        {"swiss_german": "Wie gohts der", "standard_german": "Wie geht es dir", "dialect": "VS"}
    ]

@pytest.fixture
def mock_wer_inputs() -> Dict[str, List[str]]:
    """Mock reference and hypothesis pairs for WER testing."""
    return {
        "references": ["das ist ein test", "hallo welt", "swiss german speech recognition"],
        "hypotheses": ["das ist ein test", "hallo velo", "swiss german speech"]
    }

@pytest.fixture
def mock_metadata_dict() -> List[Dict]:
    """Mock metadata as list of dicts."""
    return [
        {
            "client_id": "client_001",
            "path": "sample_be_1.wav",
            "sentence": "Herzlich willkommen in Bern",
            "accent": "BE",
            "audio_path": "/app/tests/fixtures/audio/sample_be_1.wav"
        },
        {
            "client_id": "client_002",
            "path": "sample_zh_2.wav",
            "sentence": "Hallo zusammen",
            "accent": "ZH",
            "audio_path": "/app/tests/fixtures/audio/sample_zh_2.wav"
        }
    ]
```

---

## Phase 3: Write Complete Tests (DETAILED INSTRUCTIONS)

### UNIT TESTS - COMPLETE IMPLEMENTATIONS REQUIRED

#### `tests/unit/data/test_loader.py`
```python
"""Unit tests for data loader module."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.loader import load_swiss_german_metadata, load_audio
from unittest.mock import patch, MagicMock

class TestLoadSwissGermanMetadata:
    """Test suite for load_swiss_german_metadata function."""
    
    @pytest.mark.unit
    def test_loads_valid_tsv(self, mock_swiss_german_data):
        """Test loading valid TSV file."""
        df = load_swiss_german_metadata(str(mock_swiss_german_data))
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'sentence' in df.columns
        assert 'accent' in df.columns
    
    @pytest.mark.unit
    def test_raises_on_nonexistent_file(self):
        """Test raises error for non-existent file."""
        with pytest.raises((FileNotFoundError, pd.errors.EmptyDataError)):
            load_swiss_german_metadata("nonexistent.tsv")
    
    @pytest.mark.unit
    def test_handles_empty_tsv(self, temp_dir):
        """Test handling of empty TSV file."""
        empty_tsv = temp_dir / "empty.tsv"
        empty_tsv.write_text("client_id\tpath\tsentence\n")
        
        df = load_swiss_german_metadata(str(empty_tsv))
        assert len(df) == 0
        assert list(df.columns) == ['client_id', 'path', 'sentence']

class TestLoadAudio:
    """Test suite for load_audio function."""
    
    @pytest.mark.unit
    def test_loads_valid_audio_file(self, sample_audio_path):
        """Test loading valid audio file."""
        audio = load_audio(str(sample_audio_path), sample_rate=16000)
        
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32 or audio.dtype == np.float64
        assert len(audio) > 0
    
    @pytest.mark.unit
    @patch('librosa.load')
    def test_resamples_to_target_rate(self, mock_load):
        """Test audio is resampled to target sample rate."""
        mock_load.return_value = (np.random.randn(16000).astype(np.float32), 16000)
        
        audio = load_audio("dummy.wav", sample_rate=16000)
        
        mock_load.assert_called_once_with("dummy.wav", sr=16000)
        assert isinstance(audio, np.ndarray)
    
    @pytest.mark.unit
    def test_raises_on_nonexistent_file(self):
        """Test raises RuntimeError for non-existent audio file."""
        with pytest.raises(RuntimeError, match="Error loading audio file"):
            load_audio("nonexistent.wav")
    
    @pytest.mark.unit
    @patch('librosa.load', side_effect=Exception("Corrupt file"))
    def test_handles_corrupt_audio(self, mock_load):
        """Test handling of corrupt audio files."""
        with pytest.raises(RuntimeError, match="Error loading audio file"):
            load_audio("corrupt.wav")
```

#### `tests/unit/data/test_preprocessor.py`
```python
"""Unit tests for audio preprocessor module."""
import pytest
import numpy as np
from src.data.preprocessor import AudioPreprocessor

class TestAudioPreprocessor:
    """Test suite for AudioPreprocessor class."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance."""
        return AudioPreprocessor(target_sample_rate=16000)
    
    @pytest.mark.unit
    def test_initialization(self, preprocessor):
        """Test preprocessor initialization."""
        assert preprocessor.target_sample_rate == 16000
    
    @pytest.mark.unit
    def test_normalize_audio_zero_mean(self, preprocessor):
        """Test normalization produces zero mean audio."""
        audio = np.random.randn(16000).astype(np.float32)
        normalized = preprocessor.normalize_audio(audio)
        
        assert np.abs(normalized.mean()) < 1e-6
        assert np.abs(normalized.std() - 1.0) < 1e-6
    
    @pytest.mark.unit
    def test_normalize_audio_handles_constant_signal(self, preprocessor):
        """Test normalization handles constant signal (std=0)."""
        audio = np.ones(16000, dtype=np.float32)
        normalized = preprocessor.normalize_audio(audio)
        
        # Should return unchanged when std=0
        assert np.allclose(normalized, audio)
    
    @pytest.mark.unit
    def test_normalize_audio_preserves_shape(self, preprocessor):
        """Test normalization preserves audio shape."""
        audio = np.random.randn(24000).astype(np.float32)
        normalized = preprocessor.normalize_audio(audio)
        
        assert normalized.shape == audio.shape
    
    @pytest.mark.unit
    @pytest.mark.parametrize("orig_sr,target_sr", [
        (8000, 16000),
        (22050, 16000),
        (44100, 16000),
        (48000, 16000)
    ])
    def test_resample_audio_changes_length(self, preprocessor, orig_sr, target_sr):
        """Test resampling changes audio length appropriately."""
        duration = 1.0  # 1 second
        audio = np.random.randn(int(orig_sr * duration)).astype(np.float32)
        
        preprocessor.target_sample_rate = target_sr
        resampled = preprocessor.resample_audio(audio, orig_sr)
        
        expected_length = int(target_sr * duration)
        # Allow small tolerance due to resampling
        assert abs(len(resampled) - expected_length) < 100
    
    @pytest.mark.unit
    def test_resample_audio_no_op_when_same_rate(self, preprocessor):
        """Test resampling is no-op when rates match."""
        audio = np.random.randn(16000).astype(np.float32)
        resampled = preprocessor.resample_audio(audio, 16000)
        
        assert np.array_equal(audio, resampled)
    
    @pytest.mark.unit
    def test_preprocess_pipeline(self, preprocessor, sample_audio_array):
        """Test full preprocessing pipeline."""
        processed, sr = preprocessor.preprocess(sample_audio_array, 16000)
        
        assert sr == 16000
        assert isinstance(processed, np.ndarray)
        # Should be normalized
        assert np.abs(processed.mean()) < 0.1
        assert 0.8 < processed.std() < 1.2
    
    @pytest.mark.unit
    def test_preprocess_with_resampling(self, preprocessor):
        """Test preprocessing with resampling."""
        # Create 8kHz audio
        audio_8k = np.random.randn(8000).astype(np.float32)
        
        processed, sr = preprocessor.preprocess(audio_8k, orig_sr=8000)
        
        assert sr == 16000
        assert len(processed) > len(audio_8k)  # Upsampled
```

#### `tests/unit/utils/test_audio_utils.py`
```python
"""Unit tests for audio utility functions."""
import pytest
from pathlib import Path
from src.utils.audio_utils import validate_audio_file, get_audio_duration
from unittest.mock import patch

class TestValidateAudioFile:
    """Test suite for validate_audio_file function."""
    
    @pytest.mark.unit
    def test_valid_wav_file(self, sample_audio_path):
        """Test validation passes for valid .wav file."""
        assert validate_audio_file(str(sample_audio_path)) is True
    
    @pytest.mark.unit
    def test_nonexistent_file(self):
        """Test validation fails for non-existent file."""
        assert validate_audio_file("nonexistent.wav") is False
    
    @pytest.mark.unit
    @pytest.mark.parametrize("extension", [".flac", ".wav", ".mp3"])
    def test_valid_extensions(self, temp_dir, extension):
        """Test all valid audio extensions are accepted."""
        audio_file = temp_dir / f"test{extension}"
        audio_file.touch()
        
        assert validate_audio_file(str(audio_file)) is True
    
    @pytest.mark.unit
    @pytest.mark.parametrize("extension", [".txt", ".mp4", ".pdf", ""])
    def test_invalid_extensions(self, temp_dir, extension):
        """Test invalid extensions are rejected."""
        file = temp_dir / f"test{extension}"
        file.touch()
        
        assert validate_audio_file(str(file)) is False
    
    @pytest.mark.unit
    def test_handles_exception_gracefully(self):
        """Test function handles exceptions and returns False."""
        # Invalid path that causes exception
        assert validate_audio_file("\0invalid") is False

class TestGetAudioDuration:
    """Test suite for get_audio_duration function."""
    
    @pytest.mark.unit
    def test_returns_duration_for_valid_file(self, sample_audio_path):
        """Test returns duration for valid audio file."""
        duration = get_audio_duration(str(sample_audio_path))
        
        assert duration is not None
        assert isinstance(duration, float)
        assert duration > 0
    
    @pytest.mark.unit
    def test_returns_none_for_nonexistent_file(self):
        """Test returns None for non-existent file."""
        duration = get_audio_duration("nonexistent.wav")
        assert duration is None
    
    @pytest.mark.unit
    @patch('librosa.get_duration', side_effect=Exception("Error"))
    def test_handles_librosa_error(self, mock_duration):
        """Test handles librosa errors gracefully."""
        duration = get_audio_duration("dummy.wav")
        assert duration is None
```

#### `tests/unit/backend/test_pydantic_models.py`
```python
"""Unit tests for Pydantic models."""
import pytest
from pydantic import ValidationError
from src.backend.models import EvaluateRequest, EvaluateResponse

class TestEvaluateRequest:
    """Test suite for EvaluateRequest model."""
    
    @pytest.mark.unit
    def test_valid_request(self):
        """Test valid request creation."""
        request = EvaluateRequest(
            model_type="whisper",
            model="medium",
            limit=10
        )
        
        assert request.model_type == "whisper"
        assert request.model == "medium"
        assert request.limit == 10
    
    @pytest.mark.unit
    def test_limit_optional(self):
        """Test limit is optional."""
        request = EvaluateRequest(
            model_type="wav2vec2",
            model="facebook/wav2vec2-large-xlsr-53-german"
        )
        
        assert request.limit is None
    
    @pytest.mark.unit
    def test_validates_model_type(self):
        """Test model_type validation."""
        # Should work with valid model types
        for model_type in ["whisper", "wav2vec2"]:
            request = EvaluateRequest(model_type=model_type, model="test")
            assert request.model_type == model_type

class TestEvaluateResponse:
    """Test suite for EvaluateResponse model."""
    
    @pytest.mark.unit
    def test_valid_response(self):
        """Test valid response creation."""
        response = EvaluateResponse(
            model="whisper-medium",
            total_samples=100,
            failed_samples=2,
            overall_wer=35.5,
            overall_cer=15.2,
            overall_bleu=65.3,
            per_dialect_wer={"BE": 30.0, "ZH": 35.0},
            per_dialect_cer={"BE": 12.0, "ZH": 18.0},
            per_dialect_bleu={"BE": 70.0, "ZH": 60.0}
        )
        
        assert response.model == "whisper-medium"
        assert response.total_samples == 100
        assert response.overall_wer == 35.5
        assert len(response.per_dialect_wer) == 2
    
    @pytest.mark.unit
    def test_failed_samples_defaults_to_zero(self):
        """Test failed_samples defaults to 0."""
        response = EvaluateResponse(
            model="test",
            total_samples=10,
            overall_wer=0.0,
            overall_cer=0.0,
            overall_bleu=0.0,
            per_dialect_wer={},
            per_dialect_cer={},
            per_dialect_bleu={}
        )
        
        assert response.failed_samples == 0
```

#### `tests/unit/evaluation/test_evaluator.py`
```python
"""Unit tests for ASR evaluator."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.evaluation.evaluator import ASREvaluator

class TestASREvaluatorInitialization:
    """Test ASREvaluator initialization."""
    
    @pytest.mark.unit
    def test_initialization_whisper(self):
        """Test initialization with Whisper model."""
        evaluator = ASREvaluator(model_type="whisper", model_name="tiny")
        
        assert evaluator.model_type == "whisper"
        assert evaluator.model_name == "tiny"
        assert evaluator.model is None  # Not loaded yet
    
    @pytest.mark.unit
    def test_initialization_wav2vec2(self):
        """Test initialization with Wav2Vec2 model."""
        evaluator = ASREvaluator(
            model_type="wav2vec2",
            model_name="facebook/wav2vec2-base"
        )
        
        assert evaluator.model_type == "wav2vec2"
        assert evaluator.model_name == "facebook/wav2vec2-base"

class TestASREvaluatorTranscribe:
    """Test transcription functionality."""
    
    @pytest.mark.unit
    @patch('whisper.load_model')
    @patch('whisper.transcribe')
    def test_transcribe_with_whisper(self, mock_transcribe, mock_load, sample_audio_path):
        """Test transcription with Whisper model (mocked)."""
        # Mock model loading and transcription
        mock_model = Mock()
        mock_load.return_value = mock_model
        mock_transcribe.return_value = {"text": "Das ist ein Test"}
        
        evaluator = ASREvaluator(model_type="whisper", model_name="tiny")
        evaluator.model = mock_model
        
        result = evaluator.transcribe(str(sample_audio_path))
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    @pytest.mark.unit
    def test_transcribe_raises_without_loaded_model(self, sample_audio_path):
        """Test transcribe raises error if model not loaded."""
        evaluator = ASREvaluator(model_type="whisper", model_name="tiny")
        
        with pytest.raises((RuntimeError, AttributeError)):
            evaluator.transcribe(str(sample_audio_path))
```

### INTEGRATION TESTS - COMPLETE IMPLEMENTATIONS REQUIRED

#### `tests/integration/test_data_pipeline.py`
```python
"""Integration tests for data loading pipeline."""
import pytest
import pandas as pd
from src.data.loader import load_swiss_german_metadata, load_audio
from src.data.preprocessor import AudioPreprocessor

class TestDataPipeline:
    """Test complete data pipeline integration."""
    
    @pytest.mark.integration
    def test_load_metadata_and_audio(self, mock_swiss_german_data, fixtures_root):
        """Test loading metadata then loading referenced audio files."""
        # Load metadata
        df = load_swiss_german_metadata(str(mock_swiss_german_data))
        
        assert len(df) > 0
        
        # Load first audio file
        first_audio_path = fixtures_root / "audio" / df.iloc[0]['path']
        if first_audio_path.exists():
            audio = load_audio(str(first_audio_path))
            assert len(audio) > 0
    
    @pytest.mark.integration
    def test_load_and_preprocess_audio(self, sample_audio_path):
        """Test loading and preprocessing audio."""
        # Load audio
        audio = load_audio(str(sample_audio_path), sample_rate=16000)
        
        # Preprocess
        preprocessor = AudioPreprocessor(target_sample_rate=16000)
        processed, sr = preprocessor.preprocess(audio, 16000)
        
        assert sr == 16000
        assert len(processed) == len(audio)  # Same rate, no resampling
        assert abs(processed.mean()) < 0.1  # Normalized
```

#### `tests/integration/test_backend_endpoints.py`
```python
"""Integration tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock

@pytest.fixture
def client():
    """Create FastAPI test client."""
    from src.backend.endpoints import router
    from fastapi import FastAPI
    
    app = FastAPI()
    app.include_router(router, prefix="/api")
    
    return TestClient(app)

class TestEvaluateEndpoint:
    """Test /evaluate endpoint."""
    
    @pytest.mark.integration
    @patch('src.evaluation.evaluator.ASREvaluator')
    def test_evaluate_endpoint_success(self, mock_evaluator_class, client):
        """Test successful evaluation request."""
        # Mock evaluator
        mock_evaluator = Mock()
        mock_evaluator.evaluate_dataset.return_value = {
            "total_samples": 10,
            "failed_samples": 0,
            "overall_wer": 30.0,
            "overall_cer": 15.0,
            "overall_bleu": 65.0,
            "per_dialect_wer": {"BE": 28.0},
            "per_dialect_cer": {"BE": 14.0},
            "per_dialect_bleu": {"BE": 67.0}
        }
        mock_evaluator_class.return_value = mock_evaluator
        
        response = client.post(
            "/api/evaluate",
            json={
                "model_type": "whisper",
                "model": "tiny",
                "limit": 10
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "tiny"
        assert data["total_samples"] == 10
    
    @pytest.mark.integration
    def test_evaluate_endpoint_validation_error(self, client):
        """Test endpoint with invalid request."""
        response = client.post(
            "/api/evaluate",
            json={"invalid": "data"}
        )
        
        assert response.status_code == 422  # Validation error
```

### E2E TESTS - COMPLETE IMPLEMENTATIONS REQUIRED

#### `tests/e2e/test_evaluation_workflow.py`
```python
"""End-to-end tests for complete evaluation workflow."""
import pytest
from pathlib import Path

class TestCompleteEvaluationWorkflow:
    """Test complete evaluation from data loading to results."""
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_tiny_dataset_evaluation(self, mock_swiss_german_data, temp_results_dir):
        """Test evaluation on tiny mock dataset."""
        # This would load a real tiny model and evaluate
        # Skip if resources not available
        pytest.skip("E2E test requires model loading - implement when needed")
```

---

## Phase 4: Fixture Data Files

### `tests/fixtures/data/mock_swiss_german.tsv`
```tsv
client_id	path	sentence	up_votes	down_votes	age	gender	accent	audio_path
client_001	sample_be_1.wav	Herzlich willkommen in Bern	5	0	35	male	BE	/app/tests/fixtures/audio/sample_be_1.wav
client_002	sample_zh_2.wav	Hallo zusammen	3	0	28	female	ZH	/app/tests/fixtures/audio/sample_zh_2.wav
client_003	sample_vs_3.wav	Wie geht es dir	4	0	42	male	VS	/app/tests/fixtures/audio/sample_vs_3.wav
```

### `tests/fixtures/data/mock_results.json`
```json
{
  "model": "whisper-tiny",
  "overall_wer": 28.5,
  "overall_cer": 12.3,
  "overall_bleu": 65.7,
  "per_dialect_wer": {"BE": 25.2, "ZH": 30.1, "VS": 35.8},
  "per_dialect_cer": {"BE": 12.0, "ZH": 13.0, "VS": 18.0},
  "per_dialect_bleu": {"BE": 70.0, "ZH": 65.0, "VS": 60.0},
  "samples": [
    {
      "audio_path": "sample_be_1.wav",
      "reference": "Herzlich willkommen in Bern",
      "hypothesis": "Herzlich willkommen in Bern",
      "dialect": "BE",
      "wer": 0.0,
      "cer": 0.0
    }
  ]
}
```

---

## Phase 5: Docker Compose Services

Add to `docker-compose.yml`:

```yaml
  test-unit:
    build: .
    command: ["python", "-m", "pytest", "tests/unit/", "-v", "-m", "unit"]
    volumes:
      - ./src:/app/src
      - ./tests:/app/tests
    environment:
      - PYTHONPATH=/app

  test-integration:
    build: .
    command: ["python", "-m", "pytest", "tests/integration/", "-v", "-m", "integration"]
    volumes:
      - ./data:/app/data
      - ./src:/app/src
      - ./tests:/app/tests
      - ./models:/app/models
      - huggingface-cache:/home/appuser/.cache/huggingface
    environment:
      - PYTHONPATH=/app
      - HF_TOKEN=${HF_TOKEN}

  test-e2e:
    build: .
    command: ["python", "-m", "pytest", "tests/e2e/", "-v", "-m", "e2e"]
    volumes:
      - ./data:/app/data
      - ./src:/app/src
      - ./tests:/app/tests
      - ./models:/app/models
      - ./results:/app/results
      - huggingface-cache:/home/appuser/.cache/huggingface
    environment:
      - PYTHONPATH=/app
      - HF_TOKEN=${HF_TOKEN}

  test:
    build: .
    command: ["python", "-m", "pytest", "tests/", "-v"]
    volumes:
      - ./data:/app/data            
      - ./src:/app/src
      - ./tests:/app/tests
      - ./models:/app/models
      - ./results:/app/results
      - huggingface-cache:/home/appuser/.cache/huggingface
    environment:
      - PYTHONPATH=/app
      - HF_TOKEN=${HF_TOKEN}

  test-coverage:
    build: .
    command: ["python", "-m", "pytest", "tests/", "--cov=src", "--cov-report=html", "--cov-report=term"]
    volumes:
      - ./data:/app/data
      - ./src:/app/src
      - ./tests:/app/tests
      - ./htmlcov:/app/htmlcov
      - huggingface-cache:/home/appuser/.cache/huggingface
    environment:
      - PYTHONPATH=/app
```

---

## Phase 6: Documentation

Create `tests/README.md` with comprehensive documentation (see original prompt for content).

---

## Validation Checklist

- [ ] All directory structure created
- [ ] Existing tests moved and imports updated
- [ ] **ALL unit test files have COMPLETE implementations (not placeholders)**
- [ ] **ALL integration test files have COMPLETE implementations**
- [ ] Mock fixtures properly configured
- [ ] pytest.ini with markers configured
- [ ] conftest.py files at all levels
- [ ] Docker compose services added
- [ ] Tests README documentation created
- [ ] `docker compose run --rm test-unit` passes
- [ ] `docker compose run --rm test` passes

---

## CRITICAL REMINDERS

1. **NO PLACEHOLDER TESTS** - Every test file must have real, working test implementations
2. **Read the source code** - Inspect actual implementations before writing tests
3. **Use proper mocking** - Mock external dependencies (transformers, librosa when needed)
4. **Test edge cases** - Include error handling, boundary conditions, empty inputs
5. **Follow existing patterns** - Look at test_evaluation.py for quality examples
6. **Use pytest markers** - Mark every test with appropriate decorator (@pytest.mark.unit, etc.)

## Expected Test Counts (Approximate)

- Unit tests: ~150-200 tests
- Integration tests: ~20-30 tests
- E2E tests: ~5-10 tests
- Total: ~175-240 tests

This is production-ready, comprehensive test coverage suitable for a Bachelor thesis project.
