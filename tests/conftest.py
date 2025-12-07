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


@pytest.fixture
def sample_stereo_audio_array() -> np.ndarray:
    """Generate stereo audio array for testing."""
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    mono = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    # Create stereo by stacking
    stereo = np.stack([mono, mono * 0.8], axis=0)
    return stereo
