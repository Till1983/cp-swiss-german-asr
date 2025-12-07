"""Integration test fixtures - requires data volumes, may load models."""
import pytest
from pathlib import Path


@pytest.fixture
def test_metadata_path(data_root: Path) -> Path:
    """Path to test.tsv metadata file."""
    test_path = data_root / "metadata" / "test.tsv"
    if not test_path.exists():
        pytest.skip("Test metadata not found - run with data volume")
    return test_path


@pytest.fixture
def audio_clips_dir(data_root: Path) -> Path:
    """Path to audio clips directory."""
    clips_path = data_root / "raw" / "fhnw-swiss-german-corpus" / "clips"
    if not clips_path.exists():
        pytest.skip("Audio clips directory not found - run with data volume")
    return clips_path
