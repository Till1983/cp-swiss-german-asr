"""Unit test fixtures - fast, no external dependencies."""
import pytest
from typing import Dict, List
import numpy as np


@pytest.fixture
def sample_swiss_german_texts() -> List[Dict[str, str]]:
    """Sample Swiss German sentences with Standard German translations."""
    return [
        {"swiss_german": "Grueezi mitenand", "standard_german": "Hallo zusammen", "dialect": "ZH"},
        {"swiss_german": "Haerzlich willkomme z Baern", "standard_german": "Herzlich willkommen in Bern", "dialect": "BE"},
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


@pytest.fixture
def mock_evaluation_results() -> List[Dict]:
    """Mock evaluation results for testing."""
    return [
        {
            "audio_file": "sample_be_1.wav",
            "dialect": "BE",
            "reference": "Herzlich willkommen in Bern",
            "hypothesis": "Herzlich willkommen in Bern",
            "wer": 0.0,
            "cer": 0.0,
            "bleu": 100.0
        },
        {
            "audio_file": "sample_zh_2.wav",
            "dialect": "ZH",
            "reference": "Hallo zusammen",
            "hypothesis": "Hallo zusame",
            "wer": 50.0,
            "cer": 14.3,
            "bleu": 50.0
        }
    ]


@pytest.fixture
def sample_audio_8k() -> np.ndarray:
    """Generate 8kHz sample audio for resampling tests."""
    sample_rate = 8000
    duration = 1.0
    frequency = 440.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    return audio


@pytest.fixture
def sample_audio_44k() -> np.ndarray:
    """Generate 44.1kHz sample audio for resampling tests."""
    sample_rate = 44100
    duration = 1.0
    frequency = 440.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    return audio


@pytest.fixture
def constant_audio() -> np.ndarray:
    """Generate constant audio (std=0) for normalization edge case testing."""
    return np.ones(16000, dtype=np.float32)


@pytest.fixture
def silent_audio() -> np.ndarray:
    """Generate silent audio (all zeros)."""
    return np.zeros(16000, dtype=np.float32)


@pytest.fixture
def mock_checkpoint_metrics() -> Dict[str, float]:
    """Mock checkpoint metrics for testing."""
    return {
        "wer": 25.5,
        "cer": 12.3,
        "bleu": 65.0
    }
