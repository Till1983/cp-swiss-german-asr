"""Unit test fixtures - fast, no external dependencies."""
import pytest
from typing import Dict, List
import numpy as np
import sys
from types import ModuleType

# Light-weight mocks for optional heavy deps used in training module
if 'transformers' not in sys.modules:
    transformers = ModuleType('transformers')
    transformers.__path__ = []  # mark as package
    trainer_utils = ModuleType('transformers.trainer_utils')
    integrations = ModuleType('transformers.integrations')
    # Provide expected constant used by trainer
    setattr(trainer_utils, 'PREFIX_CHECKPOINT_DIR', 'checkpoint')
    # Minimal classes used in training tests
    class Trainer:  # type: ignore
        def __init__(self, *args, **kwargs):
            self.args = kwargs
        def log_metrics(self, *args, **kwargs):
            pass
        def evaluate(self, *args, **kwargs):
            return {}
        def train(self, *args, **kwargs):
            return {}
        def save_model(self, *args, **kwargs):
            pass
    class TrainingArguments:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
    class EarlyStoppingCallback:  # type: ignore
        pass
    class TrainerCallback:  # type: ignore
        pass
    class TensorBoardCallback:  # type: ignore
        pass
    class WandbCallback:  # type: ignore
        pass
    class Wav2Vec2Processor:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()
    class WhisperProcessor:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()
    class WhisperForConditionalGeneration:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
    class Wav2Vec2ForCTC:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
    class AutoProcessor:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
    setattr(transformers, 'Trainer', Trainer)
    setattr(transformers, 'TrainingArguments', TrainingArguments)
    setattr(transformers, 'EarlyStoppingCallback', EarlyStoppingCallback)
    setattr(transformers, 'TrainerCallback', TrainerCallback)
    setattr(transformers, 'Wav2Vec2Processor', Wav2Vec2Processor)
    setattr(transformers, 'WhisperProcessor', WhisperProcessor)
    setattr(transformers, 'WhisperForConditionalGeneration', WhisperForConditionalGeneration)
    setattr(transformers, 'Wav2Vec2ForCTC', Wav2Vec2ForCTC)
    setattr(transformers, 'AutoProcessor', AutoProcessor)
    setattr(integrations, 'TensorBoardCallback', TensorBoardCallback)
    setattr(integrations, 'WandbCallback', WandbCallback)
    sys.modules['transformers'] = transformers
    sys.modules['transformers.trainer_utils'] = trainer_utils
    sys.modules['transformers.integrations'] = integrations

# Mock torch to avoid heavy import in unit tests
if 'torch' not in sys.modules:
    torch = ModuleType('torch')
    backends = ModuleType('torch.backends')
    mps = ModuleType('torch.backends.mps')
    cuda = ModuleType('torch.cuda')
    # Minimal API used in code
    setattr(torch, 'cuda', cuda)
    setattr(torch, 'backends', backends)
    setattr(backends, 'mps', mps)
    sys.modules['torch'] = torch
    sys.modules['torch.backends'] = backends
    sys.modules['torch.backends.mps'] = mps
    sys.modules['torch.cuda'] = cuda


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


@pytest.fixture
def mock_error_analysis_data() -> Dict:
    """Mock error analysis data structure."""
    return {
        "aggregate_stats": {
            "mean_wer": 28.5,
            "median_wer": 27.0,
            "std_wer": 12.3,
            "mean_cer": 14.2,
            "mean_bleu": 65.8
        },
        "error_distribution_percent": {
            "substitution": 60.0,
            "deletion": 25.0,
            "insertion": 10.0,
            "correct": 5.0
        },
        "dialect_analysis": {
            "BE": {
                "sample_count": 50,
                "mean_wer": 25.0,
                "std_wer": 10.5,
                "mean_cer": 12.0,
                "mean_bleu": 70.0,
                "std_bleu": 8.5,
                "error_distribution": {
                    "substitution": 30,
                    "deletion": 12,
                    "insertion": 5,
                    "correct": 3,
                    "sub_rate": 0.6,
                    "del_rate": 0.24,
                    "ins_rate": 0.1
                },
                "top_confusions": [
                    [["ist", "isch"], 15],
                    [["das", "dasch"], 10]
                ]
            },
            "ZH": {
                "sample_count": 45,
                "mean_wer": 30.0,
                "std_wer": 11.0,
                "mean_cer": 15.5,
                "mean_bleu": 62.0,
                "std_bleu": 9.2,
                "error_distribution": {
                    "substitution": 28,
                    "deletion": 10,
                    "insertion": 7,
                    "correct": 0,
                    "sub_rate": 0.62,
                    "del_rate": 0.22,
                    "ins_rate": 0.16
                },
                "top_confusions": [
                    [["haben", "hend"], 12]
                ]
            }
        }
    }


@pytest.fixture
def mock_model_results_csv(temp_dir) -> str:
    """Create a mock model results CSV file."""
    import pandas as pd

    csv_data = pd.DataFrame({
        "dialect": ["BE", "ZH", "VS", "OVERALL"],
        "wer": [25.0, 30.0, 28.0, 27.5],
        "cer": [12.0, 15.0, 14.0, 13.5],
        "bleu": [70.0, 65.0, 68.0, 67.5]
    })

    csv_path = temp_dir / "test_results.csv"
    csv_data.to_csv(csv_path, index=False)

    return str(csv_path)


@pytest.fixture
def sample_dialects() -> List[str]:
    """Common Swiss German dialect codes."""
    return ["BE", "ZH", "BS", "LU", "SG", "AG", "GR", "VS"]


@pytest.fixture
def mock_batch_results() -> Dict:
    """Mock batch evaluation results."""
    return {
        "overall_wer": 28.5,
        "overall_cer": 14.2,
        "overall_bleu": 67.8,
        "per_sample_wer": [0.0, 25.0, 50.0, 100.0],
        "per_sample_cer": [0.0, 12.5, 25.0, 50.0],
        "per_sample_bleu": [100.0, 75.0, 50.0, 0.0]
    }


@pytest.fixture
def mock_worst_samples_data() -> List[Dict]:
    """Mock worst performing samples."""
    return [
        {
            "dialect": "BE",
            "wer": 85.0,
            "cer": 65.0,
            "bleu": 20.0,
            "reference": "das ist ein sehr langer text mit vielen worten",
            "hypothesis": "das isch en lang text",
            "audio_file": "be_sample_1.wav"
        },
        {
            "dialect": "ZH",
            "wer": 90.0,
            "cer": 70.0,
            "bleu": 15.0,
            "reference": "grüezi mitenand wie geht es euch",
            "hypothesis": "gruezi",
            "audio_file": "zh_sample_1.wav"
        },
        {
            "dialect": "VS",
            "wer": 82.0,
            "cer": 62.0,
            "bleu": 25.0,
            "reference": "walliser dialekt ist sehr schwierig",
            "hypothesis": "walliser sehr schwer",
            "audio_file": "vs_sample_1.wav"
        }
    ]
