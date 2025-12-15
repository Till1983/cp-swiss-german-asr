"""Integration test fixtures - requires data volumes, may load models."""
import pytest
from pathlib import Path
from unittest import mock
import sys
from types import ModuleType

# NOTE: These tests run in the same process as unit tests during full-suite
# collection (e.g., coverage in Docker). Avoid overwriting any lightweight
# stubs that other conftests may have already installed.

def _ensure_mock_module(name: str) -> None:
    if name not in sys.modules:
        sys.modules[name] = mock.MagicMock()


def _ensure_module(name: str, *, is_package: bool = False) -> ModuleType:
    if name in sys.modules and isinstance(sys.modules[name], ModuleType):
        return sys.modules[name]
    module = ModuleType(name)
    if is_package:
        module.__path__ = []  # mark as package
    sys.modules[name] = module
    return module


# Mock heavy ML dependencies before any test imports
_ensure_mock_module('whisper')
_ensure_mock_module('librosa')
_ensure_mock_module('torchaudio')
_ensure_mock_module('soundfile')
_ensure_mock_module('pydub')
_ensure_mock_module('streamlit')
_ensure_mock_module('plotly')
_ensure_mock_module('plotly.graph_objects')
_ensure_mock_module('plotly.express')

# For torch/transformers, prefer minimal ModuleType stubs over MagicMock to
# avoid breaking class bases and typing at import-time.
if 'torch' not in sys.modules or isinstance(sys.modules.get('torch'), mock.MagicMock):
    torch = _ensure_module('torch')
    backends = _ensure_module('torch.backends')
    mps = _ensure_module('torch.backends.mps')
    cuda = _ensure_module('torch.cuda')
    setattr(torch, 'cuda', cuda)
    setattr(torch, 'backends', backends)
    setattr(backends, 'mps', mps)

if 'transformers' not in sys.modules or isinstance(sys.modules.get('transformers'), mock.MagicMock):
    transformers = _ensure_module('transformers', is_package=True)
    trainer_utils = _ensure_module('transformers.trainer_utils')
    integrations = _ensure_module('transformers.integrations')
    setattr(trainer_utils, 'PREFIX_CHECKPOINT_DIR', 'checkpoint')

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
