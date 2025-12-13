"""Root test fixtures - Docker-aware path resolution and common fixtures."""
import pytest
from pathlib import Path
from typing import Generator
import tempfile
import numpy as np
import sys
from types import ModuleType
from types import SimpleNamespace
import math

# ---------------------------------------------------------------------------
# Lightweight mocks for optional heavy deps (ensure availability during test
# collection, especially in container without installed ML packages).
# ---------------------------------------------------------------------------
if 'transformers' not in sys.modules:
    transformers = ModuleType('transformers')
    transformers.__path__ = []  # mark as package

    trainer_utils = ModuleType('transformers.trainer_utils')
    setattr(trainer_utils, 'PREFIX_CHECKPOINT_DIR', 'checkpoint')

    integrations = ModuleType('transformers.integrations')
    class TensorBoardCallback:  # type: ignore
        pass
    class WandbCallback:  # type: ignore
        pass
    setattr(integrations, 'TensorBoardCallback', TensorBoardCallback)
    setattr(integrations, 'WandbCallback', WandbCallback)

    class DummyTokenizer:
        def __init__(self):
            self._vocab = {'<s>':0,'</s>':1,'a':2,'b':3,'c':4}
        def get_vocab(self):
            return self._vocab
        def set_target_lang(self, lang):
            self.lang = lang

    class BaseProcessor:
        def __init__(self):
            self.tokenizer = DummyTokenizer()
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()
        def batch_decode(self, ids, skip_special_tokens=True):
            return ["decoded"]

    class Wav2Vec2Processor(BaseProcessor):  # type: ignore
        def __call__(self, waveform, return_tensors=None, sampling_rate=None):
            arr = np.array(waveform)
            tensor = sys.modules['torch'].tensor(arr)
            return {
                'input_values': tensor,
                'attention_mask': sys.modules['torch'].ones(tensor.shape[0] if tensor.shape else 1)
            }

    class WhisperProcessor(BaseProcessor):  # type: ignore
        def __call__(self, audio, sampling_rate=16000, return_tensors=None):
            feats = sys.modules['torch'].tensor(np.zeros((1,80,3000), dtype=np.float32))
            return SimpleNamespace(input_features=feats)

    class WhisperForConditionalGeneration:  # type: ignore
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()
        def to(self, device):
            return self
        def eval(self):
            return self
        def generate(self, input_features, **kwargs):
            return [[1,2,3]]

    class Wav2Vec2ForCTC:  # type: ignore
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()
        def to(self, device):
            return self
        def eval(self):
            return self
        def __call__(self, *args, **kwargs):
            return SimpleNamespace(logits=sys.modules['torch'].randn(1,10,5))

    class AutoProcessor(Wav2Vec2Processor):  # type: ignore
        pass

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

    setattr(transformers, 'Trainer', Trainer)
    setattr(transformers, 'TrainingArguments', TrainingArguments)
    setattr(transformers, 'EarlyStoppingCallback', EarlyStoppingCallback)
    setattr(transformers, 'TrainerCallback', TrainerCallback)
    setattr(transformers, 'Wav2Vec2Processor', Wav2Vec2Processor)
    setattr(transformers, 'WhisperProcessor', WhisperProcessor)
    setattr(transformers, 'WhisperForConditionalGeneration', WhisperForConditionalGeneration)
    setattr(transformers, 'Wav2Vec2ForCTC', Wav2Vec2ForCTC)
    setattr(transformers, 'AutoProcessor', AutoProcessor)

    sys.modules['transformers'] = transformers
    sys.modules['transformers.trainer_utils'] = trainer_utils
    sys.modules['transformers.integrations'] = integrations

if 'torch' not in sys.modules:
    torch = ModuleType('torch')
    backends = ModuleType('torch.backends')
    mps = ModuleType('torch.backends.mps')
    cuda = ModuleType('torch.cuda')

    class FakeTensor:
        def __init__(self, data):
            self.data = np.array(data)
        @property
        def shape(self):
            return self.data.shape
        def to(self, *args, **kwargs):
            return self
        def cpu(self):
            return self
        def numpy(self):
            return self.data
        def squeeze(self, *args, **kwargs):
            return FakeTensor(np.squeeze(self.data, *args, **kwargs))
        def mean(self, *args, **kwargs):
            axis = kwargs.get('dim', kwargs.get('axis', None))
            keep = kwargs.get('keepdim', kwargs.get('keepdims', False))
            return FakeTensor(np.mean(self.data, axis=axis, keepdims=keep))
        def argmax(self, dim=None):
            return FakeTensor(self.data.argmax(axis=dim))
        def numel(self):
            return self.data.size
        def __array__(self):
            return self.data
        def __len__(self):
            return len(self.data)
        def __iter__(self):
            return iter(self.data)

    def _make_tensor(data):
        return FakeTensor(data)

    def _normalize_shape(shape):
        # Allow torch.zeros((1, 0)) style calls
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            return tuple(shape[0])
        return shape

    def randn(*shape):
        shape = _normalize_shape(shape)
        return FakeTensor(np.random.randn(*shape))

    def ones(*shape):
        shape = _normalize_shape(shape)
        return FakeTensor(np.ones(shape))

    def zeros(*shape):
        shape = _normalize_shape(shape)
        return FakeTensor(np.zeros(shape))

    def mean(x, dim=None, keepdim=False):
        arr = x.data if isinstance(x, FakeTensor) else np.array(x)
        return FakeTensor(np.mean(arr, axis=dim, keepdims=keepdim))

    class _NoGrad:
        def __enter__(self):
            return None
        def __exit__(self, *exc):
            return False

    class _Device:
        def __init__(self, name):
            self.type = name
        def __str__(self):
            return self.type

    def device(name):
        return _Device(name)

    def is_available_false():
        return False

    setattr(cuda, 'is_available', is_available_false)
    setattr(cuda, 'empty_cache', lambda: None)
    setattr(mps, 'is_available', is_available_false)

    setattr(torch, 'cuda', cuda)
    setattr(torch, 'backends', backends)
    setattr(backends, 'mps', mps)
    setattr(torch, 'tensor', _make_tensor)
    setattr(torch, 'randn', randn)
    setattr(torch, 'ones', ones)
    setattr(torch, 'zeros', zeros)
    setattr(torch, 'mean', mean)
    setattr(torch, 'no_grad', _NoGrad)
    setattr(torch, 'device', device)

    sys.modules['torch'] = torch
    sys.modules['torch.backends'] = backends
    sys.modules['torch.backends.mps'] = mps
    sys.modules['torch.cuda'] = cuda

# torchaudio stub
if 'torchaudio' not in sys.modules:
    torchaudio = ModuleType('torchaudio')
    ta_transforms = ModuleType('torchaudio.transforms')

    def ta_load(path):
        p = Path(path)
        if not p.exists():
            raise RuntimeError(f"File not found: {path}")
        # return 1 second of 16k mono silence
        return sys.modules['torch'].tensor(np.zeros((1, 16000), dtype=np.float32)), 16000

    class Resample:
        def __init__(self, orig_freq, new_freq):
            self.orig_freq = orig_freq
            self.new_freq = new_freq
        def __call__(self, waveform):
            # keep waveform unchanged for tests
            return waveform

    setattr(ta_transforms, 'Resample', Resample)
    setattr(torchaudio, 'load', ta_load)
    setattr(torchaudio, 'transforms', ta_transforms)
    sys.modules['torchaudio'] = torchaudio
    sys.modules['torchaudio.transforms'] = ta_transforms

# librosa stub
if 'librosa' not in sys.modules:
    librosa = ModuleType('librosa')
    def lb_load(path, sr=22050):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return np.random.randn(sr).astype(np.float32), sr
    def lb_resample(y, orig_sr, target_sr):
        # simple linear resample
        if orig_sr == target_sr:
            return y
        ratio = target_sr / float(orig_sr)
        new_len = max(1, int(len(y) * ratio))
        return np.interp(
            np.linspace(0, len(y) - 1, new_len),
            np.arange(len(y)),
            y,
        ).astype(np.float32)
    def lb_get_duration(path=None, y=None, sr=22050):
        if path is None or path == "":
            return None
        if path:
            p = Path(path)
            if not p.exists() or not p.is_file():
                return None
            try:
                import wave
                with wave.open(str(p), 'rb') as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate() or sr
                    return frames / float(rate) if rate else None
            except Exception:
                return None
        if y is not None:
            return len(y) / float(sr)
        # default to 1s
        return 1.0
    setattr(librosa, 'load', lb_load)
    setattr(librosa, 'resample', lb_resample)
    setattr(librosa, 'get_duration', lb_get_duration)
    sys.modules['librosa'] = librosa

# plotly stubs
if 'plotly' not in sys.modules:
    plotly = ModuleType('plotly')
    go = ModuleType('plotly.graph_objects')
    px = ModuleType('plotly.express')

    class DummyLayout:
        def __init__(self):
            self.title = SimpleNamespace(text=None)
            self.height = None
            self.annotations = []
            self.showlegend = True
            self.yaxis = SimpleNamespace(title=SimpleNamespace(text=None))

    class DummyBar:
        def __init__(self, x=None, y=None, name=None, marker=None, marker_color=None, text=None, hovertemplate=None, texttemplate=None, customdata=None, **kwargs):
            self.x = tuple(x) if x is not None else []
            self.y = y or []
            self.name = name
            color = marker_color
            if color is None and marker is not None:
                if isinstance(marker, dict):
                    color = marker.get('color')
                else:
                    color = getattr(marker, 'color', None)
            self.marker = SimpleNamespace(color=color)
            self.text = text
            self.hovertemplate = hovertemplate
            self.texttemplate = texttemplate
            self.customdata = customdata
            self.type = 'bar'
            self.kwargs = kwargs

    class DummyFigure:
        def __init__(self, data=None):
            self.data = data or []
            self.layout = DummyLayout()
        def add_trace(self, trace):
            self.data.append(trace)
        def add_annotation(self, **kwargs):
            self.layout.annotations.append(SimpleNamespace(**kwargs))
        def update_layout(self, **kwargs):
            for k, v in kwargs.items():
                if k == 'title':
                    if isinstance(v, dict) and 'text' in v:
                        self.layout.title.text = v['text']
                    else:
                        self.layout.title.text = v
                elif k == 'height':
                    self.layout.height = v
                elif k == 'showlegend':
                    self.layout.showlegend = v
                elif k == 'annotations':
                    converted = []
                    for ann in v:
                        if isinstance(ann, dict):
                            converted.append(SimpleNamespace(**ann))
                        else:
                            converted.append(ann)
                    self.layout.annotations = converted
                elif k == 'yaxis_title' or k == 'yaxis_title_text':
                    self.layout.yaxis.title.text = v
                elif k == 'yaxis':
                    if isinstance(v, dict) and 'title' in v:
                        title = v['title']
                        if isinstance(title, dict) and 'text' in title:
                            self.layout.yaxis.title.text = title['text']
                        else:
                            self.layout.yaxis.title.text = title

    def px_bar(*args, **kwargs):
        data = kwargs.get('data_frame') or []
        fig = DummyFigure([])
        if data:
            fig.data.append(DummyBar())
        return fig

    setattr(go, 'Figure', DummyFigure)
    setattr(go, 'Bar', DummyBar)
    setattr(px, 'bar', px_bar)
    sys.modules['plotly'] = plotly
    sys.modules['plotly.graph_objects'] = go
    sys.modules['plotly.express'] = px

# streamlit stub
if 'streamlit' not in sys.modules:
    streamlit = ModuleType('streamlit')
    def cache_data(func=None, **kwargs):
        if func:
            return func
        def wrapper(f):
            return f
        return wrapper
    def warning(*args, **kwargs):
        warning.called = True
    warning.called = False
    def error(*args, **kwargs):
        error.called = True
    error.called = False
    setattr(streamlit, 'cache_data', cache_data)
    setattr(streamlit, 'warning', warning)
    setattr(streamlit, 'error', error)
    sys.modules['streamlit'] = streamlit

# whisper stub
if 'whisper' not in sys.modules:
    whisper = ModuleType('whisper')
    def load_audio(path):
        p = Path(path)
        if not p.exists():
            # return 1s zeros
            return np.zeros(16000, dtype=np.float32)
        return np.zeros(16000, dtype=np.float32)
    class DummyWhisperModel:
        def __init__(self, name, device=None):
            self.name = name
            self.device = device
        def transcribe(self, audio, **kwargs):
            return {'text': 'dummy transcription'}
    def load_model(name, device=None):
        return DummyWhisperModel(name, device)
    setattr(whisper, 'load_model', load_model)
    setattr(whisper, 'load_audio', load_audio)
    sys.modules['whisper'] = whisper

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
