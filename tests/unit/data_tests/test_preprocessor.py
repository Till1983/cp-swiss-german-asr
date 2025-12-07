"""Unit tests for audio preprocessor module."""
import pytest
import numpy as np
from src.data.preprocessor import AudioPreprocessor


class TestAudioPreprocessorInit:
    """Test AudioPreprocessor initialization."""

    @pytest.mark.unit
    def test_initialization_default_sample_rate(self):
        """Test preprocessor initialization with default sample rate."""
        preprocessor = AudioPreprocessor()

        assert preprocessor.target_sample_rate == 16000

    @pytest.mark.unit
    def test_initialization_custom_sample_rate(self):
        """Test preprocessor initialization with custom sample rate."""
        preprocessor = AudioPreprocessor(target_sample_rate=22050)

        assert preprocessor.target_sample_rate == 22050

    @pytest.mark.unit
    @pytest.mark.parametrize("sample_rate", [8000, 16000, 22050, 44100, 48000])
    def test_initialization_various_sample_rates(self, sample_rate):
        """Test preprocessor works with various sample rates."""
        preprocessor = AudioPreprocessor(target_sample_rate=sample_rate)

        assert preprocessor.target_sample_rate == sample_rate


class TestAudioPreprocessorNormalize:
    """Test AudioPreprocessor normalize_audio method."""

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance."""
        return AudioPreprocessor(target_sample_rate=16000)

    @pytest.mark.unit
    def test_normalize_audio_zero_mean(self, preprocessor, sample_audio_array):
        """Test normalization produces zero mean audio."""
        normalized = preprocessor.normalize_audio(sample_audio_array)

        assert np.abs(normalized.mean()) < 1e-6

    @pytest.mark.unit
    def test_normalize_audio_unit_variance(self, preprocessor, sample_audio_array):
        """Test normalization produces unit variance audio."""
        normalized = preprocessor.normalize_audio(sample_audio_array)

        assert np.abs(normalized.std() - 1.0) < 1e-6

    @pytest.mark.unit
    def test_normalize_audio_handles_constant_signal(self, preprocessor, constant_audio):
        """Test normalization handles constant signal (std=0)."""
        normalized = preprocessor.normalize_audio(constant_audio)

        # Should return unchanged when std=0
        assert np.allclose(normalized, constant_audio)

    @pytest.mark.unit
    def test_normalize_audio_handles_silent_audio(self, preprocessor, silent_audio):
        """Test normalization handles silent (all zeros) audio."""
        normalized = preprocessor.normalize_audio(silent_audio)

        # Silent audio has std=0, should return unchanged
        assert np.allclose(normalized, silent_audio)

    @pytest.mark.unit
    def test_normalize_audio_preserves_shape(self, preprocessor):
        """Test normalization preserves audio shape."""
        audio = np.random.randn(24000).astype(np.float32)
        normalized = preprocessor.normalize_audio(audio)

        assert normalized.shape == audio.shape

    @pytest.mark.unit
    def test_normalize_audio_preserves_dtype(self, preprocessor, sample_audio_array):
        """Test normalization preserves float dtype."""
        normalized = preprocessor.normalize_audio(sample_audio_array)

        assert np.issubdtype(normalized.dtype, np.floating)

    @pytest.mark.unit
    def test_normalize_audio_random_signal(self, preprocessor):
        """Test normalization on random signal."""
        audio = np.random.randn(16000).astype(np.float32) * 10 + 5
        normalized = preprocessor.normalize_audio(audio)

        assert np.abs(normalized.mean()) < 1e-5
        assert np.abs(normalized.std() - 1.0) < 1e-5


class TestAudioPreprocessorResample:
    """Test AudioPreprocessor resample_audio method."""

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance."""
        return AudioPreprocessor(target_sample_rate=16000)

    @pytest.mark.unit
    def test_resample_audio_no_op_when_same_rate(self, preprocessor, sample_audio_array):
        """Test resampling is no-op when rates match."""
        resampled = preprocessor.resample_audio(sample_audio_array, 16000)

        assert np.array_equal(sample_audio_array, resampled)

    @pytest.mark.unit
    @pytest.mark.parametrize("orig_sr,expected_factor", [
        (8000, 2.0),    # 8kHz -> 16kHz = 2x samples
        (22050, 0.726), # 22.05kHz -> 16kHz = ~0.73x samples
        (44100, 0.363), # 44.1kHz -> 16kHz = ~0.36x samples
        (48000, 0.333), # 48kHz -> 16kHz = ~0.33x samples
    ])
    def test_resample_audio_changes_length(self, preprocessor, orig_sr, expected_factor):
        """Test resampling changes audio length appropriately."""
        duration = 1.0
        audio = np.random.randn(int(orig_sr * duration)).astype(np.float32)

        resampled = preprocessor.resample_audio(audio, orig_sr)

        expected_length = int(16000 * duration)
        # Allow tolerance due to resampling
        assert abs(len(resampled) - expected_length) < 100

    @pytest.mark.unit
    def test_resample_upsample_8k_to_16k(self, preprocessor, sample_audio_8k):
        """Test upsampling from 8kHz to 16kHz."""
        resampled = preprocessor.resample_audio(sample_audio_8k, 8000)

        # Should approximately double in length
        assert len(resampled) > len(sample_audio_8k) * 1.8
        assert len(resampled) < len(sample_audio_8k) * 2.2

    @pytest.mark.unit
    def test_resample_downsample_44k_to_16k(self, preprocessor, sample_audio_44k):
        """Test downsampling from 44.1kHz to 16kHz."""
        resampled = preprocessor.resample_audio(sample_audio_44k, 44100)

        # Should reduce to about 16/44.1 = ~0.36 of original
        expected_ratio = 16000 / 44100
        actual_ratio = len(resampled) / len(sample_audio_44k)
        assert abs(actual_ratio - expected_ratio) < 0.05

    @pytest.mark.unit
    def test_resample_preserves_dtype(self, preprocessor, sample_audio_8k):
        """Test resampling preserves float dtype."""
        resampled = preprocessor.resample_audio(sample_audio_8k, 8000)

        assert np.issubdtype(resampled.dtype, np.floating)


class TestAudioPreprocessorPreprocess:
    """Test AudioPreprocessor preprocess method (full pipeline)."""

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance."""
        return AudioPreprocessor(target_sample_rate=16000)

    @pytest.mark.unit
    def test_preprocess_returns_tuple(self, preprocessor, sample_audio_array):
        """Test preprocess returns tuple of (audio, sample_rate)."""
        result = preprocessor.preprocess(sample_audio_array, 16000)

        assert isinstance(result, tuple)
        assert len(result) == 2

    @pytest.mark.unit
    def test_preprocess_returns_correct_sample_rate(self, preprocessor, sample_audio_array):
        """Test preprocess returns correct sample rate."""
        processed, sr = preprocessor.preprocess(sample_audio_array, 16000)

        assert sr == 16000

    @pytest.mark.unit
    def test_preprocess_normalizes_audio(self, preprocessor, sample_audio_array):
        """Test preprocessing normalizes the audio."""
        processed, sr = preprocessor.preprocess(sample_audio_array, 16000)

        # Should be normalized (approximately zero mean and unit variance)
        assert np.abs(processed.mean()) < 0.1
        assert 0.8 < processed.std() < 1.2

    @pytest.mark.unit
    def test_preprocess_with_resampling(self, preprocessor, sample_audio_8k):
        """Test preprocessing with resampling from 8kHz."""
        processed, sr = preprocessor.preprocess(sample_audio_8k, orig_sr=8000)

        assert sr == 16000
        assert len(processed) > len(sample_audio_8k)  # Upsampled

    @pytest.mark.unit
    def test_preprocess_full_pipeline_8k(self, preprocessor):
        """Test full preprocessing pipeline with 8kHz audio."""
        # Create 8kHz audio
        audio_8k = np.random.randn(8000).astype(np.float32) * 10 + 5

        processed, sr = preprocessor.preprocess(audio_8k, orig_sr=8000)

        # Should be resampled to 16kHz
        assert sr == 16000
        assert len(processed) >= 15000  # ~16000 samples

        # Should be normalized
        assert np.abs(processed.mean()) < 0.1

    @pytest.mark.unit
    def test_preprocess_no_resampling_needed(self, preprocessor, sample_audio_array):
        """Test preprocessing when no resampling is needed."""
        processed, sr = preprocessor.preprocess(sample_audio_array, orig_sr=16000)

        assert sr == 16000
        assert len(processed) == len(sample_audio_array)

    @pytest.mark.unit
    def test_preprocess_with_different_target_rate(self):
        """Test preprocessing with different target sample rate."""
        preprocessor = AudioPreprocessor(target_sample_rate=22050)
        audio = np.random.randn(16000).astype(np.float32)

        processed, sr = preprocessor.preprocess(audio, orig_sr=16000)

        assert sr == 22050
        # Should be approximately 22050/16000 = 1.38x longer
        expected_length = int(16000 * (22050 / 16000))
        assert abs(len(processed) - expected_length) < 100
