"""Unit tests for Wav2Vec2 model wrapper."""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np


class TestWav2Vec2ModelInit:
    """Test Wav2Vec2Model initialization."""

    @pytest.mark.unit
    @patch('src.models.wav2vec2_model.Wav2Vec2Processor')
    @patch('src.models.wav2vec2_model.Wav2Vec2ForCTC')
    def test_initialization_default_device(self, mock_model_class, mock_processor_class):
        """Test model initializes with default device selection."""
        mock_processor_class.from_pretrained.return_value = Mock()
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        from src.models.wav2vec2_model import Wav2Vec2Model
        model = Wav2Vec2Model(model_name="facebook/wav2vec2-base")

        assert model.model_name == "facebook/wav2vec2-base"
        mock_model.to.assert_called()
        mock_model.eval.assert_called()

    @pytest.mark.unit
    @patch('src.models.wav2vec2_model.Wav2Vec2Processor')
    @patch('src.models.wav2vec2_model.Wav2Vec2ForCTC')
    def test_initialization_with_device(self, mock_model_class, mock_processor_class):
        """Test model initializes with specified device."""
        mock_processor_class.from_pretrained.return_value = Mock()
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        from src.models.wav2vec2_model import Wav2Vec2Model
        model = Wav2Vec2Model(model_name="facebook/wav2vec2-base", device="cpu")

        assert model.device == "cpu"

    @pytest.mark.unit
    @patch('src.models.wav2vec2_model.Wav2Vec2Processor')
    @patch('src.models.wav2vec2_model.Wav2Vec2ForCTC')
    def test_initialization_stores_lm_path(self, mock_model_class, mock_processor_class):
        """Test model stores language model path."""
        mock_processor_class.from_pretrained.return_value = Mock()
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        from src.models.wav2vec2_model import Wav2Vec2Model
        model = Wav2Vec2Model(
            model_name="facebook/wav2vec2-base",
            lm_path="/path/to/lm.arpa"
        )

        assert model.lm_path == "/path/to/lm.arpa"

    @pytest.mark.unit
    @patch('src.models.wav2vec2_model.Wav2Vec2Processor')
    @patch('src.models.wav2vec2_model.Wav2Vec2ForCTC')
    def test_model_is_set_to_eval_mode(self, mock_model_class, mock_processor_class):
        """Test model is set to evaluation mode."""
        mock_processor_class.from_pretrained.return_value = Mock()
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        from src.models.wav2vec2_model import Wav2Vec2Model
        model = Wav2Vec2Model(model_name="facebook/wav2vec2-base")

        mock_model.eval.assert_called_once()

    @pytest.mark.unit
    @patch('src.models.wav2vec2_model.Wav2Vec2Processor')
    @patch('src.models.wav2vec2_model.Wav2Vec2ForCTC')
    def test_processor_loaded(self, mock_model_class, mock_processor_class):
        """Test processor is loaded from pretrained."""
        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = Mock()

        from src.models.wav2vec2_model import Wav2Vec2Model
        model = Wav2Vec2Model(model_name="facebook/wav2vec2-base")

        assert model.processor == mock_processor

    @pytest.mark.unit
    @patch('src.models.wav2vec2_model.Wav2Vec2Processor')
    @patch('src.models.wav2vec2_model.Wav2Vec2ForCTC')
    def test_initialization_device_cuda(self, mock_model_class, mock_processor_class):
        """Test model initialization detects CUDA device."""
        mock_processor_class.from_pretrained.return_value = Mock()
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        with patch('torch.cuda.is_available', return_value=True):
            from src.models.wav2vec2_model import Wav2Vec2Model
            model = Wav2Vec2Model()
            assert model.device == "cuda"

    @pytest.mark.unit
    @patch('src.models.wav2vec2_model.Wav2Vec2Processor')
    @patch('src.models.wav2vec2_model.Wav2Vec2ForCTC')
    def test_initialization_device_cpu(self, mock_model_class, mock_processor_class):
        """Test model initialization falls back to CPU."""
        mock_processor_class.from_pretrained.return_value = Mock()
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=False):
                from src.models.wav2vec2_model import Wav2Vec2Model
                model = Wav2Vec2Model()
                assert model.device == "cpu"


class TestWav2Vec2ModelTranscribe:
    """Test Wav2Vec2Model transcribe method."""

    @pytest.fixture
    def mock_wav2vec2_model(self):
        """Create a mocked Wav2Vec2Model."""
        with patch('src.models.wav2vec2_model.Wav2Vec2Processor') as mock_processor_class, \
             patch('src.models.wav2vec2_model.Wav2Vec2ForCTC') as mock_model_class:

            # Setup processor mock
            mock_processor = Mock()
            mock_processor.return_value = {"input_values": torch.randn(1, 16000)}
            mock_processor.batch_decode.return_value = ["transcribed text"]
            mock_processor_class.from_pretrained.return_value = mock_processor

            # Setup model mock
            mock_model = Mock()
            mock_logits = Mock()
            mock_logits.logits = torch.randn(1, 100, 32)
            mock_model.return_value = mock_logits
            mock_model_class.from_pretrained.return_value = mock_model

            from src.models.wav2vec2_model import Wav2Vec2Model
            model = Wav2Vec2Model(model_name="facebook/wav2vec2-base", device="cpu")

            yield model

    @pytest.mark.unit
    def test_transcribe_returns_dict(self, mock_wav2vec2_model, sample_audio_path):
        """Test transcribe returns dictionary with 'text' key."""
        result = mock_wav2vec2_model.transcribe(sample_audio_path)

        assert isinstance(result, dict)
        assert "text" in result

    @pytest.mark.unit
    def test_transcribe_nonexistent_file_raises(self, mock_wav2vec2_model):
        """Test transcribe raises FileNotFoundError for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            mock_wav2vec2_model.transcribe(Path("/nonexistent/audio.wav"))

    @pytest.mark.unit
    @patch('torchaudio.load')
    def test_transcribe_handles_different_sample_rates(self, mock_torchaudio, mock_wav2vec2_model, temp_dir):
        """Test transcribe handles audio with different sample rates."""
        # Create a dummy audio file
        audio_file = temp_dir / "test.wav"
        audio_file.touch()

        # Mock torchaudio to return 8kHz audio
        mock_torchaudio.return_value = (torch.randn(1, 8000), 8000)

        # Should not raise - resampling should occur
        result = mock_wav2vec2_model.transcribe(audio_file)
        assert "text" in result

    @pytest.mark.unit
    @patch('torchaudio.load')
    def test_transcribe_handles_stereo_audio(self, mock_torchaudio, mock_wav2vec2_model, temp_dir):
        """Test transcribe handles stereo audio by converting to mono."""
        audio_file = temp_dir / "stereo.wav"
        audio_file.touch()

        # Mock stereo audio (2 channels)
        mock_torchaudio.return_value = (torch.randn(2, 16000), 16000)

        result = mock_wav2vec2_model.transcribe(audio_file)
        assert "text" in result

    @pytest.mark.unit
    @patch('torchaudio.load')
    def test_transcribe_handles_audio_load_error(self, mock_torchaudio, mock_wav2vec2_model, temp_dir):
        """Test transcribe handles audio load errors."""
        audio_file = temp_dir / "test.wav"
        audio_file.touch()

        mock_torchaudio.side_effect = Exception("Audio load failed")

        with pytest.raises(ValueError, match="Failed to load audio"):
            mock_wav2vec2_model.transcribe(audio_file)

    @pytest.mark.unit
    @patch('torchaudio.load')
    def test_transcribe_rejects_empty_audio(self, mock_torchaudio, mock_wav2vec2_model, temp_dir):
        """Test transcribe rejects empty audio."""
        audio_file = temp_dir / "test.wav"
        audio_file.touch()

        # Mock empty audio
        empty_waveform = torch.zeros((1, 0))
        mock_torchaudio.return_value = (empty_waveform, 16000)

        with pytest.raises(ValueError, match="Audio file is empty"):
            mock_wav2vec2_model.transcribe(audio_file)

    @pytest.mark.unit
    @patch('torchaudio.load')
    def test_transcribe_uses_greedy_decoding_as_fallback(self, mock_torchaudio, mock_wav2vec2_model, temp_dir):
        """Test transcribe falls back to greedy decoding without LM."""
        audio_file = temp_dir / "test.wav"
        audio_file.touch()

        # Mock audio
        waveform = torch.randn(1, 16000)
        mock_torchaudio.return_value = (waveform, 16000)

        result = mock_wav2vec2_model.transcribe(audio_file)

        # Should use greedy decoding (batch_decode was called)
        assert mock_wav2vec2_model.processor.batch_decode.called
        assert result["text"] == "transcribed text"

class TestWav2Vec2ModelDecoderInit:
    """Test Wav2Vec2Model decoder initialization."""

    @pytest.mark.unit
    @patch('src.models.wav2vec2_model.Wav2Vec2Processor')
    @patch('src.models.wav2vec2_model.Wav2Vec2ForCTC')
    @patch('src.models.wav2vec2_model._HAS_PYCTCDECODE', True)
    def test_decoder_not_init_without_lm_path(self, mock_model_class, mock_processor_class):
        """Test decoder is not initialized without LM path."""
        mock_processor_class.from_pretrained.return_value = Mock()
        mock_model_class.from_pretrained.return_value = Mock()

        from src.models.wav2vec2_model import Wav2Vec2Model
        model = Wav2Vec2Model(model_name="facebook/wav2vec2-base", lm_path=None)

        assert model.decoder is None

    @pytest.mark.unit
    @patch('src.models.wav2vec2_model.Wav2Vec2Processor')
    @patch('src.models.wav2vec2_model.Wav2Vec2ForCTC')
    @patch('src.models.wav2vec2_model._HAS_PYCTCDECODE', False)
    def test_decoder_not_init_without_pyctcdecode(self, mock_model_class, mock_processor_class):
        """Test decoder is not initialized when pyctcdecode not available."""
        mock_processor = Mock()
        mock_processor.tokenizer.get_vocab.return_value = {"a": 0, "b": 1}
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = Mock()

        from src.models.wav2vec2_model import Wav2Vec2Model
        model = Wav2Vec2Model(
            model_name="facebook/wav2vec2-base",
            lm_path="/path/to/lm.arpa"
        )

        assert model.decoder is None


class TestWav2Vec2ModelEdgeCases:
    """Test edge cases for Wav2Vec2Model."""

    @pytest.mark.unit
    @patch('src.models.wav2vec2_model.Wav2Vec2Processor')
    @patch('src.models.wav2vec2_model.Wav2Vec2ForCTC')
    def test_handles_local_model_path(self, mock_model_class, mock_processor_class, temp_dir):
        """Test model handles local model path."""
        # Create a fake model directory
        model_dir = temp_dir / "local_model"
        model_dir.mkdir()

        mock_processor_class.from_pretrained.return_value = Mock()
        mock_model_class.from_pretrained.return_value = Mock()

        from src.models.wav2vec2_model import Wav2Vec2Model

        # This tests the local path detection logic
        model = Wav2Vec2Model(model_name=str(model_dir))

        assert model.model_name == str(model_dir)

    @pytest.mark.unit
    @patch('src.models.wav2vec2_model.Wav2Vec2Processor')
    @patch('src.models.wav2vec2_model.Wav2Vec2ForCTC')
    def test_processor_load_failure(self, mock_model_class, mock_processor_class):
        """Test appropriate error when processor fails to load."""
        mock_processor_class.from_pretrained.side_effect = OSError("Model not found")

        from src.models.wav2vec2_model import Wav2Vec2Model

        with pytest.raises(ValueError, match="Failed to load processor"):
            Wav2Vec2Model(model_name="nonexistent/model")
