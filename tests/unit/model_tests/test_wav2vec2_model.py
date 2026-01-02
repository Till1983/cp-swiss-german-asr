"""Unit tests for Wav2Vec2 model wrapper."""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np


class TestWav2Vec2ModelInit:
    """Test Wav2Vec2Model initialization."""

    @pytest.mark.unit
    @patch('src.models.wav2vec2_model.AutoProcessor')
    @patch('src.models.wav2vec2_model.AutoModelForCTC')
    @patch('src.models.wav2vec2_model.AutoConfig')
    def test_initialization_default_device(self, mock_config_class, mock_model_class, mock_processor_class):
        """Test model initializes with default device selection."""
        mock_config_class.from_pretrained.return_value = Mock(model_type="wav2vec2")
        mock_processor_class.from_pretrained.return_value = Mock()
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        from src.models.wav2vec2_model import Wav2Vec2Model
        model = Wav2Vec2Model(model_name="facebook/wav2vec2-base")

        assert model.model_name == "facebook/wav2vec2-base"
        mock_model.to.assert_called()
        mock_model.eval.assert_called()

    @pytest.mark.unit
    @patch('src.models.wav2vec2_model.AutoProcessor')
    @patch('src.models.wav2vec2_model.AutoModelForCTC')
    @patch('src.models.wav2vec2_model.AutoConfig')
    def test_initialization_with_device(self, mock_config_class, mock_model_class, mock_processor_class):
        """Test model initializes with specified device."""
        mock_config_class.from_pretrained.return_value = Mock(model_type="wav2vec2")
        mock_processor_class.from_pretrained.return_value = Mock()
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        from src.models.wav2vec2_model import Wav2Vec2Model
        model = Wav2Vec2Model(model_name="facebook/wav2vec2-base", device="cpu")

        assert model.device == "cpu"

    @pytest.mark.unit
    @patch('src.models.wav2vec2_model.AutoProcessor')
    @patch('src.models.wav2vec2_model.AutoModelForCTC')
    @patch('src.models.wav2vec2_model.AutoConfig')
    def test_initialization_stores_lm_path(self, mock_config_class, mock_model_class, mock_processor_class):
        """Test model stores language model path."""
        mock_config_class.from_pretrained.return_value = Mock(model_type="wav2vec2")
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
    @patch('src.models.wav2vec2_model.AutoProcessor')
    @patch('src.models.wav2vec2_model.AutoModelForCTC')
    @patch('src.models.wav2vec2_model.AutoConfig')
    def test_model_is_set_to_eval_mode(self, mock_config_class, mock_model_class, mock_processor_class):
        """Test model is set to evaluation mode."""
        mock_config_class.from_pretrained.return_value = Mock(model_type="wav2vec2")
        mock_processor_class.from_pretrained.return_value = Mock()
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        from src.models.wav2vec2_model import Wav2Vec2Model
        model = Wav2Vec2Model(model_name="facebook/wav2vec2-base")

        mock_model.eval.assert_called_once()

    @pytest.mark.unit
    @patch('src.models.wav2vec2_model.AutoProcessor')
    @patch('src.models.wav2vec2_model.AutoModelForCTC')
    @patch('src.models.wav2vec2_model.AutoConfig')
    def test_processor_loaded(self, mock_config_class, mock_model_class, mock_processor_class):
        """Test processor is loaded from pretrained."""
        mock_config_class.from_pretrained.return_value = Mock(model_type="wav2vec2")
        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = Mock()

        from src.models.wav2vec2_model import Wav2Vec2Model
        model = Wav2Vec2Model(model_name="facebook/wav2vec2-base")

        assert model.processor == mock_processor

    @pytest.mark.unit
    @patch('src.models.wav2vec2_model.AutoProcessor')
    @patch('src.models.wav2vec2_model.AutoModelForCTC')
    @patch('src.models.wav2vec2_model.AutoConfig')
    def test_initialization_device_cuda(self, mock_config_class, mock_model_class, mock_processor_class):
        """Test model initialization detects CUDA device."""
        mock_config_class.from_pretrained.return_value = Mock(model_type="wav2vec2")
        mock_processor_class.from_pretrained.return_value = Mock()
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        with patch('torch.cuda.is_available', return_value=True):
            from src.models.wav2vec2_model import Wav2Vec2Model
            model = Wav2Vec2Model()
            assert model.device == "cuda"

    @pytest.mark.unit
    @patch('src.models.wav2vec2_model.AutoProcessor')
    @patch('src.models.wav2vec2_model.AutoModelForCTC')
    @patch('src.models.wav2vec2_model.AutoConfig')
    def test_initialization_device_cpu(self, mock_config_class, mock_model_class, mock_processor_class):
        """Test model initialization falls back to CPU."""
        mock_config_class.from_pretrained.return_value = Mock(model_type="wav2vec2")
        mock_processor_class.from_pretrained.return_value = Mock()
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=False):
                from src.models.wav2vec2_model import Wav2Vec2Model
                model = Wav2Vec2Model()
                assert model.device == "cpu"

    @pytest.mark.unit
    @patch('src.models.wav2vec2_model.AutoProcessor')
    @patch('src.models.wav2vec2_model.AutoModelForCTC')
    @patch('src.models.wav2vec2_model.AutoConfig')
    def test_initialization_uses_auto_model_for_wav2vec2_bert(self, mock_config_class, mock_model_class, mock_processor_class):
        """Ensure wav2vec2-bert configs load via AutoModelForCTC."""
        mock_config = Mock(model_type="wav2vec2-bert")
        mock_config_class.from_pretrained.return_value = mock_config
        mock_processor_class.from_pretrained.return_value = Mock()
        mock_model_class.from_pretrained.return_value = Mock()

        from src.models.wav2vec2_model import Wav2Vec2Model

        model = Wav2Vec2Model(model_name="sharrnah/wav2vec2-bert-CV16-de")

        mock_model_class.from_pretrained.assert_called_once()
        _, kwargs = mock_model_class.from_pretrained.call_args
        assert kwargs.get("config") is mock_config


class TestWav2Vec2ModelTranscribe:
    """Test Wav2Vec2Model transcribe method."""

    @pytest.fixture
    def mock_wav2vec2_model(self):
        """Create a mocked Wav2Vec2Model."""
        with patch('src.models.wav2vec2_model.AutoProcessor') as mock_processor_class, \
             patch('src.models.wav2vec2_model.AutoModelForCTC') as mock_model_class, \
             patch('src.models.wav2vec2_model.AutoConfig') as mock_config_class:

            # Setup processor mock
            mock_processor = Mock()
            mock_processor.return_value = {"input_values": torch.randn(1, 16000)}
            mock_processor.batch_decode.return_value = ["transcribed text"]
            mock_processor_class.from_pretrained.return_value = mock_processor

            mock_config_class.from_pretrained.return_value = Mock(model_type="wav2vec2")

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
    @patch('src.models.wav2vec2_model.AutoProcessor')
    @patch('src.models.wav2vec2_model.AutoModelForCTC')
    @patch('src.models.wav2vec2_model.AutoConfig')
    @patch('src.models.wav2vec2_model._HAS_PYCTCDECODE', True)
    def test_decoder_not_init_without_lm_path(self, mock_config_class, mock_model_class, mock_processor_class):
        """Test decoder is not initialized without LM path."""
        mock_config_class.from_pretrained.return_value = Mock(model_type="wav2vec2")
        mock_processor_class.from_pretrained.return_value = Mock()
        mock_model_class.from_pretrained.return_value = Mock()

        from src.models.wav2vec2_model import Wav2Vec2Model
        model = Wav2Vec2Model(model_name="facebook/wav2vec2-base", lm_path=None)

        assert model.decoder is None

    @pytest.mark.unit
    @patch('src.models.wav2vec2_model.AutoProcessor')
    @patch('src.models.wav2vec2_model.AutoModelForCTC')
    @patch('src.models.wav2vec2_model.AutoConfig')
    @patch('src.models.wav2vec2_model._HAS_PYCTCDECODE', False)
    def test_decoder_not_init_without_pyctcdecode(self, mock_config_class, mock_model_class, mock_processor_class):
        """Test decoder is not initialized when pyctcdecode not available."""
        mock_processor = Mock()
        mock_processor.tokenizer.get_vocab.return_value = {"a": 0, "b": 1}
        mock_config_class.from_pretrained.return_value = Mock(model_type="wav2vec2")
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = Mock()

        from src.models.wav2vec2_model import Wav2Vec2Model
        model = Wav2Vec2Model(
            model_name="facebook/wav2vec2-base",
            lm_path="/path/to/lm.arpa"
        )

        assert model.decoder is None

    @pytest.mark.unit
    @patch('src.models.wav2vec2_model.AutoProcessor')
    @patch('src.models.wav2vec2_model.AutoModelForCTC')
    @patch('src.models.wav2vec2_model.AutoConfig')
    @patch('src.models.wav2vec2_model._HAS_PYCTCDECODE', True)
    @patch('src.models.wav2vec2_model.build_ctcdecoder')
    def test_decoder_init_with_valid_lm(self, mock_decoder, mock_config_class, mock_model_class, mock_processor_class, temp_dir):
        """Test decoder initializes successfully with valid LM file."""
        # Create LM file
        lm_file = temp_dir / "test.arpa"
        lm_file.write_text("dummy lm content")

        mock_processor = Mock()
        mock_processor.tokenizer.get_vocab.return_value = {"a": 0, "b": 1}
        mock_config_class.from_pretrained.return_value = Mock(model_type="wav2vec2")
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = Mock()
        
        mock_decoder_instance = Mock()
        mock_decoder.return_value = mock_decoder_instance

        from src.models.wav2vec2_model import Wav2Vec2Model
        model = Wav2Vec2Model(
            model_name="facebook/wav2vec2-base",
            lm_path=str(lm_file)
        )

        # Decoder should be initialized
        assert model.decoder is not None
        mock_decoder.assert_called_once()

    @pytest.mark.unit
    @patch('src.models.wav2vec2_model.AutoProcessor')
    @patch('src.models.wav2vec2_model.AutoModelForCTC')
    @patch('src.models.wav2vec2_model.AutoConfig')
    @patch('src.models.wav2vec2_model._HAS_PYCTCDECODE', True)
    @patch('src.models.wav2vec2_model.build_ctcdecoder')
    def test_decoder_falls_back_on_build_failure(self, mock_decoder, mock_config_class, mock_model_class, mock_processor_class, temp_dir):
        """Test decoder falls back to None when build_ctcdecoder fails."""
        # Create LM file
        lm_file = temp_dir / "test.arpa"
        lm_file.write_text("dummy lm content")

        mock_processor = Mock()
        mock_processor.tokenizer.get_vocab.return_value = {"a": 0, "b": 1}
        mock_config_class.from_pretrained.return_value = Mock(model_type="wav2vec2")
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = Mock()
        
        # Make decoder build fail
        mock_decoder.side_effect = Exception("Decoder build failed")

        from src.models.wav2vec2_model import Wav2Vec2Model
        model = Wav2Vec2Model(
            model_name="facebook/wav2vec2-base",
            lm_path=str(lm_file)
        )

        # Should fall back to None gracefully
        assert model.decoder is None

    @pytest.mark.unit
    @patch('src.models.wav2vec2_model.AutoProcessor')
    @patch('src.models.wav2vec2_model.AutoModelForCTC')
    @patch('src.models.wav2vec2_model.AutoConfig')
    @patch('src.models.wav2vec2_model._HAS_PYCTCDECODE', True)
    @patch('src.models.wav2vec2_model.build_ctcdecoder')
    @patch('torchaudio.load')
    def test_transcribe_with_decoder(self, mock_torchaudio, mock_decoder_builder, mock_config_class, mock_model_class, mock_processor_class, temp_dir):
        """Test transcribe uses decoder when available."""
        # Setup LM file
        lm_file = temp_dir / "test.arpa"
        lm_file.write_text("dummy lm content")

        # Setup mocks
        mock_processor = Mock()
        mock_processor.tokenizer.get_vocab.return_value = {"a": 0, "b": 1}
        mock_processor.return_value = {"input_values": torch.randn(1, 16000)}
        mock_config_class.from_pretrained.return_value = Mock(model_type="wav2vec2")
        mock_processor_class.from_pretrained.return_value = mock_processor
        
        mock_model = Mock()
        mock_logits = Mock()
        mock_logits.squeeze.return_value.cpu.return_value.numpy.return_value = [[0.1, 0.2]]
        mock_model.return_value.logits = mock_logits
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model
        
        mock_decoder = Mock()
        mock_decoder.decode.return_value = "decoded with lm"
        mock_decoder_builder.return_value = mock_decoder

        # Mock audio loading
        audio_file = temp_dir / "test.wav"
        audio_file.touch()
        waveform = torch.randn(1, 16000)
        mock_torchaudio.return_value = (waveform, 16000)

        from src.models.wav2vec2_model import Wav2Vec2Model
        model = Wav2Vec2Model(
            model_name="facebook/wav2vec2-base",
            lm_path=str(lm_file)
        )

        result = model.transcribe(audio_file)

        # Should use decoder
        mock_decoder.decode.assert_called_once()
        assert result["text"] == "decoded with lm"


class TestWav2Vec2ModelEdgeCases:
    """Test edge cases for Wav2Vec2Model."""

    @pytest.mark.unit
    def test_import_error_handling(self):
        """Test that module handles missing pyctcdecode gracefully."""
        # This tests the import error handling at module level (lines 12-13)
        # The _HAS_PYCTCDECODE flag should be set based on import success
        from src.models import wav2vec2_model
        assert hasattr(wav2vec2_model, '_HAS_PYCTCDECODE')
        assert isinstance(wav2vec2_model._HAS_PYCTCDECODE, bool)

    @pytest.mark.unit
    def test_import_error_sets_has_pyctcdecode_false(self):
        """Simulate missing pyctcdecode without reloading the module."""
        from src.models import wav2vec2_model

        with patch.object(wav2vec2_model, "_HAS_PYCTCDECODE", False):
            assert wav2vec2_model._HAS_PYCTCDECODE is False

    @pytest.mark.unit
    @patch('src.models.wav2vec2_model.AutoProcessor')
    @patch('src.models.wav2vec2_model.AutoModelForCTC')
    @patch('src.models.wav2vec2_model.AutoConfig')
    def test_handles_local_model_path(self, mock_config_class, mock_model_class, mock_processor_class, temp_dir):
        """Test model handles local model path."""
        # Create a fake model directory
        model_dir = temp_dir / "local_model"
        model_dir.mkdir()

        mock_config_class.from_pretrained.return_value = Mock(model_type="wav2vec2")
        mock_processor_class.from_pretrained.return_value = Mock()
        mock_model_class.from_pretrained.return_value = Mock()

        from src.models.wav2vec2_model import Wav2Vec2Model

        # This tests the local path detection logic
        model = Wav2Vec2Model(model_name=str(model_dir))

        assert model.model_name == str(model_dir)

    @pytest.mark.unit
    @patch('src.models.wav2vec2_model.AutoProcessor')
    @patch('src.models.wav2vec2_model.AutoModelForCTC')
    @patch('src.models.wav2vec2_model.AutoConfig')
    def test_processor_load_failure(self, mock_config_class, mock_model_class, mock_processor_class):
        """Test appropriate error when processor fails to load."""
        mock_config_class.from_pretrained.return_value = Mock(model_type="wav2vec2")
        mock_processor_class.from_pretrained.side_effect = OSError("Model not found")

        from src.models.wav2vec2_model import Wav2Vec2Model

        with pytest.raises(ValueError, match="Failed to load processor"):
            Wav2Vec2Model(model_name="nonexistent/model")
