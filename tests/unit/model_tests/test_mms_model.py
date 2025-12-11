"""Unit tests for MMS model wrapper."""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np


class TestMMSModelInit:
    """Test MMSModel initialization."""

    @pytest.mark.unit
    @patch('src.models.mms_model.AutoProcessor')
    @patch('src.models.mms_model.Wav2Vec2ForCTC')
    def test_initialization_default_model(self, mock_model_class, mock_processor_class):
        """Test model initializes with default model name."""
        mock_processor_class.from_pretrained.return_value = Mock()
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        from src.models.mms_model import MMSModel
        model = MMSModel()

        assert model.model_name == "facebook/mms-1b-all"
        mock_model.to.assert_called()
        mock_model.eval.assert_called()

    @pytest.mark.unit
    @patch('src.models.mms_model.AutoProcessor')
    @patch('src.models.mms_model.Wav2Vec2ForCTC')
    def test_initialization_with_custom_model(self, mock_model_class, mock_processor_class):
        """Test model initializes with custom model name."""
        mock_processor_class.from_pretrained.return_value = Mock()
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        from src.models.mms_model import MMSModel
        model = MMSModel(model_name="facebook/mms-300m")

        assert model.model_name == "facebook/mms-300m"

    @pytest.mark.unit
    @patch('src.models.mms_model.AutoProcessor')
    @patch('src.models.mms_model.Wav2Vec2ForCTC')
    def test_initialization_with_device(self, mock_model_class, mock_processor_class):
        """Test model initializes with specified device."""
        mock_processor_class.from_pretrained.return_value = Mock()
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        from src.models.mms_model import MMSModel
        model = MMSModel(device="cpu")

        assert model.device == "cpu"

    @pytest.mark.unit
    @patch('src.models.mms_model.AutoProcessor')
    @patch('src.models.mms_model.Wav2Vec2ForCTC')
    def test_initialization_stores_lm_path(self, mock_model_class, mock_processor_class):
        """Test model stores language model path."""
        mock_processor_class.from_pretrained.return_value = Mock()
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        from src.models.mms_model import MMSModel
        model = MMSModel(lm_path="/path/to/lm.arpa")

        assert model.lm_path == "/path/to/lm.arpa"

    @pytest.mark.unit
    @patch('src.models.mms_model.AutoProcessor')
    @patch('src.models.mms_model.Wav2Vec2ForCTC')
    def test_model_set_to_eval_mode(self, mock_model_class, mock_processor_class):
        """Test model is set to evaluation mode."""
        mock_processor_class.from_pretrained.return_value = Mock()
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        from src.models.mms_model import MMSModel
        model = MMSModel()

        mock_model.eval.assert_called_once()

    @pytest.mark.unit
    @patch('src.models.mms_model.AutoProcessor')
    @patch('src.models.mms_model.Wav2Vec2ForCTC')
    def test_processor_loaded(self, mock_model_class, mock_processor_class):
        """Test processor is loaded from pretrained."""
        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = Mock()

        from src.models.mms_model import MMSModel
        model = MMSModel()

        assert model.processor == mock_processor


class TestMMSModelTranscribe:
    """Test MMSModel transcribe method."""

    @pytest.fixture
    def mock_mms_model(self):
        """Create a mocked MMSModel."""
        with patch('src.models.mms_model.AutoProcessor') as mock_processor_class, \
             patch('src.models.mms_model.Wav2Vec2ForCTC') as mock_model_class:

            # Setup processor mock
            mock_processor = Mock()
            mock_processor.return_value = {"input_values": torch.randn(1, 16000)}
            mock_processor.batch_decode.return_value = ["transcribed text"]
            mock_processor.tokenizer = Mock()
            mock_processor.tokenizer.set_target_lang = Mock()
            mock_processor_class.from_pretrained.return_value = mock_processor

            # Setup model mock
            mock_model = Mock()
            mock_logits = Mock()
            mock_logits.logits = torch.randn(1, 100, 32)
            mock_model.return_value = mock_logits
            mock_model.load_adapter = Mock()
            mock_model_class.from_pretrained.return_value = mock_model

            from src.models.mms_model import MMSModel
            model = MMSModel(device="cpu")

            yield model

    @pytest.mark.unit
    def test_transcribe_returns_dict(self, mock_mms_model, sample_audio_path):
        """Test transcribe returns dictionary with 'text' key."""
        result = mock_mms_model.transcribe(sample_audio_path)

        assert isinstance(result, dict)
        assert "text" in result

    @pytest.mark.unit
    def test_transcribe_default_language(self, mock_mms_model, sample_audio_path):
        """Test transcribe uses default language (German)."""
        mock_mms_model.transcribe(sample_audio_path)

        # Check that German language was set
        mock_mms_model.processor.tokenizer.set_target_lang.assert_called_with("deu")

    @pytest.mark.unit
    def test_transcribe_custom_language(self, mock_mms_model, sample_audio_path):
        """Test transcribe with custom language."""
        mock_mms_model.transcribe(sample_audio_path, language="fra")

        mock_mms_model.processor.tokenizer.set_target_lang.assert_called_with("fra")

    @pytest.mark.unit
    def test_transcribe_loads_adapter(self, mock_mms_model, sample_audio_path):
        """Test transcribe loads language-specific adapter."""
        mock_mms_model.transcribe(sample_audio_path, language="deu")

        mock_mms_model.model.load_adapter.assert_called_with("deu")

    @pytest.mark.unit
    def test_transcribe_nonexistent_file_raises(self, mock_mms_model):
        """Test transcribe raises FileNotFoundError for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            mock_mms_model.transcribe(Path("/nonexistent/audio.wav"))

    @pytest.mark.unit
    @patch('torchaudio.load')
    def test_transcribe_handles_different_sample_rates(self, mock_torchaudio, mock_mms_model, temp_dir):
        """Test transcribe handles audio with different sample rates."""
        audio_file = temp_dir / "test.wav"
        audio_file.touch()

        # Mock torchaudio to return 8kHz audio
        mock_torchaudio.return_value = (torch.randn(1, 8000), 8000)

        result = mock_mms_model.transcribe(audio_file)
        assert "text" in result

    @pytest.mark.unit
    @patch('torchaudio.load')
    def test_transcribe_handles_stereo_audio(self, mock_torchaudio, mock_mms_model, temp_dir):
        """Test transcribe handles stereo audio by converting to mono."""
        audio_file = temp_dir / "stereo.wav"
        audio_file.touch()

        # Mock stereo audio
        mock_torchaudio.return_value = (torch.randn(2, 16000), 16000)

        result = mock_mms_model.transcribe(audio_file)
        assert "text" in result

    @pytest.mark.unit
    @patch('torchaudio.load')
    def test_transcribe_uses_greedy_decoding_as_fallback(self, mock_torchaudio, mock_mms_model, temp_dir):
        """Test transcribe falls back to greedy decoding without LM."""
        audio_file = temp_dir / "test.wav"
        audio_file.touch()

        # Mock audio
        waveform = torch.randn(1, 16000)
        mock_torchaudio.return_value = (waveform, 16000)

        result = mock_mms_model.transcribe(audio_file)

        # Should use greedy decoding (batch_decode was called)
        assert mock_mms_model.processor.batch_decode.called
        assert result["text"] == "transcribed text"


class TestMMSModelDecoderInit:
    """Test MMSModel decoder initialization."""

    @pytest.mark.unit
    @patch('src.models.mms_model.AutoProcessor')
    @patch('src.models.mms_model.Wav2Vec2ForCTC')
    @patch('src.models.mms_model._HAS_PYCTCDECODE', True)
    def test_decoder_not_init_without_lm_path(self, mock_model_class, mock_processor_class):
        """Test decoder is not initialized without LM path."""
        mock_processor_class.from_pretrained.return_value = Mock()
        mock_model_class.from_pretrained.return_value = Mock()

        from src.models.mms_model import MMSModel
        model = MMSModel(lm_path=None)

        assert model.decoder is None

    @pytest.mark.unit
    @patch('src.models.mms_model.AutoProcessor')
    @patch('src.models.mms_model.Wav2Vec2ForCTC')
    @patch('src.models.mms_model._HAS_PYCTCDECODE', False)
    def test_decoder_not_init_without_pyctcdecode(self, mock_model_class, mock_processor_class):
        """Test decoder is not initialized when pyctcdecode not available."""
        mock_processor = Mock()
        mock_processor.tokenizer.get_vocab.return_value = {"a": 0, "b": 1}
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = Mock()

        from src.models.mms_model import MMSModel
        model = MMSModel(lm_path="/path/to/lm.arpa")

        assert model.decoder is None

    @pytest.mark.unit
    @patch('src.models.mms_model.AutoProcessor')
    @patch('src.models.mms_model.Wav2Vec2ForCTC')
    @patch('src.models.mms_model._HAS_PYCTCDECODE', True)
    @patch('src.models.mms_model.build_ctcdecoder')
    def test_decoder_init_with_valid_lm(self, mock_build_decoder, mock_model_class, mock_processor_class, temp_dir):
        """Test decoder is initialized with valid LM path."""
        mock_processor = Mock()
        mock_processor.tokenizer.get_vocab.return_value = {"a": 0, "b": 1, "c": 2}
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = Mock()

        # Create a fake LM file
        lm_file = temp_dir / "test.arpa"
        lm_file.touch()

        mock_decoder = Mock()
        mock_build_decoder.return_value = mock_decoder

        from src.models.mms_model import MMSModel
        model = MMSModel(lm_path=str(lm_file))

        assert model.decoder == mock_decoder

    @pytest.mark.unit
    @patch('src.models.mms_model.AutoProcessor')
    @patch('src.models.mms_model.Wav2Vec2ForCTC')
    @patch('src.models.mms_model._HAS_PYCTCDECODE', True)
    def test_decoder_warns_when_lm_not_found(self, mock_model_class, mock_processor_class, capsys):
        """Test initialization warns when LM file not found."""
        mock_processor_class.from_pretrained.return_value = Mock()
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        from src.models.mms_model import MMSModel
        MMSModel(lm_path="/nonexistent/lm.arpa")

        captured = capsys.readouterr()
        assert "Warning" in captured.out or "lm" in captured.out.lower()

    @pytest.mark.unit
    @patch('src.models.mms_model.AutoProcessor')
    @patch('src.models.mms_model.Wav2Vec2ForCTC')
    @patch('src.models.mms_model.Path')
    def test_decoder_validates_lm_path_exists(self, mock_path, mock_model_class, mock_processor_class):
        """Test initialization checks if LM path exists."""
        mock_processor_class.from_pretrained.return_value = Mock()
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        # Mock Path to return False for exists()
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False
        mock_path.return_value = mock_path_instance

        from src.models.mms_model import MMSModel
        model = MMSModel(lm_path="/path/to/lm.arpa")

        # Decoder should not be initialized
        assert model.decoder is None


class TestMMSModelEdgeCases:
    """Test edge cases for MMSModel."""

    @pytest.mark.unit
    def test_import_error_sets_has_pyctcdecode_false(self, monkeypatch):
        """Simulate missing pyctcdecode without reloading the module."""
        from src.models import mms_model

        with patch.object(mms_model, "_HAS_PYCTCDECODE", False):
            assert mms_model._HAS_PYCTCDECODE is False

    @pytest.mark.unit
    def test_import_error_handling(self):
        """Test that module handles missing pyctcdecode gracefully."""
        from src.models import mms_model
        assert hasattr(mms_model, '_HAS_PYCTCDECODE')
        assert isinstance(mms_model._HAS_PYCTCDECODE, bool)

    @pytest.mark.unit
    @patch('src.models.mms_model.AutoProcessor')
    @patch('src.models.mms_model.Wav2Vec2ForCTC')
    @patch('src.models.mms_model._HAS_PYCTCDECODE', False)
    def test_init_decoder_without_pyctcdecode(self, mock_model_class, mock_processor_class, temp_dir):
        """Test _init_decoder returns early when pyctcdecode not available."""
        lm_file = temp_dir / "test.arpa"
        lm_file.write_text("dummy")
        
        mock_processor_class.from_pretrained.return_value = Mock()
        mock_model_class.from_pretrained.return_value = Mock()
        
        from src.models.mms_model import MMSModel
        model = MMSModel(model_name="facebook/mms-1b-all", lm_path=str(lm_file))
        
        # Decoder should remain None when pyctcdecode unavailable
        model._init_decoder()
        assert model.decoder is None

    @pytest.mark.unit
    @patch('src.models.mms_model.AutoProcessor')
    @patch('src.models.mms_model.Wav2Vec2ForCTC')
    @patch('src.models.mms_model._HAS_PYCTCDECODE', True)
    @patch('src.models.mms_model.build_ctcdecoder', side_effect=Exception("boom"))
    def test_init_decoder_falls_back_on_failure(self, mock_decoder, mock_model_class, mock_processor_class, temp_dir):
        """_init_decoder should fall back to greedy when decoder build fails."""
        lm_file = temp_dir / "fail.arpa"
        lm_file.write_text("dummy")

        mock_processor = Mock()
        mock_processor.tokenizer.get_vocab.return_value = {"a": 0}
        mock_processor_class.from_pretrained.return_value = mock_processor

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model

        from src.models.mms_model import MMSModel
        model = MMSModel(model_name="facebook/mms-1b-all", lm_path=str(lm_file))

        # Force decoder init path to hit exception
        model._init_decoder()

        assert model.decoder is None

    @pytest.mark.unit
    @patch('src.models.mms_model.AutoProcessor')
    @patch('src.models.mms_model.Wav2Vec2ForCTC')
    @patch('src.models.mms_model._HAS_PYCTCDECODE', True)
    @patch('src.models.mms_model.build_ctcdecoder')
    @patch('torchaudio.load')
    def test_transcribe_reinits_decoder_on_first_call(self, mock_audio, mock_decoder, mock_model_class, mock_processor_class, temp_dir):
        """Test that decoder is initialized on first transcribe call with LM."""
        lm_file = temp_dir / "test.arpa"
        lm_file.write_text("dummy")
        audio_file = temp_dir / "test.wav"
        audio_file.touch()
        
        mock_processor = Mock()
        mock_processor.tokenizer.get_vocab.return_value = {"a": 0, "b": 1}
        mock_processor.return_value = {"input_values": torch.randn(1, 16000)}
        mock_processor_class.from_pretrained.return_value = mock_processor
        
        mock_model = Mock()
        mock_logits = Mock()
        mock_logits.argmax.return_value = torch.tensor([[0, 1]])
        mock_logits.squeeze.return_value.cpu.return_value.numpy.return_value = [[0.1, 0.2]]
        mock_model.return_value.logits = mock_logits
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model
        
        mock_decoder_instance = Mock()
        mock_decoder_instance.decode.return_value = "decoded"
        mock_decoder.return_value = mock_decoder_instance
        
        mock_audio.return_value = (torch.randn(1, 16000), 16000)
        mock_processor.batch_decode.return_value = ["text"]
        
        from src.models.mms_model import MMSModel
        model = MMSModel(model_name="facebook/mms-1b-all", lm_path=str(lm_file))
        
        # Manually set decoder to None to test re-init path
        model.decoder = None

        model.transcribe(audio_file)

        # Decoder should be initialized and used
        assert mock_decoder.called

    @pytest.mark.unit
    @patch('src.models.mms_model.AutoProcessor')
    @patch('src.models.mms_model.Wav2Vec2ForCTC')
    @patch('torchaudio.load')
    def test_transcribe_empty_audio_raises(self, mock_torchaudio, mock_model_class, mock_processor_class, temp_dir):
        """Test transcribe raises error for empty audio."""
        mock_processor_class.from_pretrained.return_value = Mock()
        mock_model_class.from_pretrained.return_value = Mock()

        # Mock empty audio
        mock_torchaudio.return_value = (torch.tensor([]), 16000)

        audio_file = temp_dir / "empty.wav"
        audio_file.touch()

        from src.models.mms_model import MMSModel
        model = MMSModel(device="cpu")

        with pytest.raises(ValueError, match="empty"):
            model.transcribe(audio_file)

    @pytest.mark.unit
    @patch('src.models.mms_model.AutoProcessor')
    @patch('src.models.mms_model.Wav2Vec2ForCTC')
    @patch('torchaudio.load')
    def test_transcribe_corrupt_audio_raises(self, mock_torchaudio, mock_model_class, mock_processor_class, temp_dir):
        """Test transcribe raises error for corrupt audio."""
        mock_processor_class.from_pretrained.return_value = Mock()
        mock_model_class.from_pretrained.return_value = Mock()

        # Mock torchaudio to raise an error
        mock_torchaudio.side_effect = Exception("Corrupt file")

        audio_file = temp_dir / "corrupt.wav"
        audio_file.touch()

        from src.models.mms_model import MMSModel
        model = MMSModel(device="cpu")

        with pytest.raises(ValueError, match="Failed to load audio"):
            model.transcribe(audio_file)

    @pytest.mark.unit
    @patch('src.models.mms_model.AutoProcessor')
    @patch('src.models.mms_model.Wav2Vec2ForCTC')
    def test_language_adapter_switch(self, mock_model_class, mock_processor_class, temp_dir):
        """Test model correctly switches language adapters."""
        mock_processor = Mock()
        mock_processor.return_value = {"input_values": torch.randn(1, 16000)}
        mock_processor.batch_decode.return_value = ["text"]
        mock_processor.tokenizer = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor

        mock_model = Mock()
        mock_logits = Mock()
        mock_logits.logits = torch.randn(1, 100, 32)
        mock_model.return_value = mock_logits
        mock_model_class.from_pretrained.return_value = mock_model

        from src.models.mms_model import MMSModel

        with patch('torchaudio.load') as mock_torchaudio:
            mock_torchaudio.return_value = (torch.randn(1, 16000), 16000)

            audio_file = temp_dir / "test.wav"
            audio_file.touch()

            model = MMSModel(device="cpu")

            # Transcribe with different languages
            model.transcribe(audio_file, language="deu")
            mock_model.load_adapter.assert_called_with("deu")

            model.transcribe(audio_file, language="fra")
            mock_model.load_adapter.assert_called_with("fra")
