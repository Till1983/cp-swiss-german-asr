"""Unit tests for SeamlessM4T model wrapper."""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np


class TestSeamlessM4TModelInit:
    """Test SeamlessM4TModel initialization."""

    @pytest.mark.unit
    @patch('src.models.seamless_m4t_model.AutoProcessor')
    @patch('src.models.seamless_m4t_model.SeamlessM4Tv2ForSpeechToText')
    def test_initialization_default_model(self, mock_model_class, mock_processor_class):
        """Test model initializes with default model name."""
        mock_processor_class.from_pretrained.return_value = Mock()
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        from src.models.seamless_m4t_model import SeamlessM4TModel
        model = SeamlessM4TModel()

        assert model.model_name == "facebook/seamless-m4t-v2-large"
        mock_model.to.assert_called()
        mock_model.eval.assert_called()

    @pytest.mark.unit
    @patch('src.models.seamless_m4t_model.AutoProcessor')
    @patch('src.models.seamless_m4t_model.SeamlessM4Tv2ForSpeechToText')
    def test_initialization_with_custom_model(self, mock_model_class, mock_processor_class):
        """Test model initializes with custom model name."""
        mock_processor_class.from_pretrained.return_value = Mock()
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        from src.models.seamless_m4t_model import SeamlessM4TModel
        model = SeamlessM4TModel(model_name="facebook/seamless-m4t-medium")

        assert model.model_name == "facebook/seamless-m4t-medium"

    @pytest.mark.unit
    @patch('src.models.seamless_m4t_model.AutoProcessor')
    @patch('src.models.seamless_m4t_model.SeamlessM4Tv2ForSpeechToText')
    def test_initialization_with_device(self, mock_model_class, mock_processor_class):
        """Test model initializes with specified device."""
        mock_processor_class.from_pretrained.return_value = Mock()
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        from src.models.seamless_m4t_model import SeamlessM4TModel
        model = SeamlessM4TModel(device="cpu")

        assert model.device == "cpu"

    @pytest.mark.unit
    @patch('src.models.seamless_m4t_model.AutoProcessor')
    @patch('src.models.seamless_m4t_model.SeamlessM4Tv2ForSpeechToText')
    def test_initialization_half_precision_on_cpu(self, mock_model_class, mock_processor_class):
        """Test model uses float32 on CPU even with half precision requested."""
        mock_processor_class.from_pretrained.return_value = Mock()
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        from src.models.seamless_m4t_model import SeamlessM4TModel
        model = SeamlessM4TModel(device="cpu", use_half_precision=True)

        # CPU should use float32 regardless of half precision setting
        assert model.dtype == torch.float32

    @pytest.mark.unit
    @patch('src.models.seamless_m4t_model.AutoProcessor')
    @patch('src.models.seamless_m4t_model.SeamlessM4Tv2ForSpeechToText')
    def test_model_set_to_eval_mode(self, mock_model_class, mock_processor_class):
        """Test model is set to evaluation mode."""
        mock_processor_class.from_pretrained.return_value = Mock()
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        from src.models.seamless_m4t_model import SeamlessM4TModel
        model = SeamlessM4TModel()

        mock_model.eval.assert_called_once()

    @pytest.mark.unit
    @patch('src.models.seamless_m4t_model.AutoProcessor')
    @patch('src.models.seamless_m4t_model.SeamlessM4Tv2ForSpeechToText')
    def test_processor_loaded(self, mock_model_class, mock_processor_class):
        """Test processor is loaded from pretrained."""
        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = Mock()

        from src.models.seamless_m4t_model import SeamlessM4TModel
        model = SeamlessM4TModel()

        assert model.processor == mock_processor


class TestSeamlessM4TLanguageCodeMapping:
    """Test language code resolution."""

    @pytest.mark.unit
    @patch('src.models.seamless_m4t_model.AutoProcessor')
    @patch('src.models.seamless_m4t_model.SeamlessM4Tv2ForSpeechToText')
    def test_language_code_de_to_deu(self, mock_model_class, mock_processor_class):
        """Test 'de' is resolved to 'deu'."""
        mock_processor_class.from_pretrained.return_value = Mock()
        mock_model_class.from_pretrained.return_value = Mock()

        from src.models.seamless_m4t_model import SeamlessM4TModel
        model = SeamlessM4TModel(device="cpu")

        assert model._resolve_language_code("de") == "deu"

    @pytest.mark.unit
    @patch('src.models.seamless_m4t_model.AutoProcessor')
    @patch('src.models.seamless_m4t_model.SeamlessM4Tv2ForSpeechToText')
    def test_language_code_deu_unchanged(self, mock_model_class, mock_processor_class):
        """Test 'deu' remains 'deu'."""
        mock_processor_class.from_pretrained.return_value = Mock()
        mock_model_class.from_pretrained.return_value = Mock()

        from src.models.seamless_m4t_model import SeamlessM4TModel
        model = SeamlessM4TModel(device="cpu")

        assert model._resolve_language_code("deu") == "deu"

    @pytest.mark.unit
    @patch('src.models.seamless_m4t_model.AutoProcessor')
    @patch('src.models.seamless_m4t_model.SeamlessM4Tv2ForSpeechToText')
    def test_language_code_german_to_deu(self, mock_model_class, mock_processor_class):
        """Test 'german' is resolved to 'deu'."""
        mock_processor_class.from_pretrained.return_value = Mock()
        mock_model_class.from_pretrained.return_value = Mock()

        from src.models.seamless_m4t_model import SeamlessM4TModel
        model = SeamlessM4TModel(device="cpu")

        assert model._resolve_language_code("german") == "deu"

    @pytest.mark.unit
    @patch('src.models.seamless_m4t_model.AutoProcessor')
    @patch('src.models.seamless_m4t_model.SeamlessM4Tv2ForSpeechToText')
    def test_language_code_swiss_german_to_deu(self, mock_model_class, mock_processor_class):
        """Test 'gsw' (Swiss German) is mapped to 'deu'."""
        mock_processor_class.from_pretrained.return_value = Mock()
        mock_model_class.from_pretrained.return_value = Mock()

        from src.models.seamless_m4t_model import SeamlessM4TModel
        model = SeamlessM4TModel(device="cpu")

        assert model._resolve_language_code("gsw") == "deu"

    @pytest.mark.unit
    @patch('src.models.seamless_m4t_model.AutoProcessor')
    @patch('src.models.seamless_m4t_model.SeamlessM4Tv2ForSpeechToText')
    def test_language_code_unknown_passthrough(self, mock_model_class, mock_processor_class):
        """Test unknown language codes are passed through as-is."""
        mock_processor_class.from_pretrained.return_value = Mock()
        mock_model_class.from_pretrained.return_value = Mock()

        from src.models.seamless_m4t_model import SeamlessM4TModel
        model = SeamlessM4TModel(device="cpu")

        assert model._resolve_language_code("fra") == "fra"


class TestSeamlessM4TModelTranscribe:
    """Test SeamlessM4TModel transcribe method."""

    @pytest.fixture
    def mock_seamless_model(self):
        """Create a mocked SeamlessM4TModel."""
        with patch('src.models.seamless_m4t_model.AutoProcessor') as mock_processor_class, \
             patch('src.models.seamless_m4t_model.SeamlessM4Tv2ForSpeechToText') as mock_model_class:

            # Setup processor mock
            mock_processor = Mock()
            mock_processor.return_value = {"input_features": torch.randn(1, 80, 3000)}
            mock_processor.decode.return_value = "transcribed text"
            mock_processor_class.from_pretrained.return_value = mock_processor

            # Setup model mock
            mock_model = Mock()
            mock_model.generate.return_value = [torch.tensor([1, 2, 3, 4, 5])]
            mock_model_class.from_pretrained.return_value = mock_model

            from src.models.seamless_m4t_model import SeamlessM4TModel
            model = SeamlessM4TModel(device="cpu")

            yield model

    @pytest.mark.unit
    def test_transcribe_returns_dict(self, mock_seamless_model, sample_audio_path):
        """Test transcribe returns dictionary with 'text' key."""
        result = mock_seamless_model.transcribe(sample_audio_path)

        assert isinstance(result, dict)
        assert "text" in result

    @pytest.mark.unit
    def test_transcribe_default_language(self, mock_seamless_model, sample_audio_path):
        """Test transcribe uses default language (German)."""
        mock_seamless_model.transcribe(sample_audio_path)

        # Check that model.generate was called with tgt_lang="deu"
        mock_seamless_model.model.generate.assert_called()
        call_kwargs = mock_seamless_model.model.generate.call_args[1]
        assert call_kwargs.get("tgt_lang") == "deu"

    @pytest.mark.unit
    def test_transcribe_custom_language(self, mock_seamless_model, sample_audio_path):
        """Test transcribe with custom language code."""
        mock_seamless_model.transcribe(sample_audio_path, language="fra")

        call_kwargs = mock_seamless_model.model.generate.call_args[1]
        assert call_kwargs.get("tgt_lang") == "fra"

    @pytest.mark.unit
    def test_transcribe_file_not_found(self, mock_seamless_model):
        """Test transcribe raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            mock_seamless_model.transcribe(Path("/nonexistent/audio.flac"))

    @pytest.mark.unit
    def test_transcribe_generate_speech_not_passed(self, mock_seamless_model, sample_audio_path):
        """Test transcribe does NOT pass generate_speech parameter (it's not supported by the model)."""
        mock_seamless_model.transcribe(sample_audio_path)

        call_kwargs = mock_seamless_model.model.generate.call_args[1]
        # generate_speech should not be in the kwargs since the model doesn't support it
        assert "generate_speech" not in call_kwargs


class TestSeamlessM4TModelAudioProcessing:
    """Test audio loading and preprocessing."""

    @pytest.mark.unit
    @patch('src.models.seamless_m4t_model.AutoProcessor')
    @patch('src.models.seamless_m4t_model.SeamlessM4Tv2ForSpeechToText')
    @patch('src.models.seamless_m4t_model.torchaudio')
    def test_audio_resampling(self, mock_torchaudio, mock_model_class, mock_processor_class, sample_audio_path):
        """Test audio is resampled to 16kHz if needed."""
        # Setup mocks
        mock_processor = Mock()
        mock_processor.return_value = {"input_features": torch.randn(1, 80, 3000)}
        mock_processor.decode.return_value = "transcribed text"
        mock_processor_class.from_pretrained.return_value = mock_processor

        mock_model = Mock()
        mock_model.generate.return_value = [torch.tensor([1, 2, 3, 4, 5])]
        mock_model_class.from_pretrained.return_value = mock_model

        # Return audio at 44100 Hz (needs resampling)
        mock_torchaudio.load.return_value = (torch.randn(1, 44100), 44100)
        mock_resampler = Mock(return_value=torch.randn(1, 16000))
        mock_torchaudio.transforms.Resample.return_value = mock_resampler

        from src.models.seamless_m4t_model import SeamlessM4TModel
        model = SeamlessM4TModel(device="cpu")
        model.transcribe(sample_audio_path)

        # Verify resampler was created for 44100 -> 16000
        mock_torchaudio.transforms.Resample.assert_called_with(44100, 16000)
        mock_resampler.assert_called()

    @pytest.mark.unit
    @patch('src.models.seamless_m4t_model.AutoProcessor')
    @patch('src.models.seamless_m4t_model.SeamlessM4Tv2ForSpeechToText')
    @patch('src.models.seamless_m4t_model.torchaudio')
    def test_stereo_to_mono_conversion(self, mock_torchaudio, mock_model_class, mock_processor_class, sample_audio_path):
        """Test stereo audio is converted to mono."""
        # Setup mocks
        mock_processor = Mock()
        mock_processor.return_value = {"input_features": torch.randn(1, 80, 3000)}
        mock_processor.decode.return_value = "transcribed text"
        mock_processor_class.from_pretrained.return_value = mock_processor

        mock_model = Mock()
        mock_model.generate.return_value = [torch.tensor([1, 2, 3, 4, 5])]
        mock_model_class.from_pretrained.return_value = mock_model

        # Return stereo audio at 16kHz
        stereo_waveform = torch.randn(2, 16000)  # 2 channels
        mock_torchaudio.load.return_value = (stereo_waveform, 16000)

        from src.models.seamless_m4t_model import SeamlessM4TModel
        model = SeamlessM4TModel(device="cpu")
        model.transcribe(sample_audio_path)

        # Processor should receive mono audio (1D or 1 channel)
        call_args = mock_processor.call_args
        audios_arg = call_args[1].get("audios") if call_args[1] else call_args[0][0]
        # After torch.mean(dim=0) and squeeze(), should be 1D
        assert len(audios_arg.shape) == 1 or audios_arg.shape[0] == 1

    @pytest.mark.unit
    @patch('src.models.seamless_m4t_model.AutoProcessor')
    @patch('src.models.seamless_m4t_model.SeamlessM4Tv2ForSpeechToText')
    @patch('src.models.seamless_m4t_model.torchaudio')
    def test_empty_audio_raises_error(self, mock_torchaudio, mock_model_class, mock_processor_class, sample_audio_path):
        """Test empty audio raises ValueError."""
        mock_processor_class.from_pretrained.return_value = Mock()
        mock_model_class.from_pretrained.return_value = Mock()

        # Return empty waveform
        mock_torchaudio.load.return_value = (torch.tensor([]), 16000)

        from src.models.seamless_m4t_model import SeamlessM4TModel
        model = SeamlessM4TModel(device="cpu")

        with pytest.raises(ValueError, match="empty"):
            model.transcribe(sample_audio_path)
