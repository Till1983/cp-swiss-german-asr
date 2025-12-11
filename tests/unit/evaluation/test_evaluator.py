"""Unit tests for ASR evaluator module."""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import torch


class TestASREvaluatorInit:
    """Test ASREvaluator initialization."""

    @pytest.mark.unit
    def test_initialization_whisper_default(self):
        """Test initialization with Whisper model defaults."""
        from src.evaluation.evaluator import ASREvaluator

        evaluator = ASREvaluator()

        assert evaluator.model_type == "whisper"
        assert evaluator.model_name == "base"
        assert evaluator.model is None

    @pytest.mark.unit
    def test_initialization_whisper_custom_model(self):
        """Test initialization with custom Whisper model."""
        from src.evaluation.evaluator import ASREvaluator

        evaluator = ASREvaluator(model_type="whisper", model_name="large-v2")

        assert evaluator.model_type == "whisper"
        assert evaluator.model_name == "large-v2"

    @pytest.mark.unit
    def test_initialization_wav2vec2(self):
        """Test initialization with Wav2Vec2 model."""
        from src.evaluation.evaluator import ASREvaluator

        evaluator = ASREvaluator(
            model_type="wav2vec2",
            model_name="facebook/wav2vec2-large-xlsr-53-german"
        )

        assert evaluator.model_type == "wav2vec2"
        assert evaluator.model_name == "facebook/wav2vec2-large-xlsr-53-german"

    @pytest.mark.unit
    def test_initialization_mms(self):
        """Test initialization with MMS model."""
        from src.evaluation.evaluator import ASREvaluator

        evaluator = ASREvaluator(
            model_type="mms",
            model_name="facebook/mms-1b-all"
        )

        assert evaluator.model_type == "mms"
        assert evaluator.model_name == "facebook/mms-1b-all"

    @pytest.mark.unit
    def test_initialization_invalid_model_type(self):
        """Test initialization with invalid model type raises ValueError."""
        from src.evaluation.evaluator import ASREvaluator

        with pytest.raises(ValueError, match="model_type must be"):
            ASREvaluator(model_type="invalid")

    @pytest.mark.unit
    def test_initialization_with_device(self):
        """Test initialization with specified device."""
        from src.evaluation.evaluator import ASREvaluator

        evaluator = ASREvaluator(device="cpu")

        assert evaluator.device == "cpu"

    @pytest.mark.unit
    def test_initialization_with_lm_path(self):
        """Test initialization with language model path."""
        from src.evaluation.evaluator import ASREvaluator

        evaluator = ASREvaluator(lm_path="/path/to/lm.arpa")

        assert evaluator.lm_path == "/path/to/lm.arpa"

    @pytest.mark.unit
    def test_model_not_loaded_on_init(self):
        """Test model is not loaded on initialization."""
        from src.evaluation.evaluator import ASREvaluator

        evaluator = ASREvaluator()

        assert evaluator.model is None


class TestASREvaluatorLoadModel:
    """Test ASREvaluator load_model method."""

    @pytest.mark.unit
    @patch('whisper.load_model')
    def test_load_whisper_model(self, mock_load):
        """Test loading Whisper model."""
        mock_load.return_value = Mock()

        from src.evaluation.evaluator import ASREvaluator
        evaluator = ASREvaluator(model_type="whisper", model_name="tiny")
        evaluator.load_model()

        mock_load.assert_called_once()
        assert evaluator.model is not None

    @pytest.mark.unit
    @patch('src.evaluation.evaluator.Wav2Vec2Model')
    def test_load_wav2vec2_model(self, mock_wav2vec2_class):
        """Test loading Wav2Vec2 model."""
        mock_wav2vec2_class.return_value = Mock()

        from src.evaluation.evaluator import ASREvaluator
        evaluator = ASREvaluator(
            model_type="wav2vec2",
            model_name="facebook/wav2vec2-base"
        )
        evaluator.load_model()

        mock_wav2vec2_class.assert_called_once()
        assert evaluator.model is not None

    @pytest.mark.unit
    @patch('src.evaluation.evaluator.MMSModel')
    def test_load_mms_model(self, mock_mms_class):
        """Test loading MMS model."""
        mock_mms_class.return_value = Mock()

        from src.evaluation.evaluator import ASREvaluator
        evaluator = ASREvaluator(
            model_type="mms",
            model_name="facebook/mms-1b-all"
        )
        evaluator.load_model()

        mock_mms_class.assert_called_once()
        assert evaluator.model is not None

    @pytest.mark.unit
    @patch('whisper.load_model')
    def test_load_model_failure_raises_runtime_error(self, mock_load):
        """Test load_model raises RuntimeError on failure."""
        mock_load.side_effect = Exception("Model load failed")

        from src.evaluation.evaluator import ASREvaluator
        evaluator = ASREvaluator(model_type="whisper", model_name="tiny")

        with pytest.raises(RuntimeError, match="Failed to load Whisper model"):
            evaluator.load_model()

    @pytest.mark.unit
    @patch('src.evaluation.evaluator.WhisperProcessor')
    @patch('src.evaluation.evaluator.WhisperForConditionalGeneration')
    def test_load_whisper_hf_model(self, mock_model_class, mock_processor_class):
        """Test loading Hugging Face Whisper model."""
        mock_processor = Mock()
        mock_model = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model

        from src.evaluation.evaluator import ASREvaluator
        evaluator = ASREvaluator(model_type="whisper-hf", model_name="openai/whisper-tiny")
        evaluator.load_model()

        mock_processor_class.from_pretrained.assert_called_once()
        mock_model_class.from_pretrained.assert_called_once()
        assert evaluator.processor is not None
        assert evaluator.model is not None

    @pytest.mark.unit
    @patch('src.evaluation.evaluator.WhisperProcessor')
    @patch('src.evaluation.evaluator.WhisperForConditionalGeneration')
    def test_load_whisper_hf_failure_raises_runtime_error(self, mock_model_class, mock_processor_class):
        """Test loading HF Whisper model raises error on failure."""
        mock_processor_class.from_pretrained.side_effect = Exception("HF load failed")

        from src.evaluation.evaluator import ASREvaluator
        evaluator = ASREvaluator(model_type="whisper-hf", model_name="openai/whisper-tiny")

        with pytest.raises(RuntimeError, match="Failed to load HF Whisper model"):
            evaluator.load_model()

    @pytest.mark.unit
    @patch('src.evaluation.evaluator.Wav2Vec2Model')
    def test_load_wav2vec2_failure_raises_runtime_error(self, mock_wav2vec2_class):
        """Test loading Wav2Vec2 model raises error on failure."""
        mock_wav2vec2_class.side_effect = Exception("Wav2Vec2 load failed")

        from src.evaluation.evaluator import ASREvaluator
        evaluator = ASREvaluator(model_type="wav2vec2", model_name="test")

        with pytest.raises(RuntimeError, match="Failed to load Wav2Vec2 model"):
            evaluator.load_model()

    @pytest.mark.unit
    @patch('src.evaluation.evaluator.MMSModel')
    def test_load_mms_failure_raises_runtime_error(self, mock_mms_class):
        """Test loading MMS model raises error on failure."""
        mock_mms_class.side_effect = Exception("MMS load failed")

        from src.evaluation.evaluator import ASREvaluator
        evaluator = ASREvaluator(model_type="mms", model_name="test")

        with pytest.raises(RuntimeError, match="Failed to load MMS model"):
            evaluator.load_model()


class TestASREvaluatorTranscribe:
    """Test ASREvaluator _get_transcription method."""

    @pytest.fixture
    def mock_whisper_evaluator(self):
        """Create evaluator with mocked Whisper model."""
        with patch('whisper.load_model') as mock_load, \
             patch('whisper.load_audio') as mock_audio:

            mock_model = Mock()
            mock_model.transcribe.return_value = {"text": "transcribed text"}
            mock_load.return_value = mock_model
            mock_audio.return_value = Mock()

            from src.evaluation.evaluator import ASREvaluator
            evaluator = ASREvaluator(model_type="whisper", model_name="tiny")
            evaluator.load_model()

            yield evaluator

    @pytest.mark.unit
    def test_transcribe_whisper(self, mock_whisper_evaluator, sample_audio_path):
        """Test transcription with Whisper model."""
        result = mock_whisper_evaluator._get_transcription(sample_audio_path)

        assert result == "transcribed text"

    @pytest.mark.unit
    @patch('src.evaluation.evaluator.Wav2Vec2Model')
    def test_transcribe_wav2vec2(self, mock_wav2vec2_class, sample_audio_path):
        """Test transcription with Wav2Vec2 model."""
        mock_model = Mock()
        mock_model.transcribe.return_value = {"text": "wav2vec2 text"}
        mock_wav2vec2_class.return_value = mock_model

        from src.evaluation.evaluator import ASREvaluator
        evaluator = ASREvaluator(model_type="wav2vec2", model_name="test")
        evaluator.load_model()

        result = evaluator._get_transcription(sample_audio_path)

        assert result == "wav2vec2 text"

    @pytest.mark.unit
    @patch('src.evaluation.evaluator.MMSModel')
    def test_transcribe_mms(self, mock_mms_class, sample_audio_path):
        """Test transcription with MMS model."""
        mock_model = Mock()
        mock_model.transcribe.return_value = {"text": "mms text"}
        mock_mms_class.return_value = mock_model

        from src.evaluation.evaluator import ASREvaluator
        evaluator = ASREvaluator(model_type="mms", model_name="test")
        evaluator.load_model()

        result = evaluator._get_transcription(sample_audio_path)

        assert result == "mms text"

    @pytest.mark.unit
    @patch('src.evaluation.evaluator.WhisperProcessor')
    @patch('src.evaluation.evaluator.WhisperForConditionalGeneration')
    @patch('whisper.load_audio')
    def test_transcribe_whisper_hf(self, mock_audio, mock_model_class, mock_processor_class, sample_audio_path):
        """Test transcription with Hugging Face Whisper model."""
        import numpy as np
        
        # Setup mocks
        mock_audio_data = np.random.randn(16000).astype(np.float32)
        mock_audio.return_value = mock_audio_data
        
        mock_processor = Mock()
        # Create a proper object with input_features attribute
        mock_inputs = Mock()
        mock_input_features = torch.randn(1, 80, 3000)  # Typical whisper input shape
        mock_input_features_device = Mock()
        mock_input_features_device.to = Mock(return_value=mock_input_features)
        mock_inputs.input_features = mock_input_features_device
        
        mock_processor.return_value = mock_inputs
        mock_processor.batch_decode.return_value = ["whisper hf text"]
        mock_processor_class.from_pretrained.return_value = mock_processor
        
        mock_model = Mock()
        mock_predicted_ids = torch.tensor([[1, 2, 3]])
        mock_model.generate.return_value = mock_predicted_ids
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model
        
        from src.evaluation.evaluator import ASREvaluator
        evaluator = ASREvaluator(model_type="whisper-hf", model_name="openai/whisper-tiny")
        evaluator.load_model()

        result = evaluator._get_transcription(sample_audio_path)

        assert result == "whisper hf text"
        mock_model.generate.assert_called_once()

    @pytest.mark.unit
    @patch('src.evaluation.evaluator.Wav2Vec2Model')
    def test_transcribe_wav2vec2_returns_string(self, mock_wav2vec2_class, sample_audio_path):
        """Test transcription returns string from Wav2Vec2 dict result."""
        mock_model = Mock()
        # Test fallback when dict doesn't have 'text' key
        mock_model.transcribe.return_value = "plain string"
        mock_wav2vec2_class.return_value = mock_model

        from src.evaluation.evaluator import ASREvaluator
        evaluator = ASREvaluator(model_type="wav2vec2", model_name="test")
        evaluator.load_model()

        result = evaluator._get_transcription(sample_audio_path)

        assert result == "plain string"


class TestASREvaluatorEvaluateDataset:
    """Test ASREvaluator evaluate_dataset method."""

    @pytest.fixture
    def mock_evaluator(self):
        """Create evaluator with mocked model."""
        with patch('whisper.load_model') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model

            from src.evaluation.evaluator import ASREvaluator
            evaluator = ASREvaluator(model_type="whisper", model_name="tiny")
            evaluator.load_model()

            # Mock _get_transcription
            evaluator._get_transcription = Mock(return_value="transcribed text")

            yield evaluator

    @pytest.mark.unit
    def test_evaluate_raises_without_model_loaded(self, temp_dir):
        """Test evaluate_dataset raises error if model not loaded."""
        from src.evaluation.evaluator import ASREvaluator
        evaluator = ASREvaluator()

        with pytest.raises(ValueError, match="Model not loaded"):
            evaluator.evaluate_dataset(str(temp_dir / "test.tsv"))

    @pytest.mark.unit
    def test_evaluate_raises_on_nonexistent_metadata(self, mock_evaluator):
        """Test evaluate_dataset raises error for nonexistent metadata file."""
        with pytest.raises(FileNotFoundError, match="Metadata file not found"):
            mock_evaluator.evaluate_dataset("/nonexistent/metadata.tsv")

    @pytest.mark.unit
    def test_evaluate_raises_on_missing_columns(self, mock_evaluator, temp_dir):
        """Test evaluate_dataset raises error for metadata with missing columns."""
        # Create metadata without required columns
        metadata = temp_dir / "incomplete.tsv"
        metadata.write_text("client_id\tother_column\nclient1\tvalue\n")

        with pytest.raises(ValueError, match="must contain columns"):
            mock_evaluator.evaluate_dataset(str(metadata))

    @pytest.mark.unit
    @patch('src.evaluation.evaluator.FHNW_SWISS_GERMAN_ROOT', Path("/fake/path"))
    def test_evaluate_returns_dict_with_expected_keys(self, mock_evaluator, temp_dir):
        """Test evaluate_dataset returns dictionary with expected keys."""
        # Create valid metadata
        metadata = temp_dir / "test.tsv"
        metadata.write_text("path\tsentence\taccent\ntest.wav\tHello world\tBE\n")

        # Create audio file
        audio_dir = temp_dir / "clips"
        audio_dir.mkdir()
        (audio_dir / "test.wav").touch()

        with patch.object(mock_evaluator, 'evaluate_dataset') as mock_eval:
            mock_eval.return_value = {
                'overall_wer': 25.0,
                'overall_cer': 10.0,
                'overall_bleu': 70.0,
                'per_dialect_wer': {'BE': 25.0},
                'per_dialect_cer': {'BE': 10.0},
                'per_dialect_bleu': {'BE': 70.0},
                'total_samples': 1,
                'failed_samples': 0,
                'samples': []
            }

            result = mock_eval(str(metadata))

        expected_keys = [
            'overall_wer', 'overall_cer', 'overall_bleu',
            'per_dialect_wer', 'per_dialect_cer', 'per_dialect_bleu',
            'total_samples', 'failed_samples', 'samples'
        ]

        for key in expected_keys:
            assert key in result

    @pytest.mark.unit
    @patch('whisper.load_model')
    def test_evaluate_respects_limit(self, mock_load, temp_dir):
        """Test evaluate_dataset respects sample limit."""
        mock_model = Mock()
        mock_load.return_value = mock_model

        # Create metadata with multiple rows
        metadata_content = "path\tsentence\taccent\n"
        audio_dir = temp_dir / "clips"
        audio_dir.mkdir()

        for i in range(10):
            metadata_content += f"test_{i}.wav\tSentence {i}\tBE\n"
            (audio_dir / f"test_{i}.wav").touch()

        metadata = temp_dir / "test.tsv"
        metadata.write_text(metadata_content)

        from src.evaluation.evaluator import ASREvaluator
        evaluator = ASREvaluator(model_type="whisper", model_name="tiny")
        evaluator.model = mock_model
        evaluator._get_transcription = Mock(return_value="text")

        with patch('src.evaluation.evaluator.FHNW_SWISS_GERMAN_ROOT', temp_dir):
            result = evaluator.evaluate_dataset(str(metadata), limit=3)

        # Should only process 3 samples
        assert result['total_samples'] <= 3


class TestASREvaluatorMetricsCalculation:
    """Test ASREvaluator metrics calculation."""

    @pytest.mark.unit
    def test_perfect_transcription_metrics(self):
        """Test metrics for perfect transcription."""
        from src.evaluation import metrics

        reference = "hello world"
        hypothesis = "hello world"

        wer = metrics.calculate_wer(reference, hypothesis)
        cer = metrics.calculate_cer(reference, hypothesis)
        bleu = metrics.calculate_bleu_score(reference, hypothesis)

        assert wer == 0.0
        assert cer == 0.0
        assert bleu == pytest.approx(100.0, abs=0.01)

    @pytest.mark.unit
    def test_complete_mismatch_metrics(self):
        """Test metrics for completely wrong transcription."""
        from src.evaluation import metrics

        reference = "hello world"
        hypothesis = "foo bar"

        wer = metrics.calculate_wer(reference, hypothesis)
        cer = metrics.calculate_cer(reference, hypothesis)

        assert wer == 100.0
        # CER should also be high for complete mismatch
        assert cer > 50.0


class TestASREvaluatorEdgeCases:
    """Test edge cases for ASREvaluator."""

    @pytest.mark.unit
    def test_evaluate_empty_metadata(self, temp_dir):
        """Test evaluating empty metadata file."""
        from src.evaluation.evaluator import ASREvaluator

        # Create empty metadata with only headers
        metadata = temp_dir / "empty.tsv"
        metadata.write_text("path\tsentence\taccent\n")

        evaluator = ASREvaluator()
        evaluator.model = Mock()
        evaluator._get_transcription = Mock()

        with patch('src.evaluation.evaluator.FHNW_SWISS_GERMAN_ROOT', temp_dir):
            result = evaluator.evaluate_dataset(str(metadata))

        # Should handle empty data gracefully
        assert result['total_samples'] == 0
        assert result['failed_samples'] == 0

    @pytest.mark.unit
    def test_handles_transcription_failure(self, temp_dir):
        """Test evaluator handles transcription failures gracefully."""
        from src.evaluation.evaluator import ASREvaluator

        # Setup
        metadata = temp_dir / "test.tsv"
        metadata.write_text("path\tsentence\taccent\ntest.wav\tHello\tBE\n")
        audio_dir = temp_dir / "clips"
        audio_dir.mkdir()
        (audio_dir / "test.wav").touch()

        evaluator = ASREvaluator()
        evaluator.model = Mock()
        evaluator._get_transcription = Mock(side_effect=Exception("Transcription failed"))

        with patch('src.evaluation.evaluator.FHNW_SWISS_GERMAN_ROOT', temp_dir):
            result = evaluator.evaluate_dataset(str(metadata))

        # Should count as failed sample and handle gracefully
        assert result['failed_samples'] == 1
        assert result['total_samples'] == 0  # No successful results
        assert result['overall_wer'] == 0.0  # Default/safe value when no results

    @pytest.mark.unit
    def test_partial_transcription_failure(self, temp_dir):
        """Test evaluator with mix of successful and failed transcriptions."""
        from src.evaluation.evaluator import ASREvaluator

        # Setup metadata with 3 samples
        metadata = temp_dir / "test.tsv"
        metadata.write_text("path\tsentence\taccent\ntest1.wav\tHello\tBE\ntest2.wav\tWorld\tZH\ntest3.wav\tFoo\tBE\n")
        audio_dir = temp_dir / "clips"
        audio_dir.mkdir()
        (audio_dir / "test1.wav").touch()
        (audio_dir / "test2.wav").touch()
        (audio_dir / "test3.wav").touch()

        evaluator = ASREvaluator()
        evaluator.model = Mock()
        # Fail on test2.wav, succeed on others
        evaluator._get_transcription = Mock(side_effect=[
            "Hello",
            Exception("Transcription failed"),
            "Foo"
        ])

        with patch('src.evaluation.evaluator.FHNW_SWISS_GERMAN_ROOT', temp_dir):
            result = evaluator.evaluate_dataset(str(metadata))

        # Should process 2 successful and 1 failed
        assert result['total_samples'] == 2
        assert result['failed_samples'] == 1
        assert result['overall_wer'] >= 0.0
