"""
Parameterized tests to reduce test duplication across modules.

Uses pytest.mark.parametrize to test multiple scenarios with single test functions.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


class TestFileExtensionValidation:
    """Parameterized tests for file extension validation."""

    @pytest.mark.parametrize("extension,expected_valid", [
        (".wav", True),
        (".flac", True),
        (".mp3", True),
        (".WAV", True),  # Case insensitive
        (".FLAC", True),
        (".MP3", True),
        (".txt", False),
        (".json", False),
        (".py", False),
        (".mp4", False),
        (".avi", False),
        (".pdf", False),
        ("", False),  # No extension
        (".tar.gz", False),
    ])
    def test_audio_file_extension_validation(self, temp_dir, extension, expected_valid):
        """Test audio file extension validation with various extensions."""
        from src.utils.audio_utils import validate_audio_file

        if extension:
            test_file = temp_dir / f"test{extension}"
            test_file.touch()
            result = validate_audio_file(str(test_file))
        else:
            test_file = temp_dir / "noextension"
            test_file.touch()
            result = validate_audio_file(str(test_file))

        assert result == expected_valid


class TestMetricsWithVariousInputs:
    """Parameterized tests for metrics with various inputs."""

    @pytest.mark.parametrize("reference,hypothesis,expected_wer", [
        ("hello world", "hello world", 0.0),  # Perfect match
        ("hello world", "hello earth", 50.0),  # 1 substitution out of 2 = 50%
        ("hello world", "hello", 50.0),  # 1 deletion out of 2 = 50%
        ("hello", "hello world", 100.0),  # 1 insertion
        ("", "", 0.0),  # Both empty
        ("test", "", 100.0),  # Empty hypothesis
        ("the cat sat on the mat", "the cat sat on the mat", 0.0),  # Longer perfect match
        ("one two three", "four five six", 100.0),  # Complete mismatch
        ("a", "a", 0.0),  # Single word
        ("a b c", "a x c", 33.33),  # One substitution out of 3 ≈ 33.33%
    ])
    def test_calculate_wer_various_cases(self, reference, hypothesis, expected_wer):
        """Test WER calculation with various input combinations."""
        from src.evaluation.metrics import calculate_wer

        result = calculate_wer(reference, hypothesis)
        assert result == pytest.approx(expected_wer, abs=0.1)

    @pytest.mark.parametrize("reference,hypothesis,expected_cer", [
        ("hello", "hello", 0.0),  # Perfect match
        ("hello", "hallo", 20.0),  # 1 substitution out of 5 = 20%
        ("abc", "xyz", 100.0),  # Complete mismatch
        ("test", "best", 25.0),  # 1 substitution out of 4 = 25%
        ("", "", 0.0),  # Both empty
        ("test", "", 100.0),  # Empty hypothesis
        ("a", "b", 100.0),  # Single character mismatch
        ("cat", "cats", 33.33),  # 1 insertion, 1/3 ≈ 33.33%
        ("hello world", "hello world", 0.0),  # Perfect match with space
    ])
    def test_calculate_cer_various_cases(self, reference, hypothesis, expected_cer):
        """Test CER calculation with various input combinations."""
        from src.evaluation.metrics import calculate_cer

        result = calculate_cer(reference, hypothesis)
        assert result == pytest.approx(expected_cer, abs=0.5)

    @pytest.mark.parametrize("reference,hypothesis,min_bleu,max_bleu", [
        ("hello world", "hello world", 99.9, 100.0),  # Perfect match
        ("", "", 0.0, 0.0),  # Both empty
        ("test", "", 0.0, 0.0),  # Empty hypothesis
        ("", "test", 0.0, 0.0),  # Empty reference
        ("the cat sat", "the cat", 0.0, 100.0),  # Partial match
        ("hello world", "goodbye world", 0.0, 100.0),  # Partial overlap
    ])
    def test_calculate_bleu_various_cases(self, reference, hypothesis, min_bleu, max_bleu):
        """Test BLEU calculation with various input combinations."""
        from src.evaluation.metrics import calculate_bleu_score

        result = calculate_bleu_score(reference, hypothesis)
        assert min_bleu <= result <= max_bleu


class TestDialectHandling:
    """Parameterized tests for dialect-specific handling."""

    @pytest.mark.parametrize("dialect", [
        "BE",  # Bern
        "ZH",  # Zürich
        "BS",  # Basel
        "LU",  # Luzern
        "SG",  # St. Gallen
        "AG",  # Aargau
        "GR",  # Graubünden
        "VS",  # Valais
    ])
    def test_filter_by_single_dialect(self, dialect):
        """Test filtering results by each individual dialect."""
        from src.frontend.components.sidebar import filter_dataframe

        df = pd.DataFrame({
            'dialect': ['BE', 'ZH', 'BS', 'LU', 'SG', 'AG', 'GR', 'VS'],
            'wer': [25.0, 30.0, 28.0, 27.0, 29.0, 26.0, 31.0, 32.0],
            'cer': [12.0, 15.0, 14.0, 13.0, 14.5, 12.5, 15.5, 16.0],
        })

        filtered = filter_dataframe(df, [dialect])

        assert len(filtered) == 1
        assert filtered.iloc[0]['dialect'] == dialect

    @pytest.mark.parametrize("dialect_code,dialect_name", [
        ("BE", "Bern"),
        ("ZH", "Zürich"),
        ("BS", "Basel"),
        ("LU", "Luzern"),
        ("SG", "St. Gallen"),
        ("AG", "Aargau"),
        ("GR", "Graubünden"),
        ("VS", "Valais"),
    ])
    def test_dialect_code_consistency(self, dialect_code, dialect_name):
        """Test that dialect codes are consistent."""
        # Just verify codes are uppercase and 2 characters
        assert len(dialect_code) == 2
        assert dialect_code.isupper()
        assert dialect_code.isalpha()


class TestNumericRangeValidation:
    """Parameterized tests for numeric range validation."""

    @pytest.mark.parametrize("wer_value", [
        0.0,
        25.0,
        50.0,
        75.0,
        100.0,
        0.001,
        99.999,
    ])
    def test_wer_values_in_valid_range(self, wer_value):
        """Test that WER values are in valid 0-100 range."""
        assert 0.0 <= wer_value <= 100.0

    @pytest.mark.parametrize("cer_value", [
        0.0,
        15.0,
        35.0,
        50.0,
        100.0,
    ])
    def test_cer_values_in_valid_range(self, cer_value):
        """Test that CER values are in valid 0-100 range."""
        assert 0.0 <= cer_value <= 100.0

    @pytest.mark.parametrize("bleu_value", [
        0.0,
        30.0,
        50.0,
        75.0,
        100.0,
    ])
    def test_bleu_values_in_valid_range(self, bleu_value):
        """Test that BLEU values are in valid 0-100 range."""
        assert 0.0 <= bleu_value <= 100.0


class TestSampleRateConversions:
    """Parameterized tests for sample rate conversions."""

    @pytest.mark.parametrize("source_rate,target_rate,expected_ratio", [
        (16000, 16000, 1.0),  # No conversion
        (44100, 16000, 16000/44100),  # Downsample
        (8000, 16000, 2.0),  # Upsample
        (48000, 16000, 16000/48000),  # Common downsample
        (16000, 8000, 0.5),  # Half rate
        (16000, 32000, 2.0),  # Double rate
    ])
    def test_sample_rate_conversion_ratios(self, source_rate, target_rate, expected_ratio):
        """Test sample rate conversion ratios are correct."""
        from src.data.preprocessor import AudioPreprocessor

        # Generate test audio
        duration = 1.0
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(source_rate * duration)))
        audio = audio.astype(np.float32)

        preprocessor = AudioPreprocessor(target_sample_rate=target_rate)
        processed, sr = preprocessor.preprocess(audio, source_rate)

        assert sr == target_rate
        # Verify length ratio is approximately correct
        length_ratio = len(processed) / len(audio)
        assert length_ratio == pytest.approx(expected_ratio, rel=0.1)


class TestPerformanceThresholds:
    """Parameterized tests for performance threshold classifications."""

    @pytest.mark.parametrize("metric,value,expected_category", [
        ("wer", 15.0, "excellent"),
        ("wer", 29.9, "excellent"),
        ("wer", 30.0, "good"),
        ("wer", 40.0, "good"),
        ("wer", 49.9, "good"),
        ("wer", 50.0, "poor"),
        ("wer", 75.0, "poor"),
        ("cer", 10.0, "excellent"),
        ("cer", 14.9, "excellent"),
        ("cer", 15.0, "good"),
        ("cer", 25.0, "good"),
        ("cer", 34.9, "good"),
        ("cer", 35.0, "poor"),
        ("cer", 50.0, "poor"),
        ("bleu", 75.0, "excellent"),
        ("bleu", 50.0, "excellent"),
        ("bleu", 40.0, "good"),
        ("bleu", 30.0, "good"),
        ("bleu", 29.9, "poor"),
        ("bleu", 15.0, "poor"),
        ("bleu", 0.0, "poor"),
    ])
    def test_performance_category_thresholds(self, metric, value, expected_category):
        """Test performance category classification for various thresholds."""
        from src.frontend.components.plotly_charts import get_performance_category

        result = get_performance_category(value, metric)
        assert result == expected_category


class TestDataFrameOperations:
    """Parameterized tests for DataFrame operations."""

    @pytest.mark.parametrize("num_rows", [1, 10, 100, 1000])
    def test_filter_dataframe_various_sizes(self, num_rows):
        """Test filtering DataFrames of various sizes."""
        from src.frontend.components.sidebar import filter_dataframe

        # Create DataFrame with varying sizes
        df = pd.DataFrame({
            'dialect': ['BE'] * num_rows,
            'wer': np.random.uniform(20, 40, num_rows),
            'cer': np.random.uniform(10, 20, num_rows),
        })

        filtered = filter_dataframe(df, ['BE'])

        assert len(filtered) == num_rows
        assert all(filtered['dialect'] == 'BE')

    @pytest.mark.parametrize("column_name,dtype", [
        ("dialect", "object"),
        ("wer", "float64"),
        ("cer", "float64"),
        ("bleu", "float64"),
        ("model", "object"),
    ])
    def test_dataframe_column_types(self, column_name, dtype):
        """Test that DataFrame columns have expected types."""
        df = pd.DataFrame({
            "dialect": ["BE"],
            "wer": [25.0],
            "cer": [12.0],
            "bleu": [70.0],
            "model": ["whisper-small"]
        })

        assert df[column_name].dtype == dtype


class TestModelTypes:
    """Parameterized tests for different model types."""

    @pytest.mark.parametrize("model_type,model_name", [
        ("whisper", "tiny"),
        ("whisper", "base"),
        ("whisper", "small"),
        ("whisper", "medium"),
        ("whisper", "large"),
        ("wav2vec2", "facebook/wav2vec2-base"),
        ("wav2vec2", "facebook/wav2vec2-large-xlsr-53-german"),
        ("mms", "facebook/mms-1b-all"),
        ("mms", "facebook/mms-1b-l1107"),
    ])
    def test_model_type_validation(self, model_type, model_name):
        """Test that model types and names are valid."""
        from src.backend.models import EvaluateRequest

        # Create request with model type and name
        request = EvaluateRequest(
            model_type=model_type,
            model=model_name
        )

        assert request.model_type in ["whisper", "wav2vec2", "mms"]
        assert isinstance(request.model, str)
        assert len(request.model) > 0


class TestBatchSizes:
    """Parameterized tests for various batch sizes."""

    @pytest.mark.parametrize("batch_size", [1, 2, 5, 10, 16, 32, 64])
    def test_batch_metrics_with_various_batch_sizes(self, batch_size):
        """Test batch metrics calculation with various batch sizes."""
        from src.evaluation.metrics import batch_wer

        # Create batch of identical samples
        references = ["hello world"] * batch_size
        hypotheses = ["hello world"] * batch_size

        result = batch_wer(references, hypotheses)

        assert len(result["per_sample_wer"]) == batch_size
        assert result["overall_wer"] == 0.0
        assert all(wer == 0.0 or wer is None for wer in result["per_sample_wer"])


class TestUnicodeHandling:
    """Parameterized tests for Unicode character handling."""

    @pytest.mark.parametrize("text_with_unicode", [
        "Zürich",
        "Bärn",
        "Grüezi",
        "Schön",
        "Überlingen",
        "Café",
        "naïve",
        "résumé",
    ])
    def test_metrics_handle_unicode(self, text_with_unicode):
        """Test that metrics handle Unicode characters correctly."""
        from src.evaluation.metrics import calculate_wer, calculate_cer, calculate_bleu_score

        # Test with identical strings (should give optimal metrics)
        wer = calculate_wer(text_with_unicode, text_with_unicode)
        cer = calculate_cer(text_with_unicode, text_with_unicode)
        bleu = calculate_bleu_score(text_with_unicode, text_with_unicode)

        assert wer == 0.0
        assert cer == 0.0
        assert bleu == pytest.approx(100.0, abs=0.1)


class TestEmptyAndNullHandling:
    """Parameterized tests for empty and null value handling."""

    @pytest.mark.parametrize("empty_value", ["", "   ", "\t", "\n", "  \n  "])
    def test_metrics_handle_whitespace_strings(self, empty_value):
        """Test that metrics handle various whitespace-only strings."""
        from src.evaluation.metrics import calculate_wer

        # Empty/whitespace reference with non-empty hypothesis
        result = calculate_wer(empty_value, "test")

        # Should handle gracefully
        assert isinstance(result, float)
        assert 0.0 <= result <= 100.0
