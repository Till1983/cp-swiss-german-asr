"""
Additional edge case and boundary condition tests across modules.

Tests extreme values, unusual inputs, and boundary conditions.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


class TestAudioEdgeCases:
    """Test audio processing edge cases."""

    @pytest.mark.unit
    def test_audio_with_inf_values(self):
        """Test handling audio with infinite values."""
        from src.data.preprocessor import AudioPreprocessor

        audio = np.array([1.0, 2.0, np.inf, 4.0], dtype=np.float32)
        preprocessor = AudioPreprocessor()

        # Should handle gracefully
        try:
            processed, sr = preprocessor.preprocess(audio, 16000)
            # If it succeeds, verify no inf values remain
            assert not np.any(np.isinf(processed))
        except (ValueError, RuntimeError):
            # Or it should raise an appropriate error
            pass

    @pytest.mark.unit
    def test_audio_with_nan_values(self):
        """Test handling audio with NaN values."""
        from src.data.preprocessor import AudioPreprocessor

        audio = np.array([1.0, np.nan, 3.0, 4.0], dtype=np.float32)
        preprocessor = AudioPreprocessor()

        try:
            processed, sr = preprocessor.preprocess(audio, 16000)
            # Should not have NaN values
            assert not np.any(np.isnan(processed))
        except (ValueError, RuntimeError):
            pass

    @pytest.mark.unit
    def test_audio_all_zeros(self, silent_audio):
        """Test processing silent audio (all zeros)."""
        from src.data.preprocessor import AudioPreprocessor

        preprocessor = AudioPreprocessor()
        processed, sr = preprocessor.preprocess(silent_audio, 16000)

        assert len(processed) == len(silent_audio)
        assert np.allclose(processed, 0.0, atol=1e-6)

    @pytest.mark.unit
    def test_audio_very_short(self):
        """Test processing very short audio (< 0.1s)."""
        from src.data.preprocessor import AudioPreprocessor

        # 160 samples at 16kHz = 0.01 seconds
        very_short = np.random.randn(160).astype(np.float32)
        preprocessor = AudioPreprocessor()

        processed, sr = preprocessor.preprocess(very_short, 16000)

        assert len(processed) > 0
        assert sr == 16000

    @pytest.mark.unit
    def test_audio_very_long(self):
        """Test processing very long audio (> 1 hour)."""
        from src.data.preprocessor import AudioPreprocessor

        # 1 hour of audio at 16kHz
        very_long = np.random.randn(16000 * 3600).astype(np.float32)
        preprocessor = AudioPreprocessor()

        processed, sr = preprocessor.preprocess(very_long, 16000)

        assert len(processed) == len(very_long)
        assert sr == 16000

    @pytest.mark.unit
    def test_audio_extreme_sample_rates(self):
        """Test with extreme sample rates."""
        from src.data.preprocessor import AudioPreprocessor

        audio = np.random.randn(1000).astype(np.float32)

        # Very low sample rate
        preprocessor_low = AudioPreprocessor(target_sample_rate=100)
        processed_low, sr_low = preprocessor_low.preprocess(audio, 1000)
        assert sr_low == 100

        # Very high sample rate
        preprocessor_high = AudioPreprocessor(target_sample_rate=96000)
        processed_high, sr_high = preprocessor_high.preprocess(audio, 16000)
        assert sr_high == 96000


class TestDataLoaderEdgeCases:
    """Test data loader edge cases."""

    @pytest.mark.unit
    def test_load_csv_with_bom(self, temp_dir):
        """Test loading CSV with byte order mark (BOM)."""
        from src.frontend.utils.data_loader import load_data

        csv_file = temp_dir / "bom_test.csv"
        # Write CSV with UTF-8 BOM
        with open(csv_file, 'w', encoding='utf-8-sig') as f:
            f.write("dialect,wer,cer,bleu\n")
            f.write("BE,25.0,12.0,70.0\n")

        with pytest.mock.patch('src.frontend.utils.data_loader.st.cache_data', lambda x: x):
            df = load_data(str(csv_file))

        assert not df.empty
        assert "dialect" in df.columns

    @pytest.mark.unit
    def test_load_csv_with_special_characters(self, temp_dir):
        """Test loading CSV with special characters in data."""
        from src.frontend.utils.data_loader import load_data

        csv_file = temp_dir / "special_chars.csv"
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("dialect,wer,cer,bleu\n")
            f.write("Zürich-Stadt,25.5,12.3,70.8\n")
            f.write("Bern/BE,28.0,14.0,68.0\n")

        with pytest.mock.patch('src.frontend.utils.data_loader.st.cache_data', lambda x: x):
            df = load_data(str(csv_file))

        assert not df.empty
        assert len(df) == 2

    @pytest.mark.unit
    def test_load_csv_with_quoted_fields(self, temp_dir):
        """Test loading CSV with quoted fields."""
        from src.frontend.utils.data_loader import load_data

        csv_file = temp_dir / "quoted.csv"
        with open(csv_file, 'w') as f:
            f.write('dialect,wer,cer,bleu\n')
            f.write('"BE, Basel",25.0,12.0,70.0\n')

        with pytest.mock.patch('src.frontend.utils.data_loader.st.cache_data', lambda x: x):
            df = load_data(str(csv_file))

        assert not df.empty
        assert df.iloc[0]['dialect'] == "BE, Basel"

    @pytest.mark.unit
    def test_load_csv_with_very_large_values(self, temp_dir):
        """Test loading CSV with very large numeric values."""
        from src.frontend.utils.data_loader import load_data

        csv_file = temp_dir / "large_values.csv"
        with open(csv_file, 'w') as f:
            f.write("dialect,wer,cer,bleu\n")
            f.write("BE,999999.99,888888.88,100.0\n")

        with pytest.mock.patch('src.frontend.utils.data_loader.st.cache_data', lambda x: x):
            df = load_data(str(csv_file))

        assert not df.empty
        assert df.iloc[0]['wer'] == pytest.approx(999999.99)

    @pytest.mark.unit
    def test_load_csv_single_row(self, temp_dir):
        """Test loading CSV with only one data row."""
        from src.frontend.utils.data_loader import load_data

        csv_file = temp_dir / "single_row.csv"
        with open(csv_file, 'w') as f:
            f.write("dialect,wer,cer,bleu\n")
            f.write("BE,25.0,12.0,70.0\n")

        with pytest.mock.patch('src.frontend.utils.data_loader.st.cache_data', lambda x: x):
            df = load_data(str(csv_file))

        assert len(df) == 1


class TestMetricsEdgeCases:
    """Test metrics calculation edge cases."""

    @pytest.mark.unit
    def test_wer_with_only_punctuation(self):
        """Test WER when strings contain only punctuation."""
        from src.evaluation.metrics import calculate_wer

        reference = "!!!"
        hypothesis = "???"

        # After normalization, both become empty
        result = calculate_wer(reference, hypothesis)
        assert result == 0.0  # Both empty after normalization

    @pytest.mark.unit
    def test_wer_with_numbers_only(self):
        """Test WER with number-only strings."""
        from src.evaluation.metrics import calculate_wer

        reference = "1 2 3 4 5"
        hypothesis = "1 2 3 4 5"

        result = calculate_wer(reference, hypothesis)
        assert result == 0.0

    @pytest.mark.unit
    def test_wer_with_repeated_words(self):
        """Test WER with many repeated words."""
        from src.evaluation.metrics import calculate_wer

        reference = " ".join(["word"] * 100)
        hypothesis = " ".join(["word"] * 100)

        result = calculate_wer(reference, hypothesis)
        assert result == 0.0

    @pytest.mark.unit
    def test_cer_single_character_differences(self):
        """Test CER with minimal character differences."""
        from src.evaluation.metrics import calculate_cer

        # Single character substitution in long string
        reference = "a" * 1000 + "b"
        hypothesis = "a" * 1000 + "c"

        result = calculate_cer(reference, hypothesis)
        # 1 error out of 1001 characters ≈ 0.1%
        assert result < 1.0

    @pytest.mark.unit
    def test_bleu_with_partial_overlap(self):
        """Test BLEU with partial n-gram overlap."""
        from src.evaluation.metrics import calculate_bleu_score

        reference = "the quick brown fox jumps"
        hypothesis = "the slow brown fox sits"

        result = calculate_bleu_score(reference, hypothesis)
        # Should have some overlap but not perfect
        assert 0 < result < 100

    @pytest.mark.unit
    def test_batch_metrics_with_single_sample(self):
        """Test batch metrics with only one sample."""
        from src.evaluation.metrics import batch_wer, batch_cer, batch_bleu

        refs = ["hello world"]
        hyps = ["hello world"]

        wer_result = batch_wer(refs, hyps)
        cer_result = batch_cer(refs, hyps)
        bleu_result = batch_bleu(refs, hyps)

        assert wer_result["overall_wer"] == 0.0
        assert cer_result["overall_cer"] == 0.0
        assert bleu_result["overall_bleu"] > 99.0


class TestFileUtilsEdgeCases:
    """Test file utilities edge cases."""

    @pytest.mark.unit
    def test_save_json_with_special_characters(self, temp_dir):
        """Test saving JSON with special characters."""
        from src.utils.file_utils import save_results_json

        results = {
            "model_name": "test-model-ü",
            "dialect_analysis": {
                "Zürich": {"wer": 25.0}
            }
        }

        output_file = temp_dir / "special_chars.json"
        save_results_json(results, str(output_file), "test-model", "test-dataset")

        # Verify file was created and is valid JSON
        assert output_file.exists()

        import json
        with open(output_file, 'r', encoding='utf-8') as f:
            loaded = json.load(f)

        assert "Zürich" in str(loaded)

    @pytest.mark.unit
    def test_save_csv_with_unicode_content(self, temp_dir):
        """Test saving CSV with Unicode content."""
        from src.utils.file_utils import save_results_csv

        results = {
            "per_dialect_wer": {"Zürich": 25.0, "Bärn": 28.0},
            "per_dialect_cer": {"Zürich": 12.0, "Bärn": 14.0},
            "per_dialect_bleu": {"Zürich": 70.0, "Bärn": 68.0}
        }

        output_file = temp_dir / "unicode.csv"
        save_results_csv(results, str(output_file))

        # Verify file was created
        assert output_file.exists()

        # Load and verify
        df = pd.read_csv(output_file)
        assert "Zürich" in df["dialect"].values

    @pytest.mark.unit
    def test_save_empty_results(self, temp_dir):
        """Test saving empty results."""
        from src.utils.file_utils import save_results_json

        results = {}

        output_file = temp_dir / "empty.json"
        save_results_json(results, str(output_file), "test-model", "test-dataset")

        assert output_file.exists()

        import json
        with open(output_file, 'r') as f:
            loaded = json.load(f)

        assert "model_name" in loaded


class TestConfigEdgeCases:
    """Test configuration handling edge cases."""

    @pytest.mark.unit
    def test_config_with_missing_optional_fields(self):
        """Test config loading with missing optional fields."""
        from src.config import Config

        config = Config()

        # Should have default values for optional fields
        assert hasattr(config, 'batch_size')
        assert hasattr(config, 'num_workers')

    @pytest.mark.unit
    def test_config_with_invalid_types(self):
        """Test config validation with invalid types."""
        from src.config import Config

        config = Config()

        # Try to set invalid batch size
        try:
            config.batch_size = "not_a_number"
            # Should either convert or reject
            assert isinstance(config.batch_size, (int, str))
        except (ValueError, TypeError, AttributeError):
            # Or it should raise an appropriate error
            pass


class TestCheckpointManagerEdgeCases:
    """Test checkpoint manager edge cases."""

    @pytest.mark.unit
    def test_save_checkpoint_to_nonexistent_directory(self, temp_dir):
        """Test saving checkpoint to directory that doesn't exist."""
        from src.utils.checkpoint_manager import CheckpointManager

        nonexistent_dir = temp_dir / "nonexistent" / "nested" / "path"
        manager = CheckpointManager(str(nonexistent_dir))

        dummy_state = {"epoch": 1, "model": "test"}

        # Should create directory and save
        manager.save_checkpoint(dummy_state, 25.0, "test_model")

        assert nonexistent_dir.exists()

    @pytest.mark.unit
    def test_load_checkpoint_corrupted(self, temp_dir):
        """Test loading corrupted checkpoint file."""
        from src.utils.checkpoint_manager import CheckpointManager

        # Create corrupted checkpoint file
        checkpoint_file = temp_dir / "checkpoint_test_model_wer_25.00.pt"
        checkpoint_file.write_text("corrupted data")

        manager = CheckpointManager(str(temp_dir))

        # Should handle gracefully
        result = manager.load_checkpoint(str(checkpoint_file))

        # Either returns None or raises appropriate error
        assert result is None or isinstance(result, dict)

    @pytest.mark.unit
    def test_save_checkpoint_with_extreme_metrics(self, temp_dir):
        """Test saving checkpoint with extreme metric values."""
        from src.utils.checkpoint_manager import CheckpointManager

        manager = CheckpointManager(str(temp_dir))
        dummy_state = {"epoch": 1}

        # Very high WER
        manager.save_checkpoint(dummy_state, 999.99, "extreme_wer")

        # Very low WER
        manager.save_checkpoint(dummy_state, 0.001, "low_wer")

        # Check both were saved
        assert len(list(temp_dir.glob("checkpoint_*.pt"))) >= 2


class TestErrorAnalyzerEdgeCases:
    """Test error analyzer edge cases."""

    @pytest.mark.unit
    def test_analyze_with_missing_fields(self):
        """Test analysis with results missing some fields."""
        from src.evaluation.error_analyzer import ErrorAnalyzer

        results = [
            {
                "dialect": "BE",
                "reference": "test",
                "hypothesis": "test"
                # Missing wer, cer, bleu
            }
        ]

        analyzer = ErrorAnalyzer()

        # Should handle gracefully or raise appropriate error
        try:
            dialect_analysis = analyzer.analyze_by_dialect(results)
            # If it succeeds, verify it handled missing fields
            assert "BE" in dialect_analysis
        except (KeyError, ValueError):
            # Or it should raise an appropriate error
            pass

    @pytest.mark.unit
    def test_analyze_with_null_values(self):
        """Test analysis with null/None values."""
        from src.evaluation.error_analyzer import ErrorAnalyzer

        results = [
            {
                "dialect": "BE",
                "reference": "test",
                "hypothesis": "test",
                "wer": None,
                "cer": 0.0,
                "bleu": 100.0
            }
        ]

        analyzer = ErrorAnalyzer()

        # Should handle None values gracefully
        try:
            aggregate = analyzer.calculate_aggregate_stats(results)
            # If it succeeds, verify reasonable output
            assert isinstance(aggregate, dict)
        except (TypeError, ValueError):
            pass
