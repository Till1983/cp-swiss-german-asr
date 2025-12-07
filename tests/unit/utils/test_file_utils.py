"""Unit tests for file utility functions."""
import pytest
import json
import pandas as pd
from pathlib import Path
from unittest.mock import patch
from src.utils.file_utils import save_results_json, save_results_csv, ensure_log_directory


class TestSaveResultsJson:
    """Test suite for save_results_json function."""

    @pytest.fixture
    def sample_results(self):
        """Sample evaluation results for testing."""
        return {
            "overall_wer": 25.5,
            "overall_cer": 12.3,
            "overall_bleu": 65.0,
            "per_dialect_wer": {"BE": 20.0, "ZH": 30.0},
            "per_dialect_cer": {"BE": 10.0, "ZH": 15.0},
            "per_dialect_bleu": {"BE": 70.0, "ZH": 60.0}
        }

    @pytest.mark.unit
    def test_saves_json_file(self, temp_dir, sample_results):
        """Test JSON file is saved successfully."""
        output_path = temp_dir / "results.json"

        save_results_json(sample_results, str(output_path), "test-model")

        assert output_path.exists()

    @pytest.mark.unit
    def test_json_contains_expected_keys(self, temp_dir, sample_results):
        """Test saved JSON contains expected keys."""
        output_path = temp_dir / "results.json"

        save_results_json(sample_results, str(output_path), "test-model")

        with open(output_path) as f:
            data = json.load(f)

        assert 'timestamp' in data
        assert 'model_name' in data
        assert 'results' in data

    @pytest.mark.unit
    def test_json_contains_model_name(self, temp_dir, sample_results):
        """Test saved JSON contains correct model name."""
        output_path = temp_dir / "results.json"

        save_results_json(sample_results, str(output_path), "whisper-large")

        with open(output_path) as f:
            data = json.load(f)

        assert data['model_name'] == "whisper-large"

    @pytest.mark.unit
    def test_json_contains_experiment_type(self, temp_dir, sample_results):
        """Test saved JSON contains experiment type when provided."""
        output_path = temp_dir / "results.json"

        save_results_json(
            sample_results,
            str(output_path),
            "test-model",
            experiment_type="fine-tuning"
        )

        with open(output_path) as f:
            data = json.load(f)

        assert data['experiment_type'] == "fine-tuning"

    @pytest.mark.unit
    def test_json_creates_parent_directory(self, temp_dir, sample_results):
        """Test parent directory is created if it doesn't exist."""
        output_path = temp_dir / "subdir" / "nested" / "results.json"

        save_results_json(sample_results, str(output_path), "test-model")

        assert output_path.exists()

    @pytest.mark.unit
    def test_json_timestamp_format(self, temp_dir, sample_results):
        """Test timestamp has expected format."""
        output_path = temp_dir / "results.json"

        save_results_json(sample_results, str(output_path), "test-model")

        with open(output_path) as f:
            data = json.load(f)

        # Timestamp should be in YYYYMMDD_HHMMSS format
        assert len(data['timestamp']) == 15
        assert '_' in data['timestamp']

    @pytest.mark.unit
    def test_json_preserves_unicode(self, temp_dir):
        """Test Unicode characters are preserved in JSON."""
        results = {"text": "Grueezi Zuerich aeoe"}
        output_path = temp_dir / "results.json"

        save_results_json(results, str(output_path), "test-model")

        with open(output_path, encoding='utf-8') as f:
            data = json.load(f)

        assert data['results']['text'] == "Grueezi Zuerich aeoe"

    @pytest.mark.unit
    def test_json_raises_on_write_error(self, sample_results):
        """Test raises IOError on write failure."""
        # Try to write to an invalid path
        with pytest.raises(IOError, match="Failed to save results to JSON"):
            save_results_json(sample_results, "/nonexistent/dir/results.json", "test")


class TestSaveResultsCsv:
    """Test suite for save_results_csv function."""

    @pytest.fixture
    def valid_results(self):
        """Valid results with all required keys."""
        return {
            "overall_wer": 25.5,
            "overall_cer": 12.3,
            "overall_bleu": 65.0,
            "per_dialect_wer": {"BE": 20.0, "ZH": 30.0},
            "per_dialect_cer": {"BE": 10.0, "ZH": 15.0},
            "per_dialect_bleu": {"BE": 70.0, "ZH": 60.0}
        }

    @pytest.mark.unit
    def test_saves_csv_file(self, temp_dir, valid_results):
        """Test CSV file is saved successfully."""
        output_path = temp_dir / "results.csv"

        save_results_csv(valid_results, str(output_path))

        assert output_path.exists()

    @pytest.mark.unit
    def test_csv_contains_dialect_columns(self, temp_dir, valid_results):
        """Test CSV contains dialect and metric columns."""
        output_path = temp_dir / "results.csv"

        save_results_csv(valid_results, str(output_path))

        df = pd.read_csv(output_path)
        assert 'dialect' in df.columns
        assert 'wer' in df.columns
        assert 'cer' in df.columns
        assert 'bleu' in df.columns

    @pytest.mark.unit
    def test_csv_contains_overall_row(self, temp_dir, valid_results):
        """Test CSV contains OVERALL summary row."""
        output_path = temp_dir / "results.csv"

        save_results_csv(valid_results, str(output_path))

        df = pd.read_csv(output_path)
        assert 'OVERALL' in df['dialect'].values

    @pytest.mark.unit
    def test_csv_overall_values_correct(self, temp_dir, valid_results):
        """Test OVERALL row has correct values."""
        output_path = temp_dir / "results.csv"

        save_results_csv(valid_results, str(output_path))

        df = pd.read_csv(output_path)
        overall_row = df[df['dialect'] == 'OVERALL'].iloc[0]

        assert overall_row['wer'] == 25.5
        assert overall_row['cer'] == 12.3
        assert overall_row['bleu'] == 65.0

    @pytest.mark.unit
    def test_csv_creates_parent_directory(self, temp_dir, valid_results):
        """Test parent directory is created if it doesn't exist."""
        output_path = temp_dir / "subdir" / "results.csv"

        save_results_csv(valid_results, str(output_path))

        assert output_path.exists()

    @pytest.mark.unit
    def test_csv_raises_on_missing_keys(self, temp_dir):
        """Test raises ValueError when required keys are missing."""
        incomplete_results = {
            "overall_wer": 25.5,
            # Missing other required keys
        }

        with pytest.raises(IOError):  # Wrapped as IOError
            save_results_csv(incomplete_results, str(temp_dir / "results.csv"))

    @pytest.mark.unit
    def test_csv_dialect_rows_sorted(self, temp_dir, valid_results):
        """Test dialect rows are sorted alphabetically."""
        output_path = temp_dir / "results.csv"

        save_results_csv(valid_results, str(output_path))

        df = pd.read_csv(output_path)
        dialect_rows = df[df['dialect'] != 'OVERALL']['dialect'].tolist()

        assert dialect_rows == sorted(dialect_rows)


class TestEnsureLogDirectory:
    """Test suite for ensure_log_directory function."""

    @pytest.mark.unit
    def test_creates_directory(self, temp_dir):
        """Test directory is created if it doesn't exist."""
        log_path = temp_dir / "logs" / "app.log"

        ensure_log_directory(str(log_path))

        assert (temp_dir / "logs").exists()

    @pytest.mark.unit
    def test_handles_existing_directory(self, temp_dir):
        """Test doesn't raise error if directory exists."""
        log_dir = temp_dir / "logs"
        log_dir.mkdir()
        log_path = log_dir / "app.log"

        # Should not raise
        ensure_log_directory(str(log_path))

        assert log_dir.exists()

    @pytest.mark.unit
    def test_creates_nested_directories(self, temp_dir):
        """Test creates nested directories."""
        log_path = temp_dir / "a" / "b" / "c" / "app.log"

        ensure_log_directory(str(log_path))

        assert (temp_dir / "a" / "b" / "c").exists()

    @pytest.mark.unit
    def test_handles_path_object(self, temp_dir):
        """Test handles string path correctly."""
        log_path = str(temp_dir / "logs" / "app.log")

        ensure_log_directory(log_path)

        assert (temp_dir / "logs").exists()
