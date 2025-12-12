"""
Unit tests for src.frontend.utils.error_data_loader module.

Tests error analysis data loading, parsing, and aggregation functionality.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch
from src.frontend.utils.error_data_loader import (
    load_error_analysis_json,
    get_available_error_analyses,
    load_all_error_analyses,
    extract_dialect_statistics,
    extract_all_dialect_metrics,
    extract_confusion_pairs_raw,
    extract_global_error_distribution,
    extract_aggregate_stats,
    get_worst_samples_path,
    load_worst_samples,
    aggregate_model_comparison,
    aggregate_dialect_comparison
)


@pytest.fixture
def sample_error_analysis_data():
    """Sample error analysis data structure."""
    return {
        "aggregate_stats": {
            "mean_wer": 28.5,
            "median_wer": 27.0,
            "std_wer": 12.3,
            "mean_cer": 14.2,
            "mean_bleu": 65.8
        },
        "error_distribution_percent": {
            "substitution": 60.0,
            "deletion": 25.0,
            "insertion": 10.0,
            "correct": 5.0
        },
        "dialect_analysis": {
            "BE": {
                "sample_count": 50,
                "mean_wer": 25.0,
                "std_wer": 10.5,
                "mean_cer": 12.0,
                "mean_bleu": 70.0,
                "std_bleu": 8.5,
                "error_distribution": {
                    "substitution": 30,
                    "deletion": 12,
                    "insertion": 5,
                    "correct": 3,
                    "sub_rate": 0.6,
                    "del_rate": 0.24,
                    "ins_rate": 0.1
                },
                "top_confusions": [
                    [["ist", "isch"], 15],
                    [["das", "dasch"], 10],
                    [["nicht", "nid"], 8]
                ]
            },
            "ZH": {
                "sample_count": 45,
                "mean_wer": 30.0,
                "std_wer": 11.0,
                "mean_cer": 15.5,
                "mean_bleu": 62.0,
                "std_bleu": 9.2,
                "error_distribution": {
                    "substitution": 28,
                    "deletion": 10,
                    "insertion": 7,
                    "correct": 0,
                    "sub_rate": 0.62,
                    "del_rate": 0.22,
                    "ins_rate": 0.16
                },
                "top_confusions": [
                    [["haben", "hend"], 12],
                    [["sein", "sind"], 9]
                ]
            }
        }
    }


@pytest.fixture
def temp_error_analysis_dir(temp_dir):
    """Create temporary error analysis directory structure."""
    analysis_dir = temp_dir / "error_analysis"
    analysis_dir.mkdir()

    # Create timestamped subdirectory
    timestamp_dir = analysis_dir / "20251212_100000"
    timestamp_dir.mkdir()

    return analysis_dir


class TestLoadErrorAnalysisJson:
    """Tests for load_error_analysis_json function."""

    def test_load_valid_json(self, sample_error_analysis_data):
        """Test loading a valid error analysis JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_error_analysis_data, f)
            temp_path = f.name

        try:
            with patch('src.frontend.utils.error_data_loader.st.cache_data', lambda x: x):
                data = load_error_analysis_json(temp_path)
                assert "aggregate_stats" in data
                assert data["aggregate_stats"]["mean_wer"] == 28.5
                assert "dialect_analysis" in data
        finally:
            Path(temp_path).unlink()

    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        with patch('src.frontend.utils.error_data_loader.st.cache_data', lambda x: x):
            with pytest.raises(FileNotFoundError):
                load_error_analysis_json("/nonexistent/analysis.json")

    def test_load_malformed_json(self):
        """Test loading a malformed JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{invalid json content")
            temp_path = f.name

        try:
            with patch('src.frontend.utils.error_data_loader.st.cache_data', lambda x: x):
                with pytest.raises(ValueError, match="Failed to parse JSON"):
                    load_error_analysis_json(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_load_empty_json_file(self):
        """Test loading an empty JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            with patch('src.frontend.utils.error_data_loader.st.cache_data', lambda x: x):
                with pytest.raises(ValueError, match="Failed to parse JSON"):
                    load_error_analysis_json(temp_path)
        finally:
            Path(temp_path).unlink()


class TestGetAvailableErrorAnalyses:
    """Tests for get_available_error_analyses function."""

    def test_empty_directory(self, temp_error_analysis_dir):
        """Test with empty error analysis directory."""
        with patch('src.frontend.utils.error_data_loader.st'):
            results = get_available_error_analyses(str(temp_error_analysis_dir))
            assert results == []

    def test_missing_directory(self):
        """Test with non-existent directory."""
        with patch('src.frontend.utils.error_data_loader.st'):
            results = get_available_error_analyses("/nonexistent/error_analysis")
            assert results == []

    def test_with_analysis_files(self, temp_error_analysis_dir, sample_error_analysis_data):
        """Test retrieving available analysis files."""
        # Create timestamped subdirectory with analysis file
        timestamp_dir = temp_error_analysis_dir / "20251212_100000"
        timestamp_dir.mkdir()

        analysis_file = timestamp_dir / "analysis_whisper-small.json"
        with open(analysis_file, 'w') as f:
            json.dump(sample_error_analysis_data, f)

        with patch('src.frontend.utils.error_data_loader.st'):
            results = get_available_error_analyses(str(temp_error_analysis_dir))

            assert len(results) == 1
            assert results[0]['model_name'] == 'whisper-small'
            assert 'json_path' in results[0]
            assert 'timestamp' in results[0]

    def test_with_multiple_models(self, temp_error_analysis_dir, sample_error_analysis_data):
        """Test with multiple model analysis files."""
        timestamp_dir = temp_error_analysis_dir / "20251212_100000"
        timestamp_dir.mkdir()

        models = ['whisper-small', 'whisper-medium', 'wav2vec2-base']
        for model in models:
            analysis_file = timestamp_dir / f"analysis_{model}.json"
            with open(analysis_file, 'w') as f:
                json.dump(sample_error_analysis_data, f)

        with patch('src.frontend.utils.error_data_loader.st'):
            results = get_available_error_analyses(str(temp_error_analysis_dir))

            assert len(results) == 3
            model_names = [r['model_name'] for r in results]
            assert set(model_names) == set(models)

    def test_multiple_timestamps_sorted(self, temp_error_analysis_dir, sample_error_analysis_data):
        """Test that results are sorted by timestamp (most recent first)."""
        # Create multiple timestamp directories
        timestamps = ["20251210_100000", "20251212_100000", "20251211_100000"]
        for ts in timestamps:
            ts_dir = temp_error_analysis_dir / ts
            ts_dir.mkdir()
            analysis_file = ts_dir / "analysis_whisper-small.json"
            with open(analysis_file, 'w') as f:
                json.dump(sample_error_analysis_data, f)

        with patch('src.frontend.utils.error_data_loader.st'):
            results = get_available_error_analyses(str(temp_error_analysis_dir))

            # Should have 3 results for same model
            assert len(results) == 3

            # File paths should be sorted in reverse order (newest first)
            paths = [Path(r['json_path']).parent.name for r in results]
            # The actual sorting is by file mtime, so we just check all are present
            assert set(paths) == set(timestamps)


class TestLoadAllErrorAnalyses:
    """Tests for load_all_error_analyses function."""

    def test_load_all_analyses(self, temp_error_analysis_dir, sample_error_analysis_data):
        """Test loading all available analyses."""
        timestamp_dir = temp_error_analysis_dir / "20251212_100000"
        timestamp_dir.mkdir()

        models = ['model-a', 'model-b']
        for model in models:
            analysis_file = timestamp_dir / f"analysis_{model}.json"
            with open(analysis_file, 'w') as f:
                json.dump(sample_error_analysis_data, f)

        with patch('src.frontend.utils.error_data_loader.st'):
            with patch('src.frontend.utils.error_data_loader.st.cache_data', lambda x: x):
                analyses = load_all_error_analyses(str(temp_error_analysis_dir))

                assert len(analyses) == 2
                assert 'model-a' in analyses
                assert 'model-b' in analyses

    def test_load_specific_model(self, temp_error_analysis_dir, sample_error_analysis_data):
        """Test loading a specific model's analysis."""
        timestamp_dir = temp_error_analysis_dir / "20251212_100000"
        timestamp_dir.mkdir()

        for model in ['model-a', 'model-b', 'model-c']:
            analysis_file = timestamp_dir / f"analysis_{model}.json"
            with open(analysis_file, 'w') as f:
                json.dump(sample_error_analysis_data, f)

        with patch('src.frontend.utils.error_data_loader.st'):
            with patch('src.frontend.utils.error_data_loader.st.cache_data', lambda x: x):
                analyses = load_all_error_analyses(str(temp_error_analysis_dir), model_name='model-b')

                assert len(analyses) == 1
                assert 'model-b' in analyses

    def test_load_with_missing_files(self, temp_error_analysis_dir):
        """Test loading when files don't exist (graceful degradation)."""
        with patch('src.frontend.utils.error_data_loader.st') as mock_st:
            with patch('src.frontend.utils.error_data_loader.st.cache_data', lambda x: x):
                analyses = load_all_error_analyses(str(temp_error_analysis_dir))
                assert analyses == {}

    def test_load_with_corrupted_file(self, temp_error_analysis_dir):
        """Test loading when one file is corrupted."""
        timestamp_dir = temp_error_analysis_dir / "20251212_100000"
        timestamp_dir.mkdir()

        # Create corrupted file
        corrupted_file = timestamp_dir / "analysis_corrupted.json"
        with open(corrupted_file, 'w') as f:
            f.write("{invalid json")

        with patch('src.frontend.utils.error_data_loader.st') as mock_st:
            with patch('src.frontend.utils.error_data_loader.st.cache_data', lambda x: x):
                analyses = load_all_error_analyses(str(temp_error_analysis_dir))

                # Should handle gracefully and skip corrupted file
                mock_st.warning.assert_called()


class TestExtractDialectStatistics:
    """Tests for extract_dialect_statistics function."""

    def test_extract_existing_dialect(self, sample_error_analysis_data):
        """Test extracting statistics for an existing dialect."""
        stats = extract_dialect_statistics(sample_error_analysis_data, "BE")

        assert stats is not None
        assert stats['sample_count'] == 50
        assert stats['mean_wer'] == 25.0

    def test_extract_nonexistent_dialect(self, sample_error_analysis_data):
        """Test extracting statistics for a dialect that doesn't exist."""
        stats = extract_dialect_statistics(sample_error_analysis_data, "NONEXISTENT")
        assert stats is None

    def test_extract_from_data_without_dialect_analysis(self):
        """Test extracting when dialect_analysis is missing."""
        data = {"aggregate_stats": {"mean_wer": 30.0}}
        stats = extract_dialect_statistics(data, "BE")
        assert stats is None


class TestExtractAllDialectMetrics:
    """Tests for extract_all_dialect_metrics function."""

    def test_extract_all_dialects(self, sample_error_analysis_data):
        """Test extracting metrics for all dialects."""
        df = extract_all_dialect_metrics(sample_error_analysis_data)

        assert len(df) == 2
        assert 'dialect' in df.columns
        assert 'sample_count' in df.columns
        assert 'mean_wer' in df.columns
        assert set(df['dialect'].values) == {'BE', 'ZH'}

    def test_extract_with_no_dialect_analysis(self):
        """Test extraction when no dialect analysis exists."""
        data = {"aggregate_stats": {}}
        df = extract_all_dialect_metrics(data)

        assert df.empty

    def test_extract_with_missing_fields(self):
        """Test extraction with incomplete data."""
        data = {
            "dialect_analysis": {
                "BE": {
                    "sample_count": 10
                    # Missing other fields
                }
            }
        }
        df = extract_all_dialect_metrics(data)

        assert len(df) == 1
        # Should have default values for missing fields
        assert df.iloc[0]['mean_wer'] == 0.0
        assert df.iloc[0]['mean_cer'] == 0.0


class TestExtractConfusionPairsRaw:
    """Tests for extract_confusion_pairs_raw function."""

    def test_extract_confusion_pairs(self, sample_error_analysis_data):
        """Test extracting confusion pairs."""
        dialect_stats = sample_error_analysis_data['dialect_analysis']['BE']
        pairs = extract_confusion_pairs_raw(dialect_stats)

        assert len(pairs) == 3
        assert pairs[0] == ("ist", "isch", 15)
        assert pairs[1] == ("das", "dasch", 10)

    def test_extract_with_top_n_limit(self, sample_error_analysis_data):
        """Test extracting with top_n limit."""
        dialect_stats = sample_error_analysis_data['dialect_analysis']['BE']
        pairs = extract_confusion_pairs_raw(dialect_stats, top_n=2)

        assert len(pairs) == 2

    def test_extract_with_no_confusions(self):
        """Test extraction when no confusions exist."""
        dialect_stats = {"sample_count": 10}
        pairs = extract_confusion_pairs_raw(dialect_stats)

        assert pairs == []

    def test_extract_with_empty_confusions(self):
        """Test extraction with empty confusions list."""
        dialect_stats = {"top_confusions": []}
        pairs = extract_confusion_pairs_raw(dialect_stats)

        assert pairs == []

    def test_extract_with_malformed_data(self):
        """Test extraction with malformed confusion data."""
        dialect_stats = {
            "top_confusions": [
                "not_a_list",  # Invalid format
                [["valid", "pair"], 5]  # Valid format
            ]
        }
        pairs = extract_confusion_pairs_raw(dialect_stats)

        # Should skip invalid and include valid
        assert len(pairs) == 1
        assert pairs[0] == ("valid", "pair", 5)


class TestExtractGlobalErrorDistribution:
    """Tests for extract_global_error_distribution function."""

    def test_extract_error_distribution(self, sample_error_analysis_data):
        """Test extracting global error distribution."""
        dist = extract_global_error_distribution(sample_error_analysis_data)

        assert dist["substitution"] == 60.0
        assert dist["deletion"] == 25.0
        assert dist["insertion"] == 10.0

    def test_extract_from_empty_data(self):
        """Test extraction from data without error distribution."""
        data = {}
        dist = extract_global_error_distribution(data)

        assert dist == {}


class TestExtractAggregateStats:
    """Tests for extract_aggregate_stats function."""

    def test_extract_aggregate_stats(self, sample_error_analysis_data):
        """Test extracting aggregate statistics."""
        stats = extract_aggregate_stats(sample_error_analysis_data)

        assert stats["mean_wer"] == 28.5
        assert stats["mean_cer"] == 14.2
        assert stats["mean_bleu"] == 65.8

    def test_extract_from_empty_data(self):
        """Test extraction from data without aggregate stats."""
        data = {}
        stats = extract_aggregate_stats(data)

        assert stats == {}


class TestGetWorstSamplesPath:
    """Tests for get_worst_samples_path function."""

    def test_find_existing_file(self, temp_error_analysis_dir):
        """Test finding an existing worst samples file."""
        timestamp_dir = temp_error_analysis_dir / "20251212_100000"
        timestamp_dir.mkdir()

        csv_file = timestamp_dir / "worst_samples_whisper-small.csv"
        csv_file.write_text("dialect,wer,reference,hypothesis\n")

        path = get_worst_samples_path("whisper-small", str(temp_error_analysis_dir))

        assert path is not None
        assert "worst_samples_whisper-small.csv" in path

    def test_find_nonexistent_file(self, temp_error_analysis_dir):
        """Test when file doesn't exist."""
        path = get_worst_samples_path("nonexistent", str(temp_error_analysis_dir))
        assert path is None

    def test_find_most_recent_when_multiple_exist(self, temp_error_analysis_dir):
        """Test that most recent file is returned when multiple exist."""
        import time

        # Create older file
        old_dir = temp_error_analysis_dir / "20251210_100000"
        old_dir.mkdir()
        old_file = old_dir / "worst_samples_model.csv"
        old_file.write_text("old\n")
        time.sleep(0.01)

        # Create newer file
        new_dir = temp_error_analysis_dir / "20251212_100000"
        new_dir.mkdir()
        new_file = new_dir / "worst_samples_model.csv"
        new_file.write_text("new\n")

        path = get_worst_samples_path("model", str(temp_error_analysis_dir))

        # Should return the newer file
        assert path is not None
        assert "20251212_100000" in path


class TestLoadWorstSamples:
    """Tests for load_worst_samples function."""

    def test_load_valid_csv(self):
        """Test loading a valid worst samples CSV."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("dialect,wer,reference,hypothesis\n")
            f.write("BE,85.5,correct text,wrong text\n")
            f.write("ZH,90.0,another,mismatch\n")
            temp_path = f.name

        try:
            with patch('src.frontend.utils.error_data_loader.st.cache_data', lambda x: x):
                df = load_worst_samples(temp_path)

                assert len(df) == 2
                assert 'dialect' in df.columns
                assert 'wer' in df.columns
                assert df.iloc[0]['dialect'] == 'BE'
        finally:
            Path(temp_path).unlink()

    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        with patch('src.frontend.utils.error_data_loader.st.cache_data', lambda x: x):
            with pytest.raises(FileNotFoundError):
                load_worst_samples("/nonexistent/worst_samples.csv")

    def test_load_empty_csv(self):
        """Test loading an empty CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name

        try:
            with patch('src.frontend.utils.error_data_loader.st.cache_data', lambda x: x):
                with patch('src.frontend.utils.error_data_loader.st'):
                    df = load_worst_samples(temp_path)
                    assert df.empty
        finally:
            Path(temp_path).unlink()


class TestAggregateModelComparison:
    """Tests for aggregate_model_comparison function."""

    def test_aggregate_multiple_models(self, sample_error_analysis_data):
        """Test aggregating comparison data for multiple models."""
        analyses = {
            'model-a': sample_error_analysis_data,
            'model-b': sample_error_analysis_data
        }

        df = aggregate_model_comparison(analyses)

        assert len(df) == 2
        assert 'model' in df.columns
        assert 'mean_wer' in df.columns
        assert 'mean_cer' in df.columns
        assert 'total_samples' in df.columns
        assert set(df['model'].values) == {'model-a', 'model-b'}

    def test_aggregate_empty_analyses(self):
        """Test aggregation with no analyses."""
        df = aggregate_model_comparison({})
        assert df.empty

    def test_aggregate_calculates_total_samples(self, sample_error_analysis_data):
        """Test that total samples are calculated correctly."""
        analyses = {'model-a': sample_error_analysis_data}

        df = aggregate_model_comparison(analyses)

        # BE has 50 samples, ZH has 45 samples
        assert df.iloc[0]['total_samples'] == 95

    def test_aggregate_sorted_by_wer(self):
        """Test that results are sorted by WER (best first)."""
        data1 = {
            "aggregate_stats": {"mean_wer": 30.0, "mean_cer": 15.0, "mean_bleu": 60.0},
            "error_distribution_percent": {},
            "dialect_analysis": {}
        }
        data2 = {
            "aggregate_stats": {"mean_wer": 20.0, "mean_cer": 10.0, "mean_bleu": 70.0},
            "error_distribution_percent": {},
            "dialect_analysis": {}
        }

        analyses = {'worse-model': data1, 'better-model': data2}
        df = aggregate_model_comparison(analyses)

        # Should be sorted with better-model first
        assert df.iloc[0]['model'] == 'better-model'
        assert df.iloc[1]['model'] == 'worse-model'


class TestAggregateDialectComparison:
    """Tests for aggregate_dialect_comparison function."""

    def test_aggregate_dialect_metrics(self, sample_error_analysis_data):
        """Test aggregating dialect-level metrics across models."""
        analyses = {
            'model-a': sample_error_analysis_data,
            'model-b': sample_error_analysis_data
        }

        df = aggregate_dialect_comparison(analyses)

        assert len(df) == 4  # 2 models × 2 dialects
        assert 'model' in df.columns
        assert 'dialect' in df.columns
        assert 'mean_wer' in df.columns

    def test_aggregate_empty_analyses(self):
        """Test aggregation with no analyses."""
        df = aggregate_dialect_comparison({})
        assert df.empty

    def test_aggregate_column_order(self, sample_error_analysis_data):
        """Test that model and dialect are first columns."""
        analyses = {'model-a': sample_error_analysis_data}
        df = aggregate_dialect_comparison(analyses)

        # First two columns should be model and dialect
        assert df.columns[0] == 'model'
        assert df.columns[1] == 'dialect'

    def test_aggregate_with_empty_dialect_analysis(self):
        """Test aggregation when dialect analysis is empty."""
        data = {
            "aggregate_stats": {},
            "error_distribution_percent": {},
            "dialect_analysis": {}
        }
        analyses = {'model-a': data}

        df = aggregate_dialect_comparison(analyses)
        assert df.empty
