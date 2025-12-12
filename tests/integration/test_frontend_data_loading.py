"""
Integration tests for frontend data loading from actual result files.

Tests the complete flow from saved results to frontend visualization.
"""

import pytest
import json
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import patch
from src.frontend.utils.data_loader import (
    load_data,
    get_available_results,
    combine_model_results,
    combine_multiple_models
)
from src.frontend.utils.error_data_loader import (
    load_error_analysis_json,
    get_available_error_analyses,
    load_all_error_analyses,
    get_worst_samples_path,
    load_worst_samples
)


@pytest.fixture
def mock_results_directory(temp_dir):
    """Create mock results directory structure."""
    results_dir = temp_dir / "results"
    results_dir.mkdir()

    # Create timestamped subdirectory
    timestamp_dir = results_dir / "20251212_100000"
    timestamp_dir.mkdir()

    # Create model result files
    models = ["whisper-small", "whisper-medium"]
    for model in models:
        # CSV results
        csv_data = pd.DataFrame({
            "dialect": ["BE", "ZH", "VS", "OVERALL"],
            "wer": [25.0, 30.0, 28.0, 27.5],
            "cer": [12.0, 15.0, 14.0, 13.5],
            "bleu": [70.0, 65.0, 68.0, 67.5]
        })
        csv_path = timestamp_dir / f"{model}_results.csv"
        csv_data.to_csv(csv_path, index=False)

        # JSON metadata
        json_data = {
            "model_name": model,
            "timestamp": "2025-12-12T10:00:00",
            "total_samples": 100,
            "results": {
                "overall_wer": 27.5,
                "overall_cer": 13.5,
                "overall_bleu": 67.5
            }
        }
        json_path = timestamp_dir / f"{model}_results.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f)

    return results_dir


@pytest.fixture
def mock_error_analysis_directory(temp_dir):
    """Create mock error analysis directory structure."""
    analysis_dir = temp_dir / "error_analysis"
    analysis_dir.mkdir()

    timestamp_dir = analysis_dir / "20251212_100000"
    timestamp_dir.mkdir()

    # Create error analysis files
    for model in ["whisper-small", "whisper-medium"]:
        analysis_data = {
            "aggregate_stats": {
                "mean_wer": 27.5,
                "median_wer": 27.0,
                "std_wer": 5.0,
                "mean_cer": 13.5,
                "mean_bleu": 67.5
            },
            "dialect_analysis": {
                "BE": {
                    "sample_count": 30,
                    "mean_wer": 25.0,
                    "std_wer": 4.0,
                    "mean_cer": 12.0,
                    "mean_bleu": 70.0,
                    "std_bleu": 5.0,
                    "error_distribution": {
                        "substitution": 20,
                        "deletion": 8,
                        "insertion": 5,
                        "correct": 2,
                        "sub_rate": 0.6,
                        "del_rate": 0.24,
                        "ins_rate": 0.15
                    },
                    "top_confusions": [
                        [["ist", "isch"], 10],
                        [["das", "dasch"], 8]
                    ]
                },
                "ZH": {
                    "sample_count": 35,
                    "mean_wer": 30.0,
                    "std_wer": 6.0,
                    "mean_cer": 15.0,
                    "mean_bleu": 65.0,
                    "std_bleu": 7.0,
                    "error_distribution": {
                        "substitution": 25,
                        "deletion": 10,
                        "insertion": 6,
                        "correct": 1,
                        "sub_rate": 0.6,
                        "del_rate": 0.24,
                        "ins_rate": 0.14
                    },
                    "top_confusions": [
                        [["haben", "hend"], 12]
                    ]
                }
            },
            "error_distribution_percent": {
                "substitution": 60.0,
                "deletion": 24.0,
                "insertion": 14.0,
                "correct": 2.0
            }
        }

        analysis_file = timestamp_dir / f"analysis_{model}.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)

        # Create worst samples CSV
        worst_samples = pd.DataFrame({
            "dialect": ["BE", "ZH", "BE"],
            "wer": [85.0, 90.0, 82.0],
            "cer": [65.0, 70.0, 62.0],
            "reference": ["correct text one", "correct text two", "correct text three"],
            "hypothesis": ["wrong text", "incorrect", "bad transcription"],
            "audio_file": ["be_1.wav", "zh_1.wav", "be_2.wav"]
        })
        worst_csv = timestamp_dir / f"worst_samples_{model}.csv"
        worst_samples.to_csv(worst_csv, index=False)

    return analysis_dir


class TestFrontendResultsLoading:
    """Test loading results for frontend visualization."""

    @pytest.mark.integration
    def test_discover_available_results(self, mock_results_directory):
        """Test discovering available result files."""
        results = get_available_results(str(mock_results_directory))

        assert len(results) > 0
        assert "whisper-small" in results
        assert "whisper-medium" in results

    @pytest.mark.integration
    def test_load_single_model_results(self, mock_results_directory):
        """Test loading results for a single model."""
        results = get_available_results(str(mock_results_directory))
        assert "whisper-small" in results

        # Get CSV path for whisper-small
        whisper_small_files = results["whisper-small"]
        csv_path = whisper_small_files[0]["csv_path"]

        # Load data
        with patch('src.frontend.utils.data_loader.st.cache_data', lambda x: x):
            df = load_data(csv_path)

        assert not df.empty
        assert "dialect" in df.columns
        assert "wer" in df.columns
        assert "OVERALL" in df["dialect"].values

    @pytest.mark.integration
    def test_combine_multiple_model_results(self, mock_results_directory):
        """Test combining results from multiple models."""
        available_models = get_available_results(str(mock_results_directory))

        with patch('src.frontend.utils.data_loader.st'):
            with patch('src.frontend.utils.data_loader.st.cache_data', lambda x: x):
                combined_df = combine_multiple_models(
                    ["whisper-small", "whisper-medium"],
                    available_models
                )

        assert not combined_df.empty
        assert "model" in combined_df.columns
        assert set(combined_df["model"].unique()) == {"whisper-small", "whisper-medium"}

        # Should have results for both models
        assert len(combined_df) > 4  # At least 4 dialects per model

    @pytest.mark.integration
    def test_filter_results_by_dialect(self, mock_results_directory):
        """Test filtering results by specific dialect."""
        available_models = get_available_results(str(mock_results_directory))

        with patch('src.frontend.utils.data_loader.st'):
            with patch('src.frontend.utils.data_loader.st.cache_data', lambda x: x):
                combined_df = combine_multiple_models(
                    ["whisper-small"],
                    available_models
                )

        # Filter to only BE dialect
        be_results = combined_df[combined_df["dialect"] == "BE"]

        assert len(be_results) > 0
        assert all(be_results["dialect"] == "BE")

    @pytest.mark.integration
    def test_compare_model_performance(self, mock_results_directory):
        """Test comparing performance across models."""
        available_models = get_available_results(str(mock_results_directory))

        with patch('src.frontend.utils.data_loader.st'):
            with patch('src.frontend.utils.data_loader.st.cache_data', lambda x: x):
                combined_df = combine_multiple_models(
                    ["whisper-small", "whisper-medium"],
                    available_models
                )

        # Group by model and calculate mean WER
        model_performance = combined_df.groupby("model")["wer"].mean()

        assert "whisper-small" in model_performance.index
        assert "whisper-medium" in model_performance.index
        assert all(model_performance >= 0)
        assert all(model_performance <= 100)


class TestFrontendErrorAnalysisLoading:
    """Test loading error analysis for frontend."""

    @pytest.mark.integration
    def test_discover_available_analyses(self, mock_error_analysis_directory):
        """Test discovering available error analysis files."""
        analyses = get_available_error_analyses(str(mock_error_analysis_directory))

        assert len(analyses) > 0
        model_names = [a["model_name"] for a in analyses]
        assert "whisper-small" in model_names
        assert "whisper-medium" in model_names

    @pytest.mark.integration
    def test_load_single_model_analysis(self, mock_error_analysis_directory):
        """Test loading analysis for a single model."""
        with patch('src.frontend.utils.error_data_loader.st'):
            with patch('src.frontend.utils.error_data_loader.st.cache_data', lambda x: x):
                analyses = load_all_error_analyses(
                    str(mock_error_analysis_directory),
                    model_name="whisper-small"
                )

        assert len(analyses) == 1
        assert "whisper-small" in analyses

        analysis = analyses["whisper-small"]
        assert "aggregate_stats" in analysis
        assert "dialect_analysis" in analysis

    @pytest.mark.integration
    def test_load_all_model_analyses(self, mock_error_analysis_directory):
        """Test loading analyses for all models."""
        with patch('src.frontend.utils.error_data_loader.st'):
            with patch('src.frontend.utils.error_data_loader.st.cache_data', lambda x: x):
                analyses = load_all_error_analyses(str(mock_error_analysis_directory))

        assert len(analyses) >= 2
        assert "whisper-small" in analyses
        assert "whisper-medium" in analyses

    @pytest.mark.integration
    def test_load_worst_samples_for_model(self, mock_error_analysis_directory):
        """Test loading worst samples for a model."""
        # Find worst samples file
        csv_path = get_worst_samples_path(
            "whisper-small",
            str(mock_error_analysis_directory)
        )

        assert csv_path is not None

        # Load worst samples
        with patch('src.frontend.utils.error_data_loader.st.cache_data', lambda x: x):
            df = load_worst_samples(csv_path)

        assert not df.empty
        assert "wer" in df.columns
        assert "reference" in df.columns
        assert "hypothesis" in df.columns

    @pytest.mark.integration
    def test_extract_confusion_pairs(self, mock_error_analysis_directory):
        """Test extracting confusion pairs from analysis."""
        from src.frontend.utils.error_data_loader import (
            extract_dialect_statistics,
            extract_confusion_pairs_raw
        )

        with patch('src.frontend.utils.error_data_loader.st'):
            with patch('src.frontend.utils.error_data_loader.st.cache_data', lambda x: x):
                analyses = load_all_error_analyses(
                    str(mock_error_analysis_directory),
                    model_name="whisper-small"
                )

        analysis = analyses["whisper-small"]

        # Extract BE dialect confusions
        be_stats = extract_dialect_statistics(analysis, "BE")
        assert be_stats is not None

        confusions = extract_confusion_pairs_raw(be_stats, top_n=10)
        assert len(confusions) > 0

        # Verify structure
        for ref, hyp, count in confusions:
            assert isinstance(ref, str)
            assert isinstance(hyp, str)
            assert isinstance(count, (int, float))
            assert count > 0


class TestEndToEndFrontendDataFlow:
    """Test complete data flow from results to frontend."""

    @pytest.mark.integration
    def test_complete_visualization_data_pipeline(
        self,
        mock_results_directory,
        mock_error_analysis_directory
    ):
        """Test complete pipeline from results to visualization data."""
        # Step 1: Discover available results
        available_results = get_available_results(str(mock_results_directory))
        assert len(available_results) > 0

        # Step 2: Load and combine model results
        with patch('src.frontend.utils.data_loader.st'):
            with patch('src.frontend.utils.data_loader.st.cache_data', lambda x: x):
                combined_results = combine_multiple_models(
                    ["whisper-small", "whisper-medium"],
                    available_results
                )

        assert not combined_results.empty

        # Step 3: Load error analysis
        with patch('src.frontend.utils.error_data_loader.st'):
            with patch('src.frontend.utils.error_data_loader.st.cache_data', lambda x: x):
                error_analyses = load_all_error_analyses(
                    str(mock_error_analysis_directory)
                )

        assert len(error_analyses) >= 2

        # Step 4: Extract visualization data
        from src.frontend.utils.error_data_loader import (
            aggregate_model_comparison,
            aggregate_dialect_comparison
        )

        model_comparison = aggregate_model_comparison(error_analyses)
        dialect_comparison = aggregate_dialect_comparison(error_analyses)

        assert not model_comparison.empty
        assert not dialect_comparison.empty

        # Verify data is ready for visualization
        assert "model" in model_comparison.columns
        assert "mean_wer" in model_comparison.columns
        assert "model" in dialect_comparison.columns
        assert "dialect" in dialect_comparison.columns

    @pytest.mark.integration
    def test_handle_missing_results_gracefully(self, temp_dir):
        """Test that missing results are handled gracefully."""
        # Empty results directory
        empty_dir = temp_dir / "empty_results"
        empty_dir.mkdir()

        with patch('src.frontend.utils.data_loader.st'):
            results = get_available_results(str(empty_dir))

        assert results == {}

    @pytest.mark.integration
    def test_handle_partial_results(self, mock_results_directory):
        """Test handling when only some models have results."""
        available_results = get_available_results(str(mock_results_directory))

        # Try to load a model that doesn't exist along with one that does
        with patch('src.frontend.utils.data_loader.st') as mock_st:
            with patch('src.frontend.utils.data_loader.st.cache_data', lambda x: x):
                # Should not raise, but skip the missing model
                df = combine_multiple_models(
                    ["whisper-small", "nonexistent-model"],
                    available_results
                )
                
                assert not df.empty
                assert "whisper-small" in df['model'].values
                assert "nonexistent-model" not in df['model'].values

    @pytest.mark.integration
    def test_timestamp_ordering(self, temp_dir):
        """Test that most recent results are selected when multiple exist."""
        results_dir = temp_dir / "results"
        results_dir.mkdir()

        # Create multiple timestamped directories
        timestamps = ["20251210_100000", "20251212_100000", "20251211_100000"]

        for ts in timestamps:
            ts_dir = results_dir / ts
            ts_dir.mkdir()

            # Create result file
            csv_data = pd.DataFrame({
                "dialect": ["BE"],
                "wer": [25.0],
                "cer": [12.0],
                "bleu": [70.0]
            })
            csv_path = ts_dir / "whisper-small_results.csv"
            csv_data.to_csv(csv_path, index=False)

        # Get available results
        available = get_available_results(str(results_dir))

        # Should have whisper-small with multiple evaluations
        assert "whisper-small" in available
        whisper_files = available["whisper-small"]

        # Most recent should be first (sorted by timestamp)
        assert len(whisper_files) == 3
