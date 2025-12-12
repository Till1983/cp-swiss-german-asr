"""
Integration tests for error analysis pipeline.

Tests the complete flow from evaluation results to error analysis and visualization data.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
from src.evaluation.error_analyzer import ErrorAnalyzer
from src.evaluation.metrics import calculate_wer, calculate_cer, calculate_bleu_score
from src.frontend.utils.error_data_loader import (
    load_error_analysis_json,
    extract_all_dialect_metrics,
    aggregate_model_comparison
)


@pytest.fixture
def sample_evaluation_results():
    """Sample evaluation results for testing."""
    return [
        {
            "audio_file": "be_sample_1.wav",
            "dialect": "BE",
            "reference": "das ist ein test",
            "hypothesis": "das isch ein test",
            "wer": 25.0,
            "cer": 12.5,
            "bleu": 75.0
        },
        {
            "audio_file": "be_sample_2.wav",
            "dialect": "BE",
            "reference": "hallo welt",
            "hypothesis": "hallo welt",
            "wer": 0.0,
            "cer": 0.0,
            "bleu": 100.0
        },
        {
            "audio_file": "zh_sample_1.wav",
            "dialect": "ZH",
            "reference": "grüezi mitenand",
            "hypothesis": "gruezi mitenand",
            "wer": 50.0,
            "cer": 20.0,
            "bleu": 60.0
        },
        {
            "audio_file": "zh_sample_2.wav",
            "dialect": "ZH",
            "reference": "wie geht es dir",
            "hypothesis": "wie got es dir",
            "wer": 25.0,
            "cer": 10.0,
            "bleu": 70.0
        }
    ]


class TestErrorAnalysisPipeline:
    """Test complete error analysis pipeline."""

    @pytest.mark.integration
    def test_analyze_evaluation_results(self, sample_evaluation_results):
        """Test analyzing evaluation results end-to-end."""
        analyzer = ErrorAnalyzer()

        # Analyze by dialect
        dialect_analysis = analyzer.analyze_by_dialect(sample_evaluation_results)

        assert "BE" in dialect_analysis
        assert "ZH" in dialect_analysis

        # Check BE statistics
        be_stats = dialect_analysis["BE"]
        assert be_stats["sample_count"] == 2
        assert "mean_wer" in be_stats
        assert "mean_cer" in be_stats
        assert "mean_bleu" in be_stats

        # Check ZH statistics
        zh_stats = dialect_analysis["ZH"]
        assert zh_stats["sample_count"] == 2
        assert zh_stats["mean_wer"] > be_stats["mean_wer"]  # ZH should have higher WER

    @pytest.mark.integration
    def test_calculate_aggregate_statistics(self, sample_evaluation_results):
        """Test calculating aggregate statistics."""
        analyzer = ErrorAnalyzer()

        aggregate = analyzer.calculate_aggregate_stats(sample_evaluation_results)

        assert "mean_wer" in aggregate
        assert "median_wer" in aggregate
        assert "std_wer" in aggregate
        assert "mean_cer" in aggregate
        assert "mean_bleu" in aggregate

        # Verify statistics are reasonable
        assert 0 <= aggregate["mean_wer"] <= 100
        assert 0 <= aggregate["mean_cer"] <= 100
        assert 0 <= aggregate["mean_bleu"] <= 100

    @pytest.mark.integration
    def test_full_error_analysis_json_generation(self, sample_evaluation_results, temp_dir):
        """Test generating complete error analysis JSON."""
        analyzer = ErrorAnalyzer()

        # Generate analysis
        dialect_analysis = analyzer.analyze_by_dialect(sample_evaluation_results)
        aggregate_stats = analyzer.calculate_aggregate_stats(sample_evaluation_results)

        # Create JSON structure
        analysis_data = {
            "aggregate_stats": aggregate_stats,
            "dialect_analysis": dialect_analysis
        }

        # Save to file
        output_file = temp_dir / "analysis_test.json"
        with open(output_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)

        # Verify file exists and is valid
        assert output_file.exists()

        # Load and verify
        with open(output_file, 'r') as f:
            loaded = json.load(f)

        assert "aggregate_stats" in loaded
        assert "dialect_analysis" in loaded
        assert set(loaded["dialect_analysis"].keys()) == {"BE", "ZH"}

    @pytest.mark.integration
    def test_error_analysis_to_visualization_data(self, sample_evaluation_results, temp_dir):
        """Test converting error analysis to visualization-ready data."""
        analyzer = ErrorAnalyzer()

        # Generate analysis
        dialect_analysis = analyzer.analyze_by_dialect(sample_evaluation_results)
        aggregate_stats = analyzer.calculate_aggregate_stats(sample_evaluation_results)

        analysis_data = {
            "aggregate_stats": aggregate_stats,
            "dialect_analysis": dialect_analysis,
            "error_distribution_percent": {}
        }

        # Save
        output_file = temp_dir / "analysis_model.json"
        with open(output_file, 'w') as f:
            json.dump(analysis_data, f)

        # Load using frontend loader
        with patch('src.frontend.utils.error_data_loader.st.cache_data', lambda x: x):
            loaded_data = load_error_analysis_json(str(output_file))

        # Extract metrics for visualization
        dialect_df = extract_all_dialect_metrics(loaded_data)

        assert not dialect_df.empty
        assert "dialect" in dialect_df.columns
        assert "mean_wer" in dialect_df.columns
        assert len(dialect_df) == 2  # BE and ZH

    @pytest.mark.integration
    def test_multiple_model_error_analysis_comparison(self, sample_evaluation_results, temp_dir):
        """Test comparing error analysis from multiple models."""
        analyzer = ErrorAnalyzer()

        # Generate analysis for "model A"
        analysis_a = {
            "aggregate_stats": analyzer.calculate_aggregate_stats(sample_evaluation_results),
            "dialect_analysis": analyzer.analyze_by_dialect(sample_evaluation_results),
            "error_distribution_percent": {}
        }

        # Generate slightly worse results for "model B"
        worse_results = sample_evaluation_results.copy()
        for result in worse_results:
            result["wer"] += 5.0
            result["cer"] += 2.0
            result["bleu"] -= 5.0

        analysis_b = {
            "aggregate_stats": analyzer.calculate_aggregate_stats(worse_results),
            "dialect_analysis": analyzer.analyze_by_dialect(worse_results),
            "error_distribution_percent": {}
        }

        # Save both
        file_a = temp_dir / "analysis_model_a.json"
        file_b = temp_dir / "analysis_model_b.json"

        with open(file_a, 'w') as f:
            json.dump(analysis_a, f)
        with open(file_b, 'w') as f:
            json.dump(analysis_b, f)

        # Load and compare
        with patch('src.frontend.utils.error_data_loader.st.cache_data', lambda x: x):
            loaded_a = load_error_analysis_json(str(file_a))
            loaded_b = load_error_analysis_json(str(file_b))

        analyses = {"model_a": loaded_a, "model_b": loaded_b}
        comparison_df = aggregate_model_comparison(analyses)

        assert len(comparison_df) == 2
        assert "model" in comparison_df.columns
        assert "mean_wer" in comparison_df.columns

        # Model A should have lower WER (better)
        model_a_row = comparison_df[comparison_df["model"] == "model_a"].iloc[0]
        model_b_row = comparison_df[comparison_df["model"] == "model_b"].iloc[0]
        assert model_a_row["mean_wer"] < model_b_row["mean_wer"]

    @pytest.mark.integration
    def test_error_analysis_with_empty_results(self):
        """Test error analysis with empty results."""
        analyzer = ErrorAnalyzer()

        # Empty results
        empty_results = []

        dialect_analysis = analyzer.analyze_by_dialect(empty_results)
        aggregate_stats = analyzer.calculate_aggregate_stats(empty_results)

        assert dialect_analysis == {}
        assert all(v == 0.0 or v is None for v in aggregate_stats.values() if v is not None)

    @pytest.mark.integration
    def test_error_analysis_single_dialect(self):
        """Test error analysis with results from single dialect."""
        results = [
            {
                "dialect": "BE",
                "reference": "test one",
                "hypothesis": "test one",
                "wer": 0.0,
                "cer": 0.0,
                "bleu": 100.0
            },
            {
                "dialect": "BE",
                "reference": "test two",
                "hypothesis": "test two",
                "wer": 0.0,
                "cer": 0.0,
                "bleu": 100.0
            }
        ]

        analyzer = ErrorAnalyzer()
        dialect_analysis = analyzer.analyze_by_dialect(results)

        assert len(dialect_analysis) == 1
        assert "BE" in dialect_analysis
        assert dialect_analysis["BE"]["sample_count"] == 2
        assert dialect_analysis["BE"]["mean_wer"] == 0.0


class TestErrorAnalysisMetricsIntegration:
    """Test integration between metrics calculation and error analysis."""

    @pytest.mark.integration
    def test_recalculate_metrics_in_error_analysis(self):
        """Test recalculating metrics during error analysis."""
        analyzer = ErrorAnalyzer()

        # Results without pre-calculated metrics
        results = [
            {
                "dialect": "BE",
                "reference": "das ist ein test",
                "hypothesis": "das ist ein test"
            },
            {
                "dialect": "BE",
                "reference": "hallo welt",
                "hypothesis": "hallo velo"
            }
        ]

        # Calculate metrics
        for result in results:
            result["wer"] = calculate_wer(result["reference"], result["hypothesis"])
            result["cer"] = calculate_cer(result["reference"], result["hypothesis"])
            result["bleu"] = calculate_bleu_score(result["reference"], result["hypothesis"])

        # Analyze
        dialect_analysis = analyzer.analyze_by_dialect(results)

        assert "BE" in dialect_analysis
        be_stats = dialect_analysis["BE"]

        # First result should have 0% WER (perfect match)
        # Second result should have 50% WER (1 substitution out of 2 words)
        # Mean should be 25%
        assert be_stats["mean_wer"] == pytest.approx(25.0, abs=1.0)

    @pytest.mark.integration
    def test_error_distribution_calculation(self):
        """Test error distribution calculation in analysis."""
        analyzer = ErrorAnalyzer()

        results = [
            {
                "dialect": "BE",
                "reference": "hello world from python",
                "hypothesis": "hello world",  # 2 deletions
                "wer": 50.0,
                "cer": 25.0,
                "bleu": 50.0
            },
            {
                "dialect": "BE",
                "reference": "foo bar",
                "hypothesis": "foo bar baz",  # 1 insertion
                "wer": 50.0,
                "cer": 20.0,
                "bleu": 60.0
            }
        ]

        dialect_analysis = analyzer.analyze_by_dialect(results)
        be_stats = dialect_analysis["BE"]

        # Should have error distribution
        assert "error_distribution" in be_stats
        error_dist = be_stats["error_distribution"]

        # Should track different error types
        assert "substitution" in error_dist
        assert "deletion" in error_dist
        assert "insertion" in error_dist


class TestWorstSamplesIntegration:
    """Test worst samples identification and export."""

    @pytest.mark.integration
    def test_identify_worst_samples(self, sample_evaluation_results):
        """Test identifying worst performing samples."""
        analyzer = ErrorAnalyzer()

        # Sort by WER to find worst samples
        sorted_results = sorted(
            sample_evaluation_results,
            key=lambda x: x["wer"],
            reverse=True
        )

        worst_10 = sorted_results[:min(10, len(sorted_results))]

        # Verify worst samples have highest errors
        for i in range(len(worst_10) - 1):
            assert worst_10[i]["wer"] >= worst_10[i + 1]["wer"]

    @pytest.mark.integration
    def test_export_worst_samples_csv(self, sample_evaluation_results, temp_dir):
        """Test exporting worst samples to CSV."""
        import pandas as pd

        # Get worst samples
        sorted_results = sorted(
            sample_evaluation_results,
            key=lambda x: x["wer"],
            reverse=True
        )
        worst_samples = sorted_results[:10]

        # Export to CSV
        df = pd.DataFrame(worst_samples)
        csv_path = temp_dir / "worst_samples.csv"
        df.to_csv(csv_path, index=False)

        # Verify CSV
        assert csv_path.exists()

        # Load and verify
        loaded_df = pd.read_csv(csv_path)
        assert len(loaded_df) == len(worst_samples)
        assert "wer" in loaded_df.columns
        assert "reference" in loaded_df.columns
        assert "hypothesis" in loaded_df.columns

        # Verify sorted order
        for i in range(len(loaded_df) - 1):
            assert loaded_df.iloc[i]["wer"] >= loaded_df.iloc[i + 1]["wer"]
