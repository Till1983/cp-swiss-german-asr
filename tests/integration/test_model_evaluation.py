"""Integration tests for model evaluation pipeline."""
import pytest
from unittest.mock import patch, Mock
from pathlib import Path


class TestModelEvaluationIntegration:
    """Test model evaluation integration."""

    @pytest.mark.integration
    def test_metrics_module_integration(self):
        """Test metrics module functions work together."""
        from src.evaluation.metrics import (
            calculate_wer, calculate_cer, calculate_bleu_score,
            batch_wer, batch_cer, batch_bleu
        )

        references = [
            "Herzlich willkommen in Bern",
            "Hallo zusammen",
            "Wie geht es dir"
        ]
        hypotheses = [
            "Herzlich willkommen in Bern",  # Perfect match
            "Hallo zusamen",  # Typo
            "Wie get es dir"  # Missing 'h'
        ]

        # Test batch metrics
        wer_result = batch_wer(references, hypotheses)
        cer_result = batch_cer(references, hypotheses)
        bleu_result = batch_bleu(references, hypotheses)

        # Validate structure
        assert "overall_wer" in wer_result
        assert "per_sample_wer" in wer_result
        assert len(wer_result["per_sample_wer"]) == 3

        assert "overall_cer" in cer_result
        assert "overall_bleu" in bleu_result

        # First sample should be perfect
        assert wer_result["per_sample_wer"][0] == 0.0

    @pytest.mark.integration
    def test_error_analyzer_with_metrics(self):
        """Test error analyzer works with metrics results."""
        from src.evaluation.error_analyzer import ErrorAnalyzer
        from src.evaluation.metrics import calculate_wer, calculate_cer, calculate_bleu_score

        analyzer = ErrorAnalyzer()

        # Create sample results
        results = []
        test_cases = [
            ("hallo welt", "hallo welt", "BE"),
            ("guten morgen", "gueten morgen", "ZH"),
            ("auf wiedersehen", "auf wiedersehen", "VS"),
        ]

        for ref, hyp, dialect in test_cases:
            results.append({
                "reference": ref,
                "hypothesis": hyp,
                "dialect": dialect,
                "wer": calculate_wer(ref, hyp),
                "cer": calculate_cer(ref, hyp),
                "bleu": calculate_bleu_score(ref, hyp)
            })

        # Test error analyzer functions
        dialect_analysis = analyzer.analyze_by_dialect(results)
        aggregate_stats = analyzer.calculate_aggregate_stats(results)

        assert "BE" in dialect_analysis
        assert "ZH" in dialect_analysis
        assert "VS" in dialect_analysis

        assert "mean_wer" in aggregate_stats
        assert "mean_cer" in aggregate_stats
        assert "mean_bleu" in aggregate_stats

    @pytest.mark.integration
    def test_error_analyzer_alignment(self):
        """Test error analyzer alignment functionality."""
        from src.evaluation.error_analyzer import ErrorAnalyzer

        analyzer = ErrorAnalyzer()

        reference = "das ist ein test"
        hypothesis = "das isch ein tost"

        alignment = analyzer.get_alignment(reference, hypothesis)
        categories = analyzer.categorize_errors(alignment)

        # Should have correct matches and errors
        assert categories["total_errors"] > 0
        assert categories["correct"] > 0

        # Test readable format
        formatted = analyzer.format_alignment_readable(alignment)
        assert "REF:" in formatted
        assert "HYP:" in formatted


class TestEvaluatorIntegration:
    """Test evaluator integration with mocked models."""

    @pytest.mark.integration
    @patch('src.evaluation.evaluator.whisper')
    def test_whisper_evaluator_flow(self, mock_whisper, temp_dir):
        """Test Whisper evaluator flow with mocked whisper."""
        from src.evaluation.evaluator import ASREvaluator

        # Setup mocks
        mock_model = Mock()
        mock_model.transcribe.return_value = {"text": "transcribed text"}
        mock_whisper.load_model.return_value = mock_model
        mock_whisper.load_audio.return_value = Mock()

        # Create evaluator
        evaluator = ASREvaluator(model_type="whisper", model_name="tiny")
        evaluator.load_model()

        assert evaluator.model is not None

    @pytest.mark.integration
    @patch('src.evaluation.evaluator.Wav2Vec2Model')
    def test_wav2vec2_evaluator_flow(self, mock_wav2vec2_class, temp_dir):
        """Test Wav2Vec2 evaluator flow."""
        from src.evaluation.evaluator import ASREvaluator

        mock_model = Mock()
        mock_model.transcribe.return_value = {"text": "transcribed text"}
        mock_wav2vec2_class.return_value = mock_model

        evaluator = ASREvaluator(
            model_type="wav2vec2",
            model_name="facebook/wav2vec2-base"
        )
        evaluator.load_model()

        assert evaluator.model is not None

    @pytest.mark.integration
    @patch('src.evaluation.evaluator.MMSModel')
    def test_mms_evaluator_flow(self, mock_mms_class, temp_dir):
        """Test MMS evaluator flow."""
        from src.evaluation.evaluator import ASREvaluator

        mock_model = Mock()
        mock_model.transcribe.return_value = {"text": "transcribed text"}
        mock_mms_class.return_value = mock_model

        evaluator = ASREvaluator(
            model_type="mms",
            model_name="facebook/mms-1b-all"
        )
        evaluator.load_model()

        assert evaluator.model is not None


class TestResultsSavingIntegration:
    """Test results saving integration."""

    @pytest.mark.integration
    def test_save_results_json_and_csv(self, temp_dir):
        """Test saving results in both JSON and CSV formats."""
        from src.utils.file_utils import save_results_json, save_results_csv

        results = {
            "overall_wer": 30.0,
            "overall_cer": 15.0,
            "overall_bleu": 65.0,
            "per_dialect_wer": {"BE": 28.0, "ZH": 32.0},
            "per_dialect_cer": {"BE": 13.0, "ZH": 17.0},
            "per_dialect_bleu": {"BE": 68.0, "ZH": 62.0}
        }

        # Save JSON
        json_path = temp_dir / "results.json"
        save_results_json(results, str(json_path), "test-model", "integration-test")

        # Save CSV
        csv_path = temp_dir / "results.csv"
        save_results_csv(results, str(csv_path))

        # Verify files exist
        assert json_path.exists()
        assert csv_path.exists()

        # Load and verify JSON
        import json
        with open(json_path) as f:
            loaded_json = json.load(f)

        assert loaded_json["model_name"] == "test-model"
        assert loaded_json["results"]["overall_wer"] == 30.0

        # Load and verify CSV
        import pandas as pd
        df = pd.read_csv(csv_path)

        assert "OVERALL" in df["dialect"].values
        assert len(df) == 3  # 2 dialects + OVERALL
