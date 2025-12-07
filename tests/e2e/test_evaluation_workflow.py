"""End-to-end tests for complete evaluation workflow."""
import pytest
from pathlib import Path
from unittest.mock import patch, Mock


class TestCompleteEvaluationWorkflow:
    """Test complete evaluation workflow from data loading to results."""

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_full_workflow_mocked(self, temp_dir):
        """Test full evaluation workflow with mocked model."""
        from src.data.loader import load_swiss_german_metadata
        from src.evaluation.metrics import batch_wer, batch_cer, batch_bleu
        from src.utils.file_utils import save_results_json, save_results_csv

        # Create mock metadata
        metadata_path = temp_dir / "test.tsv"
        metadata_path.write_text(
            "client_id\tpath\tsentence\taccent\n"
            "c1\taudio1.wav\tHerzlich willkommen\tBE\n"
            "c2\taudio2.wav\tHallo zusammen\tZH\n"
            "c3\taudio3.wav\tWie geht es\tVS\n"
        )

        # Load metadata
        df = load_swiss_german_metadata(str(metadata_path))
        assert len(df) == 3

        # Simulate transcription results
        references = list(df['sentence'])
        hypotheses = [
            "Herzlich willkommen",  # Perfect
            "Hallo zusamen",         # Typo
            "Wie get es"             # Missing 'h'
        ]

        # Calculate metrics
        wer_result = batch_wer(references, hypotheses)
        cer_result = batch_cer(references, hypotheses)
        bleu_result = batch_bleu(references, hypotheses)

        # Build results
        results = {
            "overall_wer": wer_result["overall_wer"],
            "overall_cer": cer_result["overall_cer"],
            "overall_bleu": bleu_result["overall_bleu"],
            "per_dialect_wer": {"BE": 0.0, "ZH": 50.0, "VS": 33.33},
            "per_dialect_cer": {"BE": 0.0, "ZH": 7.69, "VS": 9.09},
            "per_dialect_bleu": {"BE": 100.0, "ZH": 50.0, "VS": 50.0}
        }

        # Save results
        save_results_json(
            results,
            str(temp_dir / "results.json"),
            "test-model",
            "e2e-test"
        )
        save_results_csv(results, str(temp_dir / "results.csv"))

        # Verify output files
        assert (temp_dir / "results.json").exists()
        assert (temp_dir / "results.csv").exists()

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_error_analysis_workflow(self, temp_dir):
        """Test error analysis workflow end-to-end."""
        from src.evaluation.error_analyzer import ErrorAnalyzer
        from src.evaluation.metrics import calculate_wer, calculate_cer, calculate_bleu_score

        # Create sample evaluation results
        samples = [
            {
                "reference": "das ist ein test",
                "hypothesis": "das isch ein test",
                "dialect": "BE"
            },
            {
                "reference": "hallo welt",
                "hypothesis": "hallo velo",
                "dialect": "ZH"
            },
            {
                "reference": "guten morgen",
                "hypothesis": "guten morgen",
                "dialect": "VS"
            },
        ]

        # Add metrics to samples
        for sample in samples:
            sample["wer"] = calculate_wer(sample["reference"], sample["hypothesis"])
            sample["cer"] = calculate_cer(sample["reference"], sample["hypothesis"])
            sample["bleu"] = calculate_bleu_score(sample["reference"], sample["hypothesis"])

        # Analyze errors
        analyzer = ErrorAnalyzer()

        # By dialect
        dialect_analysis = analyzer.analyze_by_dialect(samples)
        assert "BE" in dialect_analysis
        assert "ZH" in dialect_analysis
        assert "VS" in dialect_analysis

        # Aggregate stats
        stats = analyzer.calculate_aggregate_stats(samples)
        assert "mean_wer" in stats
        assert "std_wer" in stats
        assert "mean_bleu" in stats

        # High error samples
        high_error = analyzer.get_high_error_samples(samples, threshold=20.0)
        assert len(high_error) >= 1  # At least one sample with high WER

        # WER-BLEU correlation
        correlation = analyzer.analyze_wer_bleu_correlation(
            samples,
            wer_threshold=30.0,
            bleu_threshold=70.0
        )
        assert "summary" in correlation
        assert correlation["summary"]["total_samples"] == 3

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_confusion_pair_analysis(self):
        """Test confusion pair analysis workflow."""
        from src.evaluation.error_analyzer import ErrorAnalyzer

        analyzer = ErrorAnalyzer()

        # Generate alignments for multiple samples
        alignments = []
        test_pairs = [
            ("das ist ein test", "das isch ein test"),
            ("ich gehe nach hause", "ich gang nach hause"),
            ("das ist gut", "das isch gut"),  # Repeated pattern
        ]

        for ref, hyp in test_pairs:
            alignment = analyzer.get_alignment(ref, hyp)
            alignments.append(alignment)

        # Find confusion pairs
        confusion_pairs = analyzer.find_confusion_pairs(alignments)

        # Should find common patterns
        assert len(confusion_pairs) > 0

        # Check format
        for pair, count in confusion_pairs:
            assert isinstance(pair, tuple)
            assert len(pair) == 2
            assert isinstance(count, int)
            assert count > 0


class TestDataToResultsWorkflow:
    """Test workflow from data loading to final results."""

    @pytest.mark.e2e
    def test_metadata_to_evaluation_results(self, temp_dir):
        """Test converting metadata to evaluation results structure."""
        import pandas as pd
        from src.evaluation.metrics import calculate_wer, calculate_cer, calculate_bleu_score

        # Create mock evaluation data
        df = pd.DataFrame({
            "path": ["audio1.wav", "audio2.wav", "audio3.wav"],
            "sentence": ["test one", "test two", "test three"],
            "accent": ["BE", "ZH", "VS"],
            "hypothesis": ["test one", "test too", "test tree"]
        })

        # Process each row
        results = []
        for _, row in df.iterrows():
            result = {
                "audio_file": row["path"],
                "dialect": row["accent"],
                "reference": row["sentence"],
                "hypothesis": row["hypothesis"],
                "wer": calculate_wer(row["sentence"], row["hypothesis"]),
                "cer": calculate_cer(row["sentence"], row["hypothesis"]),
                "bleu": calculate_bleu_score(row["sentence"], row["hypothesis"])
            }
            results.append(result)

        # Aggregate results
        total_wer = sum(r["wer"] for r in results) / len(results)
        total_cer = sum(r["cer"] for r in results) / len(results)
        total_bleu = sum(r["bleu"] for r in results) / len(results)

        # Per-dialect aggregation
        per_dialect_wer = {}
        for dialect in df["accent"].unique():
            dialect_results = [r for r in results if r["dialect"] == dialect]
            per_dialect_wer[dialect] = sum(r["wer"] for r in dialect_results) / len(dialect_results)

        # Verify structure
        assert len(results) == 3
        assert results[0]["wer"] == 0.0  # Perfect match
        assert results[1]["wer"] > 0.0   # Has error
        assert len(per_dialect_wer) == 3


class TestCheckpointWorkflow:
    """Test checkpoint management workflow."""

    @pytest.mark.e2e
    def test_checkpoint_lifecycle(self, temp_dir):
        """Test full checkpoint lifecycle: register, query, update, remove."""
        with patch('src.utils.checkpoint_manager.MODELS_DIR', temp_dir / "models"), \
             patch('src.utils.checkpoint_manager.RESULTS_DIR', temp_dir / "results"):

            from src.utils.checkpoint_manager import CheckpointManager

            manager = CheckpointManager(task_name="e2e_test")

            # Register checkpoints
            metrics_list = [
                {"wer": 30.0, "cer": 15.0, "bleu": 65.0},
                {"wer": 25.0, "cer": 12.0, "bleu": 70.0},
                {"wer": 28.0, "cer": 14.0, "bleu": 68.0},
            ]

            for i, metrics in enumerate(metrics_list):
                ckpt_path = temp_dir / f"checkpoint_{i}.pt"
                ckpt_path.touch()
                manager.register_checkpoint(
                    checkpoint_path=ckpt_path,
                    metrics=metrics,
                    epoch=i + 1
                )

            # Query best checkpoint
            best_wer = manager.get_best_checkpoint(metric="wer", mode="min")
            assert best_wer["metrics"]["wer"] == 25.0

            best_bleu = manager.get_best_checkpoint(metric="bleu", mode="max")
            assert best_bleu["metrics"]["bleu"] == 70.0

            # List checkpoints
            all_checkpoints = manager.list_checkpoints()
            assert len(all_checkpoints) == 3

            # Update metrics
            manager.update_metrics("checkpoint_0.pt", {"wer": 22.0})
            updated = manager.get_checkpoint_entry("checkpoint_0.pt")
            assert updated["metrics"]["wer"] == 22.0

            # Now checkpoint_0 should be the best
            new_best = manager.get_best_checkpoint(metric="wer", mode="min")
            assert new_best["metrics"]["wer"] == 22.0

            # Remove checkpoint
            manager.remove_checkpoint("checkpoint_2.pt", delete_file=True)
            remaining = manager.list_checkpoints()
            assert len(remaining) == 2
            assert "checkpoint_2.pt" not in remaining
