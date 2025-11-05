import pytest
from src.evaluation.metrics import calculate_wer, calculate_cer, batch_wer
from src.evaluation.evaluator import ASREvaluator
from src.evaluation.metrics import (
    calculate_wer, 
    calculate_cer, 
    calculate_bleu_score,
    batch_wer, 
    batch_cer,
    batch_bleu
)


class TestCalculateWER:
    def test_calculate_wer_exact_match(self):
        """WER should be 0 for identical strings"""
        reference = "hello world"
        hypothesis = "hello world"
        assert calculate_wer(reference, hypothesis) == 0.0
    
    def test_calculate_wer_complete_mismatch(self):
        """WER should be 100 for completely different strings"""
        reference = "hello world"
        hypothesis = "foo bar"
        assert calculate_wer(reference, hypothesis) == 100.0
    
    def test_calculate_wer_partial_match(self):
        """Test known WER value for partial match"""
        reference = "hello world from python"
        hypothesis = "hello world"
        # 2 deletions out of 4 words = 50% WER
        assert calculate_wer(reference, hypothesis) == 50.0
    
    def test_calculate_wer_case_insensitive(self):
        """WER should be case insensitive"""
        reference = "Hello World"
        hypothesis = "hello world"
        assert calculate_wer(reference, hypothesis) == 0.0
    
    def test_calculate_wer_empty_reference(self):
        """WER should handle empty reference"""
        reference = ""
        hypothesis = "hello"
        assert calculate_wer(reference, hypothesis) == 100.0
    
    def test_calculate_wer_empty_hypothesis(self):
        """WER should handle empty hypothesis"""
        reference = "hello world"
        hypothesis = ""
        assert calculate_wer(reference, hypothesis) == 100.0
    
    def test_calculate_wer_both_empty(self):
        """WER should be 0 when both are empty"""
        reference = ""
        hypothesis = ""
        assert calculate_wer(reference, hypothesis) == 0.0


class TestCalculateCER:
    def test_calculate_cer_basic(self):
        """Test CER calculation"""
        reference = "hello"
        hypothesis = "hallo"
        # 1 substitution out of 5 characters = 20% CER
        assert calculate_cer(reference, hypothesis) == 20.0
    
    def test_calculate_cer_exact_match(self):
        """CER should be 0 for identical strings"""
        reference = "hello world"
        hypothesis = "hello world"
        assert calculate_cer(reference, hypothesis) == 0.0
    
    def test_calculate_cer_complete_mismatch(self):
        """CER should be 100 for completely different strings of same length"""
        reference = "abc"
        hypothesis = "xyz"
        assert calculate_cer(reference, hypothesis) == 100.0
    
    def test_calculate_cer_empty_strings(self):
        """CER should handle empty strings"""
        assert calculate_cer("", "") == 0.0
        assert calculate_cer("hello", "") == 100.0
        assert calculate_cer("", "hello") == 100.0


class TestBatchWER:
    def test_batch_wer_empty_input(self):
        """Should handle empty lists gracefully"""
        references = []
        hypotheses = []
        result = batch_wer(references, hypotheses)
        assert result["overall_wer"] == 0.0
        assert result["per_sample_wer"] == []
    
    def test_batch_wer_single_sample(self):
        """Test batch WER with single sample"""
        references = ["hello world"]
        hypotheses = ["hello world"]
        result = batch_wer(references, hypotheses)
        assert result["overall_wer"] == 0.0
        assert len(result["per_sample_wer"]) == 1
        assert result["per_sample_wer"][0] == 0.0
    
    def test_batch_wer_multiple_samples(self):
        """Test batch WER with multiple samples"""
        references = ["hello world", "foo bar", "test case"]
        hypotheses = ["hello world", "foo baz", "test"]
        result = batch_wer(references, hypotheses)
        assert "overall_wer" in result
        assert "per_sample_wer" in result
        assert len(result["per_sample_wer"]) == 3
    
    def test_batch_wer_mismatched_lengths(self):
        """Should raise ValueError for mismatched lengths"""
        references = ["hello world"]
        hypotheses = ["hello world", "extra"]
        with pytest.raises(ValueError, match="must have the same length"):
            batch_wer(references, hypotheses)


class TestBatchCER:
    def test_batch_cer_empty_input(self):
        """Should handle empty lists gracefully"""
        references = []
        hypotheses = []
        result = batch_cer(references, hypotheses)
        assert result["overall_cer"] == 0.0
        assert result["per_sample_cer"] == []
    
    def test_batch_cer_multiple_samples(self):
        """Test batch CER with multiple samples"""
        references = ["hello", "world", "test"]
        hypotheses = ["hallo", "world", "tost"]
        result = batch_cer(references, hypotheses)
        assert "overall_cer" in result
        assert "per_sample_cer" in result
        assert len(result["per_sample_cer"]) == 3


class TestBatchBLEU:
    def test_batch_bleu_empty_input(self):
        """Should handle empty lists gracefully"""
        references = []
        hypotheses = []
        result = batch_bleu(references, hypotheses)
        assert result["overall_bleu"] == 0.0
        assert result["per_sample_bleu"] == []
    
    def test_batch_bleu_exact_match(self):
        """BLEU should be 100 for exact matches"""
        references = ["hello world"]
        hypotheses = ["hello world"]
        result = batch_bleu(references, hypotheses)
        assert result["overall_bleu"] == 100.0
        assert result["per_sample_bleu"][0] == 100.0


class TestCalculateBLEU:
    def test_calculate_bleu_exact_match(self):
        """BLEU should be 100 for identical strings"""
        reference = "hello world"
        hypothesis = "hello world"
        assert calculate_bleu_score(reference, hypothesis) == 100.0
    
    def test_calculate_bleu_empty_strings(self):
        """BLEU should handle empty strings"""
        assert calculate_bleu_score("", "") == 0.0
        assert calculate_bleu_score("hello", "") == 0.0
        assert calculate_bleu_score("", "hello") == 0.0


@pytest.mark.parametrize("reference,hypothesis,expected_wer", [
    ("hello world", "hello world", 0.0),
    ("hello", "hello world", 100.0),
    ("the cat sat", "the cat", 33.33333333333333),
])
def test_calculate_wer_parametrized(reference, hypothesis, expected_wer):
    """Parametrized WER tests"""
    assert calculate_wer(reference, hypothesis) == pytest.approx(expected_wer, rel=1e-5)


@pytest.mark.parametrize("reference,hypothesis,expected_cer", [
    ("hello", "hello", 0.0),
    ("abc", "abc", 0.0),
    ("test", "best", 25.0),
])
def test_calculate_cer_parametrized(reference, hypothesis, expected_cer):
    """Parametrized CER tests"""
    assert calculate_cer(reference, hypothesis) == pytest.approx(expected_cer, rel=1e-5)
