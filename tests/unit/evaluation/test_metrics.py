import pytest
from src.evaluation.metrics import (
    calculate_wer, 
    calculate_cer, 
    calculate_bleu_score,
    batch_wer, 
    batch_cer,
    batch_bleu,
    _normalize_text
)

# ============================================================================
# CALCULATION TESTS
# ============================================================================
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

# ============================================================================
# BATCH TESTS
# ============================================================================
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
        # Use pytest.approx to account for floating-point precision in BLEU calculation
        assert result["overall_bleu"] == pytest.approx(100.0, abs=0.01)
        assert result["per_sample_bleu"][0] == pytest.approx(100.0, abs=0.01)


class TestCalculateBLEU:
    def test_calculate_bleu_exact_match(self):
        """BLEU should be 100 for identical strings"""
        reference = "hello world"
        hypothesis = "hello world"
        assert calculate_bleu_score(reference, hypothesis) == pytest.approx(100.0, abs=0.01)
    
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

# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases with special characters, unicode, numbers, etc."""
    
    def test_wer_with_punctuation(self):
        """WER should handle punctuation correctly"""
        reference = "Hello, world!"
        hypothesis = "Hello world"
        # Punctuation is stripped by normalization
        result = calculate_wer(reference, hypothesis)
        assert result >= 0.0 and result <= 100.0
    
    def test_wer_with_numbers(self):
        """WER should handle numbers"""
        reference = "I have 5 apples"
        hypothesis = "I have five apples"
        result = calculate_wer(reference, hypothesis)
        # '5' vs 'five' is a substitution
        assert result == 25.0  # 1 out of 4 words
    
    def test_cer_with_unicode(self):
        """CER should handle Unicode characters (German umlauts)"""
        reference = "Über die Brücke gehen"
        hypothesis = "Uber die Brucke gehen"
        result = calculate_cer(reference, hypothesis)
        assert result > 0.0  # Missing umlauts = errors
    
    def test_bleu_with_unicode(self):
        """BLEU should handle Unicode correctly"""
        reference = "Zürich ist schön"
        hypothesis = "Zurich ist schön"
        result = calculate_bleu_score(reference, hypothesis)
        # Should still calculate a score
        assert result >= 0.0 and result <= 100.0
    
    def test_wer_with_extra_whitespace(self):
        """WER should handle multiple spaces correctly"""
        reference = "hello    world"
        hypothesis = "hello world"
        # Normalization strips extra whitespace
        assert calculate_wer(reference, hypothesis) == 0.0
    
    def test_cer_single_character(self):
        """CER should work with single characters"""
        assert calculate_cer("a", "a") == 0.0
        assert calculate_cer("a", "b") == 100.0
    
    def test_bleu_very_short_sentence(self):
        """BLEU should handle very short sentences"""
        reference = "go"
        hypothesis = "go"
        # sentence_bleu has smoothing for short sentences
        result = calculate_bleu_score(reference, hypothesis)
        assert result == pytest.approx(100.0, rel=0.01)
    
    def test_batch_metrics_with_empty_strings_mixed(self):
        """Batch functions should handle mix of empty and non-empty strings"""
        references = ["hello world", "", "test"]
        hypotheses = ["hello world", "something", "test"]
        
        # WER should handle this
        wer_result = batch_wer(references, hypotheses)
        assert len(wer_result["per_sample_wer"]) == 3
        assert wer_result["per_sample_wer"][1] == 100.0  # Empty ref, non-empty hyp
        
        # CER should handle this
        cer_result = batch_cer(references, hypotheses)
        assert len(cer_result["per_sample_cer"]) == 3
        
        # BLEU should handle this
        bleu_result = batch_bleu(references, hypotheses)
        assert len(bleu_result["per_sample_bleu"]) == 3

    def test_batch_metrics_with_empty_strings_mixed(self):
        """
        Batch functions should filter out empty references.
        Empty references are mathematically undefined for WER/CER.
        """
        references = ["hello world", "", "test"]
        hypotheses = ["hello world", "something", "test"]
        
        # WER should filter empty references
        wer_result = batch_wer(references, hypotheses)
        
        # Should have 3 per-sample results (including None for empty ref)
        assert len(wer_result["per_sample_wer"]) == 3
        
        # Second sample should be None (filtered)
        assert wer_result["per_sample_wer"][1] is None
        
        # Overall WER calculated only on valid samples (samples 0 and 2)
        # Both are perfect matches, so WER should be 0.0
        assert wer_result["overall_wer"] == 0.0
        
        # First and third samples should have WER scores
        assert wer_result["per_sample_wer"][0] == 0.0  # Perfect match
        assert wer_result["per_sample_wer"][2] == 0.0  # Perfect match
        
        # CER should behave the same way
        cer_result = batch_cer(references, hypotheses)
        assert len(cer_result["per_sample_cer"]) == 3
        assert cer_result["per_sample_cer"][1] is None
        assert cer_result["overall_cer"] == 0.0
        
        # BLEU handles empty strings differently (returns 0.0, not None)
        bleu_result = batch_bleu(references, hypotheses)
        assert len(bleu_result["per_sample_bleu"]) == 3
        assert bleu_result["per_sample_bleu"][1] == 0.0  # Empty ref -> 0.0 BLEU
    
    def test_batch_metrics_all_empty_references(self):
        """When all references are empty, should return sensible defaults"""
        references = ["", "", ""]
        hypotheses = ["hello", "world", "test"]
        
        wer_result = batch_wer(references, hypotheses)
        assert wer_result["overall_wer"] == 0.0  # No valid samples
        assert all(x is None for x in wer_result["per_sample_wer"])
        
        cer_result = batch_cer(references, hypotheses)
        assert cer_result["overall_cer"] == 0.0  # No valid samples
        assert all(x is None for x in cer_result["per_sample_cer"])
        
        bleu_result = batch_bleu(references, hypotheses)
        assert bleu_result["overall_bleu"] == 0.0
        assert all(x == 0.0 for x in bleu_result["per_sample_bleu"])
    
    def test_batch_metrics_no_empty_references(self):
        """Normal case with no empty references should work as before"""
        references = ["hello world", "foo bar"]
        hypotheses = ["hello world", "foo baz"]
        
        wer_result = batch_wer(references, hypotheses)
        
        # No None values - all valid
        assert all(x is not None for x in wer_result["per_sample_wer"])
        assert len(wer_result["per_sample_wer"]) == 2
        
        # First sample perfect (0% WER), second has 1 substitution (50% WER)
        assert wer_result["per_sample_wer"][0] == 0.0
        assert wer_result["per_sample_wer"][1] == 50.0
        
        # Overall WER: 1 error / 4 total words = 25%
        assert wer_result["overall_wer"] == 25.0


class TestNormalizationEdgeCases:
    """Test text normalization edge cases"""
    
    def test_normalize_text_newlines(self):
        """Normalization should convert newlines to spaces"""
        text = "hello\nworld"
        normalized = _normalize_text(text)
        # Newlines are treated as whitespace separators
        assert normalized == "hello world"
    
    def test_normalize_text_tabs(self):
        """Normalization should convert tabs to spaces"""
        text = "hello\tworld"
        normalized = _normalize_text(text)
        # Tabs are treated as whitespace separators
        assert normalized == "hello world"
    
    def test_normalize_text_multiple_spaces_internal(self):
        """Internal multiple spaces should be collapsed to single space"""
        text = "hello  world"
        normalized = _normalize_text(text)
        # Spaces should be collapsed
        assert normalized == "hello world"


class TestBoundaryConditions:
    """Test boundary conditions and extreme values"""
    
    def test_wer_very_long_sentences(self):
        """WER should handle very long sentences"""
        reference = " ".join(["word"] * 1000)
        hypothesis = " ".join(["word"] * 999)  # One deletion
        result = calculate_wer(reference, hypothesis)
        assert result == pytest.approx(0.1, abs=0.01)  # 1/1000 = 0.1%
    
    def test_cer_very_long_strings(self):
        """CER should handle very long strings"""
        reference = "a" * 10000
        hypothesis = "a" * 10000
        assert calculate_cer(reference, hypothesis) == 0.0
    
    def test_bleu_identical_long_sentence(self):
        """BLEU should be 100 for identical long sentences"""
        sentence = " ".join(["word"] * 100)
        assert calculate_bleu_score(sentence, sentence) == pytest.approx(100.0, abs=0.01)


class TestConsistency:
    """Test consistency between single and batch calculations"""
    
    def test_wer_single_vs_batch_consistency(self):
        """Single WER should match batch WER for one sample"""
        reference = "hello world"
        hypothesis = "hello earth"
        
        single_wer = calculate_wer(reference, hypothesis)
        batch_result = batch_wer([reference], [hypothesis])
        
        assert single_wer == batch_result["per_sample_wer"][0]
    
    def test_cer_single_vs_batch_consistency(self):
        """Single CER should match batch CER for one sample"""
        reference = "hello"
        hypothesis = "hallo"
        
        single_cer = calculate_cer(reference, hypothesis)
        batch_result = batch_cer([reference], [hypothesis])
        
        assert single_cer == batch_result["per_sample_cer"][0]
    
    def test_bleu_single_vs_batch_consistency(self):
        """Single BLEU should match batch BLEU for one sample"""
        reference = "hello world"
        hypothesis = "hello world"
        
        single_bleu = calculate_bleu_score(reference, hypothesis)
        batch_result = batch_bleu([reference], [hypothesis])

        assert single_bleu == pytest.approx(batch_result["per_sample_bleu"][0], abs=0.01)


# ============================================================================
# PARAMETRIZED TESTS WITH MORE CASES
# ============================================================================

@pytest.mark.parametrize("reference,hypothesis,expected_wer", [
    ("hello world", "hello world", 0.0),
    ("hello", "hello world", 100.0),  # Insertion
    ("hello world", "hello", 50.0),  # Deletion
    ("the cat sat", "the cat", 33.33),  # One deletion out of 3
    ("one two three", "four five six", 100.0),  # Complete mismatch
    ("a", "a", 0.0),  # Single word
    ("a b c d e", "a b c d e", 0.0),  # Multiple exact matches
])
def test_calculate_wer_parametrized_extended(reference, hypothesis, expected_wer):
    """Extended parametrized WER tests"""
    assert calculate_wer(reference, hypothesis) == pytest.approx(expected_wer, abs=0.1)


@pytest.mark.parametrize("reference,hypothesis,expected_cer", [
    ("hello", "hello", 0.0),
    ("abc", "abc", 0.0),
    ("test", "best", 25.0),  # 1 substitution out of 4
    ("hello", "helo", 20.0),  # 1 deletion out of 5
    ("cat", "cats", 33.33),  # 1 insertion, but CER = errors / reference_length = 1/3 = 33.33%
    ("a", "b", 100.0),  # Single character mismatch
    ("ab", "ba", 100.0),  # Transposition counts as 2 edits
])
def test_calculate_cer_parametrized_extended(reference, hypothesis, expected_cer):
    """Extended parametrized CER tests"""
    assert calculate_cer(reference, hypothesis) == pytest.approx(expected_cer, abs=0.1)


@pytest.mark.parametrize("reference,hypothesis", [
    ("hello world from python", "hello world from python"),
    ("the quick brown fox", "the quick brown fox"),
    ("a b c d e f g h i j", "a b c d e f g h i j"),
])
def test_calculate_bleu_perfect_matches(reference, hypothesis):
    """BLEU should be 100 for all perfect matches"""
    assert calculate_bleu_score(reference, hypothesis) == pytest.approx(100.0, abs=0.01)


# ============================================================================
# REAL-WORLD SWISS GERMAN EXAMPLES
# ============================================================================

class TestSwissGermanRealistic:
    """Test with realistic Swiss German transcription scenarios"""
    
    def test_wer_swiss_german_dialect_variation(self):
        """Test WER with Swiss German dialect variations"""
        reference = "ich gehe nach Hause"  # Standard German
        hypothesis = "ich gang hei"  # Swiss German approximation
        result = calculate_wer(reference, hypothesis)
        # Significant difference expected
        assert result > 0.0
    
    def test_cer_swiss_german_umlauts(self):
        """Test CER with Swiss German umlauts"""
        reference = "Zürich Bär Schön"
        hypothesis = "Zurich Bar Schon"
        result = calculate_cer(reference, hypothesis)
        # Missing umlauts = character errors
        assert result > 0.0
    
    def test_bleu_partial_swiss_german_match(self):
        """Test BLEU with partial Swiss German matches"""
        reference = "der Zug fährt nach Zürich"
        hypothesis = "der Zug fahrt nach Zurich"
        result = calculate_bleu_score(reference, hypothesis)
        # Should have some overlap
        assert 0.0 < result < 100.0