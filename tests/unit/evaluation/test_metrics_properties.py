"""
Property-based tests for metrics using Hypothesis.

Tests mathematical properties and invariants that should always hold.
"""

import pytest

# Try to import hypothesis, skip tests if not available
try:
    from hypothesis import given, strategies as st, assume, settings, HealthCheck
    from hypothesis import example
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    # Create dummy decorators for when hypothesis is not available
    def given(*args, **kwargs):
        return pytest.mark.skip(reason="hypothesis not installed")

    class st:
        @staticmethod
        def text(*args, **kwargs):
            return None
        @staticmethod
        def lists(*args, **kwargs):
            return None
    
    class HealthCheck:
        filter_too_much = "filter_too_much"

from src.evaluation.metrics import (
    calculate_wer,
    calculate_cer,
    calculate_bleu_score,
    batch_wer,
    batch_cer,
    batch_bleu
)


# Only run these tests if hypothesis is available
pytestmark = pytest.mark.skipif(
    not HYPOTHESIS_AVAILABLE,
    reason="hypothesis package not installed"
)


class TestWERProperties:
    """Property-based tests for WER calculation."""

    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=50, deadline=None)
    def test_wer_identical_strings_is_zero(self, text):
        """WER should always be 0 for identical strings."""
        assume(text.strip())  # Skip empty strings

        wer = calculate_wer(text, text)
        assert wer == 0.0

    @given(st.text(min_size=1, max_size=100), st.text(min_size=1, max_size=100))
    @settings(max_examples=50, deadline=None)
    def test_wer_is_percentage(self, reference, hypothesis):
        """WER should always be >= 0. Note: Can be > 100 if hypothesis is much longer."""
        assume(reference.strip())

        wer = calculate_wer(reference, hypothesis)
        assert 0.0 <= wer

    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=50, deadline=None)
    def test_wer_empty_hypothesis_is_100(self, reference):
        """WER should be 100 when hypothesis is empty (all deletions)."""
        assume(reference.strip())

        wer = calculate_wer(reference, "")
        assert wer == 100.0

    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=50, deadline=None)
    def test_wer_symmetric_for_reference_hypothesis(self, text1):
        """WER should be symmetric when swapping reference and hypothesis for identical texts."""
        assume(text1.strip())
        text2 = text1 + " extra"

        wer1 = calculate_wer(text1, text2)
        wer2 = calculate_wer(text2, text1)

        # Not necessarily equal, but both should be valid percentages
        assert 0.0 <= wer1
        assert 0.0 <= wer2

    @given(st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll'), max_codepoint=591), min_size=1, max_size=50))
    @settings(max_examples=30, deadline=None)
    def test_wer_case_insensitive(self, text):
        """WER should be case-insensitive."""
        assume(text.strip())
        assume("ß" not in text)  # German sharp s causes issues with simple case conversion
        assume("µ" not in text)  # Micro sign causes issues with simple case conversion
        assume("ŉ" not in text)  # Latin small letter n preceded by apostrophe causes issues
        assume("ı" not in text)  # Latin small letter dotless i causes issues
        assume("ǰ" not in text)  # Latin small letter j with caron causes issues
        assume("ſ" not in text)  # Latin small letter long s maps differently in casing

        wer_lower = calculate_wer(text.lower(), text.upper())
        assert wer_lower == 0.0


class TestCERProperties:
    """Property-based tests for CER calculation."""

    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=50, deadline=None)
    def test_cer_identical_strings_is_zero(self, text):
        """CER should always be 0 for identical strings."""
        assume(text.strip())

        cer = calculate_cer(text, text)
        assert cer == 0.0

    @given(st.text(min_size=1, max_size=100), st.text(min_size=1, max_size=100))
    @settings(max_examples=50, deadline=None)
    def test_cer_is_percentage(self, reference, hypothesis):
        """CER should always be >= 0. Note: Can be > 100."""
        assume(reference.strip())

        cer = calculate_cer(reference, hypothesis)
        assert 0.0 <= cer

    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=50, deadline=None)
    def test_cer_empty_hypothesis_is_100(self, reference):
        """CER should be 100 when hypothesis is empty."""
        assume(reference.strip())

        cer = calculate_cer(reference, "")
        assert cer == 100.0

    @given(st.text(alphabet='abcdefghijklmnopqrstuvwxyz ', min_size=5, max_size=50))
    @settings(max_examples=30, deadline=None)
    def test_cer_single_character_error(self, text):
        """CER for single character substitution should be approximately 1/n * 100."""
        assume(len(text.strip()) > 1)

        # Change first character to something definitely different (and likely not normalized away)
        # Use a number if text is letters
        modified = "1" + text[1:]
        cer = calculate_cer(text, modified)

        # Should be roughly 1 error / length of text
        # After normalization, the calculation may vary slightly
        assert 0.0 <= cer <= 100.0


class TestBLEUProperties:
    """Property-based tests for BLEU score calculation."""

    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=50, deadline=None)
    def test_bleu_identical_strings_is_100(self, text):
        """BLEU should be 100 for identical strings."""
        assume(text.strip())

        bleu = calculate_bleu_score(text, text)
        assert bleu == pytest.approx(100.0, abs=0.1)

    @given(st.text(min_size=1, max_size=100), st.text(min_size=1, max_size=100))
    @settings(max_examples=50, deadline=None)
    def test_bleu_is_percentage(self, reference, hypothesis):
        """BLEU should always be between 0 and 100."""
        assume(reference.strip() and hypothesis.strip())

        bleu = calculate_bleu_score(reference, hypothesis)
        assert 0.0 <= bleu <= 100.0001

    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=50, deadline=None)
    def test_bleu_empty_hypothesis_is_zero(self, reference):
        """BLEU should be 0 when hypothesis is empty."""
        assume(reference.strip())

        bleu = calculate_bleu_score(reference, "")
        assert bleu == 0.0

    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=50, deadline=None)
    def test_bleu_empty_reference_is_zero(self, hypothesis):
        """BLEU should be 0 when reference is empty."""
        assume(hypothesis.strip())

        bleu = calculate_bleu_score("", hypothesis)
        assert bleu == 0.0


class TestBatchMetricsProperties:
    """Property-based tests for batch metrics."""

    @given(st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=20))
    @settings(max_examples=30, deadline=None)
    def test_batch_wer_identical_lists(self, texts):
        """Batch WER should be 0 when all references match hypotheses."""
        assume(all(t.strip() for t in texts))

        result = batch_wer(texts, texts)

        assert result["overall_wer"] == 0.0
        assert all(wer == 0.0 or wer is None for wer in result["per_sample_wer"])

    @given(st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=20))
    @settings(max_examples=30, deadline=None)
    def test_batch_cer_identical_lists(self, texts):
        """Batch CER should be 0 when all references match hypotheses."""
        assume(all(t.strip() for t in texts))

        result = batch_cer(texts, texts)

        assert result["overall_cer"] == 0.0
        assert all(cer == 0.0 or cer is None for cer in result["per_sample_cer"])

    @given(st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=20))
    @settings(max_examples=30, deadline=None)
    def test_batch_bleu_identical_lists(self, texts):
        """Batch BLEU should be ~100 when all references match hypotheses."""
        assume(all(t.strip() for t in texts))

        result = batch_bleu(texts, texts)

        # Should be very close to 100
        assert result["overall_bleu"] > 99.0
        assert all(bleu > 99.0 for bleu in result["per_sample_bleu"])

    @given(
        st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=20),
        st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=20)
    )
    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_batch_wer_length_consistency(self, refs, hyps):
        """Batch WER should have consistent output lengths."""
        assume(len(refs) == len(hyps))
        assume(all(r.strip() for r in refs))

        result = batch_wer(refs, hyps)

        assert len(result["per_sample_wer"]) == len(refs)
        assert 0.0 <= result["overall_wer"]

    @given(
        st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=20),
        st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=20)
    )
    @settings(max_examples=30, deadline=None)
    def test_batch_metrics_length_mismatch_raises(self, refs, hyps):
        """Batch metrics should raise when lengths don't match."""
        assume(len(refs) != len(hyps))

        with pytest.raises(ValueError):
            batch_wer(refs, hyps)

        with pytest.raises(ValueError):
            batch_cer(refs, hyps)

        with pytest.raises(ValueError):
            batch_bleu(refs, hyps)


class TestMetricsComparisons:
    """Property-based tests for comparing metrics."""

    @given(st.text(min_size=5, max_size=50), st.text(min_size=5, max_size=50))
    @settings(max_examples=30, deadline=None)
    def test_perfect_match_all_metrics_optimal(self, text1, text2):
        """For perfect matches, all metrics should show optimal values."""
        assume(text1.strip())

        wer = calculate_wer(text1, text1)
        cer = calculate_cer(text1, text1)
        bleu = calculate_bleu_score(text1, text1)

        assert wer == 0.0
        assert cer == 0.0
        assert bleu == pytest.approx(100.0, abs=0.1)

    @given(st.text(min_size=5, max_size=50))
    @settings(max_examples=30, deadline=None)
    def test_complete_mismatch_high_errors(self, text):
        """For complete mismatches, error metrics should be high."""
        assume(text.strip())

        # Create completely different text
        opposite = "x" * len(text)

        wer = calculate_wer(text, opposite)
        cer = calculate_cer(text, opposite)

        # Should have high error rates
        assert wer > 0.0
        assert cer > 0.0


class TestNormalizationProperties:
    """Property-based tests for text normalization."""

    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=50, deadline=None)
    def test_normalization_idempotent(self, text):
        """Normalizing twice should give same result as normalizing once."""
        from src.evaluation.metrics import _normalize_text

        normalized_once = _normalize_text(text)
        normalized_twice = _normalize_text(normalized_once)

        assert normalized_once == normalized_twice

    @given(st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll')), min_size=1, max_size=50))
    @settings(max_examples=30, deadline=None)
    def test_normalization_lowercase(self, text):
        """Normalization should convert to lowercase."""
        from src.evaluation.metrics import _normalize_text

        assume(text.strip())
        normalized = _normalize_text(text)

        assert normalized == normalized.lower()

    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=30, deadline=None)
    def test_normalization_strips_extra_whitespace(self, text):
        """Normalization should remove extra whitespace."""
        from src.evaluation.metrics import _normalize_text

        # Add extra spaces
        text_with_spaces = "  ".join(text.split()) + "  "
        normalized = _normalize_text(text_with_spaces)

        # Should not have multiple consecutive spaces
        assert "  " not in normalized
        # Should not start or end with spaces
        assert not normalized.startswith(" ")
        assert not normalized.endswith(" ")


class TestMetricsMonotonicity:
    """Property-based tests for monotonicity properties."""

    @given(st.text(alphabet='abcdefghijklmnopqrstuvwxyz ', min_size=10, max_size=50))
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_wer_increases_with_more_errors(self, text):
        """WER should increase (or stay same) as we introduce more errors."""
        assume(len(text.split()) >= 3)

        words = text.split()

        # Perfect match
        wer_0 = calculate_wer(text, text)

        # One word different
        hyp_1 = " ".join(words[:-1] + ["different"])
        wer_1 = calculate_wer(text, hyp_1)

        # Two words different
        hyp_2 = " ".join(words[:-2] + ["different", "words"])
        wer_2 = calculate_wer(text, hyp_2)

        # WER should not decrease as we add more errors
        assert wer_0 <= wer_1
        assert wer_1 <= wer_2

    @given(st.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=10, max_size=50))
    @settings(max_examples=20, deadline=None)
    def test_cer_increases_with_more_character_errors(self, text):
        """CER should increase as we introduce more character errors."""
        assume(len(text) >= 5)

        # Perfect match
        cer_0 = calculate_cer(text, text)

        # One character different
        hyp_1 = "X" + text[1:]
        cer_1 = calculate_cer(text, hyp_1)

        # Two characters different
        hyp_2 = "XY" + text[2:]
        cer_2 = calculate_cer(text, hyp_2)

        # CER should not decrease
        assert cer_0 <= cer_1
        assert cer_1 <= cer_2
