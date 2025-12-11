import pytest
from src.evaluation.error_analyzer import ErrorAnalyzer

class TestErrorAnalyzer:
    @pytest.fixture
    def analyzer(self):
        """Fixture to provide an ErrorAnalyzer instance."""
        return ErrorAnalyzer()

    # ============================================================================
    # ALIGNMENT TESTS
    # ============================================================================
    
    def test_get_alignment_perfect_match(self, analyzer):
        """Test alignment for identical strings."""
        ref = "hallo welt"
        hyp = "hallo welt"
        alignment = analyzer.get_alignment(ref, hyp)
        
        assert len(alignment) == 2
        assert all(item['type'] == 'correct' for item in alignment)
        assert alignment[0]['ref'] == 'hallo'
        assert alignment[0]['hyp'] == 'hallo'

    def test_get_alignment_substitution(self, analyzer):
        """Test alignment with substitutions."""
        ref = "hallo welt"
        hyp = "hallo velo"
        alignment = analyzer.get_alignment(ref, hyp)
        
        # Should be: correct, substitution
        types = [item['type'] for item in alignment]
        assert types == ['correct', 'substitution']
        assert alignment[1]['ref'] == 'welt'
        assert alignment[1]['hyp'] == 'velo'

    def test_get_alignment_deletion(self, analyzer):
        """Test alignment with deletions."""
        ref = "hallo schöne welt"
        hyp = "hallo welt"
        alignment = analyzer.get_alignment(ref, hyp)
        
        # Should contain a deletion for "schöne"
        deletions = [item for item in alignment if item['type'] == 'deletion']
        assert len(deletions) == 1
        assert deletions[0]['ref'] == 'schöne'
        assert deletions[0]['hyp'] is None

    def test_get_alignment_insertion(self, analyzer):
        """Test alignment with insertions."""
        ref = "hallo welt"
        hyp = "hallo schöne welt"
        alignment = analyzer.get_alignment(ref, hyp)
        
        # Should contain an insertion for "schöne"
        insertions = [item for item in alignment if item['type'] == 'insertion']
        assert len(insertions) == 1
        assert insertions[0]['ref'] is None
        assert insertions[0]['hyp'] == 'schöne'

    def test_get_alignment_complex_mix(self, analyzer):
        """Test a mix of errors."""
        ref = "das ist ein test"
        hyp = "das isch ei test extra"
        alignment = analyzer.get_alignment(ref, hyp)
        
        # Should have: C, S, S, C, I
        types = [item['type'] for item in alignment]
        assert 'correct' in types
        assert 'substitution' in types
        assert 'insertion' in types

    def test_get_alignment_empty_both(self, analyzer):
        """Test with both strings empty."""
        ref = ""
        hyp = ""
        alignment = analyzer.get_alignment(ref, hyp)
        assert alignment == []

    def test_get_alignment_empty_ref(self, analyzer):
        """Test with empty reference (all insertions)."""
        ref = ""
        hyp = "extra words"
        alignment = analyzer.get_alignment(ref, hyp)
        
        assert len(alignment) == 2
        assert all(item['type'] == 'insertion' for item in alignment)

    def test_get_alignment_empty_hyp(self, analyzer):
        """Test with empty hypothesis (all deletions)."""
        ref = "missing words"
        hyp = ""
        alignment = analyzer.get_alignment(ref, hyp)
        
        assert len(alignment) == 2
        assert all(item['type'] == 'deletion' for item in alignment)

    # ============================================================================
    # ERROR CATEGORIZATION TESTS
    # ============================================================================

    def test_categorize_errors_basic(self, analyzer):
        """Test error categorization counts."""
        alignment = [
            {'type': 'correct', 'ref': 'a', 'hyp': 'a'},
            {'type': 'substitution', 'ref': 'b', 'hyp': 'c'},
            {'type': 'deletion', 'ref': 'd', 'hyp': None}
        ]
        
        counts = analyzer.categorize_errors(alignment)
        
        assert counts['correct'] == 1
        assert counts['substitution'] == 1
        assert counts['deletion'] == 1
        assert counts['insertion'] == 0
        assert counts['total_errors'] == 2

    def test_categorize_errors_all_correct(self, analyzer):
        """Test with all correct matches."""
        alignment = [
            {'type': 'correct', 'ref': 'a', 'hyp': 'a'},
            {'type': 'correct', 'ref': 'b', 'hyp': 'b'}
        ]
        
        counts = analyzer.categorize_errors(alignment)
        
        assert counts['correct'] == 2
        assert counts['total_errors'] == 0

    # ============================================================================
    # CONFUSION PAIR TESTS
    # ============================================================================

    def test_find_confusion_pairs_single(self, analyzer):
        """Test finding confusion pairs from a single alignment."""
        alignments = [
            [
                {'type': 'substitution', 'ref': 'ist', 'hyp': 'isch'},
                {'type': 'substitution', 'ref': 'das', 'hyp': 'es'}
            ]
        ]
        
        pairs = analyzer.find_confusion_pairs(alignments)
        
        assert len(pairs) == 2
        assert (('ist', 'isch'), 1) in pairs
        assert (('das', 'es'), 1) in pairs

    def test_find_confusion_pairs_frequency(self, analyzer):
        """Test that pairs are sorted by frequency."""
        # Mock alignments from multiple sentences
        alignments = [
            [
                {'type': 'substitution', 'ref': 'chuchichäschtli', 'hyp': 'kasten'},
                {'type': 'correct', 'ref': 'und', 'hyp': 'und'}
            ],
            [
                {'type': 'substitution', 'ref': 'chuchichäschtli', 'hyp': 'kasten'},
                {'type': 'substitution', 'ref': 'gsi', 'hyp': 'gewesen'}
            ]
        ]
        
        pairs = analyzer.find_confusion_pairs(alignments)
        
        # Should find (chuchichäschtli, kasten) twice
        assert pairs[0] == (('chuchichäschtli', 'kasten'), 2)
        # Should find (gsi, gewesen) once
        assert (('gsi', 'gewesen'), 1) in pairs

    def test_get_high_error_samples(self, analyzer):
        """Test filtering of high error samples."""
        results = [
            {'id': 1, 'wer': 10.0},
            {'id': 2, 'wer': 40.0},
            {'id': 3, 'wer': 60.0},
            {'id': 4, 'wer': 100.0}
        ]
        
        # Default threshold 50.0
        high_err = analyzer.get_high_error_samples(results)
        assert len(high_err) == 2
        assert {r['id'] for r in high_err} == {3, 4}
        
        # Custom threshold 20.0
        high_err_custom = analyzer.get_high_error_samples(results, threshold=20.0)
        assert len(high_err_custom) == 3

    def test_analyze_by_dialect(self, analyzer):
        """Test grouping and analysis by dialect."""
        results = [
            {'dialect': 'bern', 'reference': 'a b', 'hypothesis': 'a b', 'wer': 0.0, 'cer': 0.0, 'bleu': 100.0},
            {'dialect': 'bern', 'reference': 'a b', 'hypothesis': 'a c', 'wer': 50.0, 'cer': 20.0, 'bleu': 50.0},
            {'dialect': 'zurich', 'reference': 'x y', 'hypothesis': 'x z', 'wer': 50.0, 'cer': 20.0, 'bleu': 60.0},
            # Missing dialect key should be handled
            {'reference': 'foo', 'hypothesis': 'bar', 'wer': 100.0, 'cer': 100.0, 'bleu': 0.0}
        ]
        
        analysis = analyzer.analyze_by_dialect(results)
        
        assert 'bern' in analysis
        assert 'zurich' in analysis
        assert 'unknown' in analysis
        
        # Check Bern stats
        bern_stats = analysis['bern']
        assert bern_stats['sample_count'] == 2
        assert bern_stats['mean_wer'] == 25.0
        assert bern_stats['mean_bleu'] == 75.0  # NEW
        
        # Check error distribution structure exists and is populated
        assert 'error_distribution' in bern_stats
        dist = bern_stats['error_distribution']
        assert dist['substitution'] > 0
        assert dist['correct'] > 0

    def test_calculate_aggregate_stats(self, analyzer):
        """Test calculation of mean/median/std stats including BLEU."""
        results = [
            {'wer': 10.0, 'cer': 5.0, 'bleu': 85.0},
            {'wer': 20.0, 'cer': 10.0, 'bleu': 75.0},
            {'wer': 30.0, 'cer': 15.0, 'bleu': 65.0}
        ]
        
        stats = analyzer.calculate_aggregate_stats(results)
        
        assert stats['mean_wer'] == 20.0
        assert stats['mean_cer'] == 10.0
        assert stats['mean_bleu'] == 75.0  # NEW
        assert 'std_wer' in stats
        assert 'std_cer' in stats
        assert 'std_bleu' in stats  # NEW
        
        # Check standard deviations are calculated
        assert stats['std_wer'] > 0
        assert stats['std_bleu'] > 0  # NEW

    def test_calculate_aggregate_stats_single_sample(self, analyzer):
        """Test stats with a single sample (std should be 0)."""
        results = [{'wer': 25.0, 'cer': 12.0, 'bleu': 70.0}]
        
        stats = analyzer.calculate_aggregate_stats(results)
        
        assert stats['mean_wer'] == 25.0
        assert stats['std_wer'] == 0.0
        assert stats['mean_bleu'] == 70.0  # NEW
        assert stats['std_bleu'] == 0.0  # NEW

    # ============================================================================
    # WER-BLEU CORRELATION TESTS (NEW)
    # ============================================================================

    def test_analyze_wer_bleu_correlation_basic(self, analyzer):
        """Test WER-BLEU correlation analysis with basic categories."""
        results = [
            {'wer': 60.0, 'bleu': 50.0},  # High WER, High BLEU (paraphrase)
            {'wer': 70.0, 'bleu': 20.0},  # High WER, Low BLEU (error)
            {'wer': 10.0, 'bleu': 90.0},  # Low WER, High BLEU (good)
            {'wer': 15.0, 'bleu': 25.0},  # Low WER, Low BLEU (edge case)
        ]
        
        analysis = analyzer.analyze_wer_bleu_correlation(
            results,
            wer_threshold=50.0,
            bleu_threshold=40.0
        )
        
        assert analysis['summary']['total_samples'] == 4
        assert analysis['summary']['high_wer_high_bleu_count'] == 1
        assert analysis['summary']['high_wer_low_bleu_count'] == 1
        assert analysis['summary']['low_wer_high_bleu_count'] == 1
        assert analysis['summary']['low_wer_low_bleu_count'] == 1
        assert analysis['summary']['semantic_preservation_rate'] == 25.0  # 1/4

    def test_analyze_wer_bleu_correlation_custom_thresholds(self, analyzer):
        """Test with custom thresholds."""
        results = [
            {'wer': 60.0, 'bleu': 30.0},
            {'wer': 40.0, 'bleu': 50.0},
        ]
        
        analysis = analyzer.analyze_wer_bleu_correlation(
            results,
            wer_threshold=45.0,
            bleu_threshold=35.0
        )
        
        # With these thresholds:
        # 60.0 WER, 30.0 BLEU -> high_wer_low_bleu
        # 40.0 WER, 50.0 BLEU -> low_wer_high_bleu
        assert analysis['summary']['high_wer_low_bleu_count'] == 1
        assert analysis['summary']['low_wer_high_bleu_count'] == 1
        assert analysis['summary']['semantic_preservation_rate'] == 0.0

    def test_analyze_wer_bleu_correlation_empty(self, analyzer):
        """Test with empty results list."""
        results = []
        
        analysis = analyzer.analyze_wer_bleu_correlation(results)
        
        assert analysis['summary']['total_samples'] == 0
        assert analysis['summary']['semantic_preservation_rate'] == 0.0

    def test_analyze_wer_bleu_correlation_missing_bleu(self, analyzer):
        """Test handling of missing BLEU scores."""
        results = [
            {'wer': 60.0},  # Missing BLEU should default to 0.0
            {'wer': 40.0, 'bleu': 50.0},
        ]
        
        analysis = analyzer.analyze_wer_bleu_correlation(
            results,
            wer_threshold=50.0,
            bleu_threshold=40.0
        )
        
        # Sample with missing BLEU (0.0) should be categorized as high_wer_low_bleu
        assert analysis['summary']['high_wer_low_bleu_count'] == 1
        assert analysis['summary']['low_wer_high_bleu_count'] == 1

    def test_analyze_wer_bleu_correlation_sample_limits(self, analyzer):
        """Test that only top 10 samples are returned for each category."""
        # Create 15 samples in high_wer_high_bleu category
        results = [{'wer': 60.0, 'bleu': 50.0, 'id': i} for i in range(15)]
        
        analysis = analyzer.analyze_wer_bleu_correlation(
            results,
            wer_threshold=50.0,
            bleu_threshold=40.0
        )
        
        assert analysis['summary']['high_wer_high_bleu_count'] == 15
        # But only 10 samples should be in the detailed list
        assert len(analysis['high_wer_high_bleu_samples']) == 10

    # ============================================================================
    # FORMAT TESTS
    # ============================================================================

    def test_format_alignment_readable(self, analyzer):
        """Test readable alignment formatting."""
        alignment = [
            {'type': 'correct', 'ref': 'das', 'hyp': 'das'},
            {'type': 'substitution', 'ref': 'ist', 'hyp': 'isch'},
            {'type': 'deletion', 'ref': 'ein', 'hyp': None}
        ]
        
        formatted = analyzer.format_alignment_readable(alignment)
        
        assert 'REF:' in formatted
        assert 'HYP:' in formatted
        assert 'TYPE:' in formatted
        assert 'das' in formatted
        assert 'isch' in formatted

    def test_format_alignment_empty(self, analyzer):
        """Test formatting of empty alignment."""
        alignment = []
        formatted = analyzer.format_alignment_readable(alignment)
        
        assert 'empty' in formatted.lower()

    # ============================================================================
    # SWISS GERMAN SPECIFIC TESTS
    # ============================================================================

    def test_swiss_german_umlaut_handling(self, analyzer):
        """Test that Swiss German umlauts are handled correctly."""
        ref = "chuchichäschtli"
        hyp = "kuchikastli"
        
        alignment = analyzer.get_alignment(ref, hyp)
        
        # Should produce a substitution (words differ)
        assert len(alignment) > 0

    def test_swiss_german_dialect_variations(self, analyzer):
        """Test handling of Swiss German dialect word variations."""
        results = [
            {'dialect': 'BE', 'reference': 'ich gehe', 'hypothesis': 'ich gang', 
             'wer': 50.0, 'cer': 25.0, 'bleu': 50.0},
            {'dialect': 'ZH', 'reference': 'ich gehe', 'hypothesis': 'ich gah', 
             'wer': 50.0, 'cer': 30.0, 'bleu': 45.0},
        ]
        
        analysis = analyzer.analyze_by_dialect(results)
        
        # Should create separate analyses for BE and ZH
        assert 'BE' in analysis
        assert 'ZH' in analysis
        assert analysis['BE']['sample_count'] == 1
        assert analysis['ZH']['sample_count'] == 1

    def test_get_alignment_leftover_words_after_substitution(self, analyzer):
        """Test alignment with unequal word counts after applying substitutions."""
        # This tests lines 82-87 in error_analyzer.py
        # When ref and hyp have different lengths after processing substitutions
        ref = "a b c d e"
        hyp = "a b c"  # Missing 'd e'
        
        alignment = analyzer.get_alignment(ref, hyp)
        
        # Should have correct matches for a, b, c, then deletions for d, e
        types = [item['type'] for item in alignment]
        assert types.count('correct') == 3
        assert types.count('deletion') == 2
        
        # Verify the deletion items
        deletions = [item for item in alignment if item['type'] == 'deletion']
        assert len(deletions) == 2
        assert deletions[0]['ref'] == 'd'
        assert deletions[1]['ref'] == 'e'
        
    def test_get_alignment_leftover_insertions_after_substitution(self, analyzer):
        """Test alignment with extra hyp words after applying substitutions."""
        # This also tests lines 82-87 (the elif h branch)
        ref = "a b c"
        hyp = "a b c d e"  # Extra 'd e'
        
        alignment = analyzer.get_alignment(ref, hyp)
        
        # Should have correct matches for a, b, c, then insertions for d, e
        types = [item['type'] for item in alignment]
        assert types.count('correct') == 3
        assert types.count('insertion') == 2
        
        # Verify the insertion items
        insertions = [item for item in alignment if item['type'] == 'insertion']
        assert len(insertions) == 2
        assert insertions[0]['hyp'] == 'd'
        assert insertions[1]['hyp'] == 'e'

    def test_get_alignment_substitute_branch_with_mock(self, analyzer, monkeypatch):
        """Force substitute alignment with uneven lengths to hit deletion/ insertion paths."""

        class DummyChunk:
            def __init__(self, type_, r_start, r_end, h_start, h_end):
                self.type = type_
                self.ref_start_idx = r_start
                self.ref_end_idx = r_end
                self.hyp_start_idx = h_start
                self.hyp_end_idx = h_end

        class DummyOut:
            def __init__(self):
                self.references = [["a", "b", "c"]]
                self.hypotheses = [["x", "y", "z"]]
                self.alignments = [[
                    DummyChunk('substitute', 0, 2, 0, 1),  # deletion leftover
                    DummyChunk('substitute', 2, 3, 1, 3),  # insertion leftover
                ]]

        monkeypatch.setattr('src.evaluation.error_analyzer.jiwer.process_words', lambda r, h: DummyOut())

        alignment = analyzer.get_alignment("abc", "xyz")

        types = [item['type'] for item in alignment]
        assert 'deletion' in types
        assert 'insertion' in types