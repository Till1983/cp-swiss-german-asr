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
        hyp = "das war test" # sub 'ist'->'war', del 'ein'
        
        alignment = analyzer.get_alignment(ref, hyp)
        counts = analyzer.categorize_errors(alignment)
        
        assert counts['correct'] == 2 # das, test
        assert counts['substitution'] >= 1 # ist -> war
        assert counts['deletion'] >= 1 # ein

    # ============================================================================
    # EDGE CASE TESTS
    # ============================================================================

    def test_get_alignment_empty_strings(self, analyzer):
        """Test alignment with empty inputs."""
        # Both empty
        assert analyzer.get_alignment("", "") == []
        
        # Empty ref (all insertions)
        align_ins = analyzer.get_alignment("", "a b")
        assert len(align_ins) == 2
        assert all(x['type'] == 'insertion' for x in align_ins)
        
        # Empty hyp (all deletions)
        align_del = analyzer.get_alignment("a b", "")
        assert len(align_del) == 2
        assert all(x['type'] == 'deletion' for x in align_del)

    def test_categorize_errors_logic(self, analyzer):
        """Test that categorization counts match alignment types."""
        # Manually construct an alignment list to test counting logic purely
        alignment = [
            {'type': 'correct'},
            {'type': 'substitution'},
            {'type': 'substitution'},
            {'type': 'deletion'},
            {'type': 'insertion'},
            {'type': 'insertion'},
            {'type': 'insertion'}
        ]
        counts = analyzer.categorize_errors(alignment)
        
        assert counts['correct'] == 1
        assert counts['substitution'] == 2
        assert counts['deletion'] == 1
        assert counts['insertion'] == 3
        assert counts['total_errors'] == 6

    # ============================================================================
    # ANALYSIS FEATURE TESTS
    # ============================================================================

    def test_find_confusion_pairs(self, analyzer):
        """Test identification of common substitution pairs."""
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
            {'dialect': 'bern', 'reference': 'a b', 'hypothesis': 'a b', 'wer': 0.0, 'cer': 0.0},
            {'dialect': 'bern', 'reference': 'a b', 'hypothesis': 'a c', 'wer': 50.0, 'cer': 20.0},
            {'dialect': 'zurich', 'reference': 'x y', 'hypothesis': 'x z', 'wer': 50.0, 'cer': 20.0},
            # Missing dialect key should be handled
            {'reference': 'foo', 'hypothesis': 'bar', 'wer': 100.0, 'cer': 100.0}
        ]
        
        analysis = analyzer.analyze_by_dialect(results)
        
        assert 'bern' in analysis
        assert 'zurich' in analysis
        assert 'unknown' in analysis
        
        # Check Bern stats
        bern_stats = analysis['bern']
        assert bern_stats['sample_count'] == 2
        assert bern_stats['mean_wer'] == 25.0
        
        # Check error distribution structure exists and is populated
        assert 'error_distribution' in bern_stats
        dist = bern_stats['error_distribution']
        assert dist['substitution'] > 0
        assert dist['correct'] > 0

    def test_calculate_aggregate_stats(self, analyzer):
        """Test calculation of mean/median/std stats."""
        results = [
            {'wer': 10.0, 'cer': 5.0},
            {'wer': 20.0, 'cer': 10.0},
            {'wer': 30.0, 'cer': 15.0}
        ]
        
        stats = analyzer.calculate_aggregate_stats(results)
        
        assert stats['mean_wer'] == 20.0
        assert stats['median_wer'] == 20.0
        assert stats['mean_cer'] == 10.0
        assert stats['std_wer'] == 10.0

    def test_calculate_aggregate_stats_empty(self, analyzer):
        """Test stats calculation with empty input."""
        assert analyzer.calculate_aggregate_stats([]) == {}

    def test_format_alignment_readable(self, analyzer):
        """Test string formatting of alignment."""
        alignment = [
            {'type': 'correct', 'ref': 'hello', 'hyp': 'hello'},
            {'type': 'substitution', 'ref': 'world', 'hyp': 'wrld'}
        ]
        
        output = analyzer.format_alignment_readable(alignment)
        
        # Check for key components
        assert "REF" in output
        assert "HYP" in output
        assert "TYPE" in output
        assert "hello" in output
        assert "wrld" in output
        # Check type codes
        assert "C" in output
        assert "S" in output

    # ============================================================================
    # SWISS GERMAN SPECIFIC TESTS
    # ============================================================================

    def test_swiss_german_dialect_mismatch(self, analyzer):
        """Test with realistic Swiss German dialect variations."""
        # Bernese: "I gang hei" (I go home)
        # Zurich: "Ich gah hei"
        # If model predicts Zurich dialect for Bernese audio:
        
        ref = "i gang hei"
        hyp = "ich gah hei"
        
        alignment = analyzer.get_alignment(ref, hyp)
        counts = analyzer.categorize_errors(alignment)
        
        # Expecting substitutions: i->ich, gang->gah
        assert counts['substitution'] >= 1
        assert counts['correct'] >= 1 # hei

    def test_swiss_german_umlauts(self, analyzer):
        """Test that umlauts are handled correctly in alignment."""
        ref = "grüezi mitenand"
        hyp = "gruezi mitenand" # Missing umlaut
        
        alignment = analyzer.get_alignment(ref, hyp)
        
        # Should be a substitution
        assert alignment[0]['type'] == 'substitution'
        assert alignment[0]['ref'] == 'grüezi'
        assert alignment[0]['hyp'] == 'gruezi'