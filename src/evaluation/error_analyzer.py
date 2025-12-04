import jiwer
import statistics
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any, Optional, Union
from src.evaluation.metrics import _normalize_text


class ErrorAnalyzer:
    """
    A reusable class for analyzing ASR error patterns.
    
    Provides functionality to:
    - Generate word-level alignments
    - Categorize errors (Substitution, Deletion, Insertion)
    - Identify common confusion pairs
    - Analyze performance by dialect or other metadata
    - Analyze WER-BLEU correlation for semantic preservation
    """

    def get_alignment(self, reference: str, hypothesis: str) -> List[Dict[str, Optional[str]]]:
        """
        Generate word-level alignment between reference and hypothesis.
        
        Args:
            reference: Ground truth text
            hypothesis: Predicted text
            
        Returns:
            List of dictionaries with keys: 'type', 'ref', 'hyp'
            types: 'correct', 'substitution', 'deletion', 'insertion'
        """
        # Normalize inputs using the same logic as metrics.py
        ref_norm = _normalize_text(reference)
        hyp_norm = _normalize_text(hypothesis)
        
        # Handle edge cases for empty strings
        if not ref_norm and not hyp_norm:
            return []
        
        if not ref_norm:
            # All insertions
            return [{'type': 'insertion', 'ref': None, 'hyp': w} for w in hyp_norm.split()]
            
        if not hyp_norm:
            # All deletions
            return [{'type': 'deletion', 'ref': w, 'hyp': None} for w in ref_norm.split()]

        # Use jiwer to compute alignment
        # process_words returns a WordOutput object containing alignments
        out = jiwer.process_words(ref_norm, hyp_norm)
        
        alignment = []
        
        # out.alignments is a list of lists of AlignmentChunk. 
        # Since we process one sentence pair, we take the first list.
        # out.references and out.hypotheses are also lists of lists (one per sentence)
        ref_words_list = out.references[0]
        hyp_words_list = out.hypotheses[0]
        
        for chunk in out.alignments[0]:
            type_ = chunk.type
            
            # Get the words corresponding to this chunk
            # AlignmentChunk uses ref_start_idx/ref_end_idx and hyp_start_idx/hyp_end_idx
            ref_words = ref_words_list[chunk.ref_start_idx : chunk.ref_end_idx]
            hyp_words = hyp_words_list[chunk.hyp_start_idx : chunk.hyp_end_idx]
            
            if type_ == 'equal':
                for r, h in zip(ref_words, hyp_words):
                    alignment.append({'type': 'correct', 'ref': r, 'hyp': h})
            
            elif type_ == 'substitute':
                # In complex substitutions, lengths might differ. 
                # We pair them up as much as possible.
                max_len = max(len(ref_words), len(hyp_words))
                for i in range(max_len):
                    r = ref_words[i] if i < len(ref_words) else None
                    h = hyp_words[i] if i < len(hyp_words) else None
                    
                    if r and h:
                        alignment.append({'type': 'substitution', 'ref': r, 'hyp': h})
                    elif r:
                        # Leftover reference words count as deletions
                        alignment.append({'type': 'deletion', 'ref': r, 'hyp': None})
                    elif h:
                        # Leftover hypothesis words count as insertions
                        alignment.append({'type': 'insertion', 'ref': None, 'hyp': h})
                        
            elif type_ == 'delete':
                for r in ref_words:
                    alignment.append({'type': 'deletion', 'ref': r, 'hyp': None})
                    
            elif type_ == 'insert':
                for h in hyp_words:
                    alignment.append({'type': 'insertion', 'ref': None, 'hyp': h})
                    
        return alignment

    def categorize_errors(self, alignment: List[Dict[str, Optional[str]]]) -> Dict[str, int]:
        """
        Count occurrences of each error type in an alignment.
        
        Args:
            alignment: Output from get_alignment()
            
        Returns:
            Dictionary with counts for correct, substitution, deletion, insertion, and total_errors.
        """
        counts = Counter(item['type'] for item in alignment)
        return {
            'correct': counts['correct'],
            'substitution': counts['substitution'],
            'deletion': counts['deletion'],
            'insertion': counts['insertion'],
            'total_errors': counts['substitution'] + counts['deletion'] + counts['insertion']
        }

    def find_confusion_pairs(self, alignments: List[List[Dict[str, Optional[str]]]]) -> List[Tuple[Tuple[str, str], int]]:
        """
        Identify common substitution patterns across multiple alignments.
        
        Args:
            alignments: List of alignment lists (output from get_alignment)
            
        Returns:
            List of ((ref_word, hyp_word), count) sorted by count descending.
        """
        pairs = []
        for align in alignments:
            for item in align:
                if item['type'] == 'substitution':
                    pairs.append((item['ref'], item['hyp']))
        
        return Counter(pairs).most_common()

    def get_high_error_samples(self, results: List[Dict[str, Any]], threshold: float = 50.0) -> List[Dict[str, Any]]:
        """
        Filter results for samples with WER above a certain threshold.
        
        Args:
            results: List of evaluation results
            threshold: WER threshold (default: 50.0%)
            
        Returns:
            List of samples with WER >= threshold
        """
        return [r for r in results if r.get('wer', 0) >= threshold]

    def calculate_aggregate_stats(self, results: List[Dict]) -> Dict[str, float]:
        """
        Calculate mean, median, and standard deviation for WER, CER, and BLEU.
        
        Args:
            results: List of evaluation results with wer, cer, and bleu fields
            
        Returns:
            Dictionary with aggregate statistics
        """
        wers = [r['wer'] for r in results]
        cers = [r['cer'] for r in results]
        bleus = [r.get('bleu', 0.0) for r in results]  # MODIFIED: Extract BLEU scores
        
        return {
            'mean_wer': statistics.mean(wers),
            'median_wer': statistics.median(wers),
            'std_wer': statistics.stdev(wers) if len(wers) > 1 else 0.0,
            'mean_cer': statistics.mean(cers),
            'median_cer': statistics.median(cers),
            'std_cer': statistics.stdev(cers) if len(cers) > 1 else 0.0,
            # NEW: BLEU statistics
            'mean_bleu': statistics.mean(bleus) if bleus else 0.0,
            'median_bleu': statistics.median(bleus) if bleus else 0.0,
            'std_bleu': statistics.stdev(bleus) if len(bleus) > 1 else 0.0,
        }

    def format_alignment_readable(self, alignment: List[Dict[str, Optional[str]]]) -> str:
        """
        Format alignment as a readable 3-line string for visual inspection.
        
        Args:
            alignment: Output from get_alignment()
            
        Returns:
            String with REF, HYP, and TYPE lines aligned vertically
        """
        if not alignment:
            return "REF:  (empty)\nHYP:  (empty)\nTYPE: "
        
        ref_line = "REF:  "
        hyp_line = "HYP:  "
        type_line = "TYPE: "
        
        for item in alignment:
            ref_word = item['ref'] if item['ref'] else '*' * 5
            hyp_word = item['hyp'] if item['hyp'] else '*' * 5
            type_char = item['type'][0].upper()  # C, S, D, I
            
            # Use the longer word for column width
            max_len = max(len(ref_word), len(hyp_word))
            
            ref_line += ref_word.ljust(max_len) + " "
            hyp_line += hyp_word.ljust(max_len) + " "
            type_line += type_char.ljust(max_len) + " "
        
        return ref_line.rstrip() + "\n" + hyp_line.rstrip() + "\n" + type_line.rstrip()

    def analyze_by_dialect(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Group results by dialect and compute statistics and confusion patterns for each.
        
        Args:
            results: List of evaluation results with dialect, reference, hypothesis, wer, cer, bleu fields
            
        Returns:
            Dictionary mapping dialect to analysis results
        """
        # Group by dialect
        by_dialect = defaultdict(list)
        for r in results:
            dialect = r.get('dialect', 'unknown')
            by_dialect[dialect].append(r)
        
        analysis = {}
        
        for dialect, samples in by_dialect.items():
            # Compute WER/CER/BLEU stats
            wers = [s['wer'] for s in samples]
            cers = [s['cer'] for s in samples]
            bleus = [s.get('bleu', 0.0) for s in samples]  # MODIFIED: Extract BLEU
            
            # Compute alignments to get error distribution
            alignments = [self.get_alignment(s['reference'], s['hypothesis']) for s in samples]
            all_counts = [self.categorize_errors(align) for align in alignments]
            
            total_sub = sum(counts['substitution'] for counts in all_counts)
            total_del = sum(counts['deletion'] for counts in all_counts)
            total_ins = sum(counts['insertion'] for counts in all_counts)
            total_cor = sum(counts['correct'] for counts in all_counts)
            
            # Calculate rates over ERRORS ONLY (excluding correct)
            total_errs = total_sub + total_del + total_ins
            
            analysis[dialect] = {
                'sample_count': len(samples),
                'mean_wer': statistics.mean(wers),
                'std_wer': statistics.stdev(wers) if len(wers) > 1 else 0.0,
                'mean_cer': statistics.mean(cers),
                # NEW: BLEU statistics per dialect
                'mean_bleu': statistics.mean(bleus) if bleus else 0.0,
                'std_bleu': statistics.stdev(bleus) if len(bleus) > 1 else 0.0,
                'error_distribution': {
                    'substitution': total_sub,
                    'deletion': total_del,
                    'insertion': total_ins,
                    'correct': total_cor,
                    'sub_rate': total_sub / total_errs if total_errs > 0 else 0.0,
                    'del_rate': total_del / total_errs if total_errs > 0 else 0.0,
                    'ins_rate': total_ins / total_errs if total_errs > 0 else 0.0,
                },
                'top_confusions': self.find_confusion_pairs(alignments)[:10]
            }
        
        return analysis

    def analyze_wer_bleu_correlation(
        self, 
        results: List[Dict[str, Any]], 
        wer_threshold: float = 50.0,
        bleu_threshold: float = 40.0
    ) -> Dict[str, Any]:
        """
        Analyze correlation between WER and BLEU to identify semantic preservation.
        
        High-WER + High-BLEU samples = Valid paraphrases (semantic preservation)
        High-WER + Low-BLEU samples = True errors (semantic drift)
        
        Args:
            results: List of evaluation results with wer, cer, bleu scores
            wer_threshold: WER above which samples are considered "high error" (default: 50.0)
            bleu_threshold: BLEU above which samples are considered "high similarity" (default: 40.0)
            
        Returns:
            Dictionary with categorized samples and statistics
        """
        high_wer_high_bleu = []  # Valid paraphrases
        high_wer_low_bleu = []   # True errors
        low_wer_high_bleu = []   # Good translations
        low_wer_low_bleu = []    # Edge cases
        
        for sample in results:
            wer = sample.get('wer', 0.0)
            bleu = sample.get('bleu', 0.0)
            
            if wer >= wer_threshold and bleu >= bleu_threshold:
                high_wer_high_bleu.append(sample)
            elif wer >= wer_threshold and bleu < bleu_threshold:
                high_wer_low_bleu.append(sample)
            elif wer < wer_threshold and bleu >= bleu_threshold:
                low_wer_high_bleu.append(sample)
            else:
                low_wer_low_bleu.append(sample)
        
        return {
            'summary': {
                'total_samples': len(results),
                'high_wer_high_bleu_count': len(high_wer_high_bleu),
                'high_wer_low_bleu_count': len(high_wer_low_bleu),
                'low_wer_high_bleu_count': len(low_wer_high_bleu),
                'low_wer_low_bleu_count': len(low_wer_low_bleu),
                'semantic_preservation_rate': (
                    len(high_wer_high_bleu) / len(results) * 100 
                    if results else 0.0
                )
            },
            'high_wer_high_bleu_samples': high_wer_high_bleu[:10],  # Top 10 examples
            'high_wer_low_bleu_samples': high_wer_low_bleu[:10],
        }