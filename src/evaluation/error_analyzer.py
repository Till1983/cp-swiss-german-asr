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
            results: List of result dictionaries (from evaluator.py)
            threshold: WER threshold (0-100)
            
        Returns:
            Filtered list of result dictionaries.
        """
        return [r for r in results if r.get('wer', 0) > threshold]

    def analyze_by_dialect(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Group results by dialect and calculate aggregate statistics and error patterns.
        
        Args:
            results: List of result dictionaries containing 'dialect', 'reference', 'hypothesis', 'wer', 'cer'.
            
        Returns:
            Dictionary keyed by dialect name containing stats and patterns.
        """
        by_dialect = defaultdict(list)
        for r in results:
            dialect = r.get('dialect', 'unknown')
            by_dialect[dialect].append(r)
            
        analysis = {}
        for dialect, samples in by_dialect.items():
            wers = [s['wer'] for s in samples]
            cers = [s['cer'] for s in samples]
            
            # Calculate error distribution by running alignment on all samples
            total_sub = 0
            total_del = 0
            total_ins = 0
            total_cor = 0
            
            alignments = []
            for s in samples:
                align = self.get_alignment(s['reference'], s['hypothesis'])
                alignments.append(align)
                counts = self.categorize_errors(align)
                
                total_sub += counts['substitution']
                total_del += counts['deletion']
                total_ins += counts['insertion']
                total_cor += counts['correct']
            
            total_errs = total_sub + total_del + total_ins
            total_ops = total_errs + total_cor
            
            analysis[dialect] = {
                'sample_count': len(samples),
                'mean_wer': statistics.mean(wers) if wers else 0.0,
                'std_wer': statistics.stdev(wers) if len(wers) > 1 else 0.0,
                'mean_cer': statistics.mean(cers) if cers else 0.0,
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

    def calculate_aggregate_stats(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate basic aggregate statistics (mean, median, stdev) for WER and CER.
        """
        if not results:
            return {}
            
        wers = [r['wer'] for r in results]
        cers = [r['cer'] for r in results]
        
        return {
            'mean_wer': statistics.mean(wers),
            'median_wer': statistics.median(wers),
            'std_wer': statistics.stdev(wers) if len(wers) > 1 else 0.0,
            'mean_cer': statistics.mean(cers),
            'median_cer': statistics.median(cers),
            'std_cer': statistics.stdev(cers) if len(cers) > 1 else 0.0,
        }

    def format_alignment_readable(self, alignment: List[Dict[str, Optional[str]]]) -> str:
        """
        Create a human-readable string representation of an alignment.
        
        Example output:
        REF:  hello  world
        HYP:  hello  *****
        TYPE:   C      D
        """
        ref_line = []
        hyp_line = []
        type_line = []
        
        for item in alignment:
            r = item['ref'] if item['ref'] else "*" * len(item['hyp'] or "")
            h = item['hyp'] if item['hyp'] else "*" * len(item['ref'] or "")
            
            # Determine width needed for this column
            width = max(len(r), len(h), 1)
            
            ref_line.append(f"{r:<{width}}")
            hyp_line.append(f"{h:<{width}}")
            
            # Type code: C=Correct, S=Sub, D=Del, I=Ins
            code = item['type'][0].upper()
            type_line.append(f"{code:<{width}}")
            
        return (
            f"REF:  {'  '.join(ref_line)}\n"
            f"HYP:  {'  '.join(hyp_line)}\n"
            f"TYPE: {'  '.join(type_line)}"
        )