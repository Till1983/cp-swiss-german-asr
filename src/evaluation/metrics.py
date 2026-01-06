from typing import List, Dict
import jiwer
from sacrebleu import sentence_bleu

"""
ASR Evaluation Metrics Module

This module provides functions for calculating various speech recognition
evaluation metrics including WER, CER, and BLEU score.
"""



def _normalize_text(text: str, mode: str = "asr_fair") -> str:
    """
    Normalize text for metric calculation.
    
    Modes:
    - "standard": lowercase + whitespace collapse (applied in previous analyses, preserves punctuation)
    - "asr_fair": lowercase + punctuation removal + whitespace collapse (new default for future ASR evaluations)
    """
    text = text.lower()
    
    if mode == "asr_fair":
        import string
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    return " ".join(text.split())


def _filter_empty_references(references: List[str], hypotheses: List[str]) -> tuple:
    """
    Filter out pairs where reference is empty (undefined for WER/CER).
    
    Empty references have no words to compare against, making WER/CER mathematically
    undefined. This function removes such pairs to enable valid metric calculation.
    
    Args:
        references: List of reference texts
        hypotheses: List of hypothesis texts
        
    Returns:
        Tuple of (filtered_references, filtered_hypotheses, valid_indices)
        where valid_indices maps filtered positions back to original positions
    """
    valid_pairs = []
    valid_indices = []
    
    for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
        if ref.strip():  # Only keep non-empty references
            valid_pairs.append((ref, hyp))
            valid_indices.append(i)
    
    if not valid_pairs:
        return [], [], []
    
    filtered_refs, filtered_hyps = zip(*valid_pairs)
    return list(filtered_refs), list(filtered_hyps), valid_indices


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER) between reference and hypothesis.
    
    Args:
        reference: Ground truth text
        hypothesis: Predicted text
        
    Returns:
        WER as a percentage (0-100)
    """
    reference = _normalize_text(reference)
    hypothesis = _normalize_text(hypothesis)
    
    if not reference:
        return 0.0 if not hypothesis else 100.0
    
    wer = jiwer.wer(reference, hypothesis)
    return wer * 100.0


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate (CER) between reference and hypothesis.
    
    Args:
        reference: Ground truth text
        hypothesis: Predicted text
        
    Returns:
        CER as a percentage (0-100)
    """
    reference = _normalize_text(reference)
    hypothesis = _normalize_text(hypothesis)
    
    if not reference:
        return 0.0 if not hypothesis else 100.0
    
    cer = jiwer.cer(reference, hypothesis)
    return cer * 100.0


def calculate_bleu_score(reference: str, hypothesis: str) -> float:
    """
    Calculate BLEU score between reference and hypothesis.
    
    Args:
        reference: Ground truth text
        hypothesis: Predicted text
        
    Returns:
        BLEU score (0-100)
    """
    reference = _normalize_text(reference)
    hypothesis = _normalize_text(hypothesis)
    
    if not reference or not hypothesis:
        return 0.0
    
    # Use sentence_bleu for single samples instead of corpus_bleu
    bleu = sentence_bleu(hypothesis, [reference])
    return bleu.score


def batch_wer(references: List[str], hypotheses: List[str]) -> Dict:
    """
    Calculate WER for a batch of reference-hypothesis pairs.
    
    Uses the standard aggregate WER calculation: (S+D+I) / N across all samples,
    where S=substitutions, D=deletions, I=insertions, N=total reference words.
    
    Empty references are filtered out as they are mathematically undefined for WER.
    
    Args:
        references: List of ground truth texts
        hypotheses: List of predicted texts
        
    Returns:
        Dictionary containing:
            - overall_wer: WER across all valid samples (aggregate method)
            - per_sample_wer: List of WER for each sample (including None for filtered samples)
    """
    if len(references) != len(hypotheses):
        raise ValueError("References and hypotheses must have the same length")
    
    if not references:
        return {"overall_wer": 0.0, "per_sample_wer": []}
    
    # Normalize all texts
    norm_references = [_normalize_text(ref) for ref in references]
    norm_hypotheses = [_normalize_text(hyp) for hyp in hypotheses]
    
    # Filter empty references for aggregate calculation
    filtered_refs, filtered_hyps, valid_indices = _filter_empty_references(
        norm_references, norm_hypotheses
    )
    
    # If no valid samples, return zeros
    if not filtered_refs:
        return {
            "overall_wer": 0.0,
            "per_sample_wer": [None] * len(references)  # All filtered
        }
    
    # Calculate overall WER using AGGREGATE method (standard for ASR)
    # This is: total_errors / total_reference_words across all samples
    overall_wer = jiwer.wer(filtered_refs, filtered_hyps) * 100.0
    
    # Calculate per-sample WER for analysis
    # Note: per_sample_wer is for detailed analysis, NOT for averaging
    per_sample_wer = []
    for ref, hyp in zip(norm_references, norm_hypotheses):
        if not ref:
            # Empty reference - WER undefined, mark as None
            per_sample_wer.append(None)
        else:
            wer_score = jiwer.wer(ref, hyp) * 100.0
            per_sample_wer.append(wer_score)
    
    return {
        "overall_wer": overall_wer,
        "per_sample_wer": per_sample_wer
    }

def batch_cer(references: List[str], hypotheses: List[str]) -> Dict:
    """
    Calculate CER for a batch of reference-hypothesis pairs.
    
    Uses the standard aggregate CER calculation: (S+D+I) / N across all samples,
    where S=substitutions, D=deletions, I=insertions, N=total reference characters.
    
    Empty references are filtered out as they are mathematically undefined for CER.
    
    Args:
        references: List of ground truth texts
        hypotheses: List of predicted texts
        
    Returns:
        Dictionary containing:
            - overall_cer: CER across all valid samples (aggregate method)
            - per_sample_cer: List of CER for each sample (including None for filtered samples)
    """
    if len(references) != len(hypotheses):
        raise ValueError("References and hypotheses must have the same length")
    
    if not references:
        return {"overall_cer": 0.0, "per_sample_cer": []}
    
    # Normalize all texts
    norm_references = [_normalize_text(ref) for ref in references]
    norm_hypotheses = [_normalize_text(hyp) for hyp in hypotheses]
    
    # Filter empty references for aggregate calculation
    filtered_refs, filtered_hyps, valid_indices = _filter_empty_references(
        norm_references, norm_hypotheses
    )
    
    # If no valid samples, return zeros
    if not filtered_refs:
        return {
            "overall_cer": 0.0,
            "per_sample_cer": [None] * len(references)  # All filtered
        }
    
    # Calculate overall CER using AGGREGATE method (standard for ASR)
    # This is: total_errors / total_reference_characters across all samples
    overall_cer = jiwer.cer(filtered_refs, filtered_hyps) * 100.0
    
    # Calculate per-sample CER for analysis
    # Note: per_sample_cer is for detailed analysis, NOT for averaging
    per_sample_cer = []
    for ref, hyp in zip(norm_references, norm_hypotheses):
        if not ref:
            # Empty reference - CER undefined, mark as None
            per_sample_cer.append(None)
        else:
            cer_score = jiwer.cer(ref, hyp) * 100.0
            per_sample_cer.append(cer_score)
    
    return {
        "overall_cer": overall_cer,
        "per_sample_cer": per_sample_cer
    }

def batch_bleu(references: List[str], hypotheses: List[str]) -> Dict:
    """
    Calculate BLEU score for a batch of reference-hypothesis pairs.
    
    BLEU is calculated per-sentence and then averaged (standard for sentence-level BLEU).
    Empty references or hypotheses receive a score of 0.0.
    
    Args:
        references: List of ground truth texts
        hypotheses: List of predicted texts
        
    Returns:
        Dictionary containing:
            - overall_bleu: BLEU score averaged across all samples
            - per_sample_bleu: List of BLEU scores for each sample  
    """
    if len(references) != len(hypotheses):
        raise ValueError("References and hypotheses must have the same length")
    
    if not references:
        return {"overall_bleu": 0.0, "per_sample_bleu": []}
    
    # Normalize all texts
    norm_references = [_normalize_text(ref) for ref in references]
    norm_hypotheses = [_normalize_text(hyp) for hyp in hypotheses]
    
    # Calculate per-sample BLEU using sentence_bleu
    per_sample_bleu = []
    for ref, hyp in zip(norm_references, norm_hypotheses):
        if not ref or not hyp:
            bleu_score = 0.0
        else:
            bleu_score = sentence_bleu(hyp, [ref]).score
        per_sample_bleu.append(bleu_score)
    
    # Calculate overall BLEU as mean (standard for sentence-level BLEU)
    # Note: This is different from WER/CER which use aggregate calculation
    overall_bleu = sum(per_sample_bleu) / len(per_sample_bleu) if per_sample_bleu else 0.0
    
    return {
        "overall_bleu": overall_bleu,
        "per_sample_bleu": per_sample_bleu
    }