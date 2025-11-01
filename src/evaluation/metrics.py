from typing import List, Dict
import jiwer
from sacrebleu import corpus_bleu

"""
ASR Evaluation Metrics Module

This module provides functions for calculating various speech recognition
evaluation metrics including WER, CER, and BLEU score.
"""



def _normalize_text(text: str) -> str:
    """
    Normalize text by converting to lowercase and stripping whitespace.
    
    Args:
        text: Input text string
        
    Returns:
        Normalized text string
    """
    return text.lower().strip()


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
    
    bleu = corpus_bleu([hypothesis], [[reference]])
    return bleu.score


def batch_wer(references: List[str], hypotheses: List[str]) -> Dict:
    """
    Calculate WER for a batch of reference-hypothesis pairs.
    
    Args:
        references: List of ground truth texts
        hypotheses: List of predicted texts
        
    Returns:
        Dictionary containing:
            - overall_wer: WER across all samples
            - per_sample_wer: List of WER for each sample
    """
    if len(references) != len(hypotheses):
        raise ValueError("References and hypotheses must have the same length")
    
    if not references:
        return {"overall_wer": 0.0, "per_sample_wer": []}
    
    # Normalize all texts
    norm_references = [_normalize_text(ref) for ref in references]
    norm_hypotheses = [_normalize_text(hyp) for hyp in hypotheses]
    
    # Calculate per-sample WER
    per_sample_wer = []
    for ref, hyp in zip(norm_references, norm_hypotheses):
        if not ref:
            wer_score = 0.0 if not hyp else 100.0
        else:
            wer_score = jiwer.wer(ref, hyp) * 100.0
        per_sample_wer.append(wer_score)
    
    # Calculate overall WER
    overall_wer = jiwer.wer(norm_references, norm_hypotheses) * 100.0
    
    return {
        "overall_wer": overall_wer,
        "per_sample_wer": per_sample_wer
    }