from typing import Any, List, Dict
import numpy as np
import jiwer
from sacrebleu import sentence_bleu
from sacrebleu.metrics import CHRF

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


def calculate_chrf(reference: str, hypothesis: str) -> float:
    """
    Calculate chrF score between reference and hypothesis.
    
    chrF measures character n-gram F-score, which is more robust than
    word-level metrics for morphologically rich languages like German.
    
    Args:
        reference: Ground truth text
        hypothesis: Predicted text
        
    Returns:
        chrF score (0-100)
    """
    reference = _normalize_text(reference)
    hypothesis = _normalize_text(hypothesis)
    
    if not reference or not hypothesis:
        return 0.0
    
    chrf = CHRF()
    result = chrf.sentence_score(hypothesis, [reference])
    return result.score


def batch_chrf(references: List[str], hypotheses: List[str]) -> Dict:
    """
    Calculate chrF score for a batch of reference-hypothesis pairs.
    
    Uses corpus-level chrF for overall score (consistent with WER/CER aggregate
    calculation) and sentence-level chrF for per-sample analysis.
    
    Empty references or hypotheses receive a score of 0.0.
    
    Args:
        references: List of ground truth texts
        hypotheses: List of predicted texts
        
    Returns:
        Dictionary containing:
            - overall_chrf: Corpus-level chrF score across all samples
            - per_sample_chrf: List of chrF scores for each sample
    """
    if len(references) != len(hypotheses):
        raise ValueError("References and hypotheses must have the same length")
    
    if not references:
        return {"overall_chrf": 0.0, "per_sample_chrf": []}
    
    # Normalize all texts
    norm_references = [_normalize_text(ref) for ref in references]
    norm_hypotheses = [_normalize_text(hyp) for hyp in hypotheses]
    
    chrf = CHRF()
    
    # Calculate per-sample chrF using sentence_score
    per_sample_chrf = []
    for ref, hyp in zip(norm_references, norm_hypotheses):
        if not ref or not hyp:
            chrf_score = 0.0
        else:
            chrf_score = chrf.sentence_score(hyp, [ref]).score
        per_sample_chrf.append(chrf_score)
    
    # Calculate overall chrF using corpus-level aggregation
    # Filter out empty pairs for corpus-level calculation
    valid_refs = []
    valid_hyps = []
    for ref, hyp in zip(norm_references, norm_hypotheses):
        if ref and hyp:
            valid_refs.append(ref)
            valid_hyps.append(hyp)
    
    if valid_refs:
        overall_chrf = chrf.corpus_score(valid_hyps, [valid_refs]).score
    else:
        overall_chrf = 0.0
    
    return {
        "overall_chrf": overall_chrf,
        "per_sample_chrf": per_sample_chrf
    }


def calculate_semdist(reference: str, hypothesis: str, model: Any) -> float:
    """
    Calculate Semantic Distance (SemDist) between reference and hypothesis.
    
    SemDist = 1 - cosine_similarity between sentence embeddings.
    Lower values indicate higher semantic similarity (0 = identical meaning,
    1 = completely unrelated).
    
    Args:
        reference: Ground truth text
        hypothesis: Predicted text
        model: Pre-loaded SentenceTransformer instance
        
    Returns:
        Semantic distance (0-1), lower is better
    """
    reference = _normalize_text(reference)
    hypothesis = _normalize_text(hypothesis)
    
    if not reference or not hypothesis:
        return 1.0
    
    embeddings = model.encode([reference, hypothesis])
    ref_emb = embeddings[0]
    hyp_emb = embeddings[1]
    
    # Compute cosine similarity with zero-norm guard
    norm_ref = np.linalg.norm(ref_emb)
    norm_hyp = np.linalg.norm(hyp_emb)
    
    if norm_ref == 0.0 or norm_hyp == 0.0:
        return 1.0
    
    cosine_sim = np.dot(ref_emb, hyp_emb) / (norm_ref * norm_hyp)
    # Clamp to [-1, 1] to handle floating-point imprecision
    cosine_sim = float(np.clip(cosine_sim, -1.0, 1.0))
    
    return 1.0 - cosine_sim


def batch_semdist(references: List[str], hypotheses: List[str], model: Any) -> Dict:
    """
    Calculate Semantic Distance for a batch of reference-hypothesis pairs.
    
    Batch-encodes all texts at once for efficiency. Overall SemDist is the
    mean of per-sample scores (no corpus-level aggregation applies to embeddings).
    
    Empty references or hypotheses receive a score of 1.0 (maximum distance).
    
    Args:
        references: List of ground truth texts
        hypotheses: List of predicted texts
        model: Pre-loaded SentenceTransformer instance
        
    Returns:
        Dictionary containing:
            - overall_semdist: Mean semantic distance across all samples
            - per_sample_semdist: List of semantic distances for each sample
    """
    if len(references) != len(hypotheses):
        raise ValueError("References and hypotheses must have the same length")
    
    if not references:
        return {"overall_semdist": 0.0, "per_sample_semdist": []}
    
    # Normalize all texts
    norm_references = [_normalize_text(ref) for ref in references]
    norm_hypotheses = [_normalize_text(hyp) for hyp in hypotheses]
    
    # Identify valid pairs (both non-empty) for batch encoding
    valid_indices = []
    all_texts = []
    for i, (ref, hyp) in enumerate(zip(norm_references, norm_hypotheses)):
        if ref and hyp:
            valid_indices.append(i)
            all_texts.extend([ref, hyp])
    
    # Batch encode all valid texts at once for efficiency
    if all_texts:
        all_embeddings = model.encode(all_texts)
    
    # Calculate per-sample SemDist
    per_sample_semdist = [1.0] * len(references)  # Default: maximum distance
    
    for j, idx in enumerate(valid_indices):
        ref_emb = all_embeddings[j * 2]
        hyp_emb = all_embeddings[j * 2 + 1]
        
        norm_ref = np.linalg.norm(ref_emb)
        norm_hyp = np.linalg.norm(hyp_emb)
        
        if norm_ref == 0.0 or norm_hyp == 0.0:
            per_sample_semdist[idx] = 1.0
        else:
            cosine_sim = np.dot(ref_emb, hyp_emb) / (norm_ref * norm_hyp)
            cosine_sim = float(np.clip(cosine_sim, -1.0, 1.0))
            per_sample_semdist[idx] = 1.0 - cosine_sim
    
    # Overall SemDist = mean of per-sample scores
    overall_semdist = sum(per_sample_semdist) / len(per_sample_semdist)
    
    return {
        "overall_semdist": overall_semdist,
        "per_sample_semdist": per_sample_semdist
    }