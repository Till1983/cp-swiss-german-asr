# Model Selection

## Table of Contents
- [Evaluation Methodology](#evaluation-methodology)
  - [Normalisation Modes](#normalisation-modes)
  - [Metric Definitions](#metric-definitions)
- [Final Model Suite](#final-model-suite)
- [Results – Legacy (Standard Normalisation)](#results--legacy-standard-normalisation)
- [Results – Current (ASR-Fair Normalisation, March 2026)](#results--current-asr-fair-normalisation-march-2026)
- [Model Selection Rationale](#model-selection-rationale)
- [Key Findings](#key-findings)

---

## Evaluation Methodology

### Normalisation Modes

Two normalisation strategies have been used across the lifetime of this project. Understanding the difference is essential for interpreting and comparing results.

| Mode | Operations Applied | When Used |
|------|--------------------|-----------|
| **Standard Normalisation** (legacy) | Lowercase only | All evaluations prior to January 2026 |
| **ASR-Fair Normalisation** (current default) | Lowercase + punctuation removal | All evaluations from January 2026 onward |

**Why Standard Normalisation inflated error metrics:**

Models such as `wav2vec2-german-with-lm` (LM-enhanced decoding) produce output with different punctuation patterns compared to greedy CTC decoders. Standard Normalisation preserved these punctuation tokens, causing word-boundary mismatches that inflated WER and CER and deflated BLEU for affected models. The effect is not limited to Wav2Vec2: Whisper models also produce varying punctuation, meaning Standard Normalisation inflated WER, CER, and deflated BLEU for **virtually all models** to some degree.

**ASR-Fair Normalisation** removes punctuation from both reference and hypothesis before metric computation, ensuring that punctuation style differences do not skew scores. This is the standard approach in multilingual ASR benchmarking.

> **Reading legacy results:** Any result table or model entry labelled "(Standard Normalisation)" or "legacy" is affected by this issue. The numbers are preserved for historical reference and reproducibility, but **current results (ASR-Fair) should be used for model comparison**.

---

### Metric Definitions

| Metric | Range | Direction | Description |
|--------|-------|-----------|-------------|
| **WER** (Word Error Rate) | 0–100% | Lower is better | Percentage of words incorrectly recognised/translated (Levenshtein edit distance at word level) |
| **CER** (Character Error Rate) | 0–100% | Lower is better | Percentage of characters incorrectly recognised/translated (Levenshtein edit distance at character level) |
| **BLEU** | 0–100 | Higher is better | Modified n-gram precision measuring overlap between hypothesis and reference (sentence-level mean) |
| **chrF** (Character F-score) | 0–100 | Higher is better | Character n-gram F-score (β=1). More robust than WER/CER for morphologically rich languages like German because it handles compound words and inflections gracefully. Corpus-level aggregation. |
| **SemDist** (Semantic Distance) | 0–1 | Lower is better | Cosine distance between sentence embeddings computed by `paraphrase-multilingual-mpnet-base-v2`. A value of 0 means identical meaning; 1 means completely unrelated. Captures meaning preservation even when surface form differs (important for evaluating Whisper's translation output). |

**Implementation note:** chrF uses `sacrebleu.CHRF` with default parameters (β=1, character n-gram order 6). SemDist uses `sentence-transformers` with batch encoding; if `sentence-transformers` is not installed, SemDist is skipped and reported as `null`.

---

## Final Model Suite

This document describes the seven ASR models evaluated on the Swiss German test corpus (863 samples across 17 dialects). The per-model metrics shown here use **Standard Normalisation (legacy)** — see [Results – Current](#results--current-asr-fair-normalisation-march-2026) for updated figures including chrF and SemDist.

### Whisper Models (OpenAI)

Zero-shot multilingual models trained on 680,000 hours of web data. All Whisper variants perform Swiss German-to-Standard German translation.

1. **whisper-large-v2** (1550M parameters)
   - Architecture: Encoder-decoder transformer
   - Evaluation approach: Zero-shot translation
   - Overall WER: 28.00% *(Standard Normalisation, legacy)*
   - Overall CER: 12.21% *(Standard Normalisation, legacy)*
   - Overall BLEU: 57.74 *(Standard Normalisation, legacy)*
   - Role: Best-performing model, baseline reference

2. **whisper-large-v3** (1550M parameters)
   - Architecture: Encoder-decoder transformer (updated training data)
   - Evaluation approach: Zero-shot translation
   - Overall WER: 29.53% *(Standard Normalisation, legacy)*
   - Overall CER: 13.32% *(Standard Normalisation, legacy)*
   - Overall BLEU: 56.28 *(Standard Normalisation, legacy)*
   - Role: Latest Whisper version comparison

3. **whisper-large-v3-turbo** (809M parameters)
   - Architecture: Optimised encoder-decoder with reduced decoder layers
   - Evaluation approach: Zero-shot translation
   - Overall WER: 30.94% *(Standard Normalisation, legacy)*
   - Overall CER: 13.71% *(Standard Normalisation, legacy)*
   - Overall BLEU: 53.97 *(Standard Normalisation, legacy)*
   - Role: Efficiency-accuracy trade-off evaluation

4. **whisper-medium** (769M parameters)
   - Architecture: Encoder-decoder transformer (smaller variant)
   - Evaluation approach: Zero-shot translation
   - Overall WER: 34.11% *(Standard Normalisation, legacy)*
   - Overall CER: 15.93% *(Standard Normalisation, legacy)*
   - Overall BLEU: 50.82 *(Standard Normalisation, legacy)*
   - Role: Resource-constrained deployment baseline

### Wav2Vec2 Models (German-trained)

Self-supervised speech recognition models trained on German Common Voice data. These models perform direct transcription without explicit translation.

5. **wav2vec2-1b-german-cv11** (1000M parameters)
   - HuggingFace: `aware-ai/wav2vec2-xls-r-1b-german-cv11`
   - Architecture: XLS-R 1B fine-tuned on German Common Voice v11
   - Evaluation approach: Zero-shot transcription
   - Overall WER: 72.42% *(Standard Normalisation, legacy)*
   - Overall CER: 29.24% *(Standard Normalisation, legacy)*
   - Overall BLEU: 14.93 *(Standard Normalisation, legacy)*
   - Role: Large-scale German ASR comparison

6. **wav2vec2-german-with-lm** (317M parameters)
   - HuggingFace: `aware-ai/wav2vec2-large-xlsr-53-german-with-lm`
   - Architecture: XLSR-53 with optional KenLM language model integration
   - Evaluation approach: Zero-shot transcription (LM-enhanced decoding)
   - Overall WER: 75.28% *(Standard Normalisation, legacy — see normalisation correction below)*
   - Overall CER: 31.52% *(Standard Normalisation, legacy)*
   - Overall BLEU: 13.89 *(Standard Normalisation, legacy)*
   - Role: Language model-enhanced decoding evaluation

### Multilingual / Foundation Models

7. **seamless-m4t-v2-large** (2.3B parameters)
   - HuggingFace: `facebook/seamless-m4t-v2-large`
   - Architecture: Multimodal sequence-to-sequence foundation model (speech, text, translation)
   - Evaluation approach: Zero-shot speech-to-text translation (Swiss German → Standard German)
   - Role: Comparison of a large multilingual foundation model against domain-specific Whisper variants

---

## Results – Legacy (Standard Normalisation)

> **⚠️ These results were produced using Standard Normalisation (lowercase only, punctuation preserved).** This approach inflated WER and CER and deflated BLEU for virtually all models due to punctuation mismatches between hypothesis and reference. They are preserved here for reproducibility and historical reference. **Do not use these figures for current model comparisons.**

| Model | WER (%) ↓ | CER (%) ↓ | BLEU ↑ | Samples |
|-------|-----------|-----------|--------|---------|
| whisper-large-v2 | 28.00 | 12.21 | 57.74 | 863 |
| whisper-large-v3 | 29.53 | 13.32 | 56.28 | 863 |
| whisper-large-v3-turbo | 30.94 | 13.71 | 53.97 | 863 |
| whisper-medium | 34.11 | 15.93 | 50.82 | 863 |
| wav2vec2-1b-german-cv11 | 72.42 | 29.24 | 14.93 | 863 |
| wav2vec2-german-with-lm | 75.28 | 31.52 | 13.89 | 863 |

*seamless-m4t-v2-large was not evaluated under Standard Normalisation.*

---

## Results – Current (ASR-Fair Normalisation, March 2026)

All results below use **ASR-Fair Normalisation** (lowercase + punctuation removal). This is the current default and removes the systematic bias introduced by punctuation differences. Two new metrics — **chrF** and **SemDist** — are included from this evaluation cycle onward.

Results directory: `results/metrics/20260303_105207/` (all models except turbo) and `results/metrics/20260303_121313/` (whisper-large-v3-turbo).

| Model | WER (%) ↓ | CER (%) ↓ | BLEU ↑ | chrF ↑ | SemDist ↓ | Samples |
|-------|-----------|-----------|--------|--------|-----------|---------|
| whisper-large-v2 | **25.63** | **12.18** | **58.46** | **85.72** | **0.0551** | 863 |
| whisper-large-v3 | 26.93 | 13.29 | 56.93 | 85.12 | 0.0574 | 863 |
| whisper-large-v3-turbo | 28.23 | 13.66 | 54.63 | 83.97 | 0.0586 | 863 |
| whisper-medium | 31.20 | 15.90 | 51.47 | 82.46 | 0.0695 | 863 |
| seamless-m4t-v2-large | 46.77 | 22.43 | 35.14 | 72.73 | 0.1335 | 863 |
| wav2vec2-1b-german-cv11 | 70.97 | 29.46 | 14.29 | 56.23 | 0.2857 | 863 |
| wav2vec2-german-with-lm | 70.05 | 30.60 | 16.28 | 56.50 | 0.2855 | 863 |

**Bold** = best score per metric.

### Comparing Legacy and Current Results

| Model | Δ WER (pp) | Δ CER (pp) | Δ BLEU |
|-------|------------|------------|--------|
| whisper-large-v2 | −2.37 | −0.03 | +0.72 |
| whisper-large-v3 | −2.60 | −0.03 | +0.65 |
| whisper-large-v3-turbo | −2.71 | −0.05 | +0.66 |
| whisper-medium | −2.91 | −0.03 | +0.65 |
| wav2vec2-1b-german-cv11 | −1.45 | +0.22 | −0.64 |
| wav2vec2-german-with-lm | −5.23 | −0.92 | +2.39 |

Negative Δ WER/CER = improvement (error rate lower under ASR-Fair). Positive Δ BLEU = improvement.  
`wav2vec2-german-with-lm` shows the largest WER improvement (5.23pp) because its LM-enhanced decoder produced punctuation patterns most different from the reference under Standard Normalisation.

---

## Model Selection Rationale

**Whisper models** were selected to establish zero-shot translation performance benchmarks across different model sizes. The inclusion of v2, v3, and v3-turbo variants enables analysis of training data updates and architectural optimisations.

**Wav2Vec2 models** represent state-of-the-art German speech recognition, providing a comparison point for models trained on Standard German but evaluated on dialectal speech. The 1B and large-XLSR variants differ in scale and training approach, while language model integration tests decoding strategy impact.

**SeamlessM4T** was added to compare a large multilingual foundation model against the more domain-specific Whisper variants, placing Swiss German performance in the broader context of current multilingual speech research.

## Key Findings

> All findings below are based on **ASR-Fair Normalisation** (current). Legacy findings based on Standard Normalisation are given in parentheses where they differ.

- **Whisper consistently outperforms all other models** on WER (25–31% vs. 47–71%), BLEU (51–58 vs. 14–35), chrF (82–86 vs. 56–73), and SemDist (0.055–0.070 vs. 0.134–0.286)
- **whisper-large-v2 is the best-performing model** across all five metrics (WER 25.63%, CER 12.18%, BLEU 58.46, chrF 85.72, SemDist 0.055)
- **Model size matters less than training data:** whisper-medium (769M) outperforms wav2vec2-1b-german-cv11 (1000M) by ~39pp WER, with higher chrF (+26) and much lower SemDist (0.070 vs. 0.286)
- **chrF and SemDist confirm Whisper's semantic advantage:** Whisper's SemDist scores (~0.055–0.070) indicate high meaning preservation, while Wav2Vec2 models (SemDist ~0.286) confirm significant semantic divergence from Standard German references
- **SeamlessM4T occupies the middle ground** at WER 46.77%, chrF 72.73, SemDist 0.134 — substantially behind Whisper but well ahead of Wav2Vec2 on all semantic metrics, reflecting its broader multilingual training
- **Language model integration benefits Wav2Vec2** under ASR-Fair Normalisation: wav2vec2-german-with-lm (70.05%) outperforms wav2vec2-1b-german-cv11 (70.97%); this ranking was **reversed** under Standard Normalisation (75.28% vs. 72.42%) due to punctuation penalisation

---

*Last updated: 2026-03-03 (Added SeamlessM4T; ASR-Fair Normalisation results with chrF and SemDist)*
