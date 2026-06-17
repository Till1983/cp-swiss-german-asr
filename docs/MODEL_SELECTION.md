# Model Selection

## Table of Contents
- [Evaluation Methodology](#evaluation-methodology)
  - [Normalisation Modes](#normalisation-modes)
  - [Metric Definitions](#metric-definitions)
- [Final Model Suite](#final-model-suite)
- [Results – Legacy (Standard Normalisation)](#results--legacy-standard-normalisation)
- [Results – March 2026 (ASR-Fair Normalisation, macro/sentence-mean)](#results--march-2026-asr-fair-normalisation-macrosentence-mean)
- [Results – Current (ASR-Fair Normalisation + micro/corpus aggregation, June 2026)](#results--current-asr-fair-normalisation--micro-aggregation-june-2026)
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

### Aggregation Modes

A second, independent dimension affects WER and BLEU figures: how per-utterance scores are combined into a single headline number.

| Mode | WER/CER | BLEU | When Used |
|------|---------|------|-----------|
| **Macro / sentence-mean** (superseded) | Mean of per-utterance WER/CER | Mean of per-utterance BLEU | All evaluations prior to 16 June 2026 |
| **Micro / corpus-level** (current) | (Σ errors) / (Σ reference words) | sacrebleu `corpus_bleu` — single brevity penalty over the full corpus | All evaluations from 16 June 2026 onward |

Micro/corpus aggregation is the field-standard convention (jiwer, HuggingFace `evaluate`, and sclite all default to micro WER; Papineni et al. 2002, §2.2.2, compute BLEU's brevity penalty corpus-wide specifically to avoid the harsh short-sentence penalty that sentence-mean averaging incurs on a corpus with utterances as short as 2–3 words). The switch changes WER by roughly −0.3 to −1.4pp across all seven models and moves BLEU **upward** for Whisper and SeamlessM4T (+0.5 to +1.4) while moving it **downward** for both Wav2Vec2 models (−2.0 to −2.7), since corpus pooling no longer lets sentence-level smoothing prop up poor outputs. Full per-model figures are in [Results – Current](#results--current-asr-fair-normalisation--micro-aggregation-june-2026) below.

> **Reading "March 2026" results:** Any table or figure labelled "March 2026" uses ASR-Fair Normalisation but **macro/sentence-mean aggregation** — itself superseded by the June 2026 results below for WER and BLEU. CER remained stable for most models (changes ≤|0.01|pp), except whisper-medium (−0.96pp) — check the per-model table below rather than assuming CER is unaffected.

---

### Metric Definitions

| Metric | Range | Direction | Description |
|--------|-------|-----------|-------------|
| **WER** (Word Error Rate) | 0–100% | Lower is better | Percentage of words incorrectly recognised/translated (Levenshtein edit distance at word level) |
| **CER** (Character Error Rate) | 0–100% | Lower is better | Percentage of characters incorrectly recognised/translated (Levenshtein edit distance at character level) |
| **BLEU** | 0–100 | Higher is better | Modified n-gram precision measuring overlap between hypothesis and reference. **Corpus-level** aggregation (sacrebleu `corpus_bleu`) since 16 June 2026 — see note below; prior results used sentence-level mean. |
| **chrF** (Character F-score) | 0–100 | Higher is better | Character n-gram F-score (β=1). More robust than WER/CER for morphologically rich languages like German because it handles compound words and inflections gracefully. Corpus-level aggregation. |
| **SemDist** (Semantic Distance) | 0–1 | Lower is better | Cosine distance between sentence embeddings computed by `paraphrase-multilingual-mpnet-base-v2`. A value of 0 means identical meaning; 1 means completely unrelated. Captures meaning preservation even when surface form differs (important for evaluating Whisper's translation output). |

**Implementation note:** chrF uses `sacrebleu.CHRF` with default parameters (β=1, character n-gram order 6). SemDist uses `sentence-transformers` with batch encoding; if `sentence-transformers` is not installed, SemDist is skipped and reported as `null`.

---

## Final Model Suite

This document describes the seven ASR models evaluated on the Swiss German test corpus (863 samples across 17 dialects). The per-model metrics shown here use **Standard Normalisation (legacy)** — see [Results – Current](#results--current-asr-fair-normalisation--micro-aggregation-june-2026) for updated figures including chrF and SemDist.

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

## Results – March 2026 (ASR-Fair Normalisation, macro/sentence-mean)

> **Superseded by June 2026 results below for WER and BLEU** (see [Aggregation Modes](#aggregation-modes)). CER and chrF/SemDist are unaffected for five of six models; whisper-medium's CER also moved (−0.96pp) — see the June table.

All results below use **ASR-Fair Normalisation** (lowercase + punctuation removal) and **macro/sentence-mean aggregation** for WER and BLEU. Two new metrics — **chrF** and **SemDist** — are included from this evaluation cycle onward.

Results directory: `results/metrics/20260303_105207/` (all models except turbo) and `results/metrics/20260303_121313/` (whisper-large-v3-turbo).

| Model | WER (%) ↓ | CER (%) ↓ | BLEU ↑ | chrF ↑ | SemDist ↓ | Samples |
|-------|-----------|-----------|--------|--------|-----------|---------|
| whisper-large-v2 | 25.63 | 12.18 | 58.46 | 85.72 | 0.0551 | 863 |
| whisper-large-v3 | 26.93 | 13.29 | 56.93 | 85.12 | 0.0574 | 863 |
| whisper-large-v3-turbo | 28.23 | 13.66 | 54.63 | 83.97 | 0.0586 | 863 |
| whisper-medium | 31.20 | 15.90 | 51.47 | 82.46 | 0.0695 | 863 |
| seamless-m4t-v2-large | 46.77 | 22.43 | 35.14 | 72.73 | 0.1335 | 863 |
| wav2vec2-1b-german-cv11 | 70.97 | 29.46 | 14.29 | 56.23 | 0.2857 | 863 |
| wav2vec2-german-with-lm | 70.05 | 30.60 | 16.28 | 56.50 | 0.2855 | 863 |

### Comparing Legacy and March 2026 Results

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

## Results – Current (ASR-Fair Normalisation + micro aggregation, June 2026)

**These are the authoritative figures for model comparison.** All results below use ASR-Fair Normalisation, **micro/corpus-level WER and CER**, and **corpus-level BLEU** (sacrebleu `corpus_bleu`) — see [Aggregation Modes](#aggregation-modes) for why this superseded the March 2026 macro/sentence-mean figures. chrF and SemDist are corpus-level and per-sample-mean respectively; while they use the same computation logic as March 2026, all metrics were recomputed in the June 2026 evaluation batch and may differ slightly due to rounding or batch processing differences.

Results directory: `results/error_analysis/20260616_215923/` and underlying `results/metrics/` batch of 16 June 2026.

| Model | WER (%) ↓ | CER (%) ↓ | BLEU ↑ | chrF ↑ | SemDist ↓ | Samples |
|-------|-----------|-----------|--------|--------|-----------|---------|
| whisper-large-v2 | **24.98** | **12.18** | **59.81** | 85.72 | 0.0551 | 863 |
| whisper-large-v3 | 26.23 | 13.30 | 57.82 | 85.11 | 0.0574 | 863 |
| whisper-large-v3-turbo | 27.86 | 13.67 | 55.15 | 83.97 | 0.0586 | 863 |
| whisper-medium | 30.17 | 14.94 | 52.06 | 82.63 | 0.0696 | 863 |
| seamless-m4t-v2-large | 45.40 | 22.43 | 36.49 | 72.73 | 0.1335 | 863 |
| wav2vec2-1b-german-cv11 | 70.67 | 29.46 | 11.62 | 56.23 | 0.2857 | 863 |
| wav2vec2-german-with-lm | 69.66 | 30.60 | 14.24 | 56.50 | 0.2855 | 863 |

**Bold** = best score per metric.

### Comparing March 2026 and June 2026 Results

| Model | Δ WER (pp) | Δ CER (pp) | Δ BLEU |
|-------|------------|------------|--------|
| whisper-large-v2 | −0.65 | 0.00 | +1.35 |
| whisper-large-v3 | −0.70 | +0.01 | +0.89 |
| whisper-large-v3-turbo | −0.37 | +0.01 | +0.52 |
| whisper-medium | −1.03 | −0.96 | +0.59 |
| seamless-m4t-v2-large | −1.37 | 0.00 | +1.35 |
| wav2vec2-1b-german-cv11 | −0.30 | 0.00 | **−2.67** |
| wav2vec2-german-with-lm | −0.39 | 0.00 | **−2.04** |

Negative Δ WER/CER = improvement. Positive Δ BLEU = improvement under the new corpus-level computation. Note the **sign split** on BLEU: the four strongest models (Whisper, SeamlessM4T) gain 0.5–1.4 points, while both Wav2Vec2 models *lose* 2.0–2.7 points. This is the expected and correct direction — corpus-level pooling removes the sentence-level brevity-penalty smoothing that previously inflated BLEU for outputs with many short, low-quality matches. The switch is not a uniform inflation; it makes weak models look worse and strong models look slightly better, which is what a less biased metric should do. Model ranking by WER is unchanged from the March 2026 tier.

---

## Model Selection Rationale

**Whisper models** were selected to establish zero-shot translation performance benchmarks across different model sizes. The inclusion of v2, v3, and v3-turbo variants enables analysis of training data updates and architectural optimisations.

**Wav2Vec2 models** represent state-of-the-art German speech recognition, providing a comparison point for models trained on Standard German but evaluated on dialectal speech. The 1B and large-XLSR variants differ in scale and training approach, while language model integration tests decoding strategy impact.

**SeamlessM4T** was added to compare a large multilingual foundation model against the more domain-specific Whisper variants, placing Swiss German performance in the broader context of current multilingual speech research.

## Key Findings

> All findings below are based on **ASR-Fair Normalisation + micro/corpus aggregation** (June 2026, current). March 2026 macro-aggregation findings are given in parentheses where they differ in value (rankings are unchanged throughout).

- **Whisper consistently outperforms all other models** on WER (25–30% vs. 45–71%), BLEU (52–60 vs. 12–36), chrF (83–86 vs. 56–73), and SemDist (0.055–0.070 vs. 0.134–0.286)
- **whisper-large-v2 is the best-performing model** across all five metrics (WER 24.98%, CER 12.18%, BLEU 59.81, chrF 85.72, SemDist 0.055)
- **Model size matters less than training data:** whisper-medium (769M) outperforms wav2vec2-1b-german-cv11 (1000M) by ~40pp WER, with higher chrF (+26) and much lower SemDist (0.070 vs. 0.286)
- **chrF and SemDist confirm Whisper's semantic advantage:** Whisper's SemDist scores (~0.055–0.070) indicate high meaning preservation, while Wav2Vec2 models (SemDist ~0.286) confirm significant semantic divergence from Standard German references
- **SeamlessM4T occupies the middle ground** at WER 45.40%, chrF 72.73, SemDist 0.134 — substantially behind Whisper but well ahead of Wav2Vec2 on all semantic metrics, reflecting its broader multilingual training
- **Language model integration benefits Wav2Vec2** under ASR-Fair Normalisation: wav2vec2-german-with-lm (69.66%, March 2026: 70.05%) outperforms wav2vec2-1b-german-cv11 (70.67%, March 2026: 70.97%); this ranking was **reversed** under Standard Normalisation (75.28% vs. 72.42%) due to punctuation penalisation
- **Corpus-level BLEU penalises the weakest models more than it helps the strongest:** the aggregation switch (March → June 2026) raised BLEU for Whisper and SeamlessM4T by 0.5–1.4 points but *lowered* it for both Wav2Vec2 models by 2.0–2.7 points — see [Aggregation Modes](#aggregation-modes)

---

*Last updated: 2026-06-17 (Added June 2026 micro/corpus-aggregation results tier; corrected BLEU aggregation method and metric definition)*