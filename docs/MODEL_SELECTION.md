# Model Selection

## Table of Contents
- [Final Model Suite](#final-model-suite)
- [Model Selection Rationale](#model-selection-rationale)
- [Key Findings](#key-findings)

## Final Model Suite

This document describes the six ASR models evaluated on the Swiss German test corpus (863 samples across 17 dialects).

### Whisper Models (OpenAI)

Zero-shot multilingual models trained on 680,000 hours of web data. All Whisper variants perform Swiss German-to-Standard German translation.

1. **whisper-large-v2** (1550M parameters)
   - Architecture: Encoder-decoder transformer
   - Evaluation approach: Zero-shot translation
   - Overall WER: 28.00%
   - Overall CER: 12.21%
   - Overall BLEU: 57.74
   - Role: Best-performing model, baseline reference

2. **whisper-large-v3** (1550M parameters)
   - Architecture: Encoder-decoder transformer (updated training data)
   - Evaluation approach: Zero-shot translation
   - Overall WER: 29.53%
   - Overall CER: 13.32%
   - Overall BLEU: 56.28
   - Role: Latest Whisper version comparison

3. **whisper-large-v3-turbo** (809M parameters)
   - Architecture: Optimised encoder-decoder with reduced decoder layers
   - Evaluation approach: Zero-shot translation
   - Overall WER: 30.94%
   - Overall CER: 13.71%
   - Overall BLEU: 53.97
   - Role: Efficiency-accuracy trade-off evaluation

4. **whisper-medium** (769M parameters)
   - Architecture: Encoder-decoder transformer (smaller variant)
   - Evaluation approach: Zero-shot translation
   - Overall WER: 34.11%
   - Overall CER: 15.93%
   - Overall BLEU: 50.82
   - Role: Resource-constrained deployment baseline

### Wav2Vec2 Models (German-trained)

Self-supervised speech recognition models trained on German Common Voice data. These models perform direct transcription without explicit translation.

5. **wav2vec2-1b-german-cv11** (1000M parameters)
   - HuggingFace: `aware-ai/wav2vec2-xls-r-1b-german-cv11`
   - Architecture: XLS-R 1B fine-tuned on German Common Voice v11
   - Evaluation approach: Zero-shot transcription
   - Overall WER: 72.42%
   - Overall CER: 29.24%
   - Overall BLEU: 14.93
   - Role: Large-scale German ASR comparison

6. **wav2vec2-german-with-lm** (317M parameters)
   - HuggingFace: `aware-ai/wav2vec2-large-xlsr-53-german-with-lm`
   - Architecture: XLSR-53 with optional KenLM language model integration
   - Evaluation approach: Zero-shot transcription (LM-enhanced decoding)
   - Overall WER: 75.28%
   - Overall CER: 31.52%
   - Overall BLEU: 13.89
   - Role: Language model-enhanced decoding evaluation

## Model Selection Rationale

**Whisper models** were selected to establish zero-shot translation performance benchmarks across different model sizes. The inclusion of v2, v3, and v3-turbo variants enables analysis of training data updates and architectural optimisations.

**Wav2Vec2 models** represent state-of-the-art German speech recognition, providing a comparison point for models trained on Standard German but evaluated on dialectal speech. The 1B and large-XLSR variants differ in scale and training approach, while language model integration tests decoding strategy impact.

## Key Findings

- **Whisper consistently outperforms Wav2Vec2** by 2.5-3x on WER metrics (28-34% vs. 72-75%)
- **Model size matters less than training data**: whisper-medium (769M) outperforms wav2vec2-1b-german-cv11 (1000M)
- **Translation vs. transcription task framing** explains performance gap: Whisper models benefit from explicit Swiss German→Standard German mapping in training data
- **Language model integration** provides minimal benefit for Wav2Vec2 on dialectal speech (75.28% WER with LM vs. 72.42% without) - **Important:** see correction below

---

> **⚠️ Normalisation Mode Correction (January 2026)**
> 
> The original claim that "Language model integration provides minimal benefit for Wav2Vec2 on dialectal speech (75.28% WER with LM vs. 72.42% without)" was based on **standard normalisation** which systematically penalises LM-enhanced decoding due to punctuation handling differences.
> 
> Under **ASR-fair normalisation** (lowercase + punctuation removal), the ranking reverses:
> 
> | Model | Standard WER | ASR-Fair WER |
> |-------|--------------|--------------|
> | wav2vec2-1b-german-cv11 | 72.42% | 70.97% |
> | wav2vec2-german-with-lm | 75.28% | **70.05%** |
> 
> **Corrected finding:** Language model integration provides measurable benefit under fair normalisation (70.05% vs 70.97%), with the LM-enhanced model showing the largest improvement from normalisation (5.23pp vs 1.45pp).
> 
> **Interpretation:** The LM decoder produces different punctuation/formatting patterns than greedy CTC decoding, which were penalised under standard evaluation. Both Wav2Vec2 variants remain significantly behind Whisper (~25-28% WER) regardless of normalisation.

---

### Additional Note for Whisper Section

Under ASR-fair normalization, Whisper models also improve:

| Model | Standard WER | ASR-Fair WER | Improvement |
|-------|--------------|--------------|-------------|
| whisper-large-v2 | 28.00% | 25.60% | 2.40pp |
| whisper-large-v3 | 29.53% | 26.93% | 2.60pp |
| whisper-large-v3-turbo | 30.94% | 28.23% | 2.71pp |
| whisper-medium | 34.11% | 31.20% | 2.91pp |

The improvement is smaller than for CTC models (~2.5pp vs ~5-9pp) because Whisper's punctuation more closely matches reference text, but differences in comma placement and hyphenation still contribute errors under standard normalisation.

**Conclusion:** Whisper's dominance over CTC models is maintained regardless of normalisation mode. The gap remains ~44pp under both evaluation methods.

---