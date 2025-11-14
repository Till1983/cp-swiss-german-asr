# docs/MODEL_SELECTION.md
   
   ## Final Model Suite
   
   1. **whisper-large** (1550M parameters)
      - Zero-shot baseline
      - Best accuracy: WER 30.40%
      - Role: State-of-the-art reference
   
   2. **whisper-medium** (769M parameters)
      - Zero-shot baseline
      - Balanced performance: WER 36.46%
      - Role: Accuracy-efficiency trade-off
   
   3. **whisper-small** (244M parameters)
      - Zero-shot baseline
      - Practical deployment: WER 41.01%
      - Role: Resource-constrained scenarios
   
   4. **wav2vec2-german-finetuned** (317M parameters)
      - Transfer learning: Dutch → German → Swiss German
      - Target: WER < 40%
      - Role: Cross-lingual transfer evaluation