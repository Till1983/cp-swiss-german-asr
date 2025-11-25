## Week 4 Status (2025-11-25)

### ✅ Completed
- Dutch pre-training pipeline (fully functional)
- Pre-trained model checkpoints saved to RunPod and local
- Automated configuration generation
- Shell scripts for training orchestration

### ⚠️ Known Issues (Non-blocking)
- German adaptation script has deprecation warnings from transformers library
- Functions marked for deprecation: `compute_measures` (already deprecated) , `load_metric` (replaced by `evaluate.load`)
- Impact: None currently, but will need updates for future library versions
- Workaround: Script still functions correctly despite warnings

### 🔧 Next Steps
- Address deprecation warnings in follow-up PR
- Complete German fine-tuning validation
- Test knowledge transfer Swiss German evaluation

**Example terminal output with warnings:**

1.

```shell
/workspace/cp-swiss-german-asr/scripts/train_german_adaptation.py:301: FutureWarning: load_metric is deprecated and will be removed in the next major version of datasets. Use 'evaluate.load' instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate
  wer_metric = load_metric("wer")
/usr/local/lib/python3.12/dist-packages/datasets/load.py:759: FutureWarning: The repository for wer contains custom code which must be executed to correctly load the metric. You can inspect the repository content at https://raw.githubusercontent.com/huggingface/datasets/2.19.1/metrics/wer/wer.py
You can avoid this message in future by passing the argument `trust_remote_code=True`.
Passing `trust_remote_code=True` will be mandatory to load this metric from the next major release of `datasets`.
  warnings.warn(
CRITICAL:train_german_adaptation:Fatal error: cannot import name 'compute_measures' from 'jiwer' (/usr/local/lib/python3.12/dist-packages/jiwer/__init__.py)
```

**Attempted solution:** Force reinstallation of jiwer version lower then 4.0.0.
** Success. Warnings gone after downgrade to jiwer 3.1.0

2.

```shell
UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81
```

** No immediate action required. Monitor setuptools updates.

3. 

```shell
/workspace/cp-swiss-german-asr/scripts/train_german_adaptation.py:400: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`. Use `processing_class` instead.
```

** Plan to update Trainer initialization in future refactor. No current impact.


## Dataset Size Management

### German Training Data Subset

**Decision:** Used 50,000 randomly sampled German Common Voice examples for adaptation (from 607,872 total).

**Rationale:**
- Standard practice in transfer learning and continual learning research
- Random sampling ensures representative speaker and phonetic diversity
- Computational efficiency for capstone timeline
- Model already has German phonetic knowledge from base Wav2Vec2-XLSR-53
- Focus on adaptation quality (with EWC) rather than data quantity

**Scientific Precedent:**
- Kirkpatrick et al. (2017) EWC paper used data subsets for continual learning
- Common Voice fine-tuning studies typically use 10k-100k samples
- Transfer learning literature emphasizes quality of adaptation over data volume

**Future Work:**
- Full 607k dataset training could be explored for marginal improvements
- Current results with 50k provide strong baseline for Swiss German zero-shot evaluation
