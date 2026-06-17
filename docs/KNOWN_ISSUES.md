# Known Issues and Resolutions

## Table of Contents
- [Current Status (Thesis Week 3 - 2026-06-14)](#current-status-thesis-week-3---2026-06-14)
- [Resolved Issues](#resolved-issues)
- [Best Practices Derived](#best-practices-derived)
- [References](#references)

This document tracks technical issues encountered during the Swiss German ASR project development, their solutions, and lessons learned. It serves as a troubleshooting guide and knowledge base for future work.

---

## Current Status (Thesis Week 3 - 2026-06-14)

### ✅ Completed & Resolved
- Dutch pre-training pipeline (fully functional)
- German adaptation training with EWC (memory issues resolved)
- Pre-trained model checkpoints saved to RunPod and local
- Automated configuration generation
- Shell scripts for training orchestration
- Dataset loading performance optimisations
- Path construction and TSV encoding fixes
- CUDA memory management for EWC training

### ⚠️ Non-Critical Warnings
- Deprecation warnings from transformers library (functionality intact):
  - `tokenizer` parameter deprecated in `Trainer.__init__` (use `processing_class`)
  - `torch.cuda.amp.GradScaler` deprecated (use `torch.amp.GradScaler('cuda')`)
  - `as_target_processor` deprecated in Wav2Vec2 processing
- Missing punctuation/foreign characters in vocabulary (maps to `<unk>`, expected behaviour)

---

## Resolved Issues

### 1. Dataset Loading Performance Bottleneck

**Problem:** Initial German dataset loading attempted to process 607k samples with individual file validation, causing 30-60 minute freeze with 11 GB RAM usage and no progress indication.

**Solution Implemented:**
- Changed to 150k sample limit with random sampling
- Smart validation: check 100-file sample first, only do full validation if needed
- Added skip_validation flag for pre-validated data (Dutch Fisher estimation)
- Comprehensive progress logging with tqdm progress bars
- Time estimates for all long-running operations

**Result:** Dataset loading reduced from 30-60 minutes to 2-3 minutes. 150k samples yield ~31k valid files (accounting for 61% missing due to incomplete upload).

**Lessons Learned:**
- Large dataset operations require smart sampling strategies
- Progress visibility prevents "is it frozen?" debugging sessions
- Pre-filtering or limiting is preferable to full dataset processing when constrained
- Always add progress bars to operations processing thousands of items

**Related Files:**
- `scripts/train_german_adaptation.py` (lines 227-391: prepare_dataset function)
- `configs/training/german_adaptation.yml` (limit and validation settings)

---

### 2. Incomplete Data Upload and File Availability

**Problem:** Only 435k of ~1 million German Common Voice files uploaded (8-hour timeout). Random sampling from 607k metadata resulted in 61% missing file rate.

**Solution Implemented:**
- Increased sample limit to 150k to account for missing files
- Enable validation to filter non-existent files before training
- Accept 30,708 valid samples (sufficient for adaptation given base model knowledge)

**Result:** Training proceeds with adequate data. Model already has German knowledge from Wav2Vec2-XLSR-53 base, so 31k samples sufficient for adaptation phase.

**Lessons Learned:**
- Incomplete uploads acceptable when: (1) remaining data sufficient, (2) missing data random (not biased), (3) re-upload cost exceeds benefit
- Document data availability honestly in thesis
- Verify data transfer completion before starting expensive compute jobs
- For large-scale projects, use cloud-native data sources or direct downloads when possible

**Related Files:**
- Data availability documented in Week 4 learnings
- RunPod volume: `/workspace/data/raw/cv-corpus-22.0-2025-06-20/de/clips/` (435,679 files)

---

### 3. TSV Path Encoding with Embedded Newlines

**Problem:** Audio paths extracted from TSV contained embedded newline characters (`\n`), causing file-not-found errors. Error messages showed: `common_voice_de_12345.\nmp3`

**Solution Implemented:**
- Switched from custom `load_swiss_german_metadata` to direct pandas reading:
```python
  pd.read_csv(path, sep='\t', low_memory=False, encoding='utf-8', quoting=3)
```
- Aggressive path cleaning: `path.strip().replace('\n', '').replace('\r', '').replace('\t', '')`
- Path validation logging to verify constructed paths exist

**Result:** No more newline-related path errors. Validation confirms paths properly formatted before processing.

**Lessons Learned:**
- Large TSV files (600k+ rows) can have encoding inconsistencies
- Always use `low_memory=False`, explicit encoding, and `quoting=3` (QUOTE_NONE)
- Clean extracted strings aggressively, especially file paths
- Custom loader functions may not generalize to all dataset formats/scales
- The `load_swiss_german_metadata` function was designed for smaller, cleaner Swiss German dataset

**Related Files:**
- `scripts/train_german_adaptation.py` (lines 252-268: TSV loading)
- `scripts/train_german_adaptation.py` (lines 289-313: path construction)

---

### 4. Mixed Absolute/Relative Paths in Configuration

**Problem:** YAML config mixed absolute paths (`/workspace/models/...`) with relative paths (`metadata/german/train.tsv`), breaking portability between local and RunPod environments.

**Solution Implemented:**
- Changed all YAML paths to relative format
- Smart path resolution in script: checks if absolute, otherwise prepends environment-specific base (MODELS_DIR, DATA_DIR)
- Added logging showing both config value and resolved absolute path

**Result:** Same config works on local (`/app/*`) and RunPod (`/workspace/*`) without modification. Clear logging shows resolution process.

**Lessons Learned:**
- Configuration files should use relative paths
- Let code resolve based on environment detection (`src/config.py`)
- This pattern should be used consistently across all configs
- Document the path resolution pattern for team consistency

**Related Files:**
- `configs/training/german_adaptation.yml` (all path entries now relative)
- `scripts/train_german_adaptation.py` (lines 90-98: path resolution logic)
- `src/config.py` (environment detection and base path selection)

---

### 5. Transformers Library API Version Mismatch

**Problem:** Training crashed with: `TypeError: EWCTrainer.compute_loss() got an unexpected keyword argument 'num_items_in_batch'`. Newer transformers added parameter to parent method signature.

**Solution Implemented:**
- Updated method signature: `def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None)`
- Conditional parameter passing:
```python
  if num_items_in_batch is not None:
      loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
  else:
      loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
```
- Maintains backward compatibility with older transformers versions

**Result:** Training proceeds without API errors. Compatible with current and future transformers versions.

**Lessons Learned:**
- When overriding library methods, check for new parameters in parent class
- Use conditional parameter passing and defaults (`None`) for backward compatibility
- Pin major version ranges in requirements.txt: `transformers>=4.30.0,<5.0.0`
- Test with both minimum and latest supported library versions

**Related Files:**
- `scripts/train_german_adaptation.py` (lines 148-161: compute_loss override)
- `requirements.txt` (transformers version pinning)

---

### 6. Vocabulary Check False Positives

**Problem:** Vocabulary check reported all basic German characters missing ('a', 'e', 'n', 't'), despite tokenization working (57, 74 tokens produced successfully).

**Solution Implemented:**
- Investigated tokenizer vocabulary: discovered uppercase-only (A-Z, 38 tokens total)
- Wav2Vec2 automatically uppercases text before tokenization
- Modified check to uppercase text before comparison: `text.upper()`

**Result:** Warnings reduced from 47 to 17 characters. Remaining are legitimate (punctuation, rare foreign chars like Ë, Ř, Ø → map to `<unk>`).

**Lessons Learned:**
- Understand tokenizer preprocessing pipeline before implementing validation
- Wav2Vec2 uses uppercase-only vocabularies (reduces vocab from 52 to 26 letters)
- Tokenizer's `__call__` method handles case normalisation automatically
- Sanity checks must mirror actual inference pipeline behaviour
- This specific check could be removed entirely (tokenization validation in STEP 3 already confirms correctness)

**Related Files:**
- `scripts/train_german_adaptation.py` (lines 500-523: vocabulary check)
- Base model: `aware-ai/wav2vec2-large-xlsr-53-german-with-lm` (uppercase vocab)

---

### 7. CUDA Out of Memory During EWC Training

**Problem:** Training crashed at step 49 with "CUDA out of memory" (23.56 GB / 23.57 GB). Occurred during EWC loss computation despite batch_size=8 configuration.

**Root Cause Analysis:**
- Memory breakdown: model (1.2 GB) + optimizer (2.4 GB) + batch (9.6 GB) + EWC Fisher/old_params (2.4 GB) + computation temporaries (2+ GB) = ~18+ GB
- `(fisher * (param - old_param).pow(2))` creates temporary tensors during EWC
- Memory fragmentation when operating at 99% capacity

**Solution Implemented:**
- Reduced `per_device_train_batch_size` from 8 to 4
- Increased `gradient_accumulation_steps` from 2 to 4
- Maintains effective batch size of 16 (4 × 4 = 16)
- Halves peak memory during forward pass

**Result:** Memory usage drops to ~16-18 GB, providing 6-8 GB safety margin. Training time increases ~20% (5h → 6h) due to gradient accumulation overhead.

**Lessons Learned:**
- EWC adds significant memory overhead (~2.4 GB for Fisher + old_params on GPU)
- `compute_ewc_loss` creates temporary tensors that fragment memory
- Effective batch size determines convergence, not physical batch size
- Reducing physical batch size with gradient accumulation is standard for memory-constrained training
- Leave 6-8 GB safety margin when operating near GPU memory limits
- For thesis: document effective batch size and explain memory-driven configuration as sound engineering

**Alternative Solutions (Not Implemented):**
- Move EWC computation to CPU (accepts ~10% performance penalty)
- Disable EWC temporarily (loses research contribution)
- Use larger GPU (RTX 3090 24GB → RTX 5090 32GB)

**Related Files:**
- `configs/training/german_adaptation.yml` (lines 75-77: runpod batch settings)
- `scripts/train_german_adaptation.py` (lines 134-146: compute_ewc_loss)
- Week 4 learnings documentation

**References:**
- Standard practice in ASR research (Baevski et al. 2020 - Wav2Vec2 paper)
- Common Voice fine-tuning papers often use batch_size=4-8 with gradient accumulation

---

### 8. Cross-GPU Architecture Result Variation

**Problem:** Identical evaluation code produces slightly different WER/CER/BLEU results when run on different GPU architectures (e.g., RTX 3090 vs RTX 5090 vs RTX 6000 PRO), even with deterministic settings enabled.

**Observed Behaviour:**

| Model | RTX 3090 (Ampere) | RTX 5090/6000 PRO | Δ |
|-------|-------------------|-------------------|---|
| whisper-large-v3-turbo WER | 28.2275% | 28.2258% | -0.0017pp |
| whisper-large-v3-turbo CER | 13.6589% | 13.6622% | +0.0033pp |
| whisper-medium CER | 15.8972% | 15.8988% | +0.0016pp |
| wav2vec2-1b-german-cv11 WER | 69.7233% | 69.9233% | +0.20pp |

**Root Cause:** Expected behaviour per PyTorch documentation—different GPU architectures use different cuDNN kernel implementations and floating-point accumulation paths.

**Key Finding:** PyTorch version is NOT the cause. RTX 3090 with PyTorch 2.6.0 and PyTorch 2.8.0 produce identical results. GPU architecture is the determining factor.

**Solution:** 
1. Use a single GPU architecture for all final thesis evaluations
2. Document hardware: "All evaluations conducted on NVIDIA RTX 3090 (Ampere, sm_86)"

**References:**
- [PyTorch Reproducibility Guide](https://pytorch.org/docs/stable/notes/randomness.html)
- [HuggingFace Issue #38874](https://github.com/huggingface/transformers/issues/38874)

---

### 9. RTX 5090 Non-Deterministic Initialisation

**Problem:** RTX 5090 produces different results between consecutive runs, even with identical code.

**Observed Behaviour:**
- Run 1 on RTX 5090: Results match RTX 3090 exactly
- Run 2 on RTX 5090: Results match RTX 6000 PRO (different from Run 1)
- RTX 3090 and RTX 6000 PRO: Consistent across all runs

**Root Cause:** Known Blackwell architecture issue (HuggingFace #38874).

**Solution:** Avoid RTX 5090 for reproducibility-critical evaluation. Use RTX 3090 or RTX 4090.

---

### 10. TorchCodec Required Error on PyTorch 2.8

**Problem:** Wav2Vec2 evaluation fails on PyTorch 2.8.0:
```
TorchCodec is required for load_with_torchcodec.
```

**Root Cause:** PyTorch 2.8.0 changed torchaudio's default audio loading backend.

**Affected Models:** Wav2Vec2, MMS (any model using torchaudio.load())

**Not Affected:** Whisper (uses openai-whisper's own audio loading)

**Solution:** Use `requirements.txt` (PyTorch 2.6.0) on RTX 3090 for CTC model evaluations.

---

### 11. FFmpeg Not Found on RunPod RTX 3090 Instances

**Problem:** Whisper evaluation fails with:
```
FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'
```

**Root Cause:** RunPod RTX 3090 instances do not have ffmpeg pre-installed. Whisper requires the ffmpeg CLI binary in the system PATH—pip packages like `imageio-ffmpeg` or `static-ffmpeg` do not resolve this because they install ffmpeg to Python package directories, not system PATH locations.

**Note:** RTX 5090 and RTX 6000 PRO instances have ffmpeg pre-installed.

**Solution Implemented:** The `scripts/batch_evaluation.sh` script installs ffmpeg before running evaluations:

```bash
apt-get update && apt-get install -y ffmpeg
pip install --no-cache-dir -r requirements.txt --break-system-packages
```

**Why pip-based ffmpeg packages don't work:** Whisper calls `subprocess.run(["ffmpeg", ...])` with a bare command name, relying on system PATH lookup. Pip packages provide Python APIs and bundled binaries but don't integrate with subprocess PATH resolution in containerised environments.

---

### 12. Normalisation Mode Inflated Legacy Evaluation Metrics

**Problem:** All model evaluation results prior to January 2026 were computed using **Standard Normalisation** (lowercase only, punctuation preserved). Because model outputs differ in punctuation style — particularly LM-enhanced Wav2Vec2 models — this artificially inflated WER and CER and deflated BLEU for virtually all models. The effect was most pronounced for `wav2vec2-german-with-lm` (WER inflated by 5.23pp) but affected every model to some degree (~2.5–2.9pp for Whisper models).

**Solution Implemented:**
- Default normalisation mode changed to **ASR-Fair** (lowercase + punctuation removal) in January 2026
- Old results preserved in documentation as **legacy (Standard Normalisation)** for reproducibility
- New evaluation run (March 2026) provides ASR-Fair results for all seven models
- Full breakdown in [docs/MODEL_SELECTION.md](MODEL_SELECTION.md)

**Result:** Legacy results are clearly labelled and not used for current comparisons. New results are the authoritative figures.

**Lessons Learned:**
- Punctuation handling in normalisation has outsized impact on model ranking, especially when comparing models that differ in output style (e.g., LM decoding vs. greedy CTC)
- Report normalisation mode explicitly alongside all metric figures
- Standard ASR benchmarks (e.g., SpeechBench, ESB) typically strip punctuation; align with community conventions

**Related Files:**
- `src/evaluation/metrics.py` (lines 1–50: `_normalize_text`, mode parameter)
- `docs/MODEL_SELECTION.md` (Legacy and Current results sections)

---

### 13. chrF and SemDist Metrics Added to Evaluation Pipeline (March 2026)

**Context:** The initial evaluation pipeline computed only WER, CER, and BLEU. As of March 2026, **chrF** (character n-gram F-score via `sacrebleu`) and **SemDist** (semantic distance via `sentence-transformers`) are computed for every evaluation run.

**Implementation details:**
- chrF uses `sacrebleu.CHRF` with default parameters (β=1, character n-gram order 6); corpus-level aggregation for overall score
- SemDist uses `paraphrase-multilingual-mpnet-base-v2` via `sentence-transformers`; mean of per-sample scores
- SemDist is gracefully skipped if `sentence-transformers` is not installed (reported as `null` in results)
- Both metrics are ASR-Fair normalised before computation
- `save_results_csv` and `save_results_json` detect presence of metrics conditionally, maintaining backward compatibility with older result files

**Result:** All evaluation runs from March 2026 onward include five metrics (WER, CER, BLEU, chrF, SemDist). Older result files in `results/metrics/` contain only WER, CER, BLEU — dashboard and API handle both formats.

**Lessons Learned:**
- chrF is more stable than BLEU for short sentences and morphologically rich languages (German compound words score better under character-level metrics)
- SemDist captures valid translation variants that WER/CER penalise as errors, providing a complementary view of translation quality for Whisper models

**Related Files:**
- `src/evaluation/metrics.py` (`calculate_chrf`, `batch_chrf`, `calculate_semdist`, `batch_semdist`)
- `src/evaluation/evaluator.py` (lines 216–222: SentenceTransformer loading; lines 283–298: per-sample metric computation)
- `src/utils/file_utils.py` (lines 73–107: conditional chrF/SemDist in CSV/JSON output)

---

### 14. train_whisper_on_cloud.sh — Login Shell Environment Mismatch

**Problem:** The shell script launches training via `tmux new-session ... "bash -lc ..."`.
On RunPod, a non-interactive login shell (`bash -lc`) does not inherit the same
environment as an interactive SSH session, causing the tmux session to exit
immediately with no log output and no error message surfaced locally.

**Symptom:** The wrapper reports "Session already exited within the wait window"
and "log file not found" within 12 seconds of launch.

**Workaround (current):** SSH into the pod directly and launch the tmux session
manually from the interactive shell:
```bash
tmux new-session -d -s whisper-train \
  "cd /workspace/cp-swiss-german-asr && \
   export ENVIRONMENT=runpod && \
   export PYTHONPATH=/workspace/cp-swiss-german-asr && \
   python scripts/train_whisper_swiss_german.py \
     --config configs/training/whisper_swiss_german.yml \
     [args] 2>&1 | tee /workspace/results/logs/whisper_swiss_german/run_<timestamp>.log"
```

**Root cause:** The `bash -lc` invocation sources `/etc/profile` and `~/.bash_profile`
but on this RunPod image those do not activate the conda/pip environment where
`accelerate 1.14.0` was installed, causing an import failure before Python output
begins.

**Fix needed:** Source the correct environment explicitly inside the tmux command,
or replace `bash -lc` with a `bash -c` invocation that explicitly activates the
environment. Resolve before EWC-grid runs.

---

### 15. Aggregation Method Mismatch: Macro WER + Sentence-Mean BLEU (capstone-era), Corrected to Micro/Corpus (16 June 2026)

**Problem:** The capstone evaluation pipeline computed headline WER/CER as the arithmetic mean of per-utterance rates ("macro" aggregation), while `docs/PROJECT_DISCUSSION.md` §3.4 documented the intended method as corpus-level (Σ errors / Σ reference words) — implementation and documentation disagreed. Separately, headline BLEU was computed as the mean of per-sample sentence BLEU, and the documented rationale for this ("BLEU uses per-sample aggregation due to its corpus-level bias in the sacrebleu reference implementation") was itself incorrect: Papineni et al. (2002), §2.2.2, compute BLEU's brevity penalty corpus-wide specifically to avoid the harsh penalty that sentence-level averaging imposes on short utterances, which describes a deliberate design choice, not a bias to engineer around.

**Solution Implemented:**
- `src/evaluation/evaluator.py` and `src/evaluation/error_analyzer.py` updated to enforce micro-aggregate WER/CER (16 June 2026)
- `src/evaluation/metrics.py` `batch_bleu()` updated to report corpus-level BLEU via sacrebleu `corpus_bleu` as the headline figure; per-sample BLEU retained only for descriptive analysis (`error_analyzer.py`'s WER/BLEU correlation analysis, which is explicitly documented as a legitimate use of per-sample values, separate from the headline computation)
- New evaluation run (16 June 2026) provides micro-WER + corpus-BLEU results for all seven models
- `docs/PROJECT_DISCUSSION.md` (a graded capstone deliverable, not editable) annotated with a dated erratum rather than rewritten; `docs/MODEL_SELECTION.md` and `results/error_analysis/ANALYSIS_NOTES.md` updated in place since both are living documents

**Result:** WER moved by −0.3 to −1.4pp across all seven models (consistent direction with the March 2026 normalisation fix, smaller magnitude). BLEU moved **upward** for Whisper and SeamlessM4T (+0.5 to +1.4) and **downward** for both Wav2Vec2 models (−2.0 to −2.7) — corpus pooling removes sentence-level smoothing that previously inflated BLEU for weak outputs. Model ranking by WER is unchanged. Full breakdown in [docs/MODEL_SELECTION.md](MODEL_SELECTION.md).

**Lessons Learned:**
- A documented intended method and the implemented method can silently diverge; periodically check that evaluation code actually does what its own docstrings and adjacent documentation claim
- A plausible-sounding rationale for a methodological choice is not evidence that the choice or the rationale is correct — verify against the cited primary source (here, Papineni et al. directly contradicted the documented rationale)
- BLEU and WER/CER aggregation are independent axes from normalisation mode; a fix to one does not imply the other is also correct, and figures should be checked separately on each axis
- When a metric's value is unexpectedly stable across an unrelated pipeline change, that stability is itself worth investigating — it can indicate the change didn't reach that metric's computation path, not that the metric was already correct

**Related Files:**
- `src/evaluation/evaluator.py` (micro-aggregate WER/CER enforcement, 16.06.2026 header)
- `src/evaluation/metrics.py` (`batch_bleu`, corpus-level BLEU via sacrebleu)
- `src/evaluation/error_analyzer.py` (`calculate_aggregate_stats`, micro-aggregate WER/CER; `analyze_wer_bleu_correlation`, documented as per-sample and unaffected by this change)
- `docs/MODEL_SELECTION.md` (Aggregation Modes section; March 2026 and June 2026 results tiers)
- `docs/PROJECT_DISCUSSION.md` (erratum, added 17 June 2026)
- `results/error_analysis/ANALYSIS_NOTES.md` (aggregation note, added 17 June 2026)

---

## Best Practices Derived

### Data Pipeline
1. **Use smart sampling**: Don't load entire datasets when subsets suffice
2. **Add progress bars**: Every operation > 30 seconds needs visibility (tqdm)
3. **Validate smartly**: Sample-based checks before full validation
4. **Clean data aggressively**: Strip whitespace, remove special characters from paths
5. **Use pandas parameters**: `low_memory=False`, `encoding='utf-8'`, `quoting=3` for large TSVs

### Configuration Management
1. **Relative paths only**: Let code resolve to absolute based on environment
2. **Environment detection**: Use `src/config.py` pattern for base path selection
3. **Document path resolution**: Make pattern clear for consistency
4. **Version pin carefully**: Major version ranges for libraries (e.g., `transformers>=4.30,<5.0`)

### Memory Management
1. **Leave safety margins**: 6-8 GB free on 24 GB GPU
2. **Understand overhead**: EWC, optimizer states, activation memory all add up
3. **Use gradient accumulation**: Maintain effective batch size while reducing peak memory
4. **Monitor fragmentation**: Small allocations can fail at high memory usage
5. **Document trade-offs**: Explain batch size choices as engineering decisions

### Code Quality
1. **Match preprocessing in validation**: Checks should mirror inference pipeline
2. **Conditional compatibility**: Support parameter variations across library versions
3. **Log resolution steps**: Show what configuration values resolve to
4. **Fail fast with context**: Provide actionable error messages with stack traces

---

## References

- [Wav2Vec2 Paper (Baevski et al. 2020)](https://arxiv.org/abs/2006.11477)
- [Elastic Weight Consolidation (Kirkpatrick et al. 2017)](https://arxiv.org/abs/1612.00796)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [PyTorch Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

---

**Last Updated:** 2026-03-03 (Added issues #12 and #13: Normalisation Legacy Results, chrF/SemDist Metrics Integration)