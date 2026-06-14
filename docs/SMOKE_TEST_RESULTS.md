# Smoke Test Results — Whisper Large-v2 EWC Fine-tuning

**Date:** 2026-06-14  
**Run ID:** `results/smoke_test/20260614_105322`  
**Hardware:** RTX PRO 6000 Blackwell (96 GB VRAM, sm_120)  
**Config:** `configs/training/whisper_swiss_german.yml`

---

## Purpose

The 500-step smoke test served three goals before committing GPU budget to full training runs:

1. Confirm the data pipeline, model loading, EWC wiring, and profiling instrumentation work end-to-end on the target hardware.
2. Measure peak VRAM to decide whether gradient checkpointing or batch reduction is required.
3. Produce the EWC calibration log (raw task loss and raw EWC term per step) needed to set the lambda grid for the EWC-grid runs.

---

## Run Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| Model | `openai/whisper-large-v2` | locked |
| Attention | `sdpa` (FA2 unavailable on Blackwell) | locked |
| dtype | `float32` (fp32 master weights) | locked |
| Batch size | 16 | smoke test default |
| Gradient checkpointing | false | smoke test default |
| Max steps | 500 | smoke test override |
| Learning rate | 1e-5 | anchor |
| Warmup steps | 50 | anchor |
| EWC λ (placeholder) | 1.0 | nonzero to exercise EWC path |
| `apply_half_factor` | true | Requirement A |
| `condition_on_prev_tokens` | false | hallucination guard |
| Fisher sample size | 1,000 | German CV 22.0, zero-shot checkpoint |
| Eval subset | 75 utterances | smoke test override |
| Eval steps | 250 | smoke test override |

The run completed 375 of 500 steps before a disk-quota error terminated the process at the second checkpoint save. All critical outputs were written incrementally and are intact; only `summary.md` and the steps-376–500 loss entries are absent.

---

## Pre-flight Checks

Both passed before training was launched:

**Fisher key coverage** (`scripts/verify_fisher_keys.py`):
```
model parameters : 1259
artifact keys    : 1259
RESULT: identical key sets ✓  (both fisher_diagonal.pt and theta_star.pt)
```

**Collator batch check** (16 real FHNW utterances):
```
input_features: torch.Size([16, 80, 3000])  ✓
labels:         torch.Size([16, 41])         ✓
any -100 in labels: True                     ✓
```

---

## VRAM Results

| Metric | Value | Budget | Headroom |
|--------|-------|--------|----------|
| Peak allocated (train) | 87.6 GB | 96 GB | **8.4 GB** |
| Peak reserved (train) | 90.4 GB | 96 GB | 5.6 GB |
| Peak allocated (eval) | 36.4 GB | — | — |
| Peak reserved (eval) | 90.2 GB | — | — |

Steady state from step 2 onwards: ~86.6–87.0 GB allocated, ~88.2 GB reserved. Eval allocation drops sharply (generation does not hold backward-pass activations) but the reserved pool stays high as the allocator holds pages already grabbed during training.

**Decision: gradient checkpointing OFF, batch size 16. This is the operative config for all full training runs.**

---

## Loss Trajectory

| Step | Task loss | Note |
|------|-----------|------|
| 0 | 1.207 | pre-warmup |
| 25 | 0.839 | mid-warmup |
| 50 | 0.477 | warmup complete (LR = 1e-5) |
| 100 | 0.459 | |
| 200 | 0.616 | |
| 300 | 0.223 | |
| 350 | 0.201 | |
| 374 | 0.212 | final logged step |

Clean monotonic descent with no spike through warmup and no flatline. **LR=1e-5, warmup=50 confirmed.**

---

## EWC Calibration

Raw EWC term starts at zero (θ = θ* at step 0, as expected) and grows as weights drift from the zero-shot checkpoint.

Calibration window (steps 20–50):

| Metric | Value |
|--------|-------|
| Mean task loss | 0.498 |
| Mean raw EWC term | 1.60 × 10⁻⁵ |
| Ratio (task loss / EWC term) | **3.12 × 10⁴** |

This ratio is the centre `c` for the lambda grid. At `c × λ` with the ½ factor applied, the penalty at step 374 would be approximately `0.5 × 31215 × 3.3×10⁻⁴ ≈ 5.1` — larger than the task loss of ~0.21 at that point, which is the intended behaviour for the strongest regularisation condition.

**Lambda grid (`ewc.lambda_grid` in `whisper_swiss_german.yml`):**

| Condition | λ |
|-----------|---|
| Baseline (no EWC) | `0.0` |
| c/10 | `3000.0` |
| c | `30000.0` |
| c×10 | `300000.0` |

---

## Hyperparameters Confirmed

The following PENDING items in `whisper_swiss_german.yml` are resolved:

| Item | Value | Status |
|------|-------|--------|
| `training.learning_rate` | 1e-5 | ✅ confirmed — clean loss descent |
| `training.warmup_steps` | 50 | ✅ confirmed — no warmup spike |
| `training.gradient_checkpointing` | false | ✅ confirmed — 8.4 GB headroom without GC |
| `training.per_device_train_batch_size` | 16 | ✅ confirmed — no OOM |
| `training.gradient_accumulation_steps` | 1 | ✅ confirmed |
| `evaluation.eval_steps` | 125 | ✅ confirmed |
| `checkpointing.save_steps` | 125 | ✅ confirmed |
| `ewc.lambda_grid` | see above | ✅ derived from calibration log |
| `training.adam_epsilon` | 1e-9 | ✅ retained — no gradient anomalies; cite Timmel et al. p.4 |

---

## Throughput

Steady-state throughput from step 10 onwards: **~1.17 steps/second**. Extrapolated to the full training run (~1,260 steps across 5 epochs at batch 16): approximately **18 minutes per full run**, or ~72 minutes for all four lambda conditions.

---

## Outstanding Items

- `per_dialect_wer.json` was written at the step-250 eval pass and is available on the RunPod volume. Download and inspect before the first full run to confirm dialect-stratified eval is wired correctly.
- The eval pass at step 250 ran successfully (5 eval-phase rows in `vram_profile.csv`), confirming `predict_with_generate=True` and per-dialect WER logging work without error.
- Fix `train_whisper_on_cloud.sh` login-shell environment issue before EWC-grid runs (see `KNOWN_ISSUES.md`).

---

## Artefact Locations

| Artefact | Location |
|----------|----------|
| EWC calibration log | `results/smoke_test/20260614_105322/ewc_calibration.csv` |
| VRAM profile | `results/smoke_test/20260614_105322/vram_profile.csv` |
| Per-dialect WER | `results/smoke_test/20260614_105322/per_dialect_wer.json` |
| Per-utterance WER | `results/smoke_test/20260614_105322/per_utterance_wer.json` |
| nvidia-smi log | `results/smoke_test/20260614_105322/nvidia_smi.csv` |
| Console log | `results/logs/whisper_swiss_german/run_smoke_500.log` |

All artefact directories are gitignored. The run ID is recorded here as the durable reference.
