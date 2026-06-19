# German Common Voice Forgetting Benchmark

## Table of Contents
- [Purpose](#purpose)
- [Prerequisites](#prerequisites)
- [Running the Benchmark](#running-the-benchmark)
- [Running the Significance Test](#running-the-significance-test)
- [Interpreting the Output](#interpreting-the-output)
- [Worked Example — No-EWC Baseline](#worked-example--no-ewc-baseline)
- [Extending to the EWC λ Grid](#extending-to-the-ewc-λ-grid)
- [See Also](#see-also)

## Purpose

RQ2 asks whether Elastic Weight Consolidation (EWC) mitigates catastrophic forgetting of
Standard German ASR performance during Swiss German fine-tuning. This benchmark measures
that forgetting directly: it evaluates `whisper-large-v2` (zero-shot) and each fine-tuned
variant on the same fixed German Common Voice 22.0 holdout, then runs paired significance
tests on the resulting WER.

This is a **magnitude question** (how much forgetting, is it significant), not a
characterisation question. Unlike the FHNW error analysis used for RQ1, no per-dialect
breakdown or worst-samples deep-dive is part of the standard workflow here — German CV has
no dialect axis to stratify by. A brief sanity spot-check on the worst few samples per run
is still worthwhile to rule out degenerate failure modes (hallucination loops, empty
transcriptions) before trusting the aggregate WER; see
[Interpreting the Output](#interpreting-the-output).

## Prerequisites

- One-time data setup completed: see
  [German Common Voice Forgetting Benchmark — One-Time Setup](RUNPOD_WORKFLOW_addition.md#german-common-voice-forgetting-benchmark--one-time-setup)
  in `RUNPOD_WORKFLOW_addition.md`. In particular, `data/metadata/german/test_1000_seed42.tsv` must
  exist on the RunPod volume.
- Any fine-tuned checkpoint you want to benchmark must already have a `MODEL_REGISTRY`
  entry in `scripts/evaluate_models.py`.

## Running the Benchmark

```bash
# On your laptop (runs evaluate_models.py remotely, downloads results)
./scripts/batch_evaluation_cv_ger.sh \
    --models whisper-large-v2 whisper-large-v2-swiss-german-baseline
```

- `--models` has **no default** and is required on every invocation — the model pair
  changes per run (zero-shot anchor + whichever fine-tuned/EWC variant is being scored),
  so there is no single correct default to silently fall back on.
- Defaults baked into the script (override only if needed):
  - `--test-path` → `data/metadata/german/test_1000_seed42.tsv` (the seeded holdout)
  - `--audio-base-path` → `data/raw/cv-corpus-22.0-2025-06-20/de/clips`
  - `--output-dir` → `results/metrics/cv-german`
  - `--experiment-type` → `fine-tuned`
- Results sync to `results/metrics/cv-german/<experiment-type>_<timestamp>/` locally,
  intentionally separate from `results/metrics/` (the FHNW dashboard data) so the
  forgetting-benchmark results don't leak into dashboard aggregation.
- Both models in a single invocation matters: `evaluate_significance.py` pairs samples by
  `audio_file`, which requires both models to have been evaluated against the identical
  TSV in the same run.

## Running the Significance Test

Run locally via Docker Compose — no RunPod GPU needed for this step:

```bash
docker compose run --rm api \
    python scripts/evaluate_significance.py \
    --results-a results/metrics/cv-german/<experiment-type>_<timestamp>/whisper-large-v2_results.json \
    --results-b results/metrics/cv-german/<experiment-type>_<timestamp>/whisper-large-v2-swiss-german-baseline_results.json \
    --metric wer --n-bootstrap 10000 --seed 42
```

- The `api` service is the correct target: it mounts both `scripts/` and `results/`, and
  `PYTHONPATH=/app` is already set so `src` imports resolve. `test*` services override
  `command` for pytest and are not suitable here.
- Substitute the actual `<experiment-type>_<timestamp>` directory from the prior step's
  output — `evaluate_models.py` timestamps each run, so this is not a fixed path.
- Sign convention is **`d = a - b`**. For WER (lower is better), a negative `mean_diff`
  means system A improves on system B. With `--results-a` as the zero-shot model and
  `--results-b` as the fine-tuned model, a negative `mean_diff` means the fine-tuned model
  has **higher** WER than zero-shot — i.e., forgetting occurred.
- Output is written to `results/significance/<timestamp>/significance_report.json` and
  printed to stdout.

## Interpreting the Output

The report contains two independent tests plus a reconciliation check:

- **`mapsswe`** — segment-level matched-pairs test on per-utterance error counts. No
  confidence interval (by design — it's a z-test on count differences, not a resampled
  metric). Validates that the WER shift is systematic across utterances, not driven by a
  handful of outliers.
- **`bootstrap`** — paired bootstrap directly on the aggregate WER metric you report in
  Results. This is the number to cite: `mean_diff` (in WER fraction, multiply by 100 for
  percentage points) plus a 95% CI (`ci_lower`, `ci_upper`).
- **`wer_reconciliation`** — cross-checks that the bootstrap's resampled mean difference
  matches the direct micro-aggregate WER difference computed from the two result files.
  `abs_delta` should be ~0; if it isn't, something is wrong with the pairing or the input
  files and the result should not be trusted until resolved.

**Sanity check before trusting any run's WER number:** pull the 3–5 highest-WER samples
from the fine-tuned model's results and read them. You're checking for hallucination
loops, empty transcriptions, or wrong-language output dominating the aggregate — the
capstone's AG-dialect 469% WER case was exactly this failure mode and would have silently
inflated an aggregate metric if unnoticed. If the worst samples are ordinary substitution
errors with intact sentence structure, the number is trustworthy. If something looks
degenerate, flag and document it (e.g. "N samples excluded as hallucination artifacts")
rather than silently reporting an inflated WER.

## Worked Example — No-EWC Baseline

First completed run, 19 June 2026, N=1,000, 0 failed samples for either model:

| Metric | Zero-shot | No-EWC fine-tuned | Δ |
|---|---|---|---|
| WER | 6.12% | 9.05% | +2.93pp |
| CER | 1.98% | 2.90% | +0.93pp |
| BLEU | 88.48 | 83.25 | −5.23 |
| chrF | 96.25 | 94.60 | −1.65 |
| SemDist | 0.0288 | 0.0396 | +0.0108 |

Significance test (`d = a − b`, a = zero-shot, b = no-EWC fine-tuned):

- **MAPSSWE**: statistic = −8.45, p ≈ 0.0, significant.
- **Paired bootstrap**: mean_diff = −0.0293 (−2.93pp), 95% CI [−3.61pp, −2.25pp], p ≈
  1×10⁻⁴, significant.
- **Reconciliation**: `abs_delta = 0.0` — bootstrap and direct aggregate agree exactly.

Both tests agree on direction and both are significant well beyond α=0.05. Sanity spot-
check of the 8 largest per-sample degradations found ordinary substitution/paraphrase
errors (e.g. *"Letztere veröffentlichte..."* → drifting into English-leaning phrasing such
as *"Germanic Mineralogical and Petrographic Department"*), not hallucination loops — max
WER across both models was 150%, well below the 469% runaway case documented in the
capstone error analysis. No samples excluded.

**Interpretation:** catastrophic forgetting is real, statistically robust (CI excludes
zero by a wide margin), and moderate in magnitude — not a collapse (cf. the 86–144pp WER
swings reported for Pashto fine-tuning in Rahman's results), but a clear, reproducible
degradation that gives EWC a well-defined target to mitigate. This is the baseline every
EWC λ condition is compared against.

## Extending to the EWC λ Grid

For each completed EWC λ checkpoint:

1. Add a `MODEL_REGISTRY` entry in `scripts/evaluate_models.py` (same pattern as
   `whisper-large-v2-swiss-german-baseline`).
2. Run the benchmark against the same seeded holdout:
   ```bash
   ./scripts/batch_evaluation_cv_ger.sh \
       --models whisper-large-v2 whisper-large-v2-ewc-lambda-<value>
   ```
3. Run the significance test **twice** per λ condition:
   - vs. zero-shot (`whisper-large-v2`) — establishes whether forgetting is still present
     at this λ.
   - vs. the no-EWC baseline (`whisper-large-v2-swiss-german-baseline`) — establishes
     whether EWC actually *helped* relative to not using it. This is the primary RQ2
     comparison; "improved vs. zero-shot" alone is not sufficient evidence that EWC is
     doing anything, since the no-EWC model already does too.
4. Per the committed robustness check, also report German CV WER at the common fixed step
   1,260 across all λ conditions alongside the primary best-checkpoint comparison.

## See Also
- [RUNPOD_WORKFLOW.md](RUNPOD_WORKFLOW.md) — one-time data setup for this benchmark, plus
  general RunPod training/evaluation workflow.
- [KNOWN_ISSUES.md](KNOWN_ISSUES.md) — partial German CV upload issue and other documented
  RunPod data quirks.
