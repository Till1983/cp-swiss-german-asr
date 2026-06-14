#!/usr/bin/env python3
"""train_whisper_swiss_german.py — Whisper Large-v2 + EWC fine-tuning / smoke test.

Single entry point for the FHNW Swiss German fine-tuning run and its 500-step
smoke test. The smoke test is driven by ``--max_steps`` (not a separate
script); the same invocation surface serves the GC-off probe, the GC-on
fallback, and the full run.

This script is intended to be invoked on RunPod (GPU + the real
``fisher_diagonal.pt`` / ``theta_star.pt`` live there). It is written so Part 2
is "pull, run, read the output" — the heavy/testable logic lives in
``src/training`` and ``src/data`` and is unit-tested.

Example (GC-off probe, then full smoke test):

    python scripts/train_whisper_swiss_german.py --gradient_checkpointing=false --max_steps=5
    python scripts/train_whisper_swiss_german.py --gradient_checkpointing=false --max_steps=500
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402
from transformers import TrainerCallback  # noqa: E402

import src.config as config  # noqa: E402
from src.training.whisper_setup import build_whisper_model, resolve_path  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_whisper_swiss_german")

CONFIG_PATH = config.PROJECT_ROOT / "configs" / "training" / "whisper_swiss_german.yml"


# ---------------------------------------------------------------------------
# Argument / config helpers
# ---------------------------------------------------------------------------
def _str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument(
        "--gradient_checkpointing", type=_str2bool, default=None,
        help="true|false; GC-off is probed first per the smoke-test plan",
    )
    parser.add_argument("--fisher_path", type=str, default=None,
                        help="override ewc.fisher_diagonal_path (absolute or RESULTS_DIR-relative)")
    parser.add_argument("--theta_star_path", type=str, default=None,
                        help="override ewc.theta_star_path")
    parser.add_argument("--ewc_lambda", type=float, default=None,
                        help="default from smoke_test.ewc_lambda_placeholder (1.0)")
    parser.add_argument("--eval_subset_size", type=int, default=None,
                        help="default from smoke_test.eval_subset_size (~75)")
    parser.add_argument("--smoke_test", type=_str2bool, default=True,
                        help="apply smoke_test.* overrides (default true)")
    return parser.parse_args(argv)


def load_config(path) -> dict:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    # Mirror train_german_adaptation.py: apply runpod overrides when on RunPod.
    if config.ENVIRONMENT == "runpod" and "runpod" in cfg:
        for key, value in cfg["runpod"].items():
            cfg.setdefault("environment", {})[key] = value
            logger.info("RunPod override: %s = %s", key, value)
    return cfg


def check_fisher_metadata(cfg: dict) -> None:
    """Step 4a secondary sanity check on the real fisher_metadata.json.

    Consistency check only — the authoritative source for the normalisation
    convention is scripts/compute_fisher.py's docstring, and the authoritative
    key-coverage check is Seq2SeqEWCTrainer.__init__.
    """
    meta_rel = cfg["ewc"].get("fisher_metadata_path")
    if not meta_rel:
        return
    meta_path = resolve_path(config.RESULTS_DIR, meta_rel)
    if not Path(meta_path).exists():
        logger.warning("Fisher metadata not found at %s (skipping sanity check)", meta_path)
        return
    with open(meta_path) as fh:
        meta = json.load(fh)
    expected = {
        "n_processed": cfg["ewc"].get("fisher_sample_size", 1000),
        "seed": 42,
        "attn_implementation": "sdpa",
        "dtype": "float32",
    }
    mismatches = []
    if (meta.get("n_processed") != expected["n_processed"]
            and meta.get("n_requested") != expected["n_processed"]):
        mismatches.append(
            f"n_samples={meta.get('n_processed')} (expected {expected['n_processed']})"
        )
    for k in ("seed", "attn_implementation", "dtype"):
        if meta.get(k) != expected[k]:
            mismatches.append(f"{k}={meta.get(k)} (expected {expected[k]})")
    if "1/N average of squared gradients" not in meta.get("normalisation", ""):
        mismatches.append("normalisation convention string unexpected")
    if mismatches:
        logger.warning("Fisher metadata sanity check mismatches: %s", "; ".join(mismatches))
    else:
        logger.info(
            "Fisher metadata sanity check passed (n=%s, seed=%s, %s, %s).",
            meta.get("n_processed"), meta.get("seed"),
            meta.get("attn_implementation"), meta.get("dtype"),
        )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def _dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def _append_jsonl(path: Path, obj) -> None:
    """Append one JSON record as a newline-delimited entry.

    Each call writes one line: ``{"step": N, "dialect1": wer1, ...}\\n``.
    The file is created if absent; existing lines are preserved, so the full
    per-step history survives a crash mid-training.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(obj) + "\n")


# ---------------------------------------------------------------------------
# Step counter callback
# ---------------------------------------------------------------------------
class StepCounterCallback(TrainerCallback):
    """Keep step_counter[0] current so compute_metrics can stamp JSONL records.

    ``on_step_end`` fires after each optimizer step, before
    ``_maybe_log_save_evaluate`` triggers evaluation (see Trainer
    ``_inner_training_loop``).  That means when ``compute_metrics`` runs
    during an eval pass at step N, ``step_counter[0]`` is already N — the
    write and read are in the correct order without any look-ahead.

    Threading via a single-element list is the minimal-coupling pattern:
    both ``build_compute_metrics`` and ``main`` share the same list object;
    the callback writes it, the closure reads it.
    """

    def __init__(self, counter: list) -> None:
        self._counter = counter

    def on_step_end(self, args, state, control, **kwargs):
        self._counter[0] = state.global_step


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def build_compute_metrics(processor, eval_dataset, output_dir, cfg, step_counter):
    import jiwer

    dialects = eval_dataset.dialects
    per_dialect = cfg["evaluation"].get("per_dialect_wer_logging", True)
    per_utt = cfg["evaluation"].get("per_utterance_wer_persistence", True)

    def compute_metrics(pred):
        pred_ids = pred.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        label_ids = pred.label_ids
        # Restore -100 -> pad so the tokenizer can decode references.
        label_ids = label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        overall_wer = jiwer.wer(label_str, pred_str)

        samples = []
        for i, (ref, hyp) in enumerate(zip(label_str, pred_str)):
            wer_i = jiwer.wer(ref, hyp) if ref.strip() else None
            sample = {"index": i, "reference": ref, "hypothesis": hyp, "wer": wer_i}
            if i < len(dialects):
                sample["dialect"] = dialects[i]
            samples.append(sample)

        metrics = {"wer": overall_wer}

        if per_dialect:
            groups: dict = {}
            for s in samples:
                d = s.get("dialect", "unknown")
                groups.setdefault(d, {"refs": [], "hyps": []})
                groups[d]["refs"].append(s["reference"])
                groups[d]["hyps"].append(s["hypothesis"])
            per_dialect_wer = {
                d: jiwer.wer(g["refs"], g["hyps"]) for d, g in groups.items()
            }
            for d, w in per_dialect_wer.items():
                metrics[f"wer_{d}"] = w
            # Append this eval pass as a JSONL record keyed by step.
            # StepCounterCallback.on_step_end updates step_counter[0] before
            # _maybe_log_save_evaluate triggers evaluation, so the value here
            # is always the step that triggered this eval pass.
            record = {"step": step_counter[0], **per_dialect_wer}
            _append_jsonl(Path(output_dir) / "per_dialect_wer.jsonl", record)

        if per_utt:
            # Overwrite is intentional: per-utterance breakdown is a point-in-
            # time snapshot of the most recent eval pass, not a time series.
            # The step-tagged time series lives in per_dialect_wer.jsonl.
            _dump_json(Path(output_dir) / "per_utterance_wer.json", {"samples": samples})

        return metrics

    return compute_metrics


# ---------------------------------------------------------------------------
# TrainingArguments
# ---------------------------------------------------------------------------
def build_training_arguments(cfg, args, output_dir, gradient_checkpointing, max_steps, eval_steps):
    from transformers import Seq2SeqTrainingArguments

    t = cfg["training"]
    targs = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=float(t["learning_rate"]),
        warmup_steps=t["warmup_steps"],
        weight_decay=t["weight_decay"],
        adam_beta1=t["adam_beta1"],
        adam_beta2=t["adam_beta2"],
        adam_epsilon=float(t["adam_epsilon"]),
        lr_scheduler_type=t.get("lr_scheduler_type", "linear"),
        max_grad_norm=t["max_grad_norm"],
        bf16=t.get("bf16", True),
        gradient_checkpointing=gradient_checkpointing,
        seed=t.get("seed", 42),
        predict_with_generate=t.get("predict_with_generate", True),
        generation_max_length=t.get("generation_max_length", 225),
        generation_num_beams=t.get("generation_num_beams", 1),
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=cfg["checkpointing"].get("save_steps", 125),
        save_total_limit=cfg["checkpointing"].get("save_total_limit", 3),
        logging_strategy="steps",
        logging_steps=cfg["logging"].get("logging_steps", 25),
        logging_dir=str(resolve_path(config.RESULTS_DIR, cfg["logging"]["logging_dir"])),
        report_to=cfg["logging"].get("report_to", ["tensorboard"]),
        metric_for_best_model=cfg["checkpointing"].get("metric_for_best_model", "wer"),
        greater_is_better=cfg["checkpointing"].get("greater_is_better", False),
        dataloader_num_workers=cfg.get("environment", {}).get("dataloader_num_workers", 8),
        remove_unused_columns=False,  # our dataset returns dicts the collator consumes
    )
    if max_steps is not None and max_steps > 0:
        targs["max_steps"] = max_steps
    else:
        targs["num_train_epochs"] = t.get("num_train_epochs", 5)

    return Seq2SeqTrainingArguments(**targs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)

    smoke = cfg.get("smoke_test", {})
    gradient_checkpointing = (
        args.gradient_checkpointing
        if args.gradient_checkpointing is not None
        else cfg["training"].get("gradient_checkpointing", False)
    )
    max_steps = args.max_steps if args.max_steps is not None else (
        smoke.get("max_steps") if args.smoke_test else None
    )
    ewc_lambda = args.ewc_lambda if args.ewc_lambda is not None else smoke.get(
        "ewc_lambda_placeholder", 1.0
    )
    eval_subset_size = (
        args.eval_subset_size
        if args.eval_subset_size is not None
        else (smoke.get("eval_subset_size") if args.smoke_test
              else cfg["evaluation"].get("eval_subset_size"))
    )
    eval_steps = smoke.get("eval_steps", cfg["evaluation"].get("eval_steps", 125)) \
        if args.smoke_test else cfg["evaluation"].get("eval_steps", 125)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_subdir = smoke.get("output_subdir", "smoke_test") if args.smoke_test else "baseline"
    output_dir = resolve_path(config.RESULTS_DIR, output_subdir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", output_dir)
    logger.info(
        "gradient_checkpointing=%s  max_steps=%s  ewc_lambda=%s  eval_subset_size=%s",
        gradient_checkpointing, max_steps, ewc_lambda, eval_subset_size,
    )

    check_fisher_metadata(cfg)

    # --- model + processor ---
    model, processor = build_whisper_model(
        cfg["model"], cfg.get("augmentation"), gradient_checkpointing
    )

    # --- data ---
    from src.data.whisper_collator import DataCollatorSpeechSeq2SeqWithPadding
    from src.data.whisper_dataset import WhisperSpeechDataset, load_metadata_df

    clips_dir = config.FHNW_SWISS_GERMAN_ROOT / "clips"
    train_df = load_metadata_df(resolve_path(config.DATA_DIR, cfg["data"]["train_metadata"]))
    val_df = load_metadata_df(
        resolve_path(config.DATA_DIR, cfg["data"]["val_metadata"]),
        subset_size=eval_subset_size,
        seed=cfg["training"].get("seed", 42),
    )
    train_ds = WhisperSpeechDataset(
        train_df, clips_dir, processor,
        sampling_rate=cfg["data"]["sampling_rate"],
        dialect_column=cfg["data"]["dialect_column"],
    )
    eval_ds = WhisperSpeechDataset(
        val_df, clips_dir, processor,
        sampling_rate=cfg["data"]["sampling_rate"],
        dialect_column=cfg["data"]["dialect_column"],
    )
    logger.info("Train utterances: %d | Eval utterances: %d", len(train_ds), len(eval_ds))

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # --- EWC artifacts ---
    from src.training.ewc_trainer import Seq2SeqEWCTrainer, load_fisher_and_theta

    fisher_path = resolve_path(
        config.RESULTS_DIR, args.fisher_path or cfg["ewc"]["fisher_diagonal_path"]
    )
    theta_path = resolve_path(
        config.RESULTS_DIR, args.theta_star_path or cfg["ewc"]["theta_star_path"]
    )
    logger.info("Loading Fisher: %s", fisher_path)
    logger.info("Loading theta*: %s", theta_path)
    fisher_dict, old_params = load_fisher_and_theta(fisher_path, theta_path)

    # Calibration log lives in the timestamped output dir so Part 2's checklist
    # finds it alongside the other smoke-test artifacts.
    ewc_log_path = output_dir / "ewc_calibration.csv"

    # --- callbacks / profiling ---
    from src.training.profiling import (
        CudaMemoryHistoryRecorder,
        MemoryProfilerCallback,
        NvidiaSmiLogger,
        ThroughputTracker,
    )

    vram_callback = MemoryProfilerCallback(output_dir / "vram_profile.csv")

    # Shared mutable container: StepCounterCallback.on_step_end writes
    # state.global_step after every optimizer step (before evaluation fires),
    # so the closure in build_compute_metrics always reads the current step.
    step_counter = [0]
    step_counter_callback = StepCounterCallback(step_counter)

    training_args = build_training_arguments(
        cfg, args, output_dir, gradient_checkpointing, max_steps, eval_steps
    )

    trainer = Seq2SeqEWCTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=processor.feature_extractor,
        compute_metrics=build_compute_metrics(
            processor, eval_ds, output_dir, cfg, step_counter
        ),
        fisher_dict=fisher_dict,
        old_params=old_params,
        ewc_lambda=ewc_lambda,
        apply_half_factor=cfg["ewc"].get("apply_half_factor", True),
        ewc_log_path=ewc_log_path,
        callbacks=[vram_callback, step_counter_callback],
    )

    # --- profiling lifecycle around training ---
    smi = NvidiaSmiLogger(output_dir / "nvidia_smi.csv")
    mem_hist = CudaMemoryHistoryRecorder(output_dir / "memory_snapshot.pickle")
    throughput = ThroughputTracker()
    throughput.on_first_step(0)
    smi.start()
    mem_hist.start()

    oom_error = None
    try:
        train_result = trainer.train()
        throughput.update(trainer.state.global_step)
        trainer.save_model(str(output_dir / "final_model"))
    except RuntimeError as exc:
        oom_error = str(exc)
        logger.error("Training raised RuntimeError (possible OOM):\n%s", oom_error)
        train_result = None
    finally:
        mem_hist.dump()
        mem_hist.stop()
        smi.stop()

    # --- artifacts (written regardless of how far the run got) ---
    write_outputs(output_dir, trainer, train_result, throughput, cfg, args,
                  gradient_checkpointing, max_steps, ewc_lambda, oom_error)

    if oom_error is not None:
        sys.exit(1)


def write_outputs(output_dir, trainer, train_result, throughput, cfg, args,
                  gradient_checkpointing, max_steps, ewc_lambda, oom_error):
    import csv

    # loss curve from trainer.state.log_history
    log_history = getattr(trainer.state, "log_history", []) or []
    loss_rows = [h for h in log_history if "loss" in h or "eval_loss" in h]
    _dump_json(output_dir / "loss_curve.json", log_history)
    if loss_rows:
        keys = sorted({k for r in loss_rows for k in r.keys()})
        with open(output_dir / "loss_curve.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            for r in loss_rows:
                writer.writerow(r)

    target_steps = 1260  # ~5 epochs over 4,025 utterances at batch 16
    _dump_json(output_dir / "throughput.json", throughput.summary(target_steps))

    peak_alloc = peak_reserved = None
    # Derive peak VRAM from vram_profile.csv if available (more reliable than
    # torch.cuda.max_memory_* which reflect only the peak since the last reset
    # by MemoryProfilerCallback).
    vram_profile_path = output_dir / "vram_profile.csv"
    if vram_profile_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(vram_profile_path)
            if "max_allocated_gb" in df.columns:
                peak_alloc = df["max_allocated_gb"].max()
            if "max_reserved_gb" in df.columns:
                peak_reserved = df["max_reserved_gb"].max()
        except Exception:
            pass
    # Fall back to torch CUDA stats if vram_profile.csv not available.
    if peak_alloc is None or peak_reserved is None:
        try:
            import torch
            if torch.cuda.is_available():
                if peak_alloc is None:
                    peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
                if peak_reserved is None:
                    peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
        except Exception:
            pass

    summary = [
        "# Whisper Large-v2 + EWC smoke test — summary",
        "",
        f"- attention: {cfg['model'].get('attn_implementation')} (FA2 unavailable on Blackwell)",
        f"- gradient_checkpointing: {gradient_checkpointing}",
        f"- per_device_train_batch_size: {cfg['training']['per_device_train_batch_size']}",
        f"- max_steps: {max_steps}",
        f"- ewc_lambda (placeholder): {ewc_lambda}",
        (
            f"- EWC half-factor applied: {cfg['ewc'].get('apply_half_factor', True)} "
            "(Requirement A: fisher_diagonal.pt has NO 1/2 baked in)"
        ),
        "",
        "## VRAM",
        (
            f"- peak allocated: {peak_alloc:.2f} GB"
            if peak_alloc is not None
            else "- peak allocated: n/a (vram_profile.csv not found)"
        ),
        (
            f"- peak reserved: {peak_reserved:.2f} GB"
            if peak_reserved is not None
            else "- peak reserved: n/a (vram_profile.csv not found)"
        ),
        "- budget: 96 GB (see vram_profile.csv for per-step train/eval peaks)",
        "",
        "## Outcome",
    ]
    if oom_error is not None:
        summary.append(f"- OOM / RuntimeError during training:\n```\n{oom_error}\n```")
        summary.append(
            "- ACTION: re-run with --gradient_checkpointing=true; "
            "if it still OOMs, STOP and report."
        )
    else:
        summary.append("- training completed without OOM.")
        if train_result is not None:
            summary.append(f"- final metrics: {train_result.metrics}")
        summary.append(
            "- go/no-go on LR=1e-5/warmup=50: inspect loss_curve.csv for a sane "
            "decrease through warmup (no spike/flatline) and ewc_calibration.csv "
            "for the raw EWC term scale used to centre the lambda grid."
        )

    (output_dir / "summary.md").write_text("\n".join(summary) + "\n")
    logger.info("Wrote smoke-test artifacts to %s", output_dir)


if __name__ == "__main__":
    main()