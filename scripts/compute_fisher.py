#!/usr/bin/env python3
"""
compute_fisher.py — Diagonal Fisher Information Matrix estimation for EWC.

Computes an EMPIRICAL diagonal Fisher Information Matrix for whisper-large-v2
on the ZERO-SHOT checkpoint, using ground-truth transcripts from the German
Common Voice 22.0 train split. This MUST run BEFORE any Swiss German
fine-tuning exposure (Week 3 sequencing constraint: Fisher before smoke test).

Methodology notes (for thesis citation / Methodology section):

  * Empirical Fisher. F_i = (1/N) * sum_n (d/dtheta_i log p(y_n | x_n; theta))^2,
    where y_n is the GROUND-TRUTH transcript, not a sample from the model's
    own predictive distribution. This is the standard tractable approximation
    used in most EWC implementations; sampling y ~ p_theta(y|x) from an
    autoregressive decoder for 1,000 utterances would be far more expensive
    and is not what is computed here.

  * Normalisation convention. Output is the 1/N average of squared gradients
    ONLY. No additional 1/2 factor is applied. The 1/2 in the EWC penalty
    (lambda/2 * sum_i F_i * (theta_i - theta_i*)^2) belongs to the TRAINING
    LOOP's penalty computation, not to this script's output.

  * Batch size is fixed at 1. Averaging gradients across a batch and then
    squaring is NOT the same quantity as averaging squared per-sample
    gradients (the cross-terms / AM-GM gap make these different). Batch=1
    is the only way to get a clean per-sample gradient for each accumulation
    step.

  * fp32 throughout. Independent of whatever mixed-precision scheme the
    fine-tuning run uses, Fisher and theta* are computed/stored in fp32 for
    numerical stability of the squared-gradient accumulation and for a
    consistent reference point for (theta - theta*) during training.

  * Data source. Reads data/metadata/german/train.tsv (the reformatted
    Common Voice German split with audio_path precomputed by
    prepare_common_voice.py). This is row-identical to Common Voice's own
    official train partition -- using train.tsv here, and reserving
    val.tsv/test.tsv for the RQ2 forgetting evaluation, avoids any overlap
    between Fisher-importance samples and forgetting-evaluation samples.

  * Path cleaning. prepare_common_voice.py does not strip embedded
    newline/carriage-return/tab characters from the `path` column (see
    KNOWN_ISSUES.md #3). This script re-cleans `path` before constructing
    audio_path and before manifest matching.

  * File availability. Only ~435k of ~1M German CV clips were uploaded to
    the RunPod volume. Rather than calling Path.exists() per row (which
    caused a documented 30-60 minute freeze via per-file network-volume
    stat() calls), this script does ONE bulk os.listdir() over the clips
    directory and filters via in-memory set membership.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import librosa
import pandas as pd
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("compute_fisher")


# ----------------------------------------------------------------------
# Data loading and filtering
# ----------------------------------------------------------------------

def clean_path(p: str) -> str:
    """Strip embedded newline/CR/tab characters from CV `path` entries.

    prepare_common_voice.py builds audio_path = f"{cv_root}/clips/{x}"
    without this cleaning step, so audio_path values inherited from
    data/metadata/german/train.tsv may not point at real files even when
    the underlying clip exists. We re-derive audio_path from a cleaned
    `path` instead of trusting the precomputed column.
    """
    return p.strip().replace("\n", "").replace("\r", "").replace("\t", "")


def build_clip_manifest(clips_dir: Path) -> set:
    """One bulk directory read -> set of filenames actually on disk.

    Avoids per-row Path.exists() (KNOWN_ISSUES.md issue #1: ~600k individual
    stat() calls over a network volume caused a 30-60 minute freeze).
    """
    logger.info(f"Listing clips directory: {clips_dir}")
    t0 = time.time()
    available = set(os.listdir(clips_dir))
    logger.info(f"  Found {len(available):,} files on disk in {time.time() - t0:.1f}s")
    return available


def load_filtered_pool(metadata_path: Path, clips_dir: Path) -> pd.DataFrame:
    """Load train.tsv, clean paths, keep only rows with an existing clip."""
    logger.info(f"Loading metadata: {metadata_path}")
    df = pd.read_csv(
        metadata_path, sep="\t", low_memory=False, encoding="utf-8", quoting=3
    )
    logger.info(f"  Loaded {len(df):,} rows")

    df["path"] = df["path"].astype(str).apply(clean_path)

    available = build_clip_manifest(clips_dir)
    df["available"] = df["path"].isin(available)

    n_avail = int(df["available"].sum())
    logger.info(
        f"  {n_avail:,} / {len(df):,} rows have a corresponding clip "
        f"({n_avail / len(df):.1%})"
    )

    pool = df[df["available"]].copy()
    pool["audio_path"] = pool["path"].apply(lambda p: str(clips_dir / p))
    return pool.reset_index(drop=True)


# ----------------------------------------------------------------------
# Fisher computation
# ----------------------------------------------------------------------

def compute_fisher(model, processor, sample_df, device, sample_rate, n_samples):
    """Accumulate empirical diagonal Fisher information, batch size 1."""
    fisher = {
        name: torch.zeros_like(param, dtype=torch.float32)
        for name, param in model.named_parameters()
    }

    n_processed = 0
    n_failed = 0
    start_time = time.time()

    for idx, row in sample_df.iterrows():
        try:
            audio, _ = librosa.load(row["audio_path"], sr=sample_rate)
        except Exception as e:
            logger.warning(f"  [{idx}] Failed to load {row['audio_path']}: {e}")
            n_failed += 1
            continue

        inputs = processor.feature_extractor(
            audio, sampling_rate=sample_rate, return_tensors="pt"
        )
        input_features = inputs.input_features.to(device, dtype=torch.float32)

        labels = processor.tokenizer(row["sentence"], return_tensors="pt").input_ids
        labels = labels.to(device)

        model.zero_grad(set_to_none=True)
        outputs = model(input_features=input_features, labels=labels)
        outputs.loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.detach().pow(2)

        n_processed += 1

        if n_processed % 50 == 0:
            elapsed = time.time() - start_time
            rate = n_processed / elapsed
            eta = (n_samples - n_processed) / rate
            logger.info(
                f"  [{n_processed}/{n_samples}] "
                f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining "
                f"({rate:.2f} samples/s, {n_failed} failed loads)"
            )

    if n_processed == 0:
        raise RuntimeError("No samples were successfully processed.")

    for name in fisher:
        fisher[name] /= n_processed

    elapsed_total = time.time() - start_time
    return fisher, n_processed, n_failed, elapsed_total


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute diagonal Fisher Information Matrix for EWC "
                     "(zero-shot whisper-large-v2, German CV 22.0)"
    )
    parser.add_argument("--metadata", type=Path, required=True,
                         help="Path to data/metadata/german/train.tsv")
    parser.add_argument("--clips-dir", type=Path, required=True,
                         help="Path to .../cv-corpus-22.0-2025-06-20/de/clips")
    parser.add_argument("--output-dir", type=Path, required=True,
                         help="Output directory for fisher_diagonal.pt, "
                              "theta_star.pt, fisher_metadata.json")
    parser.add_argument("--model-name", default="openai/whisper-large-v2")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--save-theta-star", action="store_true", default=True)
    parser.add_argument("--no-save-theta-star", dest="save_theta_star",
                         action="store_false")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=" * 70)
    logger.info("FISHER INFORMATION MATRIX COMPUTATION")
    logger.info("=" * 70)
    logger.info(f"Model:               {args.model_name}")
    logger.info(f"Attn implementation: {args.attn_implementation}")
    logger.info(f"Device:              {device}")
    logger.info(f"N samples:           {args.n_samples}")
    logger.info(f"Seed:                {args.seed}")
    logger.info(f"Output dir:          {args.output_dir}")
    logger.info("=" * 70)

    # 1. Data: filter to available clips, sample N with fixed seed
    pool = load_filtered_pool(args.metadata, args.clips_dir)
    if len(pool) < args.n_samples:
        raise ValueError(
            f"Only {len(pool)} valid clips available, need {args.n_samples}"
        )
    sample_df = pool.sample(n=args.n_samples, random_state=args.seed).reset_index(drop=True)
    logger.info(f"Sampled {len(sample_df)} utterances (seed={args.seed})")

    # 2. Model + processor — fp32, zero-shot checkpoint, sdpa attention
    logger.info("Loading model and processor...")
    processor = WhisperProcessor.from_pretrained(args.model_name)
    processor.tokenizer.set_prefix_tokens(language="de", task="transcribe")

    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
        attn_implementation=args.attn_implementation,
    ).to(device)
    model.eval()  # disables dropout; gradients still flow to all parameters

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model loaded: {n_params:,} parameters")

    # 3. Fisher accumulation
    sample_rate = processor.feature_extractor.sampling_rate
    fisher, n_processed, n_failed, elapsed = compute_fisher(
        model, processor, sample_df, device, sample_rate, args.n_samples
    )
    logger.info(
        f"Processed {n_processed}/{args.n_samples} samples "
        f"({n_failed} failed loads) in {elapsed:.1f}s"
    )

    # 4. Save Fisher diagonal
    fisher_path = args.output_dir / "fisher_diagonal.pt"
    torch.save(fisher, fisher_path)
    logger.info(f"Saved Fisher diagonal: {fisher_path}")

    # 5. Save theta* (reference parameters) — same loaded checkpoint,
    #    captured here so Fisher and theta* are guaranteed to come from
    #    the identical weight values.
    theta_star_file = None
    if args.save_theta_star:
        theta_star = {
            name: param.detach().clone().cpu()
            for name, param in model.named_parameters()
        }
        theta_path = args.output_dir / "theta_star.pt"
        torch.save(theta_star, theta_path)
        theta_star_file = theta_path.name
        logger.info(f"Saved theta* reference parameters: {theta_path}")

    # 6. Metadata + per-parameter summary (small, citable, easy to copy out)
    per_parameter_summary = {
        name: {"mean": t.mean().item(), "max": t.max().item()}
        for name, t in fisher.items()
    }

    metadata = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_name": args.model_name,
        "attn_implementation": args.attn_implementation,
        "dtype": "float32",
        "n_requested": args.n_samples,
        "n_processed": n_processed,
        "n_failed_loads": n_failed,
        "seed": args.seed,
        "metadata_source": str(args.metadata),
        "clips_dir": str(args.clips_dir),
        "fisher_type": "empirical (ground-truth labels, not model-sampled)",
        "normalisation": (
            "1/N average of squared gradients only; no additional 1/2 "
            "factor (applied separately in the training loop's EWC penalty)"
        ),
        "fisher_file": fisher_path.name,
        "theta_star_file": theta_star_file,
        "elapsed_seconds": elapsed,
        "per_parameter_summary": per_parameter_summary,
    }

    meta_path = args.output_dir / "fisher_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata: {meta_path}")

    logger.info("=" * 70)
    logger.info("FISHER COMPUTATION COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()