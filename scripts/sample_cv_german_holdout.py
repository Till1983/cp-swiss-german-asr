"""
sample_cv_german_holdout.py — Generate a seeded, fixed-size holdout sample
from the German Common Voice 22.0 test split for the RQ2 forgetting benchmark.

Run on RunPod (where data/metadata/german/test.tsv lives):

    python3 sample_cv_german_holdout.py \
        --input /workspace/data/metadata/german/test.tsv \
        --output /workspace/data/metadata/german/test_1000_seed42.tsv \
        --n 1000 --seed 42

Background:
- The sample is drawn ONCE and reused across every EWC lambda condition
  (no-EWC baseline, lambda in {3000, 30000, 300000}) so all forgetting
  comparisons share an identical evaluation set. Do not regenerate with
  a different seed for an existing thesis result without documenting why.
- evaluator.py's evaluate_dataset() hard-requires a column literally named
  'accent' (required_columns = {'path', 'sentence', 'accent'}), inherited
  from the FHNW Swiss German schema where 'accent' encodes the per-canton
  dialect (17 distinct values). German Common Voice has no equivalent
  stratification axis -- prepare_common_voice.py populates 'locale'
  instead, which is constant ("de") for every German CV row and therefore
  not a meaningful grouping variable even if present.
- This script adds a placeholder 'accent' column (constant "unknown") to
  satisfy the evaluator's schema contract. The resulting per_dialect_*
  output will have a single "unknown" key with N = sample size, which is
  the correct shape for a benchmark with no internal stratification.
  Do not interpret "unknown" as missing/corrupted data -- it is the
  intended placeholder for "no dialect axis exists here."
"""

import argparse
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Draw a seeded fixed-size sample from the German CV test split."
    )
    parser.add_argument("--input", type=str, required=True,
                         help="Path to source test.tsv (full German CV test split)")
    parser.add_argument("--output", type=str, required=True,
                         help="Path to write the sampled TSV")
    parser.add_argument("--n", type=int, default=1000,
                         help="Sample size (default: 1000)")
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path, sep="\t", low_memory=False,
                      encoding="utf-8", quoting=3)

    if len(df) < args.n:
        raise ValueError(
            f"Requested sample size {args.n} exceeds source row count {len(df)}"
        )

    sample = df.sample(n=args.n, random_state=args.seed)

    # Fold in the accent column required by evaluator.py's schema check.
    # German CV has no dialect axis; "unknown" is a deliberate placeholder,
    # not a missing-data marker. See module docstring.
    if "accent" not in sample.columns:
        sample = sample.copy()
        sample["accent"] = "unknown"

    sample.to_csv(output_path, sep="\t", index=False)

    print(f"Source rows       : {len(df)}")
    print(f"Sampled rows      : {len(sample)}")
    print(f"Seed              : {args.seed}")
    print(f"accent column     : {'added (constant \"unknown\")' if 'accent' not in df.columns else 'already present'}")
    print(f"Written to        : {output_path}")


if __name__ == "__main__":
    main()
