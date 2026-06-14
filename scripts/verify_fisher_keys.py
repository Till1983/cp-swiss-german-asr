#!/usr/bin/env python3
"""verify_fisher_keys.py — EWC key-coverage proxy/pre-flight check.

Compares the parameter-name key set that EWC will use against the
architecture's ``named_parameters()``:

* Part 1 (Step 0b, no GPU): pass ``--fisher-metadata`` pointing at a
  ``fisher_metadata*.json``. Its ``per_parameter_summary`` keys are compared
  against ``WhisperForConditionalGeneration(WhisperConfig.from_pretrained(...))``
  -- this downloads only the (small) config, not the multi-GB checkpoint, and
  validates Requirement C structurally before any GPU time is spent.

* Part 2 (on RunPod, real artifacts): pass ``--fisher-diagonal`` /
  ``--theta-star`` pointing at the real ``.pt`` tensors to compare the actual
  loaded key sets.

Exit code 0 = key sets identical; 1 = mismatch (the diff is printed).
"""

import argparse
import json
import sys

from transformers import WhisperConfig, WhisperForConditionalGeneration


def model_param_keys(model_name: str) -> set:
    cfg = WhisperConfig.from_pretrained(model_name)
    model = WhisperForConditionalGeneration(cfg)
    return {name for name, _ in model.named_parameters()}


def keys_from_metadata(path: str) -> set:
    with open(path) as fh:
        meta = json.load(fh)
    return set(meta["per_parameter_summary"].keys())


def keys_from_state_dict(path: str) -> set:
    import torch

    state = torch.load(path, map_location="cpu")
    return set(state.keys())


def compare(label: str, model_keys: set, other_keys: set) -> bool:
    missing = sorted(model_keys - other_keys)  # in model, absent from artifact
    extra = sorted(other_keys - model_keys)     # in artifact, absent from model
    print(f"== {label} ==")
    print(f"  model parameters : {len(model_keys)}")
    print(f"  artifact keys    : {len(other_keys)}")
    if not missing and not extra:
        print("  RESULT: identical key sets ✓")
        return True
    print("  RESULT: MISMATCH ✗")
    if missing:
        print(f"  missing from artifact ({len(missing)}): {missing[:10]}")
    if extra:
        print(f"  extra in artifact ({len(extra)}): {extra[:10]}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="openai/whisper-large-v2")
    parser.add_argument("--fisher-metadata", help="Path to fisher_metadata*.json")
    parser.add_argument("--fisher-diagonal", help="Path to fisher_diagonal.pt")
    parser.add_argument("--theta-star", help="Path to theta_star.pt")
    args = parser.parse_args()

    if not (args.fisher_metadata or args.fisher_diagonal or args.theta_star):
        parser.error(
            "provide at least one of --fisher-metadata / --fisher-diagonal / "
            "--theta-star"
        )

    print(f"Instantiating {args.model_name} from config (no checkpoint download)...")
    model_keys = model_param_keys(args.model_name)

    ok = True
    if args.fisher_metadata:
        ok &= compare(
            f"metadata: {args.fisher_metadata}",
            model_keys,
            keys_from_metadata(args.fisher_metadata),
        )
    if args.fisher_diagonal:
        ok &= compare(
            f"fisher_diagonal: {args.fisher_diagonal}",
            model_keys,
            keys_from_state_dict(args.fisher_diagonal),
        )
    if args.theta_star:
        ok &= compare(
            f"theta_star: {args.theta_star}",
            model_keys,
            keys_from_state_dict(args.theta_star),
        )

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
