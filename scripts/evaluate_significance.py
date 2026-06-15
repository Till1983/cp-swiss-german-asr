import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import jiwer
from sacrebleu import corpus_bleu
from sacrebleu.metrics import CHRF

# Add project root to path before importing from src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.config import RESULTS_DIR
from src.evaluation.significance import bootstrap_significance_test, mapsswe_test
from src.evaluation.text_normalization import normalize_text
from src.utils.logging_config import setup_logger


def _load_results_file(path: str) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    model_name = payload.get("model_name", Path(path).stem)
    results = payload.get("results", payload)
    samples = results.get("samples")

    if not isinstance(samples, list):
        raise ValueError(f"No 'results.samples' list found in {path}")

    return model_name, results, samples


def _validate_pairing(samples_a: list[dict[str, Any]], samples_b: list[dict[str, Any]]) -> None:
    if len(samples_a) != len(samples_b):
        raise ValueError(
            "Results files must contain the same number of samples for paired testing"
        )

    for i, (a, b) in enumerate(zip(samples_a, samples_b)):
        audio_a = a.get("audio_file")
        audio_b = b.get("audio_file")
        if audio_a and audio_b and audio_a != audio_b:
            raise ValueError(
                f"Mismatched audio_file at index {i}: '{audio_a}' vs '{audio_b}'"
            )

        ref_a = normalize_text(str(a.get("reference", "")), mode="asr_fair")
        ref_b = normalize_text(str(b.get("reference", "")), mode="asr_fair")
        if ref_a != ref_b:
            raise ValueError(
                "Mismatched normalized references at index "
                f"{i}: '{ref_a}' vs '{ref_b}'"
            )


def _normalize_sample_text(sample: dict[str, Any]) -> tuple[str, str]:
    reference = normalize_text(str(sample.get("reference", "")), mode="asr_fair")
    hypothesis = normalize_text(str(sample.get("hypothesis", "")), mode="asr_fair")
    return reference, hypothesis


def _derive_wer_primitives(samples: list[dict[str, Any]]) -> tuple[list[int | None], list[int | None]]:
    errors: list[int | None] = []
    ref_words: list[int | None] = []

    for sample in samples:
        reference, hypothesis = _normalize_sample_text(sample)

        if not reference:
            errors.append(None)
            ref_words.append(None)
            continue

        alignment = jiwer.process_words(reference, hypothesis)
        word_errors = int(
            alignment.substitutions + alignment.deletions + alignment.insertions
        )
        word_count = int(alignment.hits + alignment.substitutions + alignment.deletions)

        errors.append(word_errors)
        ref_words.append(word_count)

    return errors, ref_words


def _derive_text_pairs(samples: list[dict[str, Any]]) -> tuple[list[str | None], list[str | None]]:
    references: list[str | None] = []
    hypotheses: list[str | None] = []

    for sample in samples:
        reference, hypothesis = _normalize_sample_text(sample)
        if not reference:
            references.append(None)
            hypotheses.append(None)
            continue

        references.append(reference)
        hypotheses.append(hypothesis)

    return references, hypotheses


def _micro_wer_metric(data: Any) -> float:
    errors, ref_words = data
    total_words = sum(ref_words)
    if total_words == 0:
        return 0.0
    return sum(errors) / total_words


def _corpus_bleu_metric(data: Any) -> float:
    references, hypotheses = data
    return float(corpus_bleu(hypotheses, [references]).score)


def _corpus_chrf_metric(data: Any) -> float:
    references, hypotheses = data
    metric = CHRF()
    return float(metric.corpus_score(hypotheses, [references]).score)


def _bootstrap_metric_config(metric_name: str) -> Callable[[Any], float]:
    if metric_name == "wer":
        return _micro_wer_metric
    if metric_name == "bleu":
        return _corpus_bleu_metric
    if metric_name == "chrf":
        return _corpus_chrf_metric

    raise ValueError(f"Unsupported metric '{metric_name}'")


def _format_result_lines(test_label: str, result) -> list[str]:
    ci_str = (
        "N/A"
        if result.ci_lower is None or result.ci_upper is None
        else f"[{result.ci_lower:.6f}, {result.ci_upper:.6f}]"
    )
    return [
        f"{test_label}",
        f"  n_samples      : {result.n_samples}",
        f"  mean_diff(a-b) : {result.mean_diff:.6f}",
        f"  statistic      : {result.statistic:.6f}",
        f"  p_value        : {result.p_value:.6f}",
        f"  significant    : {result.is_significant}",
        f"  ci             : {ci_str}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MAPSSWE and paired bootstrap significance on saved ASR results"
    )
    parser.add_argument("--results-a", required=True, help="Path to system A JSON")
    parser.add_argument("--results-b", required=True, help="Path to system B JSON")
    parser.add_argument(
        "--metric",
        choices=["wer", "bleu", "chrf"],
        default="wer",
        help="Corpus-level metric for paired bootstrap",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULTS_DIR / "significance"),
        help="Directory to save significance report JSON",
    )

    args = parser.parse_args()
    logger = setup_logger("evaluate_significance", "logs/significance.log")

    model_a, raw_results_a, samples_a = _load_results_file(args.results_a)
    model_b, raw_results_b, samples_b = _load_results_file(args.results_b)

    _validate_pairing(samples_a, samples_b)

    errors_a, ref_words_a = _derive_wer_primitives(samples_a)
    errors_b, ref_words_b = _derive_wer_primitives(samples_b)

    mapsswe_result = mapsswe_test(errors_a, errors_b, alpha=args.alpha)

    metric_fn = _bootstrap_metric_config(args.metric)
    if args.metric == "wer":
        data_a = (errors_a, ref_words_a)
        data_b = (errors_b, ref_words_b)
    else:
        refs_a, hyps_a = _derive_text_pairs(samples_a)
        refs_b, hyps_b = _derive_text_pairs(samples_b)
        data_a = (refs_a, hyps_a)
        data_b = (refs_b, hyps_b)

    bootstrap_result = bootstrap_significance_test(
        data_a,
        data_b,
        metric_fn=metric_fn,
        n_bootstrap=args.n_bootstrap,
        alpha=args.alpha,
        seed=args.seed,
    )

    dropped_pairs = len(samples_a) - bootstrap_result.n_samples

    reconciliation = None
    if args.metric == "wer":
        aggregate_a = _micro_wer_metric((
            [e for e in errors_a if e is not None],
            [w for w in ref_words_a if w is not None],
        ))
        aggregate_b = _micro_wer_metric((
            [e for e in errors_b if e is not None],
            [w for w in ref_words_b if w is not None],
        ))
        direct_diff = aggregate_a - aggregate_b
        reconciliation = {
            "bootstrap_observed_diff": bootstrap_result.mean_diff,
            "direct_aggregate_diff": direct_diff,
            "abs_delta": abs(bootstrap_result.mean_diff - direct_diff),
            "reported_overall_wer_diff_pct_points": (
                raw_results_a.get("overall_wer", 0.0) - raw_results_b.get("overall_wer", 0.0)
            ),
            "direct_aggregate_diff_pct_points": direct_diff * 100.0,
        }

    print("=" * 72)
    print("SIGNIFICANCE REPORT")
    print("=" * 72)
    print(f"System A: {model_a}")
    print(f"System B: {model_b}")
    print("Sign convention: d = a - b")
    print("Interpretation: lower-is-better metrics improve for A when mean_diff < 0")
    print(f"Paired rows used: {bootstrap_result.n_samples} (dropped: {dropped_pairs})")
    print(f"Bootstrap metric: {args.metric}")
    print(f"Alpha: {args.alpha}")
    print(f"Bootstrap iterations: {args.n_bootstrap}")
    print(f"Seed: {args.seed}")
    print("-" * 72)

    for line in _format_result_lines("MAPSSWE (count differences)", mapsswe_result):
        print(line)
    print("-")
    for line in _format_result_lines(
        f"Paired bootstrap ({args.metric} corpus metric)", bootstrap_result
    ):
        print(line)

    if reconciliation is not None:
        print("-")
        print("WER reconciliation")
        print(
            f"  bootstrap_observed_diff      : {reconciliation['bootstrap_observed_diff']:.10f}"
        )
        print(
            f"  direct_aggregate_diff        : {reconciliation['direct_aggregate_diff']:.10f}"
        )
        print(f"  abs_delta                    : {reconciliation['abs_delta']:.10f}")
        print(
            "  direct_aggregate_diff (pct)  : "
            f"{reconciliation['direct_aggregate_diff_pct_points']:.6f}"
        )
        print(
            "  reported_overall_wer_diff (pct): "
            f"{reconciliation['reported_overall_wer_diff_pct_points']:.6f}"
        )

    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "significance_report.json"

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "system_a": model_a,
        "system_b": model_b,
        "results_a_path": args.results_a,
        "results_b_path": args.results_b,
        "sign_convention": "d = a - b",
        "interpretation": "For lower-is-better metrics (WER), A improves when mean_diff < 0",
        "alpha": args.alpha,
        "bootstrap_metric": args.metric,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "paired_samples_total": len(samples_a),
        "paired_samples_used": bootstrap_result.n_samples,
        "paired_samples_dropped": dropped_pairs,
        "mapsswe": asdict(mapsswe_result),
        "bootstrap": asdict(bootstrap_result),
        "wer_reconciliation": reconciliation,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("Significance report saved to %s", report_path)
    print("=" * 72)
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
