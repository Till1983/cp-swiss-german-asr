from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from scipy.stats import norm


@dataclass
class SignificanceResult:
    test_name: str
    p_value: float
    statistic: float
    is_significant: bool
    alpha: float
    n_samples: int
    mean_diff: float
    ci_lower: float | None = None
    ci_upper: float | None = None


def mapsswe_test(
    errors_a: list[int | None],
    errors_b: list[int | None],
    alpha: float = 0.05,
) -> SignificanceResult:
    """
    Matched-pairs test on per-segment error counts (utterance-as-segment).

    For each utterance i:
      d_i = errors_a[i] - errors_b[i]

    z = mean(d) / (std(d, ddof=1) / sqrt(n))

    Uses a two-tailed p-value from the standard normal distribution.
    Paired None values are filtered out before computing statistics.
    """
    _validate_alpha(alpha)

    if len(errors_a) != len(errors_b):
        raise ValueError("errors_a and errors_b must have the same length")

    paired = [
        (a, b)
        for a, b in zip(errors_a, errors_b)
        if a is not None and b is not None
    ]

    if len(paired) < 2:
        raise ValueError("At least 2 paired, non-missing samples are required")

    diffs = np.array([int(a) - int(b) for a, b in paired], dtype=float)
    n = diffs.size
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))

    if std_diff == 0.0:
        # Identical paired differences provide no evidence of a directional shift.
        z_stat = 0.0
        p_value = 1.0
    else:
        stderr = std_diff / np.sqrt(n)
        z_stat = mean_diff / stderr
        p_value = float(2.0 * (1.0 - norm.cdf(abs(z_stat))))

    return SignificanceResult(
        test_name="mapsswe",
        p_value=p_value,
        statistic=float(z_stat),
        is_significant=p_value < alpha,
        alpha=alpha,
        n_samples=int(n),
        mean_diff=mean_diff,
        ci_lower=None,
        ci_upper=None,
    )


def paired_bootstrap_differences(
    data_a: Any,
    data_b: Any,
    metric_fn: Callable[[Any], float],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> np.ndarray:
    """
    Return paired bootstrap metric differences using one shared index draw per sample.

    Each iteration draws a single index array and applies it to both systems:
      diff = metric_fn(resampled_a) - metric_fn(resampled_b)
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be >= 1")

    filtered_a, filtered_b = _prepare_paired_data(data_a, data_b)
    n = _data_length(filtered_a)

    if n < 2:
        raise ValueError("At least 2 paired, non-missing samples are required")

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        bootstrap_idx = rng.integers(0, n, size=n)
        resampled_a = _subset_data(filtered_a, bootstrap_idx)
        resampled_b = _subset_data(filtered_b, bootstrap_idx)
        diffs[i] = float(metric_fn(resampled_a) - metric_fn(resampled_b))

    return diffs


def bootstrap_significance_test(
    data_a: Any,
    data_b: Any,
    metric_fn: Callable[[Any], float],
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> SignificanceResult:
    """
    Paired bootstrap significance test for corpus-level metrics.

    For WER, data is typically (errors, ref_word_counts), with metric_fn computing
    micro WER = sum(errors) / sum(ref_word_counts).

    For BLEU/chrF, data is typically (references, hypotheses), with metric_fn
    recomputing the corpus-level score over each resampled set.
    """
    _validate_alpha(alpha)

    filtered_a, filtered_b = _prepare_paired_data(data_a, data_b)
    n = _data_length(filtered_a)

    if n < 2:
        raise ValueError("At least 2 paired, non-missing samples are required")

    observed_diff = float(metric_fn(filtered_a) - metric_fn(filtered_b))
    diffs = paired_bootstrap_differences(
        filtered_a,
        filtered_b,
        metric_fn,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    # Two-sided p-value via the null-centred (shifted) bootstrap distribution.
    # H0: true difference = 0. Shift the resampled diffs to centre zero, then count
    # how often the null is at least as extreme as the observed difference.
    # +1 smoothing bounds p below at 1 / (B + 1); it never reports exactly zero.
    center = float(np.mean(diffs))
    n_extreme = int(np.sum(np.abs(diffs - center) >= abs(observed_diff)))
    p_value = (1 + n_extreme) / (1 + diffs.size)

    lower_q = (alpha / 2.0) * 100.0
    upper_q = (1.0 - alpha / 2.0) * 100.0
    ci_lower, ci_upper = np.percentile(diffs, [lower_q, upper_q])

    return SignificanceResult(
        test_name="paired_bootstrap",
        p_value=float(p_value),
        statistic=observed_diff,
        is_significant=p_value < alpha,
        alpha=alpha,
        n_samples=int(n),
        mean_diff=observed_diff,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
    )


def _prepare_paired_data(data_a: Any, data_b: Any) -> tuple[Any, Any]:
    n_a = _data_length(data_a)
    n_b = _data_length(data_b)

    if n_a != n_b:
        raise ValueError("data_a and data_b must have the same paired length")

    valid_indices = []
    for i in range(n_a):
        row_a = _row_at(data_a, i)
        row_b = _row_at(data_b, i)
        if _contains_none(row_a) or _contains_none(row_b):
            continue
        valid_indices.append(i)

    filtered_a = _subset_data(data_a, valid_indices)
    filtered_b = _subset_data(data_b, valid_indices)
    return filtered_a, filtered_b


def _data_length(data: Any) -> int:
    if isinstance(data, tuple):
        if not data:
            return 0

        lengths = [len(component) for component in data]
        if len(set(lengths)) != 1:
            raise ValueError("All tuple components must have the same length")
        return lengths[0]

    if isinstance(data, np.ndarray):
        return int(data.shape[0])

    if isinstance(data, Sequence):
        return len(data)

    raise TypeError("Unsupported data container type")


def _row_at(data: Any, idx: int) -> Any:
    if isinstance(data, tuple):
        return tuple(component[idx] for component in data)
    return data[idx]


def _subset_data(data: Any, indices: Sequence[int] | np.ndarray) -> Any:
    if isinstance(indices, np.ndarray):
        idx_list = indices.tolist()
    else:
        idx_list = list(indices)

    if isinstance(data, tuple):
        return tuple([component[i] for i in idx_list] for component in data)

    return [data[i] for i in idx_list]


def _contains_none(value: Any) -> bool:
    if value is None:
        return True

    if isinstance(value, tuple):
        return any(_contains_none(v) for v in value)

    return False


def _validate_alpha(alpha: float) -> None:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in the open interval (0, 1)")
