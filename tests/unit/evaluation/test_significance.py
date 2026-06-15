import numpy as np
import pytest

from src.evaluation.significance import (
    SignificanceResult,
    bootstrap_significance_test,
    mapsswe_test,
    paired_bootstrap_differences,
)


def _micro_wer_metric(data):
    errors, ref_words = data
    total_words = sum(ref_words)
    if total_words == 0:
        return 0.0
    return sum(errors) / total_words


class TestMapssweTest:
    def test_identical_error_counts_not_significant(self):
        errors_a = [1, 0, 2, 1, 3]
        errors_b = [1, 0, 2, 1, 3]

        result = mapsswe_test(errors_a, errors_b)

        assert isinstance(result, SignificanceResult)
        assert result.p_value == 1.0
        assert result.is_significant is False
        assert result.statistic == 0.0
        assert result.mean_diff == 0.0

    def test_clearly_worse_a_yields_positive_mean_diff(self):
        errors_a = [5, 4, 6, 5, 4, 6, 5, 4]
        errors_b = [1, 0, 2, 1, 1, 2, 1, 0]

        result = mapsswe_test(errors_a, errors_b)

        assert result.mean_diff > 0
        assert result.statistic > 0
        assert result.p_value < 0.05
        assert result.is_significant is True

    def test_symmetry(self):
        errors_a = [2, 3, 4, 3, 2, 4]
        errors_b = [1, 1, 2, 1, 2, 1]

        ab = mapsswe_test(errors_a, errors_b)
        ba = mapsswe_test(errors_b, errors_a)

        assert ab.statistic == pytest.approx(-ba.statistic)
        assert ab.mean_diff == pytest.approx(-ba.mean_diff)

    def test_known_analytical_z(self):
        errors_a = [2, 3, 4]
        errors_b = [1, 1, 1]

        result = mapsswe_test(errors_a, errors_b)

        diffs = np.array([1.0, 2.0, 3.0])
        expected = float(diffs.mean() / (diffs.std(ddof=1) / np.sqrt(3)))
        assert result.statistic == pytest.approx(expected)

    def test_filters_paired_nones(self):
        errors_a = [1, None, 3, 2]
        errors_b = [0, 1, None, 1]

        result = mapsswe_test(errors_a, errors_b)

        assert result.n_samples == 2

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            mapsswe_test([1, 2], [1])

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError, match="At least 2"):
            mapsswe_test([None, 1], [None, 0])


class TestPairedBootstrapDifferences:
    def test_uses_shared_indices_for_pairing(self):
        # With paired resampling and b = a + 1 elementwise, mean(a)-mean(b) == -1 always.
        data_a = [0.0, 1.0, 2.0, 3.0]
        data_b = [1.0, 2.0, 3.0, 4.0]

        diffs = paired_bootstrap_differences(
            data_a,
            data_b,
            metric_fn=lambda x: float(np.mean(x)),
            n_bootstrap=500,
            seed=42,
        )

        assert np.allclose(diffs, -1.0)

    def test_reproducible_with_seed(self):
        data_a = [1, 2, 3, 4, 5]
        data_b = [2, 2, 3, 3, 4]

        d1 = paired_bootstrap_differences(
            data_a,
            data_b,
            metric_fn=lambda x: float(np.mean(x)),
            n_bootstrap=250,
            seed=42,
        )
        d2 = paired_bootstrap_differences(
            data_a,
            data_b,
            metric_fn=lambda x: float(np.mean(x)),
            n_bootstrap=250,
            seed=42,
        )

        assert np.array_equal(d1, d2)

    def test_validation(self):
        with pytest.raises(ValueError, match="same paired length"):
            paired_bootstrap_differences([1, 2], [1], metric_fn=lambda x: float(np.mean(x)))

        with pytest.raises(ValueError, match="At least 2"):
            paired_bootstrap_differences([None, 1], [None, 1], metric_fn=lambda x: float(np.mean(x)))


class TestBootstrapSignificanceTest:
    def test_micro_wer_observed_matches_direct_difference(self):
        # A has lower error rate than B.
        errors_a = [1, 0, 2, 1]
        errors_b = [2, 1, 3, 2]
        ref_words = [10, 8, 12, 10]

        data_a = (errors_a, ref_words)
        data_b = (errors_b, ref_words)

        result = bootstrap_significance_test(
            data_a,
            data_b,
            metric_fn=_micro_wer_metric,
            n_bootstrap=2000,
            alpha=0.05,
            seed=42,
        )

        expected_observed = _micro_wer_metric(data_a) - _micro_wer_metric(data_b)
        assert result.mean_diff == pytest.approx(expected_observed)
        assert result.statistic == pytest.approx(expected_observed)
        assert result.ci_lower is not None
        assert result.ci_upper is not None
        assert result.ci_lower < result.ci_upper

    def test_filters_paired_nones_in_tuple_data(self):
        data_a = ([1, None, 2, 3], [10, None, 10, 10])
        data_b = ([2, 1, None, 4], [10, 10, 10, 10])

        result = bootstrap_significance_test(
            data_a,
            data_b,
            metric_fn=_micro_wer_metric,
            n_bootstrap=500,
            seed=42,
        )

        assert result.n_samples == 2

    def test_identical_systems_not_significant(self):
        data_a = ([1, 2, 1, 0, 2], [10, 10, 10, 10, 10])
        data_b = ([1, 2, 1, 0, 2], [10, 10, 10, 10, 10])

        result = bootstrap_significance_test(
            data_a,
            data_b,
            metric_fn=_micro_wer_metric,
            n_bootstrap=1000,
            seed=42,
        )

        assert result.mean_diff == pytest.approx(0.0)
        assert result.is_significant is False

    def test_alpha_validation(self):
        with pytest.raises(ValueError, match="alpha"):
            bootstrap_significance_test(
                [1, 2, 3],
                [1, 2, 3],
                metric_fn=lambda x: float(np.mean(x)),
                alpha=1.0,
            )

    def test_corpus_metric_recompute_path(self):
        refs = ["alpha beta", "gamma delta", "eins zwei"]
        hyps_a = ["alpha beta", "gamma delta", "eins zwei"]
        hyps_b = ["alpha", "gamma", "eins"]

        data_a = (refs, hyps_a)
        data_b = (refs, hyps_b)

        def corpus_like_metric(data):
            r, h = data
            total_ref_tokens = sum(len(x.split()) for x in r)
            total_hyp_tokens = sum(len(x.split()) for x in h)
            return total_hyp_tokens / total_ref_tokens

        result = bootstrap_significance_test(
            data_a,
            data_b,
            metric_fn=corpus_like_metric,
            n_bootstrap=1000,
            seed=42,
        )

        assert result.mean_diff > 0


class TestBootstrapStability:
    def test_endpoint_variance_drops_with_larger_bootstrap(self):
        data_a = ([1, 0, 2, 1, 0, 2, 1, 0], [10, 9, 11, 10, 9, 11, 10, 9])
        data_b = ([2, 1, 3, 2, 1, 3, 2, 1], [10, 9, 11, 10, 9, 11, 10, 9])

        seeds = [11, 17, 23, 29, 31]

        low_left = []
        high_left = []
        for seed in seeds:
            low = bootstrap_significance_test(
                data_a,
                data_b,
                metric_fn=_micro_wer_metric,
                n_bootstrap=200,
                seed=seed,
            )
            high = bootstrap_significance_test(
                data_a,
                data_b,
                metric_fn=_micro_wer_metric,
                n_bootstrap=5000,
                seed=seed,
            )
            low_left.append(low.ci_lower)
            high_left.append(high.ci_lower)

        assert float(np.var(high_left)) < float(np.var(low_left))


class TestBootstrapSignificancePositive:
    def test_clearly_different_is_significant_and_ci_excludes_zero(self):
        errors_a = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
        errors_b = [5, 6, 5, 6, 5, 6, 5, 6, 5, 6]
        ref_words = [10] * 10
        r = bootstrap_significance_test(
            (errors_a, ref_words), (errors_b, ref_words),
            metric_fn=_micro_wer_metric, n_bootstrap=5000, seed=42,
        )
        ci_excludes_zero = not (r.ci_lower <= 0 <= r.ci_upper)
        assert r.is_significant is True
        assert ci_excludes_zero is True
        assert r.is_significant == ci_excludes_zero  # CI and p must agree