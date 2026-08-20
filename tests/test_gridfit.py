"""
Unit tests for the uniform-grid least-squares primitives.

These are the closed forms the tone and profile estimators rely on, so they are
tested directly against the naive sums they replace, including the corner cases
(a ratio of one, a ratio that underflows) where the closed form is delicate.
"""

import numpy as np
import pytest

from ringdownanalysis._gridfit import (
    UNIFORM_TOLERANCE_SAMPLES,
    BlockIndex,
    geometric_sum,
    is_uniformly_sampled,
    tone_sums,
)


class TestIsUniformlySampled:
    """The gate that decides whether the closed forms may be used."""

    def test_exact_grid_is_uniform(self):
        fs = 149.01
        assert is_uniformly_sampled(np.arange(10_000) / fs, fs) is True

    def test_large_time_origin_does_not_defeat_the_test(self):
        fs = 149.01
        t = 1.7e9 + np.arange(10_000) / fs
        assert is_uniformly_sampled(t, fs) is True

    def test_quantization_noise_inside_tolerance_is_uniform(self):
        fs = 149.01
        rng = np.random.default_rng(0)
        jitter = rng.uniform(-0.4, 0.4, size=10_000) * UNIFORM_TOLERANCE_SAMPLES / fs
        assert is_uniformly_sampled(np.arange(10_000) / fs + jitter, fs) is True

    def test_slow_rate_error_accumulates_and_is_rejected(self):
        """A tiny per-step bias is what the accumulated-offset test exists for."""
        fs = 149.01
        n = 100_000
        t = np.arange(n) * (1.0 + 1e-6) / fs
        assert np.all(np.abs(np.diff(t) - 1.0 / fs) < UNIFORM_TOLERANCE_SAMPLES / fs)
        assert is_uniformly_sampled(t, fs) is False

    def test_single_sample_is_trivially_uniform(self):
        assert is_uniformly_sampled(np.array([3.0]), 10.0) is True


class TestToneSums:
    """Dirichlet-kernel form of the undamped tone sums."""

    @pytest.mark.parametrize("angle", [0.0, 1e-14, 0.37, np.pi, 2.0 * np.pi, 5.9])
    def test_matches_direct_summation(self, angle):
        n = 2_000
        k = np.arange(n)
        cos_sum, sin_sum = tone_sums(n, np.array([angle]))
        assert cos_sum[0] == pytest.approx(float(np.sum(np.cos(angle * k))), abs=1e-9)
        assert sin_sum[0] == pytest.approx(float(np.sum(np.sin(angle * k))), abs=1e-9)


class TestGeometricSum:
    """Closed-form sum of a complex geometric series."""

    @pytest.mark.parametrize(
        "log_ratio",
        [
            0.0 + 0.0j,  # every term is one
            -1e-12 + 1e-12j,  # numerator and denominator both near zero
            -1e-9 + 0.001j,  # decay far longer than the record
            -0.003 + 0.51j,  # ordinary ring-down
            -3.0 + 2.0j,  # underflows well inside the record
        ],
    )
    def test_matches_direct_summation(self, log_ratio):
        n = 5_000
        direct = np.sum(np.exp(np.arange(n) * log_ratio))
        closed = geometric_sum(n, np.array([log_ratio]))[0]
        assert closed == pytest.approx(direct, rel=1e-13, abs=1e-300)

    def test_evaluates_a_batch_of_ratios_at_once(self):
        n = 1_000
        ratios = np.array([-0.01 + 0.3j, -0.2 + 1.1j, -1e-8 + 0.05j])
        got = geometric_sum(n, ratios)
        want = [np.sum(np.exp(np.arange(n) * z)) for z in ratios]
        assert got == pytest.approx(want, rel=1e-13)

    def test_long_decay_keeps_relative_precision(self):
        """``1 - exp(w)`` would lose eight digits here; ``expm1`` does not."""
        n = 100_000
        log_ratio = -1e-8 + 0.0j
        direct = np.sum(np.exp(np.arange(n) * log_ratio.real))
        closed = geometric_sum(n, np.array([log_ratio]))[0]
        assert closed.real == pytest.approx(direct, rel=1e-12)


class TestBlockIndex:
    """Two-factor decomposition of the geometric data projection."""

    @staticmethod
    def _direct(y, log_ratio):
        return np.array([np.sum(y * np.exp(np.arange(len(y)) * z)) for z in log_ratio])

    @pytest.mark.parametrize("n", [1, 2, 17, 256, 20_001])
    def test_matches_direct_projection(self, n):
        y = np.random.default_rng(n).standard_normal(n)
        ratios = np.array([-1e-5 + 0.7j, -0.02 + 2.1j, 0.0 + 0.0j])
        got = BlockIndex(y).project(ratios)
        want = self._direct(y, ratios)
        scale = max(float(np.max(np.abs(want))), 1e-12)
        assert np.max(np.abs(got - want)) < 1e-11 * scale

    def test_zero_padding_does_not_contribute(self):
        # 101 is prime, so the rectangle is padded by several samples.
        y = np.random.default_rng(1).standard_normal(101)
        index = BlockIndex(y)
        assert index.n_blocks * index.block > index.n
        ratios = np.array([-0.05 + 1.3j])
        assert index.project(ratios) == pytest.approx(self._direct(y, ratios), rel=1e-12)

    def test_underflowing_weights_do_not_produce_subnormals(self):
        """A fast decay must give a finite, correct sum with no subnormal drag."""
        y = np.ones(50_000)
        ratios = np.array([-1.0 + 0.0j])
        got = BlockIndex(y).project(ratios)[0]
        assert got == pytest.approx(1.0 / (1.0 - np.exp(-1.0)), rel=1e-12)
