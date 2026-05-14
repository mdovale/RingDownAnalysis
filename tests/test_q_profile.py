"""
Unit tests for profile-likelihood Q estimation.
"""

import numpy as np
import pytest

from ringdownanalysis.q_profile import ProfileQEstimator, QProfileResult


def _ringdown(
    *,
    fs: float,
    f0: float,
    tau: float,
    n_samples: int,
    noise: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(n_samples, dtype=float) / fs
    rng = np.random.default_rng(seed)
    data = np.exp(-t / tau) * np.cos(2.0 * np.pi * f0 * t)
    if noise > 0:
        data = data + noise * rng.normal(size=n_samples)
    return t, data


class TestProfileQEstimator:
    """Behavioral tests for the variable-projection Q estimator."""

    def test_well_identified_ringdown_returns_finite_profile_q(self):
        fs = 200.0
        f0 = 5.0
        tau = 0.8
        true_q = np.pi * f0 * tau
        t, data = _ringdown(fs=fs, f0=f0, tau=tau, n_samples=2000, noise=0.02, seed=1)

        result = ProfileQEstimator(n_grid=121).estimate(
            t,
            data,
            fs,
            f_init=f0,
            tau_init=tau,
        )

        assert isinstance(result, QProfileResult)
        assert result.valid is True
        assert result.status == "valid"
        assert pytest.approx(true_q, rel=0.02) == result.Q
        assert result.ci95 is not None
        assert result.ci95[0] < true_q < result.ci95[1]
        assert result.lower_limit_95 is None
        assert result.upper_limit_95 is None
        assert len(result.profile_tau) == result.n_grid
        assert len(result.profile_delta) == result.n_grid

    def test_partial_high_q_record_returns_limit_not_finite_q(self):
        fs = 200.0
        f0 = 5.0
        tau = 10_000.0
        t, data = _ringdown(fs=fs, f0=f0, tau=tau, n_samples=1000, noise=0.02, seed=3)

        result = ProfileQEstimator(n_grid=121).estimate(
            t,
            data,
            fs,
            f_init=f0,
            tau_init=tau,
        )

        assert result.valid is False
        assert result.Q is None
        assert result.status == "lower_limit"
        assert result.lower_limit_95 is not None
        assert result.lower_limit_95 > 0
        assert "profile_open_high" in result.reasons

    def test_unresolved_constant_data_returns_invalid_status(self):
        fs = 100.0
        t = np.arange(100, dtype=float) / fs
        data = np.ones_like(t)

        result = ProfileQEstimator().estimate(t, data, fs, f_init=5.0, tau_init=1.0)

        assert result.valid is False
        assert result.Q is None
        assert result.status == "invalid"
        assert "profile_no_resolved_ac_content" in result.reasons
