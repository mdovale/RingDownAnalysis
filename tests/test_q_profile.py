"""
Unit tests for profile-likelihood Q estimation.
"""

import numpy as np
import pytest

from ringdownanalysis.q_profile import ProfileQEstimator, QProfileResult, _ProfileScan


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


class TestProfileScan:
    """The closed-form uniform-grid scan must match the explicit design matrix."""

    @staticmethod
    def _reference_fit(t, data, f_hat, tau):
        """Textbook fit: build the design matrix and solve it with lstsq."""
        envelope = np.exp(-t / tau)
        omega_t = 2.0 * np.pi * f_hat * t
        design = np.column_stack(
            [envelope * np.cos(omega_t), envelope * np.sin(omega_t), np.ones_like(t)]
        )
        coeffs = np.linalg.lstsq(design, data, rcond=None)[0]
        residual = data - design @ coeffs
        return float(residual @ residual), float(np.hypot(coeffs[0], coeffs[1]))

    def test_uniform_fast_path_matches_explicit_design_matrix(self):
        fs, f0, tau = 30.0, 7.67, 370.0
        t, data = _ringdown(fs=fs, f0=f0, tau=tau, n_samples=12_000, noise=0.01, seed=11)
        scan = _ProfileScan(t, data, f0)
        assert scan.uniform is True

        taus = np.exp(np.linspace(np.log(0.1 * tau), np.log(30.0 * tau), 25))
        for fit, trial in zip(scan.fit(taus), taus, strict=True):
            rss, amplitude = self._reference_fit(t, data, f0, trial)
            assert fit.rss == pytest.approx(rss, rel=1e-9)
            assert fit.amplitude == pytest.approx(amplitude, rel=1e-9)

    def test_batched_scan_matches_one_tau_at_a_time(self):
        fs, f0, tau = 30.0, 7.67, 370.0
        t, data = _ringdown(fs=fs, f0=f0, tau=tau, n_samples=8_000, noise=0.01, seed=12)
        scan = _ProfileScan(t, data, f0)

        taus = np.exp(np.linspace(np.log(50.0), np.log(50_000.0), 17))
        batched = scan.fit(taus)
        one_by_one = [scan.fit(np.array([trial]))[0] for trial in taus]
        for many, single in zip(batched, one_by_one, strict=True):
            assert many.rss == pytest.approx(single.rss, rel=1e-12)

    def test_noiseless_record_avoids_cancellation_in_residual(self):
        """A perfect fit must not report a negative or wildly wrong residual."""
        fs, f0, tau = 30.0, 7.67, 370.0
        t, data = _ringdown(fs=fs, f0=f0, tau=tau, n_samples=6_000, noise=0.0, seed=13)
        scan = _ProfileScan(t, data, f0)

        fit = scan.fit(np.array([tau]))[0]
        reference, _ = self._reference_fit(t, data, f0, tau)
        assert fit.rss >= 0.0
        assert fit.rss < 1e-12 * float(data @ data)
        # Both are round-off; only the order of magnitude is meaningful.
        assert fit.rss == pytest.approx(reference, rel=0.5, abs=1e-16 * float(data @ data))

    def test_non_uniform_times_use_the_actual_sample_times(self):
        fs, f0, tau = 30.0, 7.67, 370.0
        t, data = _ringdown(fs=fs, f0=f0, tau=tau, n_samples=4_000, noise=0.01, seed=14)
        rng = np.random.default_rng(15)
        # A random walk in time far outside the uniform-grid tolerance.
        t_jittered = np.sort(t + np.cumsum(rng.normal(0.0, 0.05 / fs, size=len(t))))
        scan = _ProfileScan(t_jittered - t_jittered[0], data, f0)
        assert scan.uniform is False

        fit = scan.fit(np.array([tau]))[0]
        reference, _ = self._reference_fit(t_jittered - t_jittered[0], data, f0, tau)
        assert fit.rss == pytest.approx(reference, rel=1e-12)

    def test_uniform_and_jittered_estimates_agree_within_timing_tolerance(self):
        """Timing noise inside the tolerance must not move Q meaningfully."""
        fs, f0, tau = 30.0, 7.67, 370.0
        t, data = _ringdown(fs=fs, f0=f0, tau=tau, n_samples=12_000, noise=0.01, seed=16)
        rng = np.random.default_rng(17)
        # Quantization-like noise, as real phasemeter exports carry.
        t_quantized = t + rng.uniform(-1e-4, 1e-4, size=len(t)) / fs

        estimator = ProfileQEstimator(n_grid=121)
        exact = estimator.estimate(t, data, fs, f_init=f0, tau_init=tau)
        quantized = estimator.estimate(t_quantized, data, fs, f_init=f0, tau_init=tau)
        assert _ProfileScan(t_quantized - t_quantized[0], data, f0).uniform is True
        assert pytest.approx(exact.Q, rel=1e-6) == quantized.Q

    def test_degrees_of_freedom_must_be_positive(self):
        t = np.arange(3, dtype=float) / 10.0
        with pytest.raises(ValueError, match="degrees of freedom"):
            _ProfileScan(t, np.array([1.0, 0.5, -0.5]), 1.0).fit(np.array([1.0]))
