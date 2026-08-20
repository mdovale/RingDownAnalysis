"""
Tests for the nonlinear-damping / frequency-pull model (P2).

Records are 10x time-scaled versions of the E5-class controlled experiments
from the 2026-08-18 investigation notebook (see tests/test_demod.py for the
scaling rationale).
"""

import numpy as np
import pytest

from ringdownanalysis.demod import SegmentedDemodEstimator, SegmentedDemodResult
from ringdownanalysis.nonlinear import NonlinearDampingResult, fit_nonlinear_damping
from ringdownanalysis.signal import generate_pathological_ringdown

F0 = 7.6699
FS = 30.0
A0 = 600.0
SIGMA_W = 1.5
SEED = 20260818


def demod_record(**kwargs) -> SegmentedDemodResult:
    defaults = dict(
        f0=F0,
        fs=FS,
        duration=1080.0,
        a0=A0,
        sigma_white=SIGMA_W,
        rng=np.random.default_rng(SEED),
    )
    defaults.update(kwargs)
    t, x = generate_pathological_ringdown(**defaults)
    return SegmentedDemodEstimator().estimate(t, x, FS)


class TestNonlinearDampingRecovery:
    """E5-class: recover (tau0, beta) from amplitude-dependent damping."""

    TAU0 = 520.0
    BETA = 2.17e-6

    @pytest.fixture(scope="class")
    def result(self) -> NonlinearDampingResult:
        return fit_nonlinear_damping(demod_record(tau=self.TAU0, damping_beta=self.BETA))

    def test_recovers_tau0_and_beta(self, result):
        assert result.valid
        assert result.tau0 == pytest.approx(self.TAU0, rel=0.02)
        assert result.beta == pytest.approx(self.BETA, rel=0.05)

    def test_zero_amplitude_q(self, result):
        assert pytest.approx(np.pi * F0 * self.TAU0, rel=0.02) == result.Q0

    def test_model_tracks_measured_amplitude(self, result):
        assert result.log_residual_rms < 0.02
        assert len(result.model_amplitude) == result.n_segments

    def test_local_law_helpers(self, result):
        # tau(A) must decrease with amplitude for beta > 0.
        assert float(result.tau_at(500.0)) < float(result.tau_at(50.0)) < self.TAU0 * 1.02
        expected_tau = 1.0 / (1.0 / self.TAU0 + self.BETA * 100.0)
        assert float(result.tau_at(100.0)) == pytest.approx(expected_tau, rel=0.02)
        assert float(result.q_at(100.0)) == pytest.approx(np.pi * F0 * expected_tau, rel=0.02)


class TestSingleExponentialRecord:
    def test_beta_consistent_with_zero(self):
        result = fit_nonlinear_damping(demod_record(tau=370.0))
        assert result.valid
        assert result.tau0 == pytest.approx(370.0, rel=0.01)
        assert abs(result.beta) < 3.0 * result.beta_stderr + 1e-12


class TestFrequencyPull:
    def test_recovers_pull_and_zero_amplitude_frequency(self):
        result = fit_nonlinear_damping(demod_record(tau=370.0, freq_pull=-2.06e-5))
        assert result.valid
        assert result.f_pull == pytest.approx(-2.06e-5, rel=0.05)
        assert result.f_zero == pytest.approx(F0, abs=1e-4)
        assert result.f_pull_stderr is not None
        assert abs(result.f_pull) > 3.0 * result.f_pull_stderr


class TestDegenerateInputs:
    def test_too_few_segments_invalid(self):
        demod = demod_record(tau=370.0)
        result = fit_nonlinear_damping(demod, min_segments=10_000)
        assert not result.valid
        assert result.status == "invalid"
        assert "nonlinear_insufficient_decay_segments" in result.reasons

    def test_small_amplitude_span_invalid(self):
        # A window much shorter than tau spans < 1 e-fold of amplitude.
        demod = demod_record(tau=5000.0, duration=720.0)
        result = fit_nonlinear_damping(demod)
        assert not result.valid
        assert "nonlinear_amplitude_span_too_small" in result.reasons

    def test_helpers_raise_without_fit(self):
        demod = demod_record(tau=370.0)
        result = fit_nonlinear_damping(demod, min_segments=10_000)
        with pytest.raises(ValueError):
            result.tau_at(100.0)
        with pytest.raises(ValueError):
            result.q_at(100.0)
