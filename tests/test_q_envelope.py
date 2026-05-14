"""
Unit tests for peak-to-peak envelope Q diagnostics.
"""

import numpy as np
import pytest

from ringdownanalysis.q_envelope import QEnvelopeDiagnostic, q_envelope_diagnostic


def _ringdown(
    *,
    fs: float,
    f0: float,
    tau: float,
    duration: float,
    noise: float = 0.0,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(0.0, duration, 1.0 / fs)
    data = np.exp(-t / tau) * np.cos(2.0 * np.pi * f0 * t)
    if noise > 0:
        rng = np.random.default_rng(seed)
        data = data + noise * rng.normal(size=len(t))
    return t, data


class TestQEnvelopeDiagnostic:
    """Behavioral tests for the envelope tau/Q diagnostic."""

    def test_clean_ringdown_returns_envelope_tau_and_q(self):
        fs = 500.0
        f0 = 5.0
        tau = 0.8
        t, data = _ringdown(fs=fs, f0=f0, tau=tau, duration=5.0)

        result = q_envelope_diagnostic(t, data, f0)

        assert isinstance(result, QEnvelopeDiagnostic)
        assert result.valid is True
        assert result.status == "valid"
        assert result.tau == pytest.approx(tau, rel=0.2)
        assert pytest.approx(np.pi * f0 * tau, rel=0.2) == result.Q
        assert result.n_windows_used >= 5
        assert len(result.t_mid) == result.n_windows
        assert len(result.fitted_amplitude) == result.n_windows

    def test_candidate_q_agreement_is_reported(self):
        fs = 500.0
        f0 = 5.0
        tau = 0.8
        true_q = float(np.pi * f0 * tau)
        t, data = _ringdown(fs=fs, f0=f0, tau=tau, duration=5.0, noise=0.002)

        good = q_envelope_diagnostic(t, data, f0, q=true_q)
        bad = q_envelope_diagnostic(t, data, f0, q=true_q * 5.0)

        assert good.valid is True
        assert good.candidate_agrees is True
        assert good.candidate_log_rmse is not None
        assert bad.valid is True
        assert bad.status == "warning"
        assert bad.candidate_agrees is False
        assert "candidate_q_envelope_mismatch" in bad.reasons

    def test_invalid_frequency_returns_invalid_status(self):
        t = np.arange(100, dtype=float) / 100.0
        data = np.cos(2.0 * np.pi * 5.0 * t)

        result = q_envelope_diagnostic(t, data, f_hz=np.nan)

        assert result.valid is False
        assert result.status == "invalid"
        assert "envelope_frequency_missing_or_nonpositive" in result.reasons
