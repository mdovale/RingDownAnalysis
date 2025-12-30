"""
Unit tests for compatibility layer.
"""

import numpy as np

from ringdownanalysis.compat import (
    crlb_var_f_ringdown_explicit,
    db_to_lin,
    estimate_freq_dft,
    estimate_freq_nls_ringdown,
    generate_ringdown,
)


class TestCompatibility:
    """Test compatibility layer functions."""

    def test_db_to_lin(self):
        """Test dB to linear conversion."""
        assert abs(db_to_lin(0.0) - 1.0) < 1e-10
        assert abs(db_to_lin(10.0) - 10.0) < 1e-10
        assert abs(db_to_lin(20.0) - 100.0) < 1e-10

    def test_crlb_var_f_ringdown_explicit(self):
        """Test CRLB function wrapper."""
        A0 = 1.0
        sigma = 0.1
        fs = 100.0
        N = 10000
        tau = 1000.0

        var = crlb_var_f_ringdown_explicit(A0, sigma, fs, N, tau)

        assert var > 0
        assert np.isfinite(var)

    def test_generate_ringdown(self):
        """Test signal generation wrapper."""
        t, x, sigma, phi0, tau = generate_ringdown(
            f0=5.0, fs=100.0, N=1000, A0=1.0, snr_db=60.0, Q=10000.0
        )

        assert len(t) == 1000
        assert len(x) == 1000
        assert sigma > 0
        assert -np.pi <= phi0 <= np.pi
        assert tau > 0

    def test_estimate_freq_nls_ringdown(self):
        """Test NLS estimation wrapper."""
        # Generate test signal
        t, x, _, _, _ = generate_ringdown(
            f0=5.0, fs=100.0, N=10000, A0=1.0, snr_db=60.0, Q=10000.0, rng=np.random.default_rng(42)
        )

        f_est = estimate_freq_nls_ringdown(x, 100.0)

        assert isinstance(f_est, float)
        assert 0 < f_est < 50.0  # Below Nyquist
        assert abs(f_est - 5.0) < 0.1  # Close to true frequency

    def test_estimate_freq_dft(self):
        """Test DFT estimation wrapper."""
        # Generate test signal
        t, x, _, _, _ = generate_ringdown(
            f0=5.0, fs=100.0, N=10000, A0=1.0, snr_db=60.0, Q=10000.0, rng=np.random.default_rng(42)
        )

        f_est = estimate_freq_dft(x, 100.0)

        assert isinstance(f_est, float)
        assert 0 < f_est < 50.0  # Below Nyquist
        assert abs(f_est - 5.0) < 0.1  # Close to true frequency
