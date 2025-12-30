"""
Unit tests for RingDownSignal class.
"""

import numpy as np
import pytest

from ringdownanalysis.signal import RingDownSignal


class TestRingDownSignal:
    """Test RingDownSignal class."""

    def test_init(self):
        """Test signal initialization."""
        signal = RingDownSignal(f0=5.0, fs=100.0, N=1000, A0=1.0, snr_db=60.0, Q=10000.0)

        assert signal.f0 == 5.0
        assert signal.fs == 100.0
        assert signal.N == 1000
        assert signal.A0 == 1.0
        assert signal.snr_db == 60.0
        assert signal.Q == 10000.0
        assert signal.tau > 0
        assert signal.sigma > 0

    def test_init_validation(self):
        """Test input validation."""
        with pytest.raises(ValueError, match="f0 must be positive"):
            RingDownSignal(f0=-1.0, fs=100.0, N=1000)

        with pytest.raises(ValueError, match="fs must be positive"):
            RingDownSignal(f0=5.0, fs=-100.0, N=1000)

        with pytest.raises(ValueError, match="N must be positive"):
            RingDownSignal(f0=5.0, fs=100.0, N=-1000)

        with pytest.raises(ValueError, match="A0 must be positive"):
            RingDownSignal(f0=5.0, fs=100.0, N=1000, A0=-1.0)

        with pytest.raises(ValueError, match="Q must be positive"):
            RingDownSignal(f0=5.0, fs=100.0, N=1000, Q=-10000.0)

    def test_tau_computation(self):
        """Test tau computation from Q and f0."""
        signal = RingDownSignal(f0=5.0, fs=100.0, N=1000, Q=10000.0)
        expected_tau = 10000.0 / (np.pi * 5.0)
        assert abs(signal.tau - expected_tau) < 1e-10

    def test_sigma_computation(self):
        """Test sigma computation from SNR."""
        signal = RingDownSignal(f0=5.0, fs=100.0, N=1000, A0=1.0, snr_db=60.0)
        rho0 = 10.0 ** (60.0 / 10.0)
        expected_sigma = np.sqrt((1.0**2 / 2.0) / rho0)
        assert abs(signal.sigma - expected_sigma) < 1e-10

    def test_time_array(self):
        """Test time array property."""
        signal = RingDownSignal(f0=5.0, fs=100.0, N=1000)
        t = signal.t

        assert len(t) == 1000
        assert t[0] == 0.0
        assert abs(t[-1] - 9.99) < 0.01
        assert abs(signal.T - 10.0) < 0.01

    def test_generate(self):
        """Test signal generation."""
        signal = RingDownSignal(f0=5.0, fs=100.0, N=1000, A0=1.0, snr_db=60.0, Q=10000.0)
        rng = np.random.default_rng(42)
        t, x, phi0 = signal.generate(rng=rng)

        assert len(t) == 1000
        assert len(x) == 1000
        assert -np.pi <= phi0 <= np.pi
        assert signal.get_phase() == phi0
        assert np.array_equal(signal.get_signal(), x)

    def test_generate_with_phase(self):
        """Test signal generation with specified phase."""
        signal = RingDownSignal(f0=5.0, fs=100.0, N=1000)
        t, x, phi0 = signal.generate(phi0=0.5)

        assert phi0 == 0.5

    def test_signal_properties(self):
        """Test signal has expected properties."""
        signal = RingDownSignal(f0=5.0, fs=100.0, N=1000, A0=1.0, snr_db=60.0, Q=10000.0)
        rng = np.random.default_rng(42)
        t, x, phi0 = signal.generate(rng=rng)

        # Signal should decay exponentially
        # Check that later samples have smaller amplitude on average
        early_std = np.std(x[:100])
        late_std = np.std(x[-100:])
        assert late_std < early_std  # Should decay
