"""
Unit tests for frequency estimators.
"""

import numpy as np
import pytest

from ringdownanalysis.estimators import (
    DFTFrequencyEstimator,
    EstimationResult,
    NLSFrequencyEstimator,
)
from ringdownanalysis.signal import RingDownSignal


class TestNLSFrequencyEstimator:
    """Test NLSFrequencyEstimator class."""

    def test_estimate_known_tau(self):
        """Test NLS estimation with known tau."""
        # Generate test signal
        signal = RingDownSignal(f0=5.0, fs=100.0, N=10000, A0=1.0, snr_db=60.0, Q=10000.0)
        rng = np.random.default_rng(42)
        _, x, _ = signal.generate(rng=rng)

        # Estimate with known tau
        estimator = NLSFrequencyEstimator(tau_known=signal.tau)
        f_est = estimator.estimate(x, signal.fs)

        # Should be close to true frequency
        assert abs(f_est - signal.f0) < 0.01

    def test_estimate_unknown_tau(self):
        """Test NLS estimation with unknown tau."""
        # Generate test signal
        signal = RingDownSignal(f0=5.0, fs=100.0, N=10000, A0=1.0, snr_db=60.0, Q=10000.0)
        rng = np.random.default_rng(42)
        _, x, _ = signal.generate(rng=rng)

        # Estimate with unknown tau
        estimator = NLSFrequencyEstimator(tau_known=None)
        f_est = estimator.estimate(x, signal.fs)

        # Should be close to true frequency
        assert abs(f_est - signal.f0) < 0.1

    def test_estimate_returns_float(self):
        """Test that estimate returns a float."""
        signal = RingDownSignal(f0=5.0, fs=100.0, N=1000, A0=1.0, snr_db=60.0, Q=10000.0)
        rng = np.random.default_rng(42)
        _, x, _ = signal.generate(rng=rng)

        estimator = NLSFrequencyEstimator()
        f_est = estimator.estimate(x, signal.fs)

        assert isinstance(f_est, float)
        assert 0 < f_est < signal.fs / 2

    def test_estimate_full_unknown_tau(self):
        """Test NLS estimate_full with unknown tau."""
        signal = RingDownSignal(f0=5.0, fs=100.0, N=10000, A0=1.0, snr_db=60.0, Q=10000.0)
        rng = np.random.default_rng(42)
        _, x, _ = signal.generate(rng=rng)

        estimator = NLSFrequencyEstimator(tau_known=None)
        result = estimator.estimate_full(x, signal.fs)

        assert isinstance(result, EstimationResult)
        assert abs(result.f - signal.f0) < 0.1
        assert result.tau is not None
        assert result.Q is not None
        assert abs(result.tau - signal.tau) < signal.tau * 0.2  # Within 20%
        assert abs(result.Q - signal.Q) < signal.Q * 0.2  # Within 20%

    def test_estimate_full_known_tau(self):
        """Test NLS estimate_full with known tau."""
        signal = RingDownSignal(f0=5.0, fs=100.0, N=10000, A0=1.0, snr_db=60.0, Q=10000.0)
        rng = np.random.default_rng(42)
        _, x, _ = signal.generate(rng=rng)

        estimator = NLSFrequencyEstimator(tau_known=signal.tau)
        result = estimator.estimate_full(x, signal.fs)

        assert isinstance(result, EstimationResult)
        assert abs(result.f - signal.f0) < 0.01
        assert result.tau == signal.tau
        assert result.Q is not None
        assert abs(result.Q - signal.Q) < signal.Q * 0.01  # Very close with known tau


class TestDFTFrequencyEstimator:
    """Test DFTFrequencyEstimator class."""

    def test_estimate_kaiser(self):
        """Test DFT estimation with Kaiser window."""
        # Generate test signal
        signal = RingDownSignal(f0=5.0, fs=100.0, N=10000, A0=1.0, snr_db=60.0, Q=10000.0)
        rng = np.random.default_rng(42)
        _, x, _ = signal.generate(rng=rng)

        # Estimate with Kaiser window
        estimator = DFTFrequencyEstimator(window="kaiser", kaiser_beta=9.0)
        f_est = estimator.estimate(x, signal.fs)

        # Should be close to true frequency
        assert abs(f_est - signal.f0) < 0.1

    def test_estimate_hann(self):
        """Test DFT estimation with Hann window."""
        signal = RingDownSignal(f0=5.0, fs=100.0, N=10000, A0=1.0, snr_db=60.0, Q=10000.0)
        rng = np.random.default_rng(42)
        _, x, _ = signal.generate(rng=rng)

        estimator = DFTFrequencyEstimator(window="hann")
        f_est = estimator.estimate(x, signal.fs)

        assert isinstance(f_est, float)
        assert 0 < f_est < signal.fs / 2

    def test_estimate_zeropad(self):
        """Test DFT estimation with zero-padding."""
        signal = RingDownSignal(f0=5.0, fs=100.0, N=1000, A0=1.0, snr_db=60.0, Q=10000.0)
        rng = np.random.default_rng(42)
        _, x, _ = signal.generate(rng=rng)

        estimator = DFTFrequencyEstimator(window="kaiser", use_zeropad=True, pad_factor=4)
        f_est = estimator.estimate(x, signal.fs)

        assert isinstance(f_est, float)
        assert 0 < f_est < signal.fs / 2

    def test_estimate_invalid_window(self):
        """Test that invalid window raises error."""
        estimator = DFTFrequencyEstimator(window="invalid")
        signal = RingDownSignal(f0=5.0, fs=100.0, N=1000)
        rng = np.random.default_rng(42)
        _, x, _ = signal.generate(rng=rng)

        with pytest.raises(ValueError, match="Unknown window"):
            estimator.estimate(x, signal.fs)

    def test_estimate_returns_float(self):
        """Test that estimate returns a float."""
        signal = RingDownSignal(f0=5.0, fs=100.0, N=1000, A0=1.0, snr_db=60.0, Q=10000.0)
        rng = np.random.default_rng(42)
        _, x, _ = signal.generate(rng=rng)

        estimator = DFTFrequencyEstimator()
        f_est = estimator.estimate(x, signal.fs)

        assert isinstance(f_est, float)
        assert 0 < f_est < signal.fs / 2

    def test_estimate_full(self):
        """Test DFT estimate_full (two-step: DFT for frequency, NLS for tau)."""
        signal = RingDownSignal(f0=5.0, fs=100.0, N=10000, A0=1.0, snr_db=60.0, Q=10000.0)
        rng = np.random.default_rng(42)
        _, x, _ = signal.generate(rng=rng)

        estimator = DFTFrequencyEstimator(window="rect")
        result = estimator.estimate_full(x, signal.fs)

        assert isinstance(result, EstimationResult)
        assert abs(result.f - signal.f0) < 0.1
        # tau and Q may be None if NLS step fails, but if successful should be reasonable
        if result.tau is not None:
            assert result.tau > 0
            assert result.Q is not None
            assert result.Q > 0
