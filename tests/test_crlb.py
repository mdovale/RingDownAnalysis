"""
Unit tests for CRLBCalculator class.
"""

import numpy as np
import pytest

from ringdownanalysis.crlb import CRLBCalculator


class TestCRLBCalculator:
    """Test CRLBCalculator class."""

    def test_variance(self):
        """Test CRLB variance calculation."""
        A0 = 1.0
        sigma = 0.1
        fs = 100.0
        N = 10000
        tau = 1000.0

        var = CRLBCalculator.variance(A0, sigma, fs, N, tau)

        assert var > 0
        assert np.isfinite(var)

    def test_standard_deviation(self):
        """Test CRLB standard deviation calculation."""
        A0 = 1.0
        sigma = 0.1
        fs = 100.0
        N = 10000
        tau = 1000.0

        std = CRLBCalculator.standard_deviation(A0, sigma, fs, N, tau)
        var = CRLBCalculator.variance(A0, sigma, fs, N, tau)

        assert abs(std - np.sqrt(var)) < 1e-10

    def test_variance_validation(self):
        """Test input validation."""
        with pytest.raises(ValueError, match="A0 must be positive"):
            CRLBCalculator.variance(-1.0, 0.1, 100.0, 1000, 100.0)

        with pytest.raises(ValueError, match="sigma must be positive"):
            CRLBCalculator.variance(1.0, -0.1, 100.0, 1000, 100.0)

        with pytest.raises(ValueError, match="fs must be positive"):
            CRLBCalculator.variance(1.0, 0.1, -100.0, 1000, 100.0)

        with pytest.raises(ValueError, match="N must be positive"):
            CRLBCalculator.variance(1.0, 0.1, 100.0, -1000, 100.0)

        with pytest.raises(ValueError, match="tau must be positive"):
            CRLBCalculator.variance(1.0, 0.1, 100.0, 1000, -100.0)

    def test_variance_scaling(self):
        """Test that variance scales appropriately with parameters."""
        A0 = 1.0
        sigma = 0.1
        fs = 100.0
        N = 10000
        tau = 1000.0

        var1 = CRLBCalculator.variance(A0, sigma, fs, N, tau)

        # Higher SNR should give lower variance
        var2 = CRLBCalculator.variance(A0, sigma / 2.0, fs, N, tau)
        assert var2 < var1

        # Longer observation time should give lower variance
        var3 = CRLBCalculator.variance(A0, sigma, fs, N * 2, tau)
        assert var3 < var1

    def test_q_variance(self):
        """Test CRLB Q variance calculation."""
        A0 = 1.0
        sigma = 0.1
        fs = 100.0
        N = 10000
        tau = 1000.0
        f0 = 5.0

        var_q = CRLBCalculator.q_variance(A0, sigma, fs, N, tau, f0)

        assert var_q > 0
        assert np.isfinite(var_q)

    def test_q_standard_deviation(self):
        """Test CRLB Q standard deviation calculation."""
        A0 = 1.0
        sigma = 0.1
        fs = 100.0
        N = 10000
        tau = 1000.0
        f0 = 5.0

        std_q = CRLBCalculator.q_standard_deviation(A0, sigma, fs, N, tau, f0)
        var_q = CRLBCalculator.q_variance(A0, sigma, fs, N, tau, f0)

        assert abs(std_q - np.sqrt(var_q)) < 1e-10

    def test_q_variance_validation(self):
        """Test input validation for Q CRLB."""
        A0 = 1.0
        sigma = 0.1
        fs = 100.0
        N = 1000
        tau = 100.0
        f0 = 5.0

        with pytest.raises(ValueError, match="A0 must be positive"):
            CRLBCalculator.q_variance(-1.0, sigma, fs, N, tau, f0)

        with pytest.raises(ValueError, match="sigma must be positive"):
            CRLBCalculator.q_variance(A0, -0.1, fs, N, tau, f0)

        with pytest.raises(ValueError, match="fs must be positive"):
            CRLBCalculator.q_variance(A0, sigma, -100.0, N, tau, f0)

        with pytest.raises(ValueError, match="N must be positive"):
            CRLBCalculator.q_variance(A0, sigma, fs, -1000, tau, f0)

        with pytest.raises(ValueError, match="tau must be positive"):
            CRLBCalculator.q_variance(A0, sigma, fs, N, -100.0, f0)

        with pytest.raises(ValueError, match="f0 must be positive"):
            CRLBCalculator.q_variance(A0, sigma, fs, N, tau, -5.0)

    def test_q_variance_scaling(self):
        """Test that Q variance scales appropriately with parameters."""
        A0 = 1.0
        sigma = 0.1
        fs = 100.0
        N = 10000
        tau = 1000.0
        f0 = 5.0

        var_q1 = CRLBCalculator.q_variance(A0, sigma, fs, N, tau, f0)

        # Higher SNR should give lower variance
        var_q2 = CRLBCalculator.q_variance(A0, sigma / 2.0, fs, N, tau, f0)
        assert var_q2 < var_q1

        # Longer observation time should give lower variance
        var_q3 = CRLBCalculator.q_variance(A0, sigma, fs, N * 2, tau, f0)
        assert var_q3 < var_q1

        # Higher Q (via higher f0 or tau) should give higher variance (due to 1 + 4Q^2 term)
        var_q4 = CRLBCalculator.q_variance(A0, sigma, fs, N, tau, f0 * 2.0)
        assert var_q4 > var_q1
