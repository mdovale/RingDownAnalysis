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

