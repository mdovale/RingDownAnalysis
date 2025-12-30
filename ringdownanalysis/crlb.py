"""
Cramér-Rao Lower Bound (CRLB) calculation for ring-down signals.
"""

import numpy as np


class CRLBCalculator:
    """
    Calculates the Cramér-Rao Lower Bound for frequency estimation variance
    with ring-down signals, using explicit Fisher information matrix.
    """
    
    @staticmethod
    def variance(
        A0: float,
        sigma: float,
        fs: float,
        N: int,
        tau: float,
    ) -> float:
        """
        Calculate CRLB for frequency estimation variance.
        
        For the ring-down model with known tau, the Fisher information matrix
        elements involve weighted sums S_0, S_1, S_2 of exp(-2t_n/tau) with
        different powers of t_n.
        
        Parameters:
        -----------
        A0 : float
            Initial amplitude of the sinusoid
        sigma : float
            Standard deviation of additive white Gaussian noise
        fs : float
            Sampling frequency (Hz)
        N : int
            Number of samples
        tau : float
            Decay time constant (s)
        
        Returns:
        --------
        float
            Lower bound on Var(f_hat)
        """
        if A0 <= 0:
            raise ValueError("A0 must be positive")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if fs <= 0:
            raise ValueError("fs must be positive")
        if N <= 0:
            raise ValueError("N must be positive")
        if tau <= 0:
            raise ValueError("tau must be positive")
        
        Ts = 1.0 / fs
        t = np.arange(N) * Ts
        
        # Calculate weighted sums explicitly
        # S_0 = sum_n exp(-2t_n/tau)
        # S_1 = sum_n t_n exp(-2t_n/tau)
        # S_2 = sum_n t_n^2 exp(-2t_n/tau)
        # Optimize by computing t**2 once and reusing
        exp_factor = np.exp(-2.0 * t / tau)
        t_squared = t * t  # Compute once and reuse
        S_0 = np.sum(exp_factor)
        S_1 = np.sum(t * exp_factor)
        S_2 = np.sum(t_squared * exp_factor)
        
        # Effective Fisher information for omega (frequency in rad/s)
        # I_eff(omega) = (A0^2/sigma^2) * (S_2 - S_1^2/S_0)
        # This comes from the Schur complement accounting for nuisance parameters
        I_eff_omega = (A0**2 / sigma**2) * (S_2 - S_1**2 / S_0)
        
        if I_eff_omega < 1e-30:
            # Fallback for degenerate case
            return np.inf
        
        # CRLB for frequency in Hz: Var(f) = Var(omega)/(2pi)^2
        crlb_var_f = 1.0 / ((2.0 * np.pi) ** 2 * I_eff_omega)
        
        return crlb_var_f
    
    @staticmethod
    def standard_deviation(
        A0: float,
        sigma: float,
        fs: float,
        N: int,
        tau: float,
    ) -> float:
        """
        Calculate CRLB for frequency estimation standard deviation.
        
        Parameters:
        -----------
        A0 : float
            Initial amplitude of the sinusoid
        sigma : float
            Standard deviation of additive white Gaussian noise
        fs : float
            Sampling frequency (Hz)
        N : int
            Number of samples
        tau : float
            Decay time constant (s)
        
        Returns:
        --------
        float
            Lower bound on std(f_hat)
        """
        var = CRLBCalculator.variance(A0, sigma, fs, N, tau)
        if np.isinf(var):
            return np.inf
        return np.sqrt(var)

