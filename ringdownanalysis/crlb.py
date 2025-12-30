"""
Cramér-Rao Lower Bound (CRLB) calculation for ring-down signals.
"""

from typing import Tuple

import numpy as np


class CRLBCalculator:
    """
    Calculates the Cramér-Rao Lower Bound for frequency and quality factor (Q)
    estimation variance with ring-down signals, using explicit Fisher information matrix.
    """

    @staticmethod
    def _compute_weighted_sums(
        fs: float,
        N: int,
        tau: float,
    ) -> Tuple[float, float, float]:
        """
        Compute weighted sums S_0, S_1, S_2 used in CRLB calculations.

        Parameters:
        -----------
        fs : float
            Sampling frequency (Hz)
        N : int
            Number of samples
        tau : float
            Decay time constant (s)

        Returns:
        --------
        Tuple[float, float, float]
            (S_0, S_1, S_2) where:
            S_0 = sum_n exp(-2t_n/tau)
            S_1 = sum_n t_n exp(-2t_n/tau)
            S_2 = sum_n t_n^2 exp(-2t_n/tau)
        """
        Ts = 1.0 / fs
        t = np.arange(N) * Ts

        exp_factor = np.exp(-2.0 * t / tau)
        t_squared = t * t
        S_0 = np.sum(exp_factor)
        S_1 = np.sum(t * exp_factor)
        S_2 = np.sum(t_squared * exp_factor)

        return S_0, S_1, S_2

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

        # Calculate weighted sums
        S_0, S_1, S_2 = CRLBCalculator._compute_weighted_sums(fs, N, tau)

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

    @staticmethod
    def q_variance(
        A0: float,
        sigma: float,
        fs: float,
        N: int,
        tau: float,
        f0: float,
    ) -> float:
        """
        Calculate CRLB for quality factor (Q) estimation variance.

        For the ring-down model, the CRLB for Q depends on the covariance of
        (omega, tau) estimates. Under the high-SNR, many-cycle approximation,
        omega and tau are asymptotically uncorrelated after marginalizing
        nuisance parameters (A_0, phi).

        The CRLB for Q is given by:
        σ_Q^2 ≥ (σ^2 τ^2) / (4 A_0^2 ΔS_2) * (1 + 4Q^2)

        where ΔS_2 = S_2 - S_1^2/S_0 and Q = π f_0 τ.

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
        f0 : float
            Resonance frequency (Hz)

        Returns:
        --------
        float
            Lower bound on Var(Q_hat)
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
        if f0 <= 0:
            raise ValueError("f0 must be positive")

        # Calculate weighted sums
        S_0, S_1, S_2 = CRLBCalculator._compute_weighted_sums(fs, N, tau)

        # ΔS_2 = S_2 - S_1^2/S_0
        Delta_S2 = S_2 - S_1**2 / S_0

        if Delta_S2 < 1e-30:
            # Fallback for degenerate case
            return np.inf

        # Quality factor: Q = π f_0 τ
        Q = np.pi * f0 * tau

        # CRLB for Q: σ_Q^2 ≥ (σ^2 τ^2) / (4 A_0^2 ΔS_2) * (1 + 4Q^2)
        crlb_var_q = (sigma**2 * tau**2) / (4.0 * A0**2 * Delta_S2) * (1.0 + 4.0 * Q**2)

        return crlb_var_q

    @staticmethod
    def q_standard_deviation(
        A0: float,
        sigma: float,
        fs: float,
        N: int,
        tau: float,
        f0: float,
    ) -> float:
        """
        Calculate CRLB for quality factor (Q) estimation standard deviation.

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
        f0 : float
            Resonance frequency (Hz)

        Returns:
        --------
        float
            Lower bound on std(Q_hat)
        """
        var = CRLBCalculator.q_variance(A0, sigma, fs, N, tau, f0)
        if np.isinf(var):
            return np.inf
        return np.sqrt(var)
