"""
Compatibility layer for legacy function-based API.
"""

from typing import Optional

import numpy as np

from .crlb import CRLBCalculator
from .estimators import DFTFrequencyEstimator, NLSFrequencyEstimator
from .monte_carlo import MonteCarloAnalyzer
from .signal import RingDownSignal


# Utility functions
def db_to_lin(x_db: float) -> float:
    """Convert dB to linear scale."""
    return 10.0 ** (x_db / 10.0)


# CRLB functions
def crlb_var_f_ringdown_explicit(A0: float, sigma: float, fs: float, N: int, tau: float) -> float:
    """
    Cramér-Rao lower bound for frequency estimation variance with ring-down,
    calculated from explicit Fisher information matrix.

    This is a compatibility wrapper around CRLBCalculator.
    """
    return CRLBCalculator.variance(A0, sigma, fs, N, tau)


# Signal generation
def generate_ringdown(
    f0: float,
    fs: float,
    N: int,
    A0: float = 1.0,
    snr_db: float = 60.0,
    Q: float = 10000.0,
    rng: Optional[np.random.Generator] = None,
):
    """
    Generate a noisy ring-down signal (exponentially decaying sinusoid).

    This is a compatibility wrapper around RingDownSignal.
    """
    signal = RingDownSignal(f0=f0, fs=fs, N=N, A0=A0, snr_db=snr_db, Q=Q)
    t, x, phi0 = signal.generate(rng=rng)
    return t, x, signal.sigma, phi0, signal.tau


# Frequency estimation
def estimate_freq_nls_ringdown(
    x: np.ndarray, fs: float, tau_known: Optional[float] = None
) -> float:
    """
    Estimate frequency using nonlinear least squares with ring-down model.

    This is a compatibility wrapper around NLSFrequencyEstimator.
    """
    estimator = NLSFrequencyEstimator(tau_known=tau_known)
    return estimator.estimate(x, fs)


def estimate_freq_dft(
    x: np.ndarray,
    fs: float,
    window: str = "kaiser",
    kaiser_beta: float = 9.0,
) -> float:
    """
    Estimate frequency using DFT peak fitting with Lorentzian function.

    This is a compatibility wrapper around DFTFrequencyEstimator.
    """
    estimator = DFTFrequencyEstimator(
        window=window,
        use_zeropad=False,
        pad_factor=1,
        lorentzian_points=7,
        kaiser_beta=kaiser_beta,
    )
    return estimator.estimate(x, fs)


def estimate_freq_dft_optimized(
    x: np.ndarray,
    fs: float,
    window: str = "kaiser",
    use_zeropad: bool = True,
    pad_factor: int = 4,
    lorentzian_points: int = 7,
    kaiser_beta: float = 9.0,
) -> float:
    """
    Optimized DFT-based frequency estimation with Lorentzian fitting for ring-down signals.

    This is a compatibility wrapper around DFTFrequencyEstimator.
    """
    estimator = DFTFrequencyEstimator(
        window=window,
        use_zeropad=use_zeropad,
        pad_factor=pad_factor,
        lorentzian_points=lorentzian_points,
        kaiser_beta=kaiser_beta,
    )
    return estimator.estimate(x, fs)


# Monte Carlo analysis
def monte_carlo_analysis(
    f0: float = 5.0,
    fs: float = 100.0,
    N: int = 1_000_000,
    A0: float = 1.0,
    snr_db: float = 60.0,
    Q: float = 10000.0,
    n_mc: int = 100,
    seed: int = 42,
    n_workers: Optional[int] = None,
    timeout_per_trial: float = 30.0,
):
    """
    Perform Monte Carlo analysis of frequency estimation methods for ring-down signals.

    This is a compatibility wrapper around MonteCarloAnalyzer.
    """
    analyzer = MonteCarloAnalyzer()
    return analyzer.run(
        f0=f0,
        fs=fs,
        N=N,
        A0=A0,
        snr_db=snr_db,
        Q=Q,
        n_mc=n_mc,
        seed=seed,
        n_workers=n_workers,
        timeout_per_trial=timeout_per_trial,
    )
