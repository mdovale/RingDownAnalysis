"""
Ring-down analysis package for frequency estimation of ring-down signals.

This package provides both object-oriented and function-based APIs for:
- Ring-down signal generation
- Frequency estimation (NLS and DFT methods)
- CRLB calculation
- Monte Carlo analysis
- Real data analysis
"""

# Core classes
from .signal import RingDownSignal
from .estimators import FrequencyEstimator, NLSFrequencyEstimator, DFTFrequencyEstimator
from .crlb import CRLBCalculator
from .data_loader import RingDownDataLoader
from .analyzer import RingDownAnalyzer
from .monte_carlo import MonteCarloAnalyzer
from .batch_analyzer import BatchRingDownAnalyzer

# Compatibility layer (legacy function-based API)
from .compat import (
    db_to_lin,
    crlb_var_f_ringdown_explicit,
    generate_ringdown,
    estimate_freq_nls_ringdown,
    estimate_freq_dft,
    estimate_freq_dft_optimized,
    monte_carlo_analysis,
)

__all__ = [
    # Classes
    'RingDownSignal',
    'FrequencyEstimator',
    'NLSFrequencyEstimator',
    'DFTFrequencyEstimator',
    'CRLBCalculator',
    'RingDownDataLoader',
    'RingDownAnalyzer',
    'MonteCarloAnalyzer',
    'BatchRingDownAnalyzer',
    # Compatibility functions
    'db_to_lin',
    'crlb_var_f_ringdown_explicit',
    'generate_ringdown',
    'estimate_freq_nls_ringdown',
    'estimate_freq_dft',
    'estimate_freq_dft_optimized',
    'monte_carlo_analysis',
]

