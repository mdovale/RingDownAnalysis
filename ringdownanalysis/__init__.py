"""
Ring-down analysis package for frequency estimation of ring-down signals.

This package provides both object-oriented and function-based APIs for:
- Ring-down signal generation
- Frequency estimation (NLS and DFT methods)
- CRLB calculation
- Monte Carlo analysis
- Real data analysis
"""

import logging

from .analyzer import RingDownAnalyzer
from .batch_analyzer import BatchRingDownAnalyzer
from .crlb import CRLBCalculator
from .data_loader import RingDownDataLoader
from .estimators import (
    DFTFrequencyEstimator,
    EstimationResult,
    FrequencyEstimator,
    NLSFrequencyEstimator,
)
from .monte_carlo import MonteCarloAnalyzer

# Core classes
from .signal import RingDownSignal

# Configure package logger
_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())  # Default: no output unless configured

# Compatibility layer (legacy function-based API)
from .compat import (
    crlb_var_f_ringdown_explicit,
    db_to_lin,
    estimate_freq_dft,
    estimate_freq_dft_optimized,
    estimate_freq_nls_ringdown,
    generate_ringdown,
    monte_carlo_analysis,
)

# Plotting functions
from .plots import (
    plot_aggregate_results,
    plot_individual_results,
    plot_performance_comparison,
    plot_q_individual_results,
    plot_q_performance_comparison,
)

__all__ = [
    # Classes
    "RingDownSignal",
    "FrequencyEstimator",
    "NLSFrequencyEstimator",
    "DFTFrequencyEstimator",
    "EstimationResult",
    "CRLBCalculator",
    "RingDownDataLoader",
    "RingDownAnalyzer",
    "MonteCarloAnalyzer",
    "BatchRingDownAnalyzer",
    # Compatibility functions
    "db_to_lin",
    "crlb_var_f_ringdown_explicit",
    "generate_ringdown",
    "estimate_freq_nls_ringdown",
    "estimate_freq_dft",
    "estimate_freq_dft_optimized",
    "monte_carlo_analysis",
    # Plotting functions
    "plot_individual_results",
    "plot_aggregate_results",
    "plot_performance_comparison",
    "plot_q_individual_results",
    "plot_q_performance_comparison",
]
