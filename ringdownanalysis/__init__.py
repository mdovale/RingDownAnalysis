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
from .batch_analyzer import BatchRingDownAnalyzer, ProcessResult
from .compat import (
    crlb_var_f_ringdown_explicit,
    db_to_lin,
    estimate_freq_dft,
    estimate_freq_dft_optimized,
    estimate_freq_nls_ringdown,
    generate_ringdown,
    monte_carlo_analysis,
)
from .crlb import CRLBCalculator
from .data_loader import RingDownDataLoader
from .estimators import (
    DFTFrequencyEstimator,
    EstimationResult,
    FrequencyEstimator,
    NLSFrequencyEstimator,
)
from .monte_carlo import MonteCarloAnalyzer
from .plots import (
    plot_aggregate_results,
    plot_individual_results,
    plot_performance_comparison,
    plot_q_individual_results,
    plot_q_performance_comparison,
)
from .signal import RingDownSignal

# Configure package logger
_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())  # Default: no output unless configured


def configure_logging(
    level: int = logging.INFO,
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> None:
    """
    Configure logging for the ringdownanalysis package with a default console handler.

    Call this at application startup for easier debugging. By default, the package
    uses NullHandler (no output) unless explicitly configured.

    Parameters:
    -----------
    level : int
        Logging level (default: logging.INFO)
    format_string : str
        Log message format (default includes timestamp, logger name, level, message)

    Example:
    --------
    >>> import logging
    >>> from ringdownanalysis import configure_logging, BatchRingDownAnalyzer
    >>> configure_logging(level=logging.INFO)
    >>> analyzer = BatchRingDownAnalyzer()
    >>> results = analyzer.process_directory("data")
    """
    logging.basicConfig(level=level, format=format_string)


__all__ = [
    # Logging
    "configure_logging",
    # Classes
    "ProcessResult",
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
