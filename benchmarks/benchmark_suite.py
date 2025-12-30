"""
Comprehensive benchmark suite for RingDownAnalysis.

Defines critical workloads and representative input sizes for performance testing.
Uses pytest-benchmark for reliable, repeatable timing measurements.
"""

import numpy as np
import pytest
from pathlib import Path
from typing import Dict, List

from ringdownanalysis import (
    RingDownAnalyzer,
    BatchRingDownAnalyzer,
    MonteCarloAnalyzer,
    NLSFrequencyEstimator,
    DFTFrequencyEstimator,
    RingDownSignal,
    CRLBCalculator,
)
from ringdownanalysis.estimators import (
    _estimate_initial_parameters_from_dft,
    _estimate_initial_tau_from_envelope,
    _fit_lorentzian_to_peak,
)
from ringdownanalysis.data_loader import RingDownDataLoader


# ============================================================================
# Critical Workload Definitions
# ============================================================================

# Representative input sizes based on real-world usage
WORKLOAD_SIZES = {
    "small": {
        "N": 10_000,      # ~100 seconds at 100 Hz
        "n_files": 5,
        "n_mc": 10,
    },
    "medium": {
        "N": 100_000,     # ~1000 seconds at 100 Hz
        "n_files": 20,
        "n_mc": 50,
    },
    "large": {
        "N": 1_000_000,   # ~10000 seconds at 100 Hz (matching LaTeX doc)
        "n_files": 50,
        "n_mc": 100,
    },
    "xlarge": {
        "N": 10_000_000,  # Stress test
        "n_files": 100,
        "n_mc": 500,
    },
}

# Standard signal parameters (matching LaTeX document)
STANDARD_PARAMS = {
    "f0": 5.0,           # Hz
    "fs": 100.0,         # Hz
    "A0": 1.0,
    "snr_db": 60.0,      # dB
    "Q": 10000.0,
}


# ============================================================================
# Fixtures for test data
# ============================================================================

@pytest.fixture(scope="session")
def rng():
    """Shared random number generator for deterministic benchmarks."""
    return np.random.default_rng(42)


@pytest.fixture
def small_signal(rng):
    """Small signal workload."""
    params = STANDARD_PARAMS.copy()
    params["N"] = WORKLOAD_SIZES["small"]["N"]
    signal = RingDownSignal(**params)
    t, x, _ = signal.generate(rng=rng)
    return t, x, signal


@pytest.fixture
def medium_signal(rng):
    """Medium signal workload."""
    params = STANDARD_PARAMS.copy()
    params["N"] = WORKLOAD_SIZES["medium"]["N"]
    signal = RingDownSignal(**params)
    t, x, _ = signal.generate(rng=rng)
    return t, x, signal


@pytest.fixture
def large_signal(rng):
    """Large signal workload."""
    params = STANDARD_PARAMS.copy()
    params["N"] = WORKLOAD_SIZES["large"]["N"]
    signal = RingDownSignal(**params)
    t, x, _ = signal.generate(rng=rng)
    return t, x, signal


@pytest.fixture
def xlarge_signal(rng):
    """Extra large signal workload."""
    params = STANDARD_PARAMS.copy()
    params["N"] = WORKLOAD_SIZES["xlarge"]["N"]
    signal = RingDownSignal(**params)
    t, x, _ = signal.generate(rng=rng)
    return t, x, signal


# ============================================================================
# Core Operation Benchmarks
# ============================================================================

class TestSignalGeneration:
    """Benchmark signal generation operations."""
    
    def test_signal_generation_small(self, benchmark, rng):
        """Benchmark small signal generation."""
        params = STANDARD_PARAMS.copy()
        params["N"] = WORKLOAD_SIZES["small"]["N"]
        
        def generate():
            signal = RingDownSignal(**params)
            return signal.generate(rng=rng)
        
        result = benchmark(generate)
        assert len(result[1]) == WORKLOAD_SIZES["small"]["N"]
    
    def test_signal_generation_medium(self, benchmark, rng):
        """Benchmark medium signal generation."""
        params = STANDARD_PARAMS.copy()
        params["N"] = WORKLOAD_SIZES["medium"]["N"]
        
        def generate():
            signal = RingDownSignal(**params)
            return signal.generate(rng=rng)
        
        result = benchmark(generate)
        assert len(result[1]) == WORKLOAD_SIZES["medium"]["N"]
    
    def test_signal_generation_large(self, benchmark, rng):
        """Benchmark large signal generation."""
        params = STANDARD_PARAMS.copy()
        params["N"] = WORKLOAD_SIZES["large"]["N"]
        
        def generate():
            signal = RingDownSignal(**params)
            return signal.generate(rng=rng)
        
        result = benchmark(generate)
        assert len(result[1]) == WORKLOAD_SIZES["large"]["N"]


class TestFrequencyEstimation:
    """Benchmark frequency estimation methods."""
    
    def test_dft_estimation_small(self, benchmark, small_signal):
        """Benchmark DFT estimation on small signal."""
        _, x, signal = small_signal
        estimator = DFTFrequencyEstimator(window="kaiser")
        
        def estimate():
            return estimator.estimate(x, signal.fs)
        
        result = benchmark(estimate)
        assert isinstance(result, float)
    
    def test_dft_estimation_medium(self, benchmark, medium_signal):
        """Benchmark DFT estimation on medium signal."""
        _, x, signal = medium_signal
        estimator = DFTFrequencyEstimator(window="kaiser")
        
        def estimate():
            return estimator.estimate(x, signal.fs)
        
        result = benchmark(estimate)
        assert isinstance(result, float)
    
    def test_dft_estimation_large(self, benchmark, large_signal):
        """Benchmark DFT estimation on large signal."""
        _, x, signal = large_signal
        estimator = DFTFrequencyEstimator(window="kaiser")
        
        def estimate():
            return estimator.estimate(x, signal.fs)
        
        result = benchmark(estimate)
        assert isinstance(result, float)
    
    def test_nls_estimation_small(self, benchmark, small_signal):
        """Benchmark NLS estimation on small signal."""
        _, x, signal = small_signal
        estimator = NLSFrequencyEstimator(tau_known=None)
        
        def estimate():
            return estimator.estimate(x, signal.fs)
        
        result = benchmark(estimate)
        assert isinstance(result, float)
    
    def test_nls_estimation_medium(self, benchmark, medium_signal):
        """Benchmark NLS estimation on medium signal."""
        _, x, signal = medium_signal
        estimator = NLSFrequencyEstimator(tau_known=None)
        
        def estimate():
            return estimator.estimate(x, signal.fs)
        
        result = benchmark(estimate)
        assert isinstance(result, float)
    
    def test_nls_estimation_large(self, benchmark, large_signal):
        """Benchmark NLS estimation on large signal."""
        _, x, signal = large_signal
        estimator = NLSFrequencyEstimator(tau_known=None)
        
        def estimate():
            return estimator.estimate(x, signal.fs)
        
        result = benchmark(estimate)
        assert isinstance(result, float)


class TestInitialParameterEstimation:
    """Benchmark initial parameter estimation helpers."""
    
    def test_estimate_initial_params_small(self, benchmark, small_signal):
        """Benchmark initial parameter estimation on small signal."""
        _, x, signal = small_signal
        
        def estimate():
            return _estimate_initial_parameters_from_dft(x, signal.fs)
        
        result = benchmark(estimate)
        assert len(result) == 4
    
    def test_estimate_initial_params_medium(self, benchmark, medium_signal):
        """Benchmark initial parameter estimation on medium signal."""
        _, x, signal = medium_signal
        
        def estimate():
            return _estimate_initial_parameters_from_dft(x, signal.fs)
        
        result = benchmark(estimate)
        assert len(result) == 4
    
    def test_estimate_initial_params_large(self, benchmark, large_signal):
        """Benchmark initial parameter estimation on large signal."""
        _, x, signal = large_signal
        
        def estimate():
            return _estimate_initial_parameters_from_dft(x, signal.fs)
        
        result = benchmark(estimate)
        assert len(result) == 4
    
    def test_estimate_tau_from_envelope_small(self, benchmark, small_signal):
        """Benchmark tau estimation from envelope on small signal."""
        t, x, _ = small_signal
        
        def estimate():
            return _estimate_initial_tau_from_envelope(x, t)
        
        result = benchmark(estimate)
        assert isinstance(result, float)


class TestCRLBCalculation:
    """Benchmark CRLB calculations."""
    
    def test_crlb_calculation_small(self, benchmark, small_signal):
        """Benchmark CRLB calculation for small signal."""
        _, x, signal = small_signal
        calc = CRLBCalculator()
        
        def calculate():
            return calc.variance(signal.A0, signal.sigma, signal.fs, signal.N, signal.tau)
        
        result = benchmark(calculate)
        assert np.isfinite(result) or np.isinf(result)
    
    def test_crlb_calculation_medium(self, benchmark, medium_signal):
        """Benchmark CRLB calculation for medium signal."""
        _, x, signal = medium_signal
        calc = CRLBCalculator()
        
        def calculate():
            return calc.variance(signal.A0, signal.sigma, signal.fs, signal.N, signal.tau)
        
        result = benchmark(calculate)
        assert np.isfinite(result) or np.isinf(result)
    
    def test_crlb_calculation_large(self, benchmark, large_signal):
        """Benchmark CRLB calculation for large signal."""
        _, x, signal = large_signal
        calc = CRLBCalculator()
        
        def calculate():
            return calc.variance(signal.A0, signal.sigma, signal.fs, signal.N, signal.tau)
        
        result = benchmark(calculate)
        assert np.isfinite(result) or np.isinf(result)


class TestFullAnalysisPipeline:
    """Benchmark complete analysis pipeline."""
    
    def test_analyze_synthetic_small(self, benchmark, rng):
        """Benchmark full analysis on small synthetic signal."""
        params = STANDARD_PARAMS.copy()
        params["N"] = WORKLOAD_SIZES["small"]["N"]
        signal = RingDownSignal(**params)
        t, x, _ = signal.generate(rng=rng)
        
        # Create temporary CSV file
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            for ti, xi in zip(t, x):
                f.write(f"{ti:.6f},0,0,{xi:.6f}\n")
            temp_path = f.name
        
        try:
            analyzer = RingDownAnalyzer()
            
            def analyze():
                return analyzer.analyze_file(temp_path)
            
            result = benchmark(analyze)
            assert 'f_nls' in result
        finally:
            os.unlink(temp_path)
    
    def test_analyze_synthetic_medium(self, benchmark, rng):
        """Benchmark full analysis on medium synthetic signal."""
        params = STANDARD_PARAMS.copy()
        params["N"] = WORKLOAD_SIZES["medium"]["N"]
        signal = RingDownSignal(**params)
        t, x, _ = signal.generate(rng=rng)
        
        # Create temporary CSV file
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            for ti, xi in zip(t[::10], x[::10]):  # Subsample to avoid huge files
                f.write(f"{ti:.6f},0,0,{xi:.6f}\n")
            temp_path = f.name
        
        try:
            analyzer = RingDownAnalyzer()
            
            def analyze():
                return analyzer.analyze_file(temp_path)
            
            result = benchmark(analyze)
            assert 'f_nls' in result
        finally:
            os.unlink(temp_path)


class TestBatchOperations:
    """Benchmark batch processing operations."""
    
    def test_consistency_analysis_small(self, benchmark, rng):
        """Benchmark consistency analysis on small batch."""
        n_files = WORKLOAD_SIZES["small"]["n_files"]
        analyzer = BatchRingDownAnalyzer()
        
        # Create mock results
        np.random.seed(42)
        analyzer.results = [
            {
                'f_nls': 5.0 + np.random.randn() * 0.001,
                'f_dft': 5.0 + np.random.randn() * 0.001,
            }
            for _ in range(n_files)
        ]
        
        def analyze():
            return analyzer.consistency_analysis()
        
        result = benchmark(analyze)
        assert 'nls_mean' in result
    
    def test_consistency_analysis_medium(self, benchmark, rng):
        """Benchmark consistency analysis on medium batch."""
        n_files = WORKLOAD_SIZES["medium"]["n_files"]
        analyzer = BatchRingDownAnalyzer()
        
        np.random.seed(42)
        analyzer.results = [
            {
                'f_nls': 5.0 + np.random.randn() * 0.001,
                'f_dft': 5.0 + np.random.randn() * 0.001,
            }
            for _ in range(n_files)
        ]
        
        def analyze():
            return analyzer.consistency_analysis()
        
        result = benchmark(analyze)
        assert 'nls_mean' in result
    
    def test_crlb_comparison_small(self, benchmark, rng):
        """Benchmark CRLB comparison on small batch."""
        n_files = WORKLOAD_SIZES["small"]["n_files"]
        analyzer = BatchRingDownAnalyzer()
        
        np.random.seed(42)
        analyzer.results = [
            {
                'f_nls': 5.0 + np.random.randn() * 0.001,
                'f_dft': 5.0 + np.random.randn() * 0.001,
                'crlb_std_f': 0.0001 * (1.0 + np.random.rand()),
            }
            for _ in range(n_files)
        ]
        
        def analyze():
            return analyzer.crlb_comparison_analysis()
        
        result = benchmark(analyze)
        assert 'frequency_diffs' in result
    
    def test_q_factor_calculation_small(self, benchmark, rng):
        """Benchmark Q factor calculation on small batch."""
        n_files = WORKLOAD_SIZES["small"]["n_files"]
        analyzer = BatchRingDownAnalyzer()
        
        np.random.seed(42)
        analyzer.results = [
            {
                'f_nls': 5.0 + np.random.randn() * 0.001,
                'tau_est': 100.0 + np.random.randn() * 1.0,
            }
            for _ in range(n_files)
        ]
        
        def calculate():
            return analyzer.calculate_q_factors()
        
        result = benchmark(calculate)
        assert len(result) == n_files


class TestMonteCarlo:
    """Benchmark Monte Carlo analysis."""
    
    @pytest.mark.slow
    def test_monte_carlo_small(self, benchmark, rng):
        """Benchmark small Monte Carlo run."""
        n_mc = WORKLOAD_SIZES["small"]["n_mc"]
        analyzer = MonteCarloAnalyzer()
        
        def run_mc():
            return analyzer.run(
                **STANDARD_PARAMS,
                N=WORKLOAD_SIZES["small"]["N"],
                n_mc=n_mc,
                seed=42,
                n_workers=1,  # Sequential for benchmarking
            )
        
        result = benchmark.pedantic(run_mc, rounds=1, iterations=1)
        assert 'errors_nls' in result
    
    @pytest.mark.slow
    def test_monte_carlo_medium(self, benchmark, rng):
        """Benchmark medium Monte Carlo run."""
        n_mc = WORKLOAD_SIZES["medium"]["n_mc"]
        analyzer = MonteCarloAnalyzer()
        
        def run_mc():
            return analyzer.run(
                **STANDARD_PARAMS,
                N=WORKLOAD_SIZES["medium"]["N"],
                n_mc=n_mc,
                seed=42,
                n_workers=1,  # Sequential for benchmarking
            )
        
        result = benchmark.pedantic(run_mc, rounds=1, iterations=1)
        assert 'errors_nls' in result


# ============================================================================
# Utility functions for reporting
# ============================================================================

def get_workload_sizes() -> Dict:
    """Get workload size definitions."""
    return WORKLOAD_SIZES.copy()


def get_standard_params() -> Dict:
    """Get standard signal parameters."""
    return STANDARD_PARAMS.copy()

