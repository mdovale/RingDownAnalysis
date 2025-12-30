"""
Performance optimization tests and benchmarks.

This module tests that optimizations preserve correctness and measures performance improvements.
"""

import time

import numpy as np
import pytest

from ringdownanalysis.batch_analyzer import BatchRingDownAnalyzer
from ringdownanalysis.estimators import (
    _estimate_initial_parameters_from_dft,
    _estimate_initial_tau_from_envelope,
    _fit_lorentzian_to_peak,
)


class TestCorrectness:
    """Test that optimizations preserve correctness."""

    def test_consistency_analysis_pairwise_diffs(self):
        """Test that pairwise differences are computed correctly."""
        analyzer = BatchRingDownAnalyzer()

        # Create mock results
        analyzer.results = [
            {"f_nls": 5.0, "f_dft": 5.001},
            {"f_nls": 5.1, "f_dft": 5.101},
            {"f_nls": 5.2, "f_dft": 5.201},
        ]

        result = analyzer.consistency_analysis()

        # Check that pairwise differences are correct
        nls_diffs = result["nls_pairwise_diffs"]
        dft_diffs = result["dft_pairwise_diffs"]

        # For 3 realizations, we should have 3 pairwise comparisons: (0,1), (0,2), (1,2)
        assert len(nls_diffs) == 3
        assert len(dft_diffs) == 3

        # Check specific values
        assert abs(nls_diffs[0] - 0.1) < 1e-10  # |5.0 - 5.1|
        assert abs(nls_diffs[1] - 0.2) < 1e-10  # |5.0 - 5.2|
        assert abs(nls_diffs[2] - 0.1) < 1e-10  # |5.1 - 5.2|

        # Check indices
        indices = result["nls_pairwise_indices"]
        assert len(indices) == 3
        assert (0, 1) in indices
        assert (0, 2) in indices
        assert (1, 2) in indices

    def test_consistency_analysis_with_nans(self):
        """Test that NaN handling is preserved."""
        analyzer = BatchRingDownAnalyzer()

        # Create results with NaN values
        analyzer.results = [
            {"f_nls": 5.0, "f_dft": 5.001},
            {"f_nls": np.nan, "f_dft": 5.101},
            {"f_nls": 5.2, "f_dft": np.nan},
        ]

        result = analyzer.consistency_analysis()

        # Should still compute pairwise differences (NaNs will propagate)
        assert len(result["nls_pairwise_diffs"]) == 3
        assert len(result["dft_pairwise_diffs"]) == 3

        # Check that NaN values are handled correctly
        # Pairwise comparisons: (0,1), (0,2), (1,2)
        # For (0,1): |5.0 - nan| = nan
        # For (0,2): |5.0 - 5.2| = 0.2 (both valid, so result is valid)
        # For (1,2): |nan - 5.2| = nan
        assert np.isnan(result["nls_pairwise_diffs"][0])  # |5.0 - nan|
        assert abs(result["nls_pairwise_diffs"][1] - 0.2) < 1e-10  # |5.0 - 5.2|
        assert np.isnan(result["nls_pairwise_diffs"][2])  # |nan - 5.2|

    def test_calculate_q_factors(self):
        """Test that Q factors are calculated correctly."""
        analyzer = BatchRingDownAnalyzer()

        analyzer.results = [
            {"f_nls": 5.0, "tau_est": 100.0},
            {"f_nls": 10.0, "tau_est": 50.0},
        ]

        q_factors = analyzer.calculate_q_factors()

        # Q = π * f * τ
        expected_q1 = np.pi * 5.0 * 100.0
        expected_q2 = np.pi * 10.0 * 50.0

        assert abs(q_factors[0] - expected_q1) < 1e-10
        assert abs(q_factors[1] - expected_q2) < 1e-10

        # Check that Q is stored in results
        assert abs(analyzer.results[0]["Q"] - expected_q1) < 1e-10
        assert abs(analyzer.results[1]["Q"] - expected_q2) < 1e-10

    def test_crlb_comparison_analysis(self):
        """Test that CRLB comparison analysis is correct."""
        analyzer = BatchRingDownAnalyzer()

        analyzer.results = [
            {"f_nls": 5.0, "f_dft": 5.001, "crlb_std_f": 0.0001},
            {"f_nls": 5.1, "f_dft": 5.101, "crlb_std_f": 0.0002},
            {"f_nls": 5.2, "f_dft": 5.201, "crlb_std_f": np.inf},
        ]

        result = analyzer.crlb_comparison_analysis()

        # Check frequency differences
        diffs = result["frequency_diffs"]
        assert len(diffs) == 3
        assert abs(diffs[0] - 0.001) < 1e-10
        assert abs(diffs[1] - 0.001) < 1e-10
        assert abs(diffs[2] - 0.001) < 1e-10

        # Check ratios
        ratios = result["ratios"]
        assert abs(ratios[0] - 10.0) < 1e-10  # 0.001 / 0.0001
        assert abs(ratios[1] - 5.0) < 1e-10  # 0.001 / 0.0002
        assert np.isnan(ratios[2])  # Division by inf results in nan

        # Check valid ratios
        valid_ratios = result["valid_ratios"]
        assert len(valid_ratios) == 2
        # Check that both expected values are in the array using np.isin
        assert np.any(np.isclose(valid_ratios, 10.0))
        assert np.any(np.isclose(valid_ratios, 5.0))

    def test_crlb_comparison_with_zero_crlb(self):
        """Test CRLB comparison with zero CRLB values."""
        analyzer = BatchRingDownAnalyzer()

        analyzer.results = [
            {"f_nls": 5.0, "f_dft": 5.001, "crlb_std_f": 0.0},
            {"f_nls": 5.1, "f_dft": 5.101, "crlb_std_f": 0.0001},
        ]

        result = analyzer.crlb_comparison_analysis()

        # Zero CRLB should result in NaN ratio
        ratios = result["ratios"]
        assert np.isnan(ratios[0])
        assert not np.isnan(ratios[1])

    def test_estimate_tau_from_envelope(self):
        """Test that tau estimation from envelope works correctly."""
        # Create a simple decaying signal
        t = np.linspace(0, 10, 10000)
        tau_true = 2.0
        x = np.exp(-t / tau_true) * np.cos(2 * np.pi * 5.0 * t)

        tau_est = _estimate_initial_tau_from_envelope(x, t)

        # Should be in reasonable range (not exact due to windowing)
        assert 0.5 * tau_true < tau_est < 5.0 * tau_true

    def test_fit_lorentzian_to_peak(self):
        """Test Lorentzian fitting to peak."""
        # Create a simple power spectrum with a peak
        N = 1000
        fs = 100.0
        f0 = 10.0
        k0 = int(f0 * N / fs)

        # Create Lorentzian-like power spectrum
        k_indices = np.arange(N // 2 + 1)
        f_bins = k_indices * fs / N
        gamma = 1.0
        P = 1.0 / ((f_bins - f0) ** 2 + (gamma / 2.0) ** 2) + 0.01

        # Test fitting
        delta = _fit_lorentzian_to_peak(P, k0, fs, N, n_points=7)

        # Delta should be small (peak should be close to k0)
        assert abs(delta) < 0.5

    def test_estimate_initial_parameters_from_dft(self):
        """Test initial parameter estimation from DFT."""
        # Create a simple signal
        fs = 100.0
        N = 10000
        t = np.arange(N) / fs
        f0 = 10.0
        A0 = 1.0
        phi0 = 0.5
        c0 = 0.1

        x = A0 * np.cos(2 * np.pi * f0 * t + phi0) + c0

        f_est, phi_est, A_est, c_est = _estimate_initial_parameters_from_dft(x, fs)

        # Frequency should be close
        assert abs(f_est - f0) < 0.1

        # Amplitude should be reasonable
        assert 0.5 * A0 < A_est < 2.0 * A0


class TestPerformance:
    """Benchmark performance improvements."""

    def benchmark_consistency_analysis(self, n_realizations: int = 100):
        """Benchmark consistency analysis."""
        analyzer = BatchRingDownAnalyzer()

        # Create mock results
        np.random.seed(42)
        analyzer.results = [
            {
                "f_nls": 5.0 + np.random.randn() * 0.001,
                "f_dft": 5.0 + np.random.randn() * 0.001,
            }
            for _ in range(n_realizations)
        ]

        # Time the analysis
        start = time.perf_counter()
        result = analyzer.consistency_analysis()
        elapsed = time.perf_counter() - start

        # Verify correctness
        assert len(result["nls_pairwise_diffs"]) == n_realizations * (n_realizations - 1) // 2

        return elapsed

    def benchmark_calculate_q_factors(self, n_results: int = 1000):
        """Benchmark Q factor calculation."""
        analyzer = BatchRingDownAnalyzer()

        np.random.seed(42)
        analyzer.results = [
            {
                "f_nls": 5.0 + np.random.randn() * 0.001,
                "tau_est": 100.0 + np.random.randn() * 1.0,
            }
            for _ in range(n_results)
        ]

        start = time.perf_counter()
        q_factors = analyzer.calculate_q_factors()
        elapsed = time.perf_counter() - start

        assert len(q_factors) == n_results

        return elapsed

    def benchmark_crlb_comparison(self, n_results: int = 1000):
        """Benchmark CRLB comparison analysis."""
        analyzer = BatchRingDownAnalyzer()

        np.random.seed(42)
        analyzer.results = [
            {
                "f_nls": 5.0 + np.random.randn() * 0.001,
                "f_dft": 5.0 + np.random.randn() * 0.001,
                "crlb_std_f": 0.0001 * (1.0 + np.random.rand()),
            }
            for _ in range(n_results)
        ]

        start = time.perf_counter()
        result = analyzer.crlb_comparison_analysis()
        elapsed = time.perf_counter() - start

        assert len(result["frequency_diffs"]) == n_results

        return elapsed

    def test_performance_improvements(self):
        """Run performance benchmarks and report results."""
        print("\n" + "=" * 70)
        print("Performance Benchmarks")
        print("=" * 70)

        # Benchmark consistency analysis
        elapsed_consistency = self.benchmark_consistency_analysis(n_realizations=100)
        print(f"Consistency analysis (100 realizations): {elapsed_consistency * 1000:.2f} ms")

        # Benchmark Q factors
        elapsed_q = self.benchmark_calculate_q_factors(n_results=1000)
        print(f"Q factor calculation (1000 results): {elapsed_q * 1000:.2f} ms")

        # Benchmark CRLB comparison
        elapsed_crlb = self.benchmark_crlb_comparison(n_results=1000)
        print(f"CRLB comparison (1000 results): {elapsed_crlb * 1000:.2f} ms")

        print("=" * 70)

        # Assert that operations complete in reasonable time
        assert elapsed_consistency < 1.0, "Consistency analysis too slow"
        assert elapsed_q < 0.1, "Q factor calculation too slow"
        assert elapsed_crlb < 0.1, "CRLB comparison too slow"


class TestDtypePreservation:
    """Test that dtypes are preserved correctly."""

    def test_array_dtypes(self):
        """Test that array dtypes are correct."""
        analyzer = BatchRingDownAnalyzer()

        analyzer.results = [
            {"f_nls": 5.0, "f_dft": 5.001, "tau_est": 100.0, "crlb_std_f": 0.0001},
            {"f_nls": 5.1, "f_dft": 5.101, "tau_est": 101.0, "crlb_std_f": 0.0002},
        ]

        # Test consistency analysis
        result = analyzer.consistency_analysis()
        assert result["nls_pairwise_diffs"].dtype == np.float64
        assert result["dft_pairwise_diffs"].dtype == np.float64

        # Test Q factors
        q_factors = analyzer.calculate_q_factors()
        assert isinstance(q_factors[0], (float, np.floating))

        # Test CRLB comparison
        crlb_result = analyzer.crlb_comparison_analysis()
        assert crlb_result["frequency_diffs"].dtype == np.float64
        assert crlb_result["crlb_stds"].dtype == np.float64
        assert crlb_result["ratios"].dtype == np.float64


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
