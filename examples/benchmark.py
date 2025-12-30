"""
Benchmark script for assessing performance of NLS and DFT frequency estimation methods.

This script creates a synthetic ring-down signal with noise and 1 million points,
then times the NLS and DFT methods separately (5 times each) and reports their
completion time and standard deviation.
"""

import time

import numpy as np

from ringdownanalysis import DFTFrequencyEstimator, NLSFrequencyEstimator, RingDownSignal


def run_benchmark():
    """Run benchmark comparing NLS and DFT methods."""

    print("=" * 70)
    print("Ring-Down Analysis Performance Benchmark")
    print("=" * 70)
    print()

    # Signal parameters
    f0 = 5.0  # Frequency (Hz)
    fs = 100.0  # Sampling frequency (Hz)
    N = 1_000_000  # Number of samples (1 million points)
    A0 = 1.0  # Initial amplitude
    snr_db = 60.0  # Initial SNR (dB)
    Q = 10000.0  # Quality factor

    print("Signal parameters:")
    print(f"  f0 = {f0} Hz")
    print(f"  fs = {fs} Hz")
    print(f"  N = {N:,} samples")
    print(f"  A0 = {A0}")
    print(f"  SNR = {snr_db} dB")
    print(f"  Q = {Q:.0e}")
    print()

    # Generate synthetic signal
    print("Generating synthetic ring-down signal...")
    signal = RingDownSignal(f0=f0, fs=fs, N=N, A0=A0, snr_db=snr_db, Q=Q)
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    t, x, phi0 = signal.generate(rng=rng)
    print(f"  Signal generated: {len(x):,} samples")
    print(f"  Duration: {t[-1]:.2f} s")
    print()

    # Initialize estimators
    nls_estimator = NLSFrequencyEstimator(tau_known=None)
    dft_estimator = DFTFrequencyEstimator(window="rect")

    # Benchmark NLS method
    print("Benchmarking NLS method...")
    nls_times = []
    nls_results = []

    for i in range(5):
        start_time = time.perf_counter()
        f_nls = nls_estimator.estimate(x, fs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        nls_times.append(elapsed)
        nls_results.append(f_nls)
        print(f"  Run {i + 1}/5: {elapsed:.4f} s (f_est = {f_nls:.6f} Hz)")

    nls_mean = np.mean(nls_times)
    nls_std = np.std(nls_times)
    nls_mean_result = np.mean(nls_results)

    print("  NLS Results:")
    print(f"    Mean time: {nls_mean:.4f} s")
    print(f"    Std dev:   {nls_std:.4f} s")
    print(f"    Mean frequency estimate: {nls_mean_result:.6f} Hz")
    print(f"    Error: {abs(nls_mean_result - f0):.6e} Hz")
    print()

    # Benchmark DFT method
    print("Benchmarking DFT method...")
    dft_times = []
    dft_results = []

    for i in range(5):
        start_time = time.perf_counter()
        f_dft = dft_estimator.estimate(x, fs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        dft_times.append(elapsed)
        dft_results.append(f_dft)
        print(f"  Run {i + 1}/5: {elapsed:.4f} s (f_est = {f_dft:.6f} Hz)")

    dft_mean = np.mean(dft_times)
    dft_std = np.std(dft_times)
    dft_mean_result = np.mean(dft_results)

    print("  DFT Results:")
    print(f"    Mean time: {dft_mean:.4f} s")
    print(f"    Std dev:   {dft_std:.4f} s")
    print(f"    Mean frequency estimate: {dft_mean_result:.6f} Hz")
    print(f"    Error: {abs(dft_mean_result - f0):.6e} Hz")
    print()

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"True frequency: {f0:.6f} Hz")
    print()
    print("NLS Method:")
    print(f"  Mean execution time: {nls_mean:.4f} ± {nls_std:.4f} s")
    print(f"  Frequency estimate:  {nls_mean_result:.6f} Hz")
    print(f"  Error:               {abs(nls_mean_result - f0):.6e} Hz")
    print()
    print("DFT Method:")
    print(f"  Mean execution time: {dft_mean:.4f} ± {dft_std:.4f} s")
    print(f"  Frequency estimate:  {dft_mean_result:.6f} Hz")
    print(f"  Error:               {abs(dft_mean_result - f0):.6e} Hz")
    print()

    speedup = nls_mean / dft_mean
    print(f"Speedup (NLS/DFT): {speedup:.2f}x")
    if speedup > 1:
        print(f"  → DFT is {speedup:.2f}x faster than NLS")
    else:
        print(f"  → NLS is {1 / speedup:.2f}x faster than DFT")
    print()

    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
