"""
Usage examples for the ring-down analysis package.

This demonstrates both the object-oriented API and the compatibility layer.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ringdownanalysis import (
    # OOP API
    RingDownSignal,
    NLSFrequencyEstimator,
    DFTFrequencyEstimator,
    CRLBCalculator,
    MonteCarloAnalyzer,
    RingDownAnalyzer,
    RingDownDataLoader,
    # Compatibility API
    generate_ringdown,
    estimate_freq_nls_ringdown,
    estimate_freq_dft,
    monte_carlo_analysis,
)
# Import plotting functions from legacy module
from ringdownanalysis.legacy_ring_down_mc import (
    plot_individual_results,
    plot_aggregate_results,
    plot_performance_comparison,
)


def example_oop_signal_generation():
    """Example: Generate ring-down signal using OOP API."""
    print("=" * 70)
    print("Example 1: Signal Generation (OOP API)")
    print("=" * 70)
    
    # Create signal with specified parameters (matching LaTeX document)
    signal = RingDownSignal(
        f0=5.0,      # Frequency (Hz)
        fs=100.0,    # Sampling frequency (Hz)
        N=1_000_000, # Number of samples (matching LaTeX: N = 10^6)
        A0=1.0,      # Initial amplitude
        snr_db=60.0, # Initial SNR (dB)
        Q=10000.0,   # Quality factor
    )
    
    print(f"Signal parameters:")
    print(f"  f0 = {signal.f0} Hz")
    print(f"  fs = {signal.fs} Hz")
    print(f"  N = {signal.N} samples")
    print(f"  A0 = {signal.A0}")
    print(f"  Q = {signal.Q}")
    print(f"  tau = {signal.tau:.2f} s")
    print(f"  sigma = {signal.sigma:.6f}")
    print(f"  T = {signal.T:.2f} s")
    
    # Generate signal
    rng = np.random.default_rng(42)
    t, x, phi0 = signal.generate(rng=rng)
    
    print(f"\nGenerated signal:")
    print(f"  Length: {len(x)} samples")
    print(f"  Initial phase: {phi0:.4f} rad")
    print(f"  Signal mean: {np.mean(x):.6f}")
    print(f"  Signal std: {np.std(x):.6f}")
    print()


def example_oop_frequency_estimation():
    """Example: Estimate frequency using OOP API."""
    print("=" * 70)
    print("Example 2: Frequency Estimation (OOP API)")
    print("=" * 70)
    
    # Generate test signal (matching LaTeX document parameters)
    signal = RingDownSignal(f0=5.0, fs=100.0, N=1_000_000, A0=1.0, snr_db=60.0, Q=10000.0)
    rng = np.random.default_rng(42)
    _, x, _ = signal.generate(rng=rng)
    
    # NLS estimator
    nls_estimator = NLSFrequencyEstimator(tau_known=None)
    f_nls = nls_estimator.estimate(x, signal.fs)
    
    # DFT estimator
    dft_estimator = DFTFrequencyEstimator(window="kaiser", kaiser_beta=9.0)
    f_dft = dft_estimator.estimate(x, signal.fs)
    
    print(f"True frequency: {signal.f0:.6f} Hz")
    print(f"NLS estimate:    {f_nls:.6f} Hz (error: {abs(f_nls - signal.f0):.6e} Hz)")
    print(f"DFT estimate:    {f_dft:.6f} Hz (error: {abs(f_dft - signal.f0):.6e} Hz)")
    print()


def example_oop_crlb():
    """Example: Calculate CRLB using OOP API."""
    print("=" * 70)
    print("Example 3: CRLB Calculation (OOP API)")
    print("=" * 70)
    
    signal = RingDownSignal(f0=5.0, fs=100.0, N=1_000_000, A0=1.0, snr_db=60.0, Q=10000.0)
    
    crlb_calc = CRLBCalculator()
    crlb_var = crlb_calc.variance(signal.A0, signal.sigma, signal.fs, signal.N, signal.tau)
    crlb_std = crlb_calc.standard_deviation(signal.A0, signal.sigma, signal.fs, signal.N, signal.tau)
    
    print(f"Signal parameters:")
    print(f"  A0 = {signal.A0}")
    print(f"  sigma = {signal.sigma:.6f}")
    print(f"  fs = {signal.fs} Hz")
    print(f"  N = {signal.N}")
    print(f"  tau = {signal.tau:.2f} s")
    print(f"\nCRLB:")
    print(f"  Variance: {crlb_var:.6e} Hz²")
    print(f"  Std dev:  {crlb_std:.6e} Hz")
    print()


def example_oop_monte_carlo():
    """Example: Run Monte Carlo analysis using OOP API."""
    print("=" * 70)
    print("Example 4: Monte Carlo Analysis (OOP API)")
    print("=" * 70)
    
    analyzer = MonteCarloAnalyzer()
    
    results = analyzer.run(
        f0=5.0,
        fs=100.0,
        N=1_000_000,  # Matching LaTeX document: N = 10^6
        A0=1.0,
        snr_db=60.0,
        Q=10000.0,
        n_mc=50,  # Small number for quick example
        seed=42,
    )
    
    print(f"\nMonte Carlo Results:")
    print(f"  True frequency: {results['f0']:.6f} Hz")
    print(f"  NLS std: {results['stats']['nls']['std']:.6e} Hz")
    print(f"  DFT std: {results['stats']['dft']['std']:.6e} Hz")
    print(f"  CRLB std: {results['crlb_std']:.6e} Hz")
    print()


def example_compat_api():
    """Example: Use compatibility layer (function-based API)."""
    print("=" * 70)
    print("Example 5: Compatibility Layer (Function-based API)")
    print("=" * 70)
    
    # Generate signal using function API (matching LaTeX document parameters)
    t, x, sigma, phi0, tau = generate_ringdown(
        f0=5.0, fs=100.0, N=1_000_000, A0=1.0, snr_db=60.0, Q=10000.0,
        rng=np.random.default_rng(42)
    )
    
    # Estimate frequency using function API
    f_nls = estimate_freq_nls_ringdown(x, 100.0)
    f_dft = estimate_freq_dft(x, 100.0)
    
    print(f"Generated signal: {len(x)} samples")
    print(f"NLS estimate: {f_nls:.6f} Hz")
    print(f"DFT estimate: {f_dft:.6f} Hz")
    print()


def example_data_analysis():
    """Example: Analyze real data file (if available)."""
    print("=" * 70)
    print("Example 6: Real Data Analysis (OOP API)")
    print("=" * 70)
    
    # This would work if data files exist
    # analyzer = RingDownAnalyzer()
    # results = analyzer.analyze_file("data/some_file.csv")
    # print(f"Filename: {results['filename']}")
    # print(f"NLS frequency: {results['f_nls']:.6f} Hz")
    # print(f"DFT frequency: {results['f_dft']:.6f} Hz")
    
    print("(Skipped - no data files in example)")
    print()


def example_generate_latex_figures():
    """
    Generate figures for the LaTeX technical note.
    
    This function runs Monte Carlo analysis with parameters matching the LaTeX document
    and generates the three figures that are included in the document:
    - freq_estimation_ringdown_v6_individual.pdf
    - freq_estimation_ringdown_v6_aggregate.pdf
    - freq_estimation_ringdown_v6_performance.pdf
    
    Parameters match the LaTeX document:
    - f0 = 5.0 Hz
    - fs = 100.0 Hz
    - N = 10^6 samples (T = 10000 s)
    - initial SNR = 60 dB
    - Q = 10^4 (tau = 636.6 s)
    - n_mc = 100 trials
    """
    print("=" * 70)
    print("Example 7: Generate LaTeX Document Figures")
    print("=" * 70)
    
    # Parameters matching the LaTeX document (Section: Numerical analysis)
    f0 = 5.0          # Hz
    fs = 100.0         # Hz
    N = 1_000_000      # samples (T = 10000 s)
    A0 = 1.0           # Initial amplitude
    snr_db = 60.0      # Initial SNR (dB)
    Q = 10000.0        # Quality factor (tau = 636.6 s)
    n_mc = 100         # Monte Carlo trials
    
    print(f"Running Monte Carlo analysis with parameters matching LaTeX document:")
    print(f"  f0 = {f0} Hz")
    print(f"  fs = {fs} Hz")
    print(f"  N = {N} samples (T = {N/fs:.0f} s)")
    print(f"  initial SNR = {snr_db} dB")
    print(f"  Q = {Q:.0e} (tau = {Q/(np.pi*f0):.1f} s)")
    print(f"  n_mc = {n_mc} trials")
    print()
    
    # Run Monte Carlo analysis using OOP API
    analyzer = MonteCarloAnalyzer()
    results = analyzer.run(
        f0=f0,
        fs=fs,
        N=N,
        A0=A0,
        snr_db=snr_db,
        Q=Q,
        n_mc=n_mc,
        seed=42,
    )
    
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent.parent / "docs" / "tn"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print()
    print("=" * 70)
    print("Generating figures...")
    print("=" * 70)
    
    # Generate and save figures
    fig1 = plot_individual_results(results)
    fig1_path = output_dir / "freq_estimation_ringdown_v6_individual.pdf"
    fig1.savefig(fig1_path, bbox_inches="tight")
    print(f"  Saved: {fig1_path}")
    plt.close(fig1)
    
    fig2 = plot_aggregate_results(results)
    fig2_path = output_dir / "freq_estimation_ringdown_v6_aggregate.pdf"
    fig2.savefig(fig2_path, bbox_inches="tight")
    print(f"  Saved: {fig2_path}")
    plt.close(fig2)
    
    fig3 = plot_performance_comparison(results)
    fig3_path = output_dir / "freq_estimation_ringdown_v6_performance.pdf"
    fig3.savefig(fig3_path, bbox_inches="tight")
    print(f"  Saved: {fig3_path}")
    plt.close(fig3)
    
    print()
    print("All figures generated successfully!")
    print()


if __name__ == "__main__":
    # Run all examples
    example_oop_signal_generation()
    example_oop_frequency_estimation()
    example_oop_crlb()
    example_oop_monte_carlo()
    example_compat_api()
    example_data_analysis()
    
    # Generate LaTeX figures (can be run separately if needed)
    # Uncomment the line below to generate figures for the LaTeX document
    # example_generate_latex_figures()
    
    print("=" * 70)
    print("All examples completed!")
    print("=" * 70)
    print()
    print("Note: To generate figures for the LaTeX document, uncomment")
    print("      the call to example_generate_latex_figures() above.")

