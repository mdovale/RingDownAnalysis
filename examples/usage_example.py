"""
Usage examples for the ring-down analysis package.

This demonstrates both the object-oriented API and the compatibility layer.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Import plotting functions and style (style applies automatically on import)
from ringdownanalysis import (
    CRLBCalculator,
    DFTFrequencyEstimator,
    MonteCarloAnalyzer,
    NLSFrequencyEstimator,
    RingDownSignal,
    plots,
)
from ringdownanalysis.plots import (
    plot_aggregate_results,
    plot_individual_results,
    plot_performance_comparison,
    plot_q_individual_results,
    plot_q_performance_comparison,
)


def example_signal_generation():
    """Example: Generate ring-down signal."""
    print("=" * 70)
    print("Example 1: Signal Generation")
    print("=" * 70)

    # Create signal with specified parameters (matching LaTeX document)
    signal = RingDownSignal(
        f0=5.0,  # Frequency (Hz)
        fs=100.0,  # Sampling frequency (Hz)
        N=1_000_000,  # Number of samples (matching LaTeX: N = 10^6)
        A0=1.0,  # Initial amplitude
        snr_db=60.0,  # Initial SNR (dB)
        Q=10000.0,  # Quality factor
    )

    print("Signal parameters:")
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

    print("\nGenerated signal:")
    print(f"  Length: {len(x)} samples")
    print(f"  Initial phase: {phi0:.4f} rad")
    print(f"  Signal mean: {np.mean(x):.6f}")
    print(f"  Signal std: {np.std(x):.6f}")
    print()


def example_frequency_estimation():
    """Example: Estimate frequency."""
    print("=" * 70)
    print("Example 2: Frequency Estimation")
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


def example_crlb():
    """Example: Calculate CRLB."""
    print("=" * 70)
    print("Example 3: CRLB Calculation")
    print("=" * 70)

    signal = RingDownSignal(f0=5.0, fs=100.0, N=1_000_000, A0=1.0, snr_db=60.0, Q=10000.0)

    crlb_calc = CRLBCalculator()
    crlb_var = crlb_calc.variance(signal.A0, signal.sigma, signal.fs, signal.N, signal.tau)
    crlb_std = crlb_calc.standard_deviation(
        signal.A0, signal.sigma, signal.fs, signal.N, signal.tau
    )

    crlb_var_q = crlb_calc.q_variance(
        signal.A0, signal.sigma, signal.fs, signal.N, signal.tau, signal.f0
    )
    crlb_std_q = crlb_calc.q_standard_deviation(
        signal.A0, signal.sigma, signal.fs, signal.N, signal.tau, signal.f0
    )

    print("Signal parameters:")
    print(f"  A0 = {signal.A0}")
    print(f"  sigma = {signal.sigma:.6f}")
    print(f"  fs = {signal.fs} Hz")
    print(f"  N = {signal.N}")
    print(f"  tau = {signal.tau:.2f} s")
    print(f"  f0 = {signal.f0} Hz")
    print(f"  Q = {signal.Q:.1e}")
    print("\nFrequency CRLB:")
    print(f"  Variance: {crlb_var:.6e} Hz²")
    print(f"  Std dev:  {crlb_std:.6e} Hz")
    print("\nQ CRLB:")
    print(f"  Variance: {crlb_var_q:.6e}")
    print(f"  Std dev:  {crlb_std_q:.6e}")
    print()


def example_monte_carlo():
    """Example: Run Monte Carlo analysis."""
    print("=" * 70)
    print("Example 4: Monte Carlo Analysis")
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

    print("\nMonte Carlo Results:")
    print(f"  True frequency: {results['f0']:.6f} Hz")
    print(f"  True Q: {results['Q']:.1e}")
    print(f"  NLS std: {results['stats']['nls']['std']:.6e} Hz")
    print(f"  DFT std: {results['stats']['dft']['std']:.6e} Hz")
    print(f"  CRLB std: {results['crlb_std']:.6e} Hz")
    if "errors_q_nls" in results and len(results["errors_q_nls"]) > 0:
        print(f"  NLS Q std: {results['stats']['q_nls']['std']:.6e}")
        print(f"  CRLB Q std: {results['crlb_std_q']:.6e}")
    print()


def example_data_analysis():
    """Example: Analyze real data file (if available)."""
    print("=" * 70)
    print("Example 6: Real Data Analysis")
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
    # Apply consistent plotting style
    plots.apply_plotting_style()

    print("=" * 70)
    print("Example 7: Generate LaTeX Document Figures")
    print("=" * 70)

    # Parameters matching the LaTeX document (Section: Numerical analysis)
    f0 = 5.0  # Hz
    fs = 100.0  # Hz
    N = 1_000_000  # samples (T = 10000 s)
    A0 = 1.0  # Initial amplitude
    snr_db = 60.0  # Initial SNR (dB)
    Q = 10000.0  # Quality factor (tau = 636.6 s)
    n_mc = 100  # Monte Carlo trials

    print("Running Monte Carlo analysis with parameters matching LaTeX document:")
    print(f"  f0 = {f0} Hz")
    print(f"  fs = {fs} Hz")
    print(f"  N = {N} samples (T = {N / fs:.0f} s)")
    print(f"  initial SNR = {snr_db} dB")
    print(f"  Q = {Q:.0e} (tau = {Q / (np.pi * f0):.1f} s)")
    print(f"  n_mc = {n_mc} trials")
    print()

    # Run Monte Carlo analysis
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

    # Generate and save frequency estimation figures
    axes1 = plot_individual_results(results)
    fig1 = axes1[0].figure if isinstance(axes1, np.ndarray) else axes1.figure
    fig1_path = output_dir / "freq_estimation_ringdown_v6_individual.pdf"
    fig1.savefig(fig1_path, bbox_inches="tight")
    print(f"  Saved: {fig1_path}")
    plt.close(fig1)

    axes2 = plot_aggregate_results(results)
    fig2 = axes2[0].figure if isinstance(axes2, np.ndarray) else axes2.figure
    fig2_path = output_dir / "freq_estimation_ringdown_v6_aggregate.pdf"
    fig2.savefig(fig2_path, bbox_inches="tight")
    print(f"  Saved: {fig2_path}")
    plt.close(fig2)

    axes3 = plot_performance_comparison(results)
    fig3 = axes3[0].figure if isinstance(axes3, np.ndarray) else axes3.figure
    fig3_path = output_dir / "freq_estimation_ringdown_v6_performance.pdf"
    fig3.savefig(fig3_path, bbox_inches="tight")
    print(f"  Saved: {fig3_path}")
    plt.close(fig3)

    # Generate and save Q estimation figures
    if "errors_q_nls" in results and len(results["errors_q_nls"]) > 0:
        axes4 = plot_q_individual_results(results)
        fig4 = axes4.figure  # Single axis
        fig4_path = output_dir / "q_estimation_ringdown_v6_individual.pdf"
        fig4.savefig(fig4_path, bbox_inches="tight")
        print(f"  Saved: {fig4_path}")
        plt.close(fig4)

        axes5 = plot_q_performance_comparison(results)
        fig5 = axes5[0].figure if isinstance(axes5, np.ndarray) else axes5.figure
        fig5_path = output_dir / "q_estimation_ringdown_v6_performance.pdf"
        fig5.savefig(fig5_path, bbox_inches="tight")
        print(f"  Saved: {fig5_path}")
        plt.close(fig5)
    else:
        print("  Warning: No Q estimation data available, skipping Q figures")

    print()
    print("All figures generated successfully!")
    print()


if __name__ == "__main__":
    # Run all examples
    # example_signal_generation()
    # example_frequency_estimation()
    # example_crlb()
    # example_monte_carlo()
    # example_data_analysis()

    # Generate LaTeX figures (can be run separately if needed)
    # Uncomment the line below to generate figures for the LaTeX document
    example_generate_latex_figures()

    print("=" * 70)
    print("All examples completed!")
    print("=" * 70)
    print()
    print("Note: To generate figures for the LaTeX document, uncomment")
    print("      the call to example_generate_latex_figures() above.")
