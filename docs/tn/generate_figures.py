"""
Generate all figures for the technical note LaTeX document.

This script consolidates figure generation from:
- examples/usage_example.py (Monte Carlo analysis figures)
- examples/crlb_scaling_figures.py (CRLB scaling figures)

All figures are saved to the same directory as this script.
"""

import sys
from pathlib import Path

# Add project root to path so we can import ringdownanalysis
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Import from ringdownanalysis package
from ringdownanalysis import (
    CRLBCalculator,
    DFTFrequencyEstimator,
    MonteCarloAnalyzer,
    NLSFrequencyEstimator,
    ProfileQEstimator,
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


# CRLB scaling plotting functions (copied from examples/crlb_scaling_figures.py)
def plot_frequency_crlb_vs_tau_ratio(params=None, ax=None, figsize=None, dpi=None, *args, **kwargs):
    """
    Plot frequency CRLB as a function of T/tau ratio.

    Shows the transition from slow-decay (T << tau) to rapid-decay (T >> tau) regimes.
    """
    # Default parameters
    if params is None:
        params = {}

    f0 = params.get("f0", 5.0)  # Hz
    fs = params.get("fs", 100.0)  # Hz
    A0 = params.get("A0", 1.0)

    # Handle SNR or sigma
    if "sigma" in params:
        sigma = params["sigma"]
    elif "SNR" in params:
        snr_db = params["SNR"]
        sigma = np.sqrt(A0**2 / (2 * 10 ** (snr_db / 10)))
    else:
        snr_db = 60.0
        sigma = np.sqrt(A0**2 / (2 * 10 ** (snr_db / 10)))  # SNR = 60 dB

    # Handle tau or Q
    if "Q" in params:
        Q = params["Q"]
        tau = Q / (np.pi * f0)
    elif "tau" in params:
        tau = params["tau"]
    else:
        tau = 100.0  # s (fixed decay time)

    # Vary observation time T = N/fs
    tau_ratios = np.logspace(-1, 1.5, 50)  # T/tau from 0.1 to ~31.6
    T_values = tau_ratios * tau
    N_values = (T_values * fs).astype(int)
    N_values = np.maximum(N_values, 10)  # Minimum 10 samples
    N_values = np.minimum(N_values, 1_000_000)  # Maximum 1M samples

    # Calculate CRLB for each T/tau
    crlb_f = np.zeros_like(tau_ratios)
    for i, (N, T) in enumerate(zip(N_values, T_values)):
        try:
            crlb_f[i] = CRLBCalculator.standard_deviation(A0, sigma, fs, N, tau)
        except (ValueError, OverflowError):
            crlb_f[i] = np.nan

    # Asymptotic limits
    rho_0 = A0**2 / (2 * sigma**2)
    T_s = 1.0 / fs
    crlb_slow_decay = np.sqrt(12.0 / ((2 * np.pi) ** 2 * rho_0 * T_s**2 * N_values**3))
    crlb_rapid_decay_value = np.sqrt(8.0 * T_s / ((2 * np.pi) ** 2 * rho_0 * tau**3))
    crlb_rapid_decay = np.full_like(tau_ratios, crlb_rapid_decay_value)

    # Create figure or use provided axes
    created_fig = False
    if ax is None:
        if figsize is None:
            figsize = (5, 3)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        created_fig = True
    else:
        fig = ax.figure

    # Filter out NaN values for plotting
    valid = ~np.isnan(crlb_f)
    ax.loglog(
        tau_ratios[valid], crlb_f[valid], "b-", linewidth=2, label="Exact CRLB", *args, **kwargs
    )
    ax.loglog(
        tau_ratios,
        crlb_slow_decay,
        "r--",
        linewidth=1.5,
        alpha=0.7,
        label="Slow-decay approximation ($T \\ll \\tau$)",
    )
    ax.axhline(
        crlb_rapid_decay_value,
        color="lime",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Rapid-decay approximation ($T \\gg \\tau$)",
    )

    ax.set_xlabel("$T / \\tau$")
    ax.set_ylabel("$\\sigma_f$ (Hz)")
    ax.set_title("Frequency CRLB scaling with observation time")
    plots.apply_legend(ax)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([tau_ratios.min(), tau_ratios.max()])

    if created_fig:
        plt.tight_layout()
    return ax


def plot_frequency_crlb_vs_snr(params=None, ax=None, figsize=None, dpi=None, *args, **kwargs):
    """
    Plot frequency CRLB as a function of initial SNR.

    Shows the 1/sqrt(SNR) scaling relationship.
    """
    # Default parameters
    if params is None:
        params = {}

    f0 = params.get("f0", 5.0)  # Hz
    fs = params.get("fs", 100.0)  # Hz
    A0 = params.get("A0", 1.0)
    N = params.get("N", 100000)  # samples

    # Handle tau or Q
    if "Q" in params:
        Q = params["Q"]
        tau = Q / (np.pi * f0)
    elif "tau" in params:
        tau = params["tau"]
    else:
        tau = 100.0  # s

    # Vary SNR
    snr_db_values = np.linspace(20, 80, 50)
    snr_linear = 10 ** (snr_db_values / 10)

    # Calculate CRLB for each SNR
    crlb_f = np.zeros_like(snr_db_values)
    for i, snr_lin in enumerate(snr_linear):
        try:
            sigma = np.sqrt(A0**2 / (2 * snr_lin))
            crlb_f[i] = CRLBCalculator.standard_deviation(A0, sigma, fs, N, tau)
        except (ValueError, OverflowError):
            crlb_f[i] = np.nan

    # Theoretical scaling: sigma_f ~ 1/sqrt(SNR)
    ref_idx = min(25, len(snr_linear) - 1)
    ref_snr = snr_linear[ref_idx]
    ref_crlb = crlb_f[ref_idx]
    theoretical = ref_crlb * np.sqrt(ref_snr / snr_linear)

    # Create figure or use provided axes
    created_fig = False
    if ax is None:
        if figsize is None:
            figsize = (5, 3)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        created_fig = True
    else:
        fig = ax.figure

    # Filter out NaN values for plotting
    valid = ~np.isnan(crlb_f)
    ax.semilogy(
        snr_db_values[valid], crlb_f[valid], "b-", linewidth=2, label="Exact CRLB", *args, **kwargs
    )
    ax.semilogy(
        snr_db_values,
        theoretical,
        "r--",
        linewidth=1.5,
        alpha=0.7,
        label="$\\propto 1/\\sqrt{\\rho_0}$",
    )

    ax.set_xlabel("Initial SNR (dB)")
    ax.set_ylabel("$\\sigma_f$ (Hz)")
    ax.set_title("Frequency CRLB scaling with initial SNR")
    plots.apply_legend(ax)
    ax.grid(True, alpha=0.3)

    if created_fig:
        plt.tight_layout()
    return ax


def plot_q_crlb_vs_q(params=None, ax=None, figsize=None, dpi=None, *args, **kwargs):
    """
    Plot Q-factor CRLB (relative error) as a function of Q.

    Shows the transition from low-Q to high-Q regimes.
    """
    # Default parameters
    if params is None:
        params = {}

    f0 = params.get("f0", 5.0)  # Hz
    fs = params.get("fs", 100.0)  # Hz
    A0 = params.get("A0", 1.0)
    N = params.get("N", 100000)  # samples

    # Handle SNR or sigma
    if "sigma" in params:
        sigma = params["sigma"]
    elif "SNR" in params:
        snr_db = params["SNR"]
        sigma = np.sqrt(A0**2 / (2 * 10 ** (snr_db / 10)))
    else:
        snr_db = 60.0
        sigma = np.sqrt(A0**2 / (2 * 10 ** (snr_db / 10)))  # SNR = 60 dB

    T = N / fs

    # Vary Q
    Q_values = np.logspace(1, 5, 50)  # Q from 10 to 100000
    tau_values = Q_values / (np.pi * f0)

    # Calculate CRLB for each Q
    crlb_q = np.zeros_like(Q_values)
    for i, (Q, tau) in enumerate(zip(Q_values, tau_values)):
        try:
            crlb_q[i] = CRLBCalculator.q_standard_deviation(A0, sigma, fs, N, tau, f0)
        except (ValueError, OverflowError):
            crlb_q[i] = np.nan

    # Relative error
    rel_error = crlb_q / Q_values

    # High-Q scaling is only valid when T >> tau (rapid-decay regime)
    T_tau_ratios = T / tau_values
    rapid_decay_mask = T_tau_ratios > 3.0

    # Only compute high-Q scaling where valid (T >> tau)
    high_q_scaling = np.full_like(Q_values, np.nan)
    if np.any(rapid_decay_mask):
        ref_indices = np.where(rapid_decay_mask)[0]
        if len(ref_indices) > 0:
            ref_idx = ref_indices[len(ref_indices) // 2]
            ref_Q = Q_values[ref_idx]
            ref_rel = rel_error[ref_idx]
            high_q_scaling[rapid_decay_mask] = ref_rel * np.sqrt(ref_Q / Q_values[rapid_decay_mask])

    # Create figure or use provided axes
    created_fig = False
    if ax is None:
        if figsize is None:
            figsize = (5, 3)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        created_fig = True
    else:
        fig = ax.figure

    # Filter out NaN values for plotting
    valid = ~np.isnan(rel_error)
    ax.loglog(
        Q_values[valid], rel_error[valid], "b-", linewidth=2, label="Exact CRLB", *args, **kwargs
    )

    # Plot high-Q scaling only where valid
    high_q_valid = ~np.isnan(high_q_scaling)
    if np.any(high_q_valid):
        ax.loglog(
            Q_values[high_q_valid],
            high_q_scaling[high_q_valid],
            "r--",
            linewidth=1.5,
            alpha=0.7,
            label="High-$Q$ scaling ($\\propto 1/\\sqrt{Q}$, $T \\gg \\tau$)",
        )

    ax.set_xlabel("Quality factor $Q$")
    ax.set_ylabel("$\\sigma_Q / Q$ (relative error)")
    ax.set_title("Q-factor CRLB relative error scaling with $Q$")
    plots.apply_legend(ax)
    ax.grid(True, alpha=0.3)

    if created_fig:
        plt.tight_layout()
    return ax


def plot_q_crlb_vs_tau_ratio(params=None, ax=None, figsize=None, dpi=None, *args, **kwargs):
    """
    Plot Q-factor CRLB (relative error) as a function of T/tau ratio.

    Shows how observation time affects Q estimation accuracy.
    """
    # Default parameters
    if params is None:
        params = {}

    f0 = params.get("f0", 5.0)  # Hz
    fs = params.get("fs", 100.0)  # Hz
    A0 = params.get("A0", 1.0)

    # Handle SNR or sigma
    if "sigma" in params:
        sigma = params["sigma"]
    elif "SNR" in params:
        snr_db = params["SNR"]
        sigma = np.sqrt(A0**2 / (2 * 10 ** (snr_db / 10)))
    else:
        snr_db = 60.0
        sigma = np.sqrt(A0**2 / (2 * 10 ** (snr_db / 10)))  # SNR = 60 dB

    # Handle tau or Q
    if "tau" in params:
        tau = params["tau"]
        Q = tau * np.pi * f0
    elif "Q" in params:
        Q = params["Q"]
        tau = Q / (np.pi * f0)  # Fixed tau
    else:
        Q = 10_000.0  # Fixed Q
        tau = Q / (np.pi * f0)  # Fixed tau

    # Vary observation time T = N/fs
    tau_ratios = np.logspace(-1, 1.5, 50)  # T/tau from 0.1 to ~31.6
    T_values = tau_ratios * tau
    N_values = (T_values * fs).astype(int)
    N_values = np.maximum(N_values, 10)  # Minimum 10 samples
    N_values = np.minimum(N_values, 1_000_000)  # Maximum 1M samples

    # Calculate CRLB for each T/tau
    crlb_q = np.zeros_like(tau_ratios)
    for i, N in enumerate(N_values):
        try:
            crlb_q[i] = CRLBCalculator.q_standard_deviation(A0, sigma, fs, N, tau, f0)
        except (ValueError, OverflowError):
            crlb_q[i] = np.nan

    # Relative error
    rel_error = crlb_q / Q

    # Create figure or use provided axes
    created_fig = False
    if ax is None:
        if figsize is None:
            figsize = (5, 3)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        created_fig = True
    else:
        fig = ax.figure

    # Filter out NaN values for plotting
    valid = ~np.isnan(rel_error)
    ax.loglog(
        tau_ratios[valid], rel_error[valid], "b-", linewidth=2, label="Exact CRLB", *args, **kwargs
    )

    ax.set_xlabel("$T / \\tau$")
    ax.set_ylabel("$\\sigma_Q / Q$ (relative error)")
    ax.set_title("Q-factor CRLB relative error scaling with observation time")
    plots.apply_legend(ax)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([tau_ratios.min(), tau_ratios.max()])

    # Add vertical line at T/tau = 1
    ax.axvline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.text(
        1.0,
        rel_error.min() * 1.5,
        "$T = \\tau$",
        rotation=90,
        verticalalignment="bottom",
        alpha=0.7,
    )

    if created_fig:
        plt.tight_layout()
    return ax


def run_q_estimator_comparison(
    *,
    f0,
    fs,
    n_samples,
    A0,
    snr_db,
    Q,
    n_trials,
    seed,
):
    """Run repeated example-parameter records through the three Q estimators."""
    signal_template = RingDownSignal(f0=f0, fs=fs, N=n_samples, A0=A0, snr_db=snr_db, Q=Q)
    tau = signal_template.tau
    sigma = signal_template.sigma
    true_q = signal_template.Q
    nls_estimator = NLSFrequencyEstimator(tau_known=None)
    dft_estimator = DFTFrequencyEstimator(window="rect", use_zeropad=False)
    profile_estimator = ProfileQEstimator(n_grid=81)
    rng = np.random.default_rng(seed)
    results = {
        "true_q": true_q,
        "tau": tau,
        "duration": n_samples / fs,
        "sigma": sigma,
        "errors": {
            "NLS": [],
            "DFT+NLS": [],
            "Profile": [],
        },
        "finite_values": {
            "NLS": [],
            "DFT+NLS": [],
            "Profile": [],
        },
        "profile_statuses": [],
        "profile_ci_coverage": [],
        "profile_ci_widths": [],
        "profile_lower_limits": [],
        "analysis_failures": 0,
        "n_trials": n_trials,
    }

    for trial_idx in range(n_trials):
        phase = rng.uniform(-np.pi, np.pi)
        trial_rng = np.random.default_rng(int(seed + 10_000 + trial_idx))
        signal = RingDownSignal(f0=f0, fs=fs, N=n_samples, A0=A0, snr_db=snr_db, Q=Q)
        t, x, _ = signal.generate(phi0=phase, rng=trial_rng)

        try:
            nls_result = nls_estimator.estimate_full(
                x,
                fs,
                tau_init=tau,
                max_nfev=150,
            )
            if nls_result.Q is not None and np.isfinite(nls_result.Q):
                q_value = float(nls_result.Q)
                results["finite_values"]["NLS"].append(q_value)
                results["errors"]["NLS"].append(q_value - true_q)
        except Exception as exc:
            results["analysis_failures"] += 1
            results["profile_statuses"].append(f"nls_failed:{type(exc).__name__}")
            continue

        try:
            dft_result = dft_estimator.estimate_full(
                x,
                fs,
                tau_init=tau,
                max_nfev=300,
            )
            if dft_result.Q is not None and np.isfinite(dft_result.Q):
                q_value = float(dft_result.Q)
                results["finite_values"]["DFT+NLS"].append(q_value)
                results["errors"]["DFT+NLS"].append(q_value - true_q)
        except Exception:
            dft_result = None

        f_profile = nls_result.f
        if (not np.isfinite(f_profile) or f_profile <= 0) and dft_result is not None:
            f_profile = dft_result.f
        try:
            profile_result = profile_estimator.estimate(
                t,
                x,
                fs,
                f_init=f_profile,
                tau_init=tau,
                tau_bounds=(tau / 2.0, tau * 2.0),
                n_grid=81,
            )
        except Exception as exc:
            results["profile_statuses"].append(f"profile_failed:{type(exc).__name__}")
            continue

        profile_status = profile_result.status
        results["profile_statuses"].append(profile_status)
        q_profile = profile_result.Q
        if profile_result.valid and q_profile is not None and np.isfinite(q_profile):
            q_profile = float(q_profile)
            results["finite_values"]["Profile"].append(q_profile)
            results["errors"]["Profile"].append(q_profile - true_q)
            ci95 = profile_result.ci95
            if ci95 is not None:
                ci_low, ci_high = ci95
                results["profile_ci_coverage"].append(ci_low <= true_q <= ci_high)
                results["profile_ci_widths"].append(float(ci_high - ci_low))
        lower_limit = profile_result.lower_limit_95
        if lower_limit is not None and np.isfinite(lower_limit):
            results["profile_lower_limits"].append(float(lower_limit))

    for key in ("errors", "finite_values"):
        for method, values in results[key].items():
            results[key][method] = np.array(values, dtype=float)
    results["profile_lower_limits"] = np.array(results["profile_lower_limits"], dtype=float)
    results["profile_ci_widths"] = np.array(results["profile_ci_widths"], dtype=float)
    results["profile_ci_coverage"] = np.array(results["profile_ci_coverage"], dtype=bool)
    return results


def q_estimator_stats(comparison, crlb_std_q):
    """Summarize estimator errors for the three-way Q comparison."""
    rows = []
    for method, errors in comparison["errors"].items():
        if len(errors) == 0:
            rows.append((method, np.nan, np.nan, np.nan, np.nan, 0))
            continue
        std = np.std(errors, ddof=1) if len(errors) > 1 else np.nan
        bias = np.mean(errors)
        rmse = np.sqrt(np.mean(errors**2))
        efficiency = crlb_std_q / std if np.isfinite(std) and std > 0 else np.nan
        rows.append((method, std, bias, rmse, efficiency, len(errors)))
    return rows


def plot_q_estimator_three_way_comparison(comparison, crlb_std_q, ax=None, figsize=None, dpi=None):
    """Plot example-parameter diagnostics for the three Q estimators."""
    if ax is None:
        if figsize is None:
            figsize = (6.5, 5.2)
        fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    else:
        if not isinstance(ax, np.ndarray) or ax.size != 4:
            raise ValueError("ax must be an array of 4 axes for this plot")
        axes = ax.reshape(2, 2)

    colors = {
        "NLS": "tab:blue",
        "DFT+NLS": "tab:orange",
        "Profile": "tab:green",
    }
    methods = ["NLS", "DFT+NLS", "Profile"]

    ax_hist = axes[0, 0]
    for method in methods:
        errors = comparison["errors"][method]
        if len(errors) == 0:
            continue
        ax_hist.hist(
            errors,
            bins=20,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=colors[method],
            label=f"{method} (n={len(errors)})",
        )
    if np.isfinite(crlb_std_q):
        ax_hist.axvline(crlb_std_q, color="gray", linestyle="--", linewidth=1.2, label="Q-CRLB")
        ax_hist.axvline(-crlb_std_q, color="gray", linestyle="--", linewidth=1.2)
    ax_hist.axvline(0.0, color="black", linestyle=":", linewidth=1.0)
    ax_hist.set_xlabel("$Q$ error")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Error distribution")
    plots.apply_legend(ax_hist)
    ax_hist.grid(True, alpha=0.3)

    ax_std = axes[0, 1]
    stats = q_estimator_stats(comparison, crlb_std_q)
    x_pos = np.arange(len(methods))
    stds = [row[1] for row in stats]
    labels = [f"{row[0]}\n(n={row[5]})" for row in stats]
    ax_std.bar(
        x_pos,
        stds,
        color=[colors[method] for method in methods],
        alpha=0.75,
        edgecolor="black",
    )
    if np.isfinite(crlb_std_q):
        ax_std.axhline(crlb_std_q, color="gray", linestyle="--", linewidth=1.5, label="Q-CRLB")
    ax_std.set_xticks(x_pos)
    ax_std.set_xticklabels(labels)
    ax_std.set_ylabel("Std. dev. of $Q$ error")
    ax_std.set_title("Precision")
    ax_std.set_yscale("log")
    plots.apply_legend(ax_std)
    ax_std.grid(True, alpha=0.3, axis="y")

    ax_bias = axes[1, 0]
    biases = [row[2] for row in stats]
    rmses = [row[3] for row in stats]
    width = 0.34
    ax_bias.bar(
        x_pos - width / 2,
        np.abs(biases),
        width,
        color="lightgray",
        edgecolor="black",
        label="|bias|",
    )
    ax_bias.bar(
        x_pos + width / 2,
        rmses,
        width,
        color=[colors[method] for method in methods],
        alpha=0.75,
        edgecolor="black",
        label="RMSE",
    )
    ax_bias.set_xticks(x_pos)
    ax_bias.set_xticklabels(methods)
    ax_bias.set_ylabel("$Q$ error magnitude")
    ax_bias.set_title("Bias and RMSE")
    ax_bias.set_yscale("log")
    plots.apply_legend(ax_bias)
    ax_bias.grid(True, alpha=0.3, axis="y")

    ax_status = axes[1, 1]
    finite_rates = [
        len(comparison["finite_values"][method]) / comparison["n_trials"] for method in methods
    ]
    ax_status.bar(
        x_pos,
        finite_rates,
        color=[colors[method] for method in methods],
        alpha=0.75,
        edgecolor="black",
    )
    ax_status.set_xticks(x_pos)
    ax_status.set_xticklabels(methods)
    ax_status.set_ylim(0.0, 1.05)
    ax_status.set_ylabel("Finite estimate rate")
    coverage = comparison["profile_ci_coverage"]
    widths = comparison["profile_ci_widths"]
    if len(coverage) > 0:
        coverage_pct = 100.0 * float(np.mean(coverage))
        median_width = float(np.median(widths)) if len(widths) > 0 else np.nan
        ax_status.text(
            0.5,
            0.08,
            f"Profile 95% CI coverage: {coverage_pct:.0f}%\nmedian CI width: {median_width:.3g}",
            ha="center",
            va="bottom",
            transform=ax_status.transAxes,
        )
    ax_status.set_title("Profile interval diagnostics")
    ax_status.grid(True, alpha=0.3, axis="y")

    for axis in axes.ravel():
        axis.tick_params(axis="x", labelrotation=0)

    if ax is None:
        fig.tight_layout()
        return axes
    return axes


def generate_q_estimator_comparison_figures(output_dir):
    """
    Compare raw NLS, raw DFT+NLS, and profile-likelihood Q estimators.

    The comparison uses the same parameter set as the main technical-note
    example, including Q = 10000.
    """
    print()
    print("=" * 70)
    print("Generating Three-Way Q Estimator Comparison")
    print("=" * 70)

    plots.apply_plotting_style()

    f0 = 5.0
    fs = 100.0
    n_samples = 1_000_000
    A0 = 1.0
    snr_db = 60.0
    Q = 10_000.0
    n_trials = 20

    comparison = run_q_estimator_comparison(
        f0=f0,
        fs=fs,
        n_samples=n_samples,
        A0=A0,
        snr_db=snr_db,
        Q=Q,
        n_trials=n_trials,
        seed=20260514,
    )

    crlb_var_q = CRLBCalculator.q_variance(
        A0,
        comparison["sigma"],
        fs,
        n_samples,
        comparison["tau"],
        f0,
    )
    crlb_std_q = float(np.sqrt(crlb_var_q)) if np.isfinite(crlb_var_q) else np.inf

    print(
        "  Example parameters: "
        f"f0={f0:g} Hz, fs={fs:g} Hz, N={n_samples}, SNR={snr_db:g} dB, "
        f"Q={Q:.0f}, tau={comparison['tau']:.1f} s, n_trials={n_trials}"
    )
    for method, std, bias, rmse, efficiency, n_used in q_estimator_stats(comparison, crlb_std_q):
        print(
            f"    {method}: n={n_used}, std={std:.3e}, bias={bias:.3e}, "
            f"rmse={rmse:.3e}, efficiency={efficiency:.3f}"
        )
    profile_valid = comparison["profile_statuses"].count("valid")
    print(f"  Profile valid count: {profile_valid}/{n_trials}")

    axes = plot_q_estimator_three_way_comparison(comparison, crlb_std_q)
    fig = axes[0, 0].figure
    fig_path = output_dir / "q_estimator_three_way_comparison.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"  Saved: {fig_path}")
    plt.close(fig)


def generate_monte_carlo_figures(output_dir):
    """
    Generate Monte Carlo analysis figures for frequency and Q estimation.

    Parameters:
    -----------
    output_dir : Path
        Directory to save figures to.
    """
    print("=" * 70)
    print("Generating Monte Carlo Analysis Figures")
    print("=" * 70)

    # Parameters matching the LaTeX document (Section: Numerical analysis)
    f0 = 5.0  # Hz
    fs = 100.0  # Hz
    N = 1_000_000  # samples (T = 10000 s)
    A0 = 1.0  # Initial amplitude
    snr_db = 60.0  # Initial SNR (dB)
    Q = 10_000.0  # Quality factor (tau = 636.6 s)
    n_mc = 100  # Monte Carlo trials

    print("Running Monte Carlo analysis with parameters matching LaTeX document:")
    print(f"  f0 = {f0} Hz")
    print(f"  fs = {fs} Hz")
    print(f"  N = {N} samples (T = {N / fs:.0f} s)")
    print(f"  initial SNR = {snr_db} dB")
    print(f"  Q = {Q:.0e} (tau = {Q / (np.pi * f0):.1f} s)")
    print(f"  n_mc = {n_mc} trials")
    print()

    # Apply plotting style
    plots.apply_plotting_style()

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

    print()
    print("Generating frequency estimation figures...")

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
    print()
    print("Generating Q estimation figures...")
    has_q_nls = "errors_q_nls" in results and len(results["errors_q_nls"]) > 0
    has_q_dft = "errors_q_dft" in results and len(results["errors_q_dft"]) > 0
    if has_q_nls or has_q_dft:
        axes4 = plot_q_individual_results(results)
        fig4 = axes4[0].figure if isinstance(axes4, np.ndarray) else axes4.figure
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


def generate_crlb_scaling_figures(output_dir):
    """
    Generate CRLB scaling figures.

    Parameters:
    -----------
    output_dir : Path
        Directory to save figures to.
    """
    print()
    print("=" * 70)
    print("Generating CRLB Scaling Figures")
    print("=" * 70)

    # Apply plotting style
    plots.apply_plotting_style()

    # Figure 1: Frequency CRLB vs T/tau
    print("  Generating frequency CRLB vs T/tau...")
    ax1 = plot_frequency_crlb_vs_tau_ratio()
    fig1 = ax1.figure
    fig1_path = output_dir / "crlb_freq_vs_tau_ratio.pdf"
    fig1.savefig(fig1_path, bbox_inches="tight")
    print(f"    Saved: {fig1_path}")
    plt.close(fig1)

    # Figure 2: Frequency CRLB vs SNR
    print("  Generating frequency CRLB vs SNR...")
    ax2 = plot_frequency_crlb_vs_snr()
    fig2 = ax2.figure
    fig2_path = output_dir / "crlb_freq_vs_snr.pdf"
    fig2.savefig(fig2_path, bbox_inches="tight")
    print(f"    Saved: {fig2_path}")
    plt.close(fig2)

    # Figure 3: Q CRLB vs Q
    print("  Generating Q CRLB vs Q...")
    ax3 = plot_q_crlb_vs_q()
    fig3 = ax3.figure
    fig3_path = output_dir / "crlb_q_vs_q.pdf"
    fig3.savefig(fig3_path, bbox_inches="tight")
    print(f"    Saved: {fig3_path}")
    plt.close(fig3)

    # Figure 4: Q CRLB vs T/tau
    print("  Generating Q CRLB vs T/tau...")
    ax4 = plot_q_crlb_vs_tau_ratio()
    fig4 = ax4.figure
    fig4_path = output_dir / "crlb_q_vs_tau_ratio.pdf"
    fig4.savefig(fig4_path, bbox_inches="tight")
    print(f"    Saved: {fig4_path}")
    plt.close(fig4)


def main():
    """Generate all figures for the technical note."""
    # Output directory is the same as this script
    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating all figures for technical note...")
    print(f"Output directory: {output_dir}")
    print()

    # Generate Monte Carlo figures
    generate_monte_carlo_figures(output_dir)

    # Generate three-way Q estimator comparison figures
    generate_q_estimator_comparison_figures(output_dir)

    # Generate CRLB scaling figures
    generate_crlb_scaling_figures(output_dir)

    print()
    print("=" * 70)
    print("All figures generated successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
