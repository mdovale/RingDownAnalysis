"""
Plotting utilities and style configuration for ring-down analysis.

This module provides consistent matplotlib styling across all plots in the codebase,
as well as plotting functions for visualizing Monte Carlo analysis results.
"""

import matplotlib.pyplot as plt
import numpy as np


def get_default_rc():
    """
    Get the default matplotlib rcParams for consistent plotting style.

    Returns:
        dict: Dictionary of rcParams to apply for consistent plotting style.
    """
    return {
        "figure.dpi": 150,
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.prop_cycle": plt.cycler(
            "color",
            [
                "#000000",
                "#DC143C",
                "#00BFFF",
                "#FFD700",
                "#32CD32",
                "#FF69B4",
                "#FF4500",
                "#1E90FF",
                "#8A2BE2",
                "#FFA07A",
                "#8B0000",
            ],
        ),
    }


# Legend styling parameters
legend_params = {
    "loc": "best",
    "fontsize": 8,
    "frameon": True,
}


def apply_legend(ax):
    """Apply a consistent legend style to a Matplotlib Axes object.

    Parameters
    ----------
    ax : Axes | list[Axes]
        The Axes object(s) to which the legend will be applied.
        If a list is provided, the legend is applied to the last Axes.

    Returns
    -------
    Legend | None
        The created Legend object, or None if no legend could be created
        (e.g., no labels exist on the axes).

    Notes
    -----
    The legend style is defined by the module-level `legend_params` dictionary.
    """
    if isinstance(ax, list):
        ax = ax[-1]

    legend = ax.legend(**legend_params)
    if legend is not None:  # Check if a legend was actually created
        frame = legend.get_frame()
        frame.set_alpha(1.0)
        frame.set_edgecolor("black")
        frame.set_linewidth(0.7)
        try:
            frame.set_boxstyle("Square")
        except AttributeError:
            # Safe fallback for older matplotlib versions
            pass
    return legend


def apply_plotting_style():
    """
    Apply the default plotting style to matplotlib rcParams.

    This function updates the global matplotlib rcParams with the default
    style configuration. It can be called explicitly, but is also called
    automatically when this module is imported.
    """
    default_rc = get_default_rc()
    plt.rcParams.update(default_rc)


# Apply default styles automatically when module is imported
apply_plotting_style()
default_rc = get_default_rc()

# ============================================================================
# Monte Carlo analysis plotting functions
# ============================================================================


def plot_individual_results(results: dict, ax=None, figsize=None, dpi=None, *args, **kwargs):
    """Plot error distributions for each method separately."""
    errors_nls = results["errors_nls"]
    errors_dft = results["errors_dft"]
    crlb_std = results["crlb_std"]

    if ax is None:
        if figsize is None:
            figsize = (6.5, 4.66)
        fig, axes = plt.subplots(2, 1, figsize=figsize, dpi=dpi)
    else:
        # If ax is provided, it should be an array of axes
        if not isinstance(ax, np.ndarray) or ax.size != 2:
            raise ValueError("ax must be an array of 2 axes for this plot")
        axes = ax
        fig = axes[0].figure

    # NLS
    ax_plot = axes[0]
    ax_plot.hist(
        errors_nls,
        bins=30,
        density=True,
        color="blue",
        alpha=0.6,
        edgecolor="black",
        *args,
        **kwargs,
    )
    ax_plot.axvline(0, color="tomato", linestyle="--", label="Zero error")
    ax_plot.axvline(crlb_std, color="lime", linestyle="--", label=f"CRLB std = {crlb_std:.2e}")
    ax_plot.axvline(-crlb_std, color="lime", linestyle="--")
    ax_plot.set_xlabel("Frequency error (Hz)")
    ax_plot.set_ylabel("Probability density")
    ax_plot.set_title(
        f"Nonlinear Least Squares (NLS)\nstd = {results['stats']['nls']['std']:.6e} Hz"
    )
    apply_legend(ax_plot)
    ax_plot.grid(True, alpha=0.3)

    # DFT
    ax_plot = axes[1]
    ax_plot.hist(
        errors_dft,
        bins=30,
        density=True,
        color="blue",
        alpha=0.6,
        edgecolor="black",
        *args,
        **kwargs,
    )
    ax_plot.axvline(0, color="tomato", linestyle="--", label="Zero error")
    ax_plot.axvline(crlb_std, color="lime", linestyle="--", label=f"CRLB std = {crlb_std:.2e}")
    ax_plot.axvline(-crlb_std, color="lime", linestyle="--")
    ax_plot.set_xlabel("Frequency error (Hz)")
    ax_plot.set_ylabel("Probability density")
    ax_plot.set_title(f"DFT std = {results['stats']['dft']['std']:.6e} Hz")
    apply_legend(ax_plot)
    ax_plot.grid(True, alpha=0.3)

    if ax is None:
        plt.tight_layout()
    return axes


def plot_aggregate_results(results: dict, ax=None, figsize=None, dpi=None, *args, **kwargs):
    """Plot comparison of all methods together."""
    errors_nls = results["errors_nls"]
    errors_dft = results["errors_dft"]
    crlb_std = results["crlb_std"]

    if ax is None:
        if figsize is None:
            figsize = (6.5, 4.66)
        fig, axes = plt.subplots(
            2, 1, figsize=figsize, dpi=dpi, gridspec_kw={"height_ratios": [1, 0.4]}
        )
    else:
        # If ax is provided, it should be an array of axes
        if not isinstance(ax, np.ndarray) or ax.size != 2:
            raise ValueError("ax must be an array of 2 axes for this plot")
        axes = ax
        fig = axes[0].figure

    # Histogram comparison
    ax_plot = axes[0]
    ax_plot.hist(
        errors_nls,
        bins=30,
        density=True,
        color="blue",
        alpha=0.6,
        label=f"NLS (std={results['stats']['nls']['std']:.2e})",
        edgecolor="black",
        *args,
        **kwargs,
    )
    ax_plot.hist(
        errors_dft,
        bins=30,
        density=True,
        color="blue",
        alpha=0.6,
        label=f"DFT (std={results['stats']['dft']['std']:.2e})",
        edgecolor="black",
        *args,
        **kwargs,
    )
    ax_plot.axvline(0, color="tomato", linestyle="--", linewidth=2, label="Zero error")
    ax_plot.axvline(
        crlb_std, color="lime", linestyle="--", linewidth=2, label=f"CRLB std = {crlb_std:.2e}"
    )
    ax_plot.axvline(-crlb_std, color="lime", linestyle="--", linewidth=2)
    ax_plot.set_xlabel("Frequency error (Hz)")
    ax_plot.set_ylabel("Probability density")
    ax_plot.set_title("Error Distribution Comparison")
    apply_legend(ax_plot)
    ax_plot.grid(True, alpha=0.3)

    # Box plot comparison
    ax_plot = axes[1]
    data = [errors_nls, errors_dft]
    bp = ax_plot.boxplot(data, tick_labels=["NLS", "DFT"], patch_artist=True, *args, **kwargs)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_alpha(0.7)
    ax_plot.axhline(0, color="tomato", linestyle="--", linewidth=2, label="Zero error")
    ax_plot.axhline(
        crlb_std, color="lime", linestyle="--", linewidth=2, label=f"CRLB std = {crlb_std:.2e}"
    )
    ax_plot.axhline(-crlb_std, color="lime", linestyle="--", linewidth=2)
    ax_plot.set_ylabel("Frequency error (Hz)")
    ax_plot.grid(True, alpha=0.3, axis="y")

    if ax is None:
        plt.tight_layout()
    return axes


def plot_performance_comparison(results: dict, ax=None, figsize=None, dpi=None, *args, **kwargs):
    """Plot performance metrics comparison."""
    stats = results["stats"]
    crlb_std = results["crlb_std"]

    methods = ["NLS", "DFT"]
    stds = [stats["nls"]["std"], stats["dft"]["std"]]
    efficiencies = [crlb_std / s for s in stds]  # Efficiency = CRLB/std

    if ax is None:
        if figsize is None:
            figsize = (6.5, 3)
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    else:
        # If ax is provided, it should be an array of axes
        if not isinstance(ax, np.ndarray) or ax.size != 2:
            raise ValueError("ax must be an array of 2 axes for this plot")
        axes = ax
        fig = axes[0].figure

    # Standard deviation comparison
    ax_plot = axes[0]
    x_pos = np.arange(len(methods))
    bars = ax_plot.bar(x_pos, stds, alpha=0.7, edgecolor="black", *args, **kwargs)
    ax_plot.axhline(
        crlb_std, color="lime", linestyle="--", linewidth=2, label=f"CRLB = {crlb_std:.2e}"
    )
    ax_plot.set_xticks(x_pos)
    ax_plot.set_xticklabels(methods)
    ax_plot.set_ylabel("Standard deviation (Hz)")
    ax_plot.set_title("Frequency Estimation Uncertainty")
    ax_plot.set_yscale("log")
    apply_legend(ax_plot)
    ax_plot.grid(True, alpha=0.3, axis="y")

    # Efficiency comparison
    ax_plot = axes[1]
    bars = ax_plot.bar(x_pos, efficiencies, alpha=0.7, edgecolor="black", *args, **kwargs)
    ax_plot.axhline(1.0, color="tomato", linestyle="--", linewidth=2, label="CRLB (efficiency = 1)")
    ax_plot.set_xticks(x_pos)
    ax_plot.set_xticklabels(methods)
    ax_plot.set_ylabel("Efficiency (CRLB / std)")
    ax_plot.set_title("Statistical Efficiency")
    apply_legend(ax_plot)
    ax_plot.grid(True, alpha=0.3, axis="y")

    if ax is None:
        plt.tight_layout()
    return axes


def plot_q_individual_results(results: dict, ax=None, figsize=None, dpi=None, *args, **kwargs):
    """Plot Q error distributions for NLS method."""
    if "errors_q_nls" not in results or len(results["errors_q_nls"]) == 0:
        # Return empty axes if no Q data
        if ax is None:
            if figsize is None:
                figsize = (6.5, 3)
            fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
            plt.tight_layout()
        else:
            fig = ax.figure
        ax.text(
            0.5,
            0.5,
            "No Q estimation data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Q Estimation Error Distribution")
        return ax

    errors_q_nls = results["errors_q_nls"]
    crlb_std_q = results.get("crlb_std_q")

    created_fig = False
    if ax is None:
        if figsize is None:
            figsize = (6.5, 3)
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        created_fig = True
    else:
        fig = ax.figure

    ax.hist(
        errors_q_nls,
        bins=30,
        density=True,
        color="blue",
        alpha=0.6,
        edgecolor="black",
        *args,
        **kwargs,
    )
    ax.axvline(0, color="tomato", linestyle="--", label="Zero error")
    if crlb_std_q is not None and np.isfinite(crlb_std_q):
        ax.axvline(crlb_std_q, color="lime", linestyle="--", label=f"CRLB std = {crlb_std_q:.2e}")
        ax.axvline(-crlb_std_q, color="lime", linestyle="--")
    ax.set_xlabel("Q error")
    ax.set_ylabel("Probability density")
    ax.set_title(
        f"Nonlinear Least Squares (NLS) Q Estimation\nstd = {results['stats']['q_nls']['std']:.6e}"
    )
    apply_legend(ax)
    ax.grid(True, alpha=0.3)

    if created_fig:
        plt.tight_layout()
    return ax


def plot_q_performance_comparison(results: dict, ax=None, figsize=None, dpi=None, *args, **kwargs):
    """Plot Q performance metrics comparison."""
    if "errors_q_nls" not in results or len(results["errors_q_nls"]) == 0:
        # Return empty axes if no Q data
        if ax is None:
            if figsize is None:
                figsize = (6.5, 3)
            fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
            plt.tight_layout()
        else:
            fig = ax.figure
        ax.text(
            0.5,
            0.5,
            "No Q estimation data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Q Estimation Performance")
        return ax

    stats = results["stats"]
    crlb_std_q = results.get("crlb_std_q")

    if crlb_std_q is None or not np.isfinite(crlb_std_q):
        # Return empty axes if no CRLB data
        if ax is None:
            if figsize is None:
                figsize = (6.5, 3)
            fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
            plt.tight_layout()
        else:
            fig = ax.figure
        ax.text(
            0.5, 0.5, "No Q-CRLB data available", ha="center", va="center", transform=ax.transAxes
        )
        ax.set_title("Q Estimation Performance")
        return ax

    methods = ["NLS"]
    stds = [stats["q_nls"]["std"]]
    efficiencies = [crlb_std_q / stats["q_nls"]["std"]] if stats["q_nls"]["std"] > 0 else [0.0]

    if ax is None:
        if figsize is None:
            figsize = (6.5, 3)
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    else:
        # If ax is provided, it should be an array of axes
        if not isinstance(ax, np.ndarray) or ax.size != 2:
            raise ValueError("ax must be an array of 2 axes for this plot")
        axes = ax
        fig = axes[0].figure

    # Standard deviation comparison
    ax_plot = axes[0]
    x_pos = np.arange(len(methods))
    bars = ax_plot.bar(x_pos, stds, alpha=0.7, edgecolor="black", *args, **kwargs)
    ax_plot.axhline(
        crlb_std_q, color="lime", linestyle="--", linewidth=2, label=f"CRLB = {crlb_std_q:.2e}"
    )
    ax_plot.set_xticks(x_pos)
    ax_plot.set_xticklabels(methods)
    ax_plot.set_ylabel("Standard deviation")
    ax_plot.set_title("Q Estimation Uncertainty")
    ax_plot.set_yscale("log")
    apply_legend(ax_plot)
    ax_plot.grid(True, alpha=0.3, axis="y")

    # Efficiency comparison
    ax_plot = axes[1]
    bars = ax_plot.bar(x_pos, efficiencies, alpha=0.7, edgecolor="black", *args, **kwargs)
    ax_plot.axhline(1.0, color="tomato", linestyle="--", linewidth=2, label="CRLB (efficiency = 1)")
    ax_plot.set_xticks(x_pos)
    ax_plot.set_xticklabels(methods)
    ax_plot.set_ylabel("Efficiency (CRLB / std)")
    ax_plot.set_title("Statistical Efficiency")
    apply_legend(ax_plot)
    ax_plot.grid(True, alpha=0.3, axis="y")

    if ax is None:
        plt.tight_layout()
    return axes
