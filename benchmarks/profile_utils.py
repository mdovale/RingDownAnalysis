"""
Profiling utilities using cProfile and pstats.

Provides functions for profiling critical code paths and analyzing bottlenecks.
"""

import cProfile
import io
import pstats
from contextlib import redirect_stdout
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ringdownanalysis import (
    BatchRingDownAnalyzer,
    DFTFrequencyEstimator,
    MonteCarloAnalyzer,
    NLSFrequencyEstimator,
    RingDownAnalyzer,
    RingDownSignal,
)


class Profiler:
    """Wrapper for cProfile with convenient analysis methods."""

    def __init__(self):
        self.profiler = cProfile.Profile()
        self.stats: Optional[pstats.Stats] = None

    def profile(self, func: Callable, *args, **kwargs):
        """Profile a function call."""
        self.profiler.enable()
        try:
            result = func(*args, **kwargs)
        finally:
            self.profiler.disable()
        return result

    def get_stats(self) -> pstats.Stats:
        """Get pstats.Stats object for analysis."""
        if self.stats is None:
            self.stats = pstats.Stats(self.profiler)
        return self.stats

    def print_stats(self, sort_by: str = "cumulative", lines: int = 20):
        """Print top statistics."""
        stats = self.get_stats()
        stats.sort_stats(sort_by)
        stats.print_stats(lines)

    def get_top_functions(
        self,
        sort_by: str = "cumulative",
        top_n: int = 20,
    ) -> List[Tuple[str, int, float, float, str]]:
        """
        Get top N functions by time.

        Returns:
        --------
        List of tuples: (filename:line(function), ncalls, tottime, cumtime, callers)
        """
        stats = self.get_stats()
        stats.sort_stats(sort_by)

        # Capture stats output
        stream = io.StringIO()
        with redirect_stdout(stream):
            stats.print_stats(top_n)
        stream.seek(0)

        # Parse output (simplified - pstats doesn't have a clean API for this)
        top_functions = []
        lines = stream.readlines()

        # Skip header lines
        start_idx = 0
        for i, line in enumerate(lines):
            if "ncalls" in line and "tottime" in line:
                start_idx = i + 1
                break

        # Parse function lines
        for line in lines[start_idx : start_idx + top_n]:
            if not line.strip() or line.startswith("---"):
                break
            parts = line.split()
            if len(parts) >= 5:
                try:
                    ncalls = parts[0] if "/" not in parts[0] else parts[0].split("/")[0]
                    tottime = float(parts[1])
                    cumtime = float(parts[2])
                    func_name = " ".join(parts[4:]) if len(parts) > 4 else ""
                    top_functions.append((func_name, ncalls, tottime, cumtime, ""))
                except (ValueError, IndexError):
                    continue

        return top_functions

    def save_stats(self, filepath: str):
        """Save stats to file."""
        stats = self.get_stats()
        stats.dump_stats(filepath)

    def load_stats(self, filepath: str):
        """Load stats from file."""
        self.stats = pstats.Stats(filepath)

    def get_bottlenecks(
        self,
        min_time: float = 0.01,
        min_calls: int = 1,
    ) -> List[Dict]:
        """
        Identify bottlenecks based on time thresholds.

        Parameters:
        -----------
        min_time : float
            Minimum cumulative time (seconds) to consider a bottleneck
        min_calls : int
            Minimum number of calls to consider

        Returns:
        --------
        List of dicts with bottleneck information
        """
        stats = self.get_stats()
        stats.sort_stats("cumulative")

        bottlenecks = []
        stream = io.StringIO()
        # Redirect stdout to capture print_stats output
        with redirect_stdout(stream):
            stats.print_stats(1000)  # Get many functions
        stream.seek(0)
        lines = stream.readlines()

        # Find header
        start_idx = 0
        for i, line in enumerate(lines):
            if "ncalls" in line and "tottime" in line:
                start_idx = i + 1
                break

        # Parse and filter
        for line in lines[start_idx:]:
            if not line.strip() or line.startswith("---"):
                break
            parts = line.split()
            if len(parts) >= 5:
                try:
                    ncalls_str = parts[0]
                    if "/" in ncalls_str:
                        ncalls = int(ncalls_str.split("/")[0])
                    else:
                        ncalls = int(ncalls_str) if ncalls_str.isdigit() else 1

                    tottime = float(parts[1])
                    cumtime = float(parts[2])
                    func_name = " ".join(parts[4:]) if len(parts) > 4 else ""

                    # Filter by thresholds
                    if cumtime >= min_time and ncalls >= min_calls:
                        bottlenecks.append(
                            {
                                "function": func_name,
                                "ncalls": ncalls,
                                "tottime": tottime,
                                "cumtime": cumtime,
                                "tottime_per_call": tottime / ncalls if ncalls > 0 else 0,
                                "cumtime_per_call": cumtime / ncalls if ncalls > 0 else 0,
                            }
                        )
                except (ValueError, IndexError, ZeroDivisionError):
                    continue

        return sorted(bottlenecks, key=lambda x: x["cumtime"], reverse=True)


# ============================================================================
# Profiling functions for critical workloads
# ============================================================================


def profile_single_file_analysis(
    filepath: Optional[str] = None,
    N: int = 100_000,
    save_to: Optional[str] = None,
) -> Profiler:
    """
    Profile single file analysis workflow.

    Parameters:
    -----------
    filepath : str, optional
        Path to real data file. If None, creates synthetic data.
    N : int
        Number of samples for synthetic data (if filepath is None)
    save_to : str, optional
        Path to save profile stats

    Returns:
    --------
    Profiler instance with results
    """
    profiler = Profiler()

    if filepath is None:
        # Create synthetic data
        import os
        import tempfile

        signal = RingDownSignal(f0=5.0, fs=100.0, N=N, A0=1.0, snr_db=60.0, Q=10000.0)
        rng = np.random.default_rng(42)
        t, x, _ = signal.generate(rng=rng)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            for ti, xi in zip(t[:: max(1, N // 10000)], x[:: max(1, N // 10000)]):
                f.write(f"{ti:.6f},0,0,{xi:.6f}\n")
            filepath = f.name

        def analyze():
            analyzer = RingDownAnalyzer()
            return analyzer.analyze_file(filepath)

        result = profiler.profile(analyze)

        # Cleanup
        os.unlink(filepath)
    else:

        def analyze():
            analyzer = RingDownAnalyzer()
            return analyzer.analyze_file(filepath)

        result = profiler.profile(analyze)

    if save_to:
        profiler.save_stats(save_to)

    return profiler


def profile_frequency_estimation(
    N: int = 1_000_000,
    method: str = "both",
    save_to: Optional[str] = None,
) -> Profiler:
    """
    Profile frequency estimation methods.

    Parameters:
    -----------
    N : int
        Number of samples
    method : str
        'nls', 'dft', or 'both'
    save_to : str, optional
        Path to save profile stats

    Returns:
    --------
    Profiler instance with results
    """
    profiler = Profiler()

    signal = RingDownSignal(f0=5.0, fs=100.0, N=N, A0=1.0, snr_db=60.0, Q=10000.0)
    rng = np.random.default_rng(42)
    _, x, _ = signal.generate(rng=rng)

    def estimate():
        if method in ("nls", "both"):
            nls_est = NLSFrequencyEstimator(tau_known=None)
            f_nls = nls_est.estimate(x, signal.fs)
        if method in ("dft", "both"):
            dft_est = DFTFrequencyEstimator(window="kaiser")
            f_dft = dft_est.estimate(x, signal.fs)

    profiler.profile(estimate)

    if save_to:
        profiler.save_stats(save_to)

    return profiler


def profile_batch_analysis(
    n_files: int = 20,
    N_per_file: int = 100_000,
    save_to: Optional[str] = None,
) -> Profiler:
    """
    Profile batch analysis workflow.

    Parameters:
    -----------
    n_files : int
        Number of files to process
    N_per_file : int
        Number of samples per file
    save_to : str, optional
        Path to save profile stats

    Returns:
    --------
    Profiler instance with results
    """
    profiler = Profiler()

    # Create synthetic files
    import os
    import shutil
    import tempfile

    temp_dir = tempfile.mkdtemp()
    filepaths = []

    try:
        signal = RingDownSignal(f0=5.0, fs=100.0, N=N_per_file, A0=1.0, snr_db=60.0, Q=10000.0)
        rng = np.random.default_rng(42)

        for i in range(n_files):
            t, x, _ = signal.generate(rng=np.random.default_rng(42 + i))
            filepath = os.path.join(temp_dir, f"test_{i}.csv")
            with open(filepath, "w") as f:
                for ti, xi in zip(
                    t[:: max(1, N_per_file // 10000)], x[:: max(1, N_per_file // 10000)]
                ):
                    f.write(f"{ti:.6f},0,0,{xi:.6f}\n")
            filepaths.append(filepath)

        def analyze():
            analyzer = BatchRingDownAnalyzer()
            return analyzer.process_files(filepaths, verbose=False, n_jobs=1)

        result = profiler.profile(analyze)

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)

    if save_to:
        profiler.save_stats(save_to)

    return profiler


def profile_monte_carlo(
    n_mc: int = 50,
    N: int = 1_000_000,
    save_to: Optional[str] = None,
) -> Profiler:
    """
    Profile Monte Carlo analysis.

    Parameters:
    -----------
    n_mc : int
        Number of Monte Carlo trials
    N : int
        Number of samples per trial
    save_to : str, optional
        Path to save profile stats

    Returns:
    --------
    Profiler instance with results
    """
    profiler = Profiler()

    def run_mc():
        analyzer = MonteCarloAnalyzer()
        return analyzer.run(
            f0=5.0,
            fs=100.0,
            N=N,
            A0=1.0,
            snr_db=60.0,
            Q=10000.0,
            n_mc=n_mc,
            seed=42,
            n_workers=1,
        )

    result = profiler.profile(run_mc)

    if save_to:
        profiler.save_stats(save_to)

    return profiler


# ============================================================================
# Analysis and reporting
# ============================================================================


def analyze_bottlenecks(
    profiler: Profiler,
    min_time: float = 0.01,
    min_calls: int = 1,
) -> Dict:
    """
    Analyze bottlenecks from profiler results.

    Returns:
    --------
    Dict with bottleneck analysis
    """
    bottlenecks = profiler.get_bottlenecks(min_time=min_time, min_calls=min_calls)

    # Categorize bottlenecks
    scipy_funcs = [b for b in bottlenecks if "scipy" in b["function"].lower()]
    numpy_funcs = [b for b in bottlenecks if "numpy" in b["function"].lower()]
    ringdown_funcs = [b for b in bottlenecks if "ringdown" in b["function"].lower()]
    other_funcs = [
        b
        for b in bottlenecks
        if not any(x in b["function"].lower() for x in ["scipy", "numpy", "ringdown"])
    ]

    total_time = sum(b["cumtime"] for b in bottlenecks)

    return {
        "total_bottlenecks": len(bottlenecks),
        "total_time": total_time,
        "scipy_functions": scipy_funcs,
        "numpy_functions": numpy_funcs,
        "ringdown_functions": ringdown_funcs,
        "other_functions": other_funcs,
        "top_10": bottlenecks[:10],
        "all_bottlenecks": bottlenecks,
    }


def print_bottleneck_report(analysis: Dict):
    """Print formatted bottleneck report."""
    print("=" * 80)
    print("BOTTLENECK ANALYSIS REPORT")
    print("=" * 80)
    print(f"\nTotal bottlenecks found: {analysis['total_bottlenecks']}")
    print(f"Total time in bottlenecks: {analysis['total_time']:.4f} seconds")

    print("\n" + "-" * 80)
    print("TOP 10 BOTTLENECKS")
    print("-" * 80)
    for i, b in enumerate(analysis["top_10"], 1):
        print(f"\n{i}. {b['function']}")
        print(f"   Calls: {b['ncalls']}")
        print(f"   Total time: {b['tottime']:.4f} s")
        print(f"   Cumulative time: {b['cumtime']:.4f} s")
        print(f"   Time per call: {b['tottime_per_call']:.6f} s")

    if analysis["scipy_functions"]:
        print("\n" + "-" * 80)
        print("SCIPY FUNCTIONS")
        print("-" * 80)
        scipy_time = sum(b["cumtime"] for b in analysis["scipy_functions"])
        print(
            f"Total time in scipy: {scipy_time:.4f} s ({scipy_time / analysis['total_time'] * 100:.1f}%)"
        )
        for b in analysis["scipy_functions"][:5]:
            print(f"  {b['function']}: {b['cumtime']:.4f} s")

    if analysis["numpy_functions"]:
        print("\n" + "-" * 80)
        print("NUMPY FUNCTIONS")
        print("-" * 80)
        numpy_time = sum(b["cumtime"] for b in analysis["numpy_functions"])
        print(
            f"Total time in numpy: {numpy_time:.4f} s ({numpy_time / analysis['total_time'] * 100:.1f}%)"
        )
        for b in analysis["numpy_functions"][:5]:
            print(f"  {b['function']}: {b['cumtime']:.4f} s")

    if analysis["ringdown_functions"]:
        print("\n" + "-" * 80)
        print("RINGDOWN FUNCTIONS (OUR CODE)")
        print("-" * 80)
        ringdown_time = sum(b["cumtime"] for b in analysis["ringdown_functions"])
        print(
            f"Total time in ringdown: {ringdown_time:.4f} s ({ringdown_time / analysis['total_time'] * 100:.1f}%)"
        )
        for b in analysis["ringdown_functions"][:5]:
            print(f"  {b['function']}: {b['cumtime']:.4f} s")

    print("\n" + "=" * 80)
