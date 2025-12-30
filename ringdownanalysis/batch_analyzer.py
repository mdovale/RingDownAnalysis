"""
Batch analysis and statistics for ring-down measurement data.

This module provides functionality for analyzing multiple ring-down data files,
computing summary statistics, Q factor analysis, consistency analysis, and
comparison with CRLB bounds.
"""

import glob
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .analyzer import RingDownAnalyzer

logger = logging.getLogger(__name__)


def _process_single_file(filepath: str) -> Dict:
    """
    Helper function to process a single file in parallel.

    Creates its own analyzer instance to avoid pickling issues.

    Parameters:
    -----------
    filepath : str
        Path to file to process

    Returns:
    --------
    Dict
        Result dictionary from analyzer.analyze_file
    """
    analyzer = RingDownAnalyzer()
    return analyzer.analyze_file(filepath)


class BatchRingDownAnalyzer:
    """
    Batch analysis for multiple ring-down measurement files.

    Extends RingDownAnalyzer with capabilities for:
    - Batch processing multiple files
    - Summary statistics and tables
    - Q factor analysis
    - Consistency analysis across realizations
    - CRLB comparison analysis
    """

    def __init__(
        self,
        analyzer: Optional[RingDownAnalyzer] = None,
    ):
        """
        Initialize batch analyzer.

        Parameters:
        -----------
        analyzer : RingDownAnalyzer, optional
            RingDownAnalyzer instance to use. If None, creates default.
        """
        self.analyzer = analyzer or RingDownAnalyzer()
        self.results: List[Dict] = []

    def process_files(
        self,
        filepaths: List[str],
        verbose: bool = True,
        n_jobs: Optional[int] = None,
    ) -> List[Dict]:
        """
        Process multiple data files and store results.

        Parameters:
        -----------
        filepaths : List[str]
            List of file paths to process
        verbose : bool
            Print progress information (default: True)
        n_jobs : int, optional
            Number of parallel workers. If None or 1, processes sequentially.
            If > 1, uses ProcessPoolExecutor for parallel processing.
            If -1, uses all available CPU cores.

        Returns:
        --------
        List[Dict]
            List of result dictionaries (same as RingDownAnalyzer.analyze_file)
        """
        self.results = []

        if not filepaths:
            return self.results

        # Determine number of workers
        if n_jobs is None or n_jobs == 1:
            # Sequential processing
            if logger.isEnabledFor(logging.INFO):
                logger.info(
                    "batch_processing_start",
                    extra={
                        "event": "batch_processing_start",
                        "n_files": len(filepaths),
                        "mode": "sequential",
                    },
                )

            for filepath in filepaths:
                try:
                    if verbose:
                        print(f"Processing {Path(filepath).name}...")

                    result = self.analyzer.analyze_file(filepath)
                    self.results.append(result)

                    if verbose:
                        print(f"  Sampling frequency: {result['fs']:.2f} Hz")
                        print(f"  Estimated tau: {result['tau_est']:.2f} s")
                        print(
                            f"  Cropped to: {result['T_crop']:.2f} s "
                            f"({result['N_crop']} samples, "
                            f"{result['N_crop'] / result['N'] * 100:.1f}% of original)"
                        )
                        print(f"  NLS frequency: {result['f_nls']:.6f} Hz")
                        print(f"  DFT frequency: {result['f_dft']:.6f} Hz")
                        print(f"  Difference: {abs(result['f_nls'] - result['f_dft']):.6e} Hz")
                        print(f"  CRLB std: {result['crlb_std_f']:.6e} Hz")
                except Exception as e:
                    logger.error(
                        "file_processing_error",
                        extra={
                            "event": "file_processing_error",
                            "filepath": str(filepath),
                            "error_type": type(e).__name__,
                            "error_msg": str(e),
                        },
                        exc_info=True,
                    )
                    if verbose:
                        print(f"  Error processing {Path(filepath).name}: {e}")
                        import traceback

                        traceback.print_exc()
        else:
            # Parallel processing
            if n_jobs == -1:
                n_jobs = os.cpu_count() or 1

            if logger.isEnabledFor(logging.INFO):
                logger.info(
                    "batch_processing_start",
                    extra={
                        "event": "batch_processing_start",
                        "n_files": len(filepaths),
                        "mode": "parallel",
                        "n_workers": n_jobs,
                    },
                )

            if verbose:
                print(f"Processing {len(filepaths)} files using {n_jobs} workers...")

            # Create a dictionary to map results back to original order
            filepath_to_index = {fp: i for i, fp in enumerate(filepaths)}
            results_dict = {}
            errors = []

            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # Submit all tasks
                future_to_filepath = {
                    executor.submit(_process_single_file, filepath): filepath
                    for filepath in filepaths
                }

                # Process completed tasks as they finish
                for future in as_completed(future_to_filepath):
                    filepath = future_to_filepath[future]
                    try:
                        result = future.result()
                        idx = filepath_to_index[filepath]
                        results_dict[idx] = result

                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                "file_processed",
                                extra={
                                    "event": "file_processed",
                                    "filepath": str(filepath),
                                    "progress": f"{len(results_dict)}/{len(filepaths)}",
                                },
                            )

                        if verbose:
                            print(
                                f"Completed {Path(filepath).name} "
                                f"({len(results_dict)}/{len(filepaths)})"
                            )
                            print(f"  Sampling frequency: {result['fs']:.2f} Hz")
                            print(f"  Estimated tau: {result['tau_est']:.2f} s")
                            print(
                                f"  Cropped to: {result['T_crop']:.2f} s "
                                f"({result['N_crop']} samples, "
                                f"{result['N_crop'] / result['N'] * 100:.1f}% of original)"
                            )
                            print(f"  NLS frequency: {result['f_nls']:.6f} Hz")
                            print(f"  DFT frequency: {result['f_dft']:.6f} Hz")
                            print(f"  Difference: {abs(result['f_nls'] - result['f_dft']):.6e} Hz")
                            print(f"  CRLB std: {result['crlb_std_f']:.6e} Hz")
                    except Exception as e:
                        errors.append((filepath, e))
                        logger.error(
                            "file_processing_error",
                            extra={
                                "event": "file_processing_error",
                                "filepath": str(filepath),
                                "error_type": type(e).__name__,
                                "error_msg": str(e),
                            },
                            exc_info=True,
                        )
                        if verbose:
                            print(f"  Error processing {Path(filepath).name}: {e}")
                            import traceback

                            traceback.print_exc()

            # Reconstruct results in original order
            self.results = [results_dict[i] for i in sorted(results_dict.keys())]

            if errors:
                logger.warning(
                    "batch_processing_errors",
                    extra={
                        "event": "batch_processing_errors",
                        "n_errors": len(errors),
                        "n_total": len(filepaths),
                    },
                )
                if verbose:
                    print(f"\n{len(errors)} file(s) failed to process")

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "batch_processing_complete",
                extra={
                    "event": "batch_processing_complete",
                    "n_successful": len(self.results),
                    "n_total": len(filepaths),
                },
            )

        if verbose:
            print(f"\nSuccessfully processed {len(self.results)} files")

        return self.results

    def process_directory(
        self,
        directory: str,
        pattern: str = "*",
        verbose: bool = True,
        n_jobs: Optional[int] = None,
    ) -> List[Dict]:
        """
        Process all data files in a directory.

        Parameters:
        -----------
        directory : str
            Directory path containing data files
        pattern : str
            Glob pattern for file matching (default: "*")
        verbose : bool
            Print progress information (default: True)
        n_jobs : int, optional
            Number of parallel workers. If None or 1, processes sequentially.
            If > 1, uses ProcessPoolExecutor for parallel processing.
            If -1, uses all available CPU cores.

        Returns:
        --------
        List[Dict]
            List of result dictionaries
        """
        csv_files = sorted(glob.glob(str(Path(directory) / f"{pattern}.csv")))
        mat_files = sorted(glob.glob(str(Path(directory) / f"{pattern}.mat")))
        all_files = csv_files + mat_files

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "directory_scan_complete",
                extra={
                    "event": "directory_scan_complete",
                    "directory": str(directory),
                    "pattern": pattern,
                    "n_csv": len(csv_files),
                    "n_mat": len(mat_files),
                    "n_total": len(all_files),
                },
            )

        if verbose:
            print(f"Found {len(csv_files)} CSV files and {len(mat_files)} MAT files")

        return self.process_files(all_files, verbose=verbose, n_jobs=n_jobs)

    def calculate_q_factors(self) -> List[float]:
        """
        Calculate Q factors for all processed results.

        Q = π * f * τ

        Returns:
        --------
        List[float]
            Q factor for each result
        """
        # Vectorized Q factor calculation
        f_nls_all = np.array([r["f_nls"] for r in self.results], dtype=float)
        tau_est_all = np.array([r["tau_est"] for r in self.results], dtype=float)
        q_factors = (np.pi * f_nls_all * tau_est_all).tolist()

        # Store Q factors in results
        for r, q in zip(self.results, q_factors):
            r["Q"] = q

        return q_factors

    def get_summary_table(self) -> Dict:
        """
        Create a summary table with all analysis results.

        Returns:
        --------
        Dict
            Dictionary with 'data' (list of dicts) and 'columns' (list of column names)
            Suitable for creating pandas DataFrame
        """
        if not self.results:
            return {"data": [], "columns": []}

        summary_data = []
        for r in self.results:
            summary_data.append(
                {
                    "Filename": r["filename"],
                    "Type": r["type"],
                    "N (samples)": r["N"],
                    "N_crop (samples)": r["N_crop"],
                    "T (s)": f"{r['T']:.2f}",
                    "T_crop (s)": f"{r['T_crop']:.2f}",
                    "fs (Hz)": f"{r['fs']:.2f}",
                    "tau_est (s)": f"{r['tau_est']:.2f}",
                    "f_NLS (Hz)": f"{r['f_nls']:.6f}",
                    "f_DFT (Hz)": f"{r['f_dft']:.6f}",
                    "|f_NLS - f_DFT| (Hz)": f"{abs(r['f_nls'] - r['f_dft']):.6e}",
                    "CRLB std (Hz)": f"{r['crlb_std_f']:.6e}",
                    "A0_est": f"{r['A0_est']:.4f}",
                    "sigma_est": f"{r['sigma_est']:.6e}",
                }
            )

        # Add Q factor if calculated
        if "Q" in self.results[0]:
            for i, r in enumerate(self.results):
                summary_data[i]["Q"] = f"{r['Q']:.2e}"

        columns = list(summary_data[0].keys()) if summary_data else []

        return {"data": summary_data, "columns": columns}

    def consistency_analysis(self) -> Dict:
        """
        Perform consistency analysis across all realizations.

        Computes:
        - Pairwise differences for NLS and DFT methods
        - Statistics (mean, median, std, min, max) for each method
        - Standard deviation across realizations
        - Coefficient of variation

        Returns:
        --------
        Dict
            Dictionary with analysis results including:
            - 'nls_pairwise_diffs': array of pairwise differences
            - 'dft_pairwise_diffs': array of pairwise differences
            - 'nls_statistics': dict with mean, median, std, min, max
            - 'dft_statistics': dict with mean, median, std, min, max
            - 'nls_std_across_realizations': float
            - 'dft_std_across_realizations': float
            - 'nls_mean': float
            - 'dft_mean': float
            - 'nls_cv': float (coefficient of variation)
            - 'dft_cv': float
            - 'nls_span': float (max - min)
            - 'dft_span': float
        """
        if not self.results:
            return {}

        n_realizations = len(self.results)

        # Extract frequencies - vectorized extraction
        f_nls_all = np.array([r["f_nls"] for r in self.results], dtype=float)
        f_dft_all = np.array([r["f_dft"] for r in self.results], dtype=float)

        # Compute pairwise differences using vectorized operations
        # Create upper triangular indices for pairwise comparisons
        i_indices, j_indices = np.triu_indices(n_realizations, k=1)
        nls_pairwise_diffs = np.abs(f_nls_all[i_indices] - f_nls_all[j_indices])
        dft_pairwise_diffs = np.abs(f_dft_all[i_indices] - f_dft_all[j_indices])

        # Statistics for pairwise differences
        # Handle empty arrays to avoid RuntimeWarning
        if len(nls_pairwise_diffs) > 0:
            nls_stats = {
                "mean": np.mean(nls_pairwise_diffs),
                "median": np.median(nls_pairwise_diffs),
                "std": np.std(nls_pairwise_diffs),
                "min": np.min(nls_pairwise_diffs),
                "max": np.max(nls_pairwise_diffs),
            }
        else:
            nls_stats = {
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
            }

        if len(dft_pairwise_diffs) > 0:
            dft_stats = {
                "mean": np.mean(dft_pairwise_diffs),
                "median": np.median(dft_pairwise_diffs),
                "std": np.std(dft_pairwise_diffs),
                "min": np.min(dft_pairwise_diffs),
                "max": np.max(dft_pairwise_diffs),
            }
        else:
            dft_stats = {
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
            }

        # Statistics across realizations
        nls_mean = np.mean(f_nls_all)
        dft_mean = np.mean(f_dft_all)
        nls_std_across = np.std(f_nls_all)
        dft_std_across = np.std(f_dft_all)

        nls_cv = nls_std_across / nls_mean if nls_mean > 0 else np.inf
        dft_cv = dft_std_across / dft_mean if dft_mean > 0 else np.inf

        nls_span = np.max(f_nls_all) - np.min(f_nls_all)
        dft_span = np.max(f_dft_all) - np.min(f_dft_all)

        return {
            "n_realizations": n_realizations,
            "n_pairwise_comparisons": len(nls_pairwise_diffs),
            "nls_pairwise_diffs": nls_pairwise_diffs,
            "dft_pairwise_diffs": dft_pairwise_diffs,
            "nls_pairwise_indices": list(zip(i_indices, j_indices)),
            "dft_pairwise_indices": list(zip(i_indices, j_indices)),
            "nls_statistics": nls_stats,
            "dft_statistics": dft_stats,
            "nls_mean": nls_mean,
            "dft_mean": dft_mean,
            "nls_std_across_realizations": nls_std_across,
            "dft_std_across_realizations": dft_std_across,
            "nls_cv": nls_cv,
            "dft_cv": dft_cv,
            "nls_span": nls_span,
            "dft_span": dft_span,
            "nls_range": (np.min(f_nls_all), np.max(f_nls_all)),
            "dft_range": (np.min(f_dft_all), np.max(f_dft_all)),
        }

    def crlb_comparison_analysis(self) -> Dict:
        """
        Compare frequency estimation differences with CRLB.

        Computes:
        - Frequency differences between NLS and DFT
        - Ratio of differences to CRLB
        - Statistics comparing differences to CRLB

        Returns:
        --------
        Dict
            Dictionary with analysis results including:
            - 'frequency_diffs': array of |f_NLS - f_DFT|
            - 'crlb_stds': array of CRLB standard deviations
            - 'ratios': array of |f_NLS - f_DFT| / CRLB_std
            - 'crlb_statistics': dict with mean, min, max CRLB
            - 'ratio_statistics': dict with mean, median, min, max ratios
        """
        if not self.results:
            return {}

        # Vectorized extraction and computation
        f_nls_all = np.array([r["f_nls"] for r in self.results], dtype=float)
        f_dft_all = np.array([r["f_dft"] for r in self.results], dtype=float)
        crlb_stds = np.array([r["crlb_std_f"] for r in self.results], dtype=float)

        # Compute differences vectorized
        diffs = np.abs(f_nls_all - f_dft_all)

        # Compute ratios (difference / CRLB) vectorized
        # Use np.divide with where to handle division by zero and inf
        ratios = np.divide(
            diffs,
            crlb_stds,
            out=np.full_like(diffs, np.nan, dtype=float),
            where=(crlb_stds > 0) & np.isfinite(crlb_stds),
        )
        valid_ratios = ratios[np.isfinite(ratios)]

        # Vectorized CRLB statistics
        valid_crlb = crlb_stds[np.isfinite(crlb_stds)]
        if len(valid_crlb) > 0:
            crlb_stats = {
                "mean": np.mean(valid_crlb),
                "min": np.min(valid_crlb),
                "max": np.max(valid_crlb),
            }
        else:
            crlb_stats = {
                "mean": np.nan,
                "min": np.nan,
                "max": np.nan,
            }

        ratio_stats = {}
        if len(valid_ratios) > 0:
            ratio_stats = {
                "mean": np.mean(valid_ratios),
                "median": np.median(valid_ratios),
                "min": np.min(valid_ratios),
                "max": np.max(valid_ratios),
            }
        else:
            ratio_stats = {
                "mean": np.nan,
                "median": np.nan,
                "min": np.nan,
                "max": np.nan,
            }

        return {
            "frequency_diffs": np.array(diffs),
            "crlb_stds": np.array(crlb_stds),
            "ratios": ratios,
            "valid_ratios": valid_ratios,
            "crlb_statistics": crlb_stats,
            "ratio_statistics": ratio_stats,
        }

    def get_q_factor_statistics(self) -> Dict:
        """
        Calculate Q factor statistics.

        Returns:
        --------
        Dict
            Dictionary with Q factor statistics:
            - 'values': array of Q factors
            - 'mean': float
            - 'std': float
            - 'min': float
            - 'max': float
            - 'range': float
        """
        if not self.results:
            return {}

        # Ensure Q factors are calculated
        if "Q" not in self.results[0]:
            self.calculate_q_factors()

        q_values = np.array([r["Q"] for r in self.results], dtype=float)

        return {
            "values": q_values,
            "mean": np.mean(q_values),
            "std": np.std(q_values),
            "min": np.min(q_values),
            "max": np.max(q_values),
            "range": np.max(q_values) - np.min(q_values),
        }

    def get_consistency_table(self) -> Dict:
        """
        Create a table showing frequency estimates and deviations from mean.

        Returns:
        --------
        Dict
            Dictionary with 'data' (list of dicts) and 'columns' (list of column names)
            Suitable for creating pandas DataFrame
        """
        if not self.results:
            return {"data": [], "columns": []}

        consistency = self.consistency_analysis()
        nls_mean = consistency["nls_mean"]
        dft_mean = consistency["dft_mean"]

        consistency_data = []
        for i, r in enumerate(self.results):
            consistency_data.append(
                {
                    "Index": i,
                    "Filename": Path(r["filename"]).name[:40],
                    "f_NLS (Hz)": f"{r['f_nls']:.9f}",
                    "f_DFT (Hz)": f"{r['f_dft']:.9f}",
                    "Deviation from NLS mean (Hz)": f"{(r['f_nls'] - nls_mean):.6e}",
                    "Deviation from DFT mean (Hz)": f"{(r['f_dft'] - dft_mean):.6e}",
                    "CRLB std (Hz)": f"{r['crlb_std_f']:.6e}",
                }
            )

        columns = list(consistency_data[0].keys()) if consistency_data else []

        return {"data": consistency_data, "columns": columns}
