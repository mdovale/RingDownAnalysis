"""
Batch analysis and statistics for ring-down measurement data.

This module provides functionality for analyzing multiple ring-down data files,
computing summary statistics, Q factor analysis, consistency analysis, and
comparison with plug-in uncertainty diagnostics.
"""

from __future__ import annotations

import glob
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .analyzer import RingDownAnalyzer

logger = logging.getLogger(__name__)


def _result_uncertainty_std(result: dict) -> float:
    """Return the preferred per-record frequency uncertainty summary."""
    if "uncertainty_std_f" in result:
        return float(result["uncertainty_std_f"])
    if "plugin_crlb_std_f" in result:
        return float(result["plugin_crlb_std_f"])
    return float(result["crlb_std_f"])


@dataclass
class ProcessResult:
    """
    Result of batch file processing with both successes and failures.

    Provides list-like access to successful results for backward compatibility
    while exposing failed files for observability.

    Attributes:
    -----------
    results : List[Dict]
        Successfully processed results (same as RingDownAnalyzer.analyze_file output)
    failed_files : List[Tuple[str, BaseException]]
        List of (filepath, exception) for files that failed to process
    """

    results: list[dict]
    failed_files: list[tuple[str, BaseException]]

    def __len__(self) -> int:
        """Return number of successful results."""
        return len(self.results)

    def __iter__(self):
        """Iterate over successful results."""
        return iter(self.results)

    def __getitem__(self, index):
        """Index into successful results."""
        return self.results[index]

    @property
    def successes(self) -> list[dict]:
        """Alias for results (successful analyses)."""
        return self.results

    @property
    def failures(self) -> list[tuple[str, BaseException]]:
        """Alias for failed_files."""
        return self.failed_files

    def has_failures(self) -> bool:
        """Return True if any files failed to process."""
        return len(self.failed_files) > 0


def _process_single_file(filepath: str, analyzer: RingDownAnalyzer | None = None) -> dict:
    """
    Helper function to process a single file in parallel.

    Uses the configured analyzer when supplied so sequential and parallel runs
    preserve estimator settings.

    Parameters:
    -----------
    filepath : str
        Path to file to process

    Returns:
    --------
    Dict
        Result dictionary from analyzer.analyze_file
    """
    active_analyzer = analyzer or RingDownAnalyzer()
    return active_analyzer.analyze_file(filepath)


def _format_optional_float(value, fmt: str) -> str:
    """Format numeric display values while preserving missing estimates."""
    if value is None:
        return "—"
    value = float(value)
    if not np.isfinite(value):
        return str(value)
    return format(value, fmt)


def _result_q_value(
    result: dict, *, include_invalid: bool = False, q_preference: str = "demod"
) -> tuple[float | None, bool, str]:
    """
    Return the preferred Q value plus validity metadata for a result.

    q_preference="demod" prefers a valid segmented-demodulation Q (the
    drift-immune estimator); when the demod estimate is not valid the profile
    logic below still applies, so a coherent record with a valid (gated)
    profile Q is not lost. q_preference="profile" preserves the historical
    behavior.
    """
    if q_preference not in ("profile", "demod"):
        raise ValueError(f"q_preference must be 'profile' or 'demod', got {q_preference!r}")

    if q_preference == "demod" and "Q_demod" in result:
        q_demod = result.get("Q_demod")
        demod_valid = bool(result.get("Q_demod_valid", False))
        demod_status = str(result.get("Q_demod_status", "valid" if demod_valid else "invalid"))
        if demod_valid and q_demod is not None and np.isfinite(q_demod):
            return float(q_demod), True, demod_status
        if include_invalid and q_demod is not None and np.isfinite(q_demod):
            return float(q_demod), False, demod_status
        # No usable demod Q: fall through to the profile-based selection.

    has_profile = "Q_profile_valid" in result
    if has_profile:
        q_valid = bool(result.get("Q_profile_valid", False))
        q_status = str(result.get("Q_profile_status", "valid" if q_valid else "invalid"))
        q = result.get("Q_profile")
        # Only a valid profile Q is batch-preferred; warning/limit/invalid
        # profile values (e.g. envelope_mismatch demotions) are skipped.
        if q_valid and q is not None and np.isfinite(q):
            return float(q), True, q_status
        if not include_invalid:
            return None, False, q_status

        q = result.get("Q_profile_raw")
        if q is None or not np.isfinite(q):
            q = result.get("Q_nls_raw")
        if q is None or not np.isfinite(q):
            return None, False, q_status
        return float(q), False, q_status

    has_validity = "Q_nls_valid" in result
    q_valid = bool(result.get("Q_nls_valid", True))
    q_status = str(result.get("Q_nls_status", "valid" if q_valid else "invalid"))

    if has_validity and not q_valid and not include_invalid:
        return None, False, q_status

    q = result.get("Q_nls")
    if q is None and include_invalid:
        q = result.get("Q_nls_raw")
    if q is None and not has_validity:
        tau_for_q = result.get("tau_nls") or result.get("tau_model") or result["tau_est"]
        q = np.pi * result["f_nls"] * tau_for_q

    if q is None or not np.isfinite(q):
        return None, False, q_status

    return float(q), q_valid, q_status


class BatchRingDownAnalyzer:
    """
    Batch analysis for multiple ring-down measurement files.

    Extends RingDownAnalyzer with capabilities for:
    - Batch processing multiple files
    - Summary statistics and tables
    - Q factor analysis
    - Consistency analysis across realizations
    - Plug-in uncertainty comparison analysis
    """

    def __init__(
        self,
        analyzer: RingDownAnalyzer | None = None,
        *,
        q_preference: str = "demod",
    ):
        """
        Initialize batch analyzer.

        Parameters:
        -----------
        analyzer : RingDownAnalyzer, optional
            RingDownAnalyzer instance to use. If None, creates default.
        q_preference : str
            Which estimator supplies the aggregate per-record Q:
            "demod" (default; prefer the drift-immune segmented-demodulation
            Q, recommended for real long-record data) or "profile" (historical
            behavior). When the preferred estimator has no valid Q the
            profile/NLS selection logic still applies.
        """
        if q_preference not in ("profile", "demod"):
            raise ValueError(f"q_preference must be 'profile' or 'demod', got {q_preference!r}")
        self.analyzer = analyzer or RingDownAnalyzer()
        self.q_preference = q_preference
        self.results: list[dict] = []

    def process_files(
        self,
        filepaths: list[str],
        verbose: bool = True,
        n_jobs: int | None = None,
    ) -> ProcessResult:
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
        ProcessResult
            Object with `.results` (successful analyses), `.failed_files` (list of
            (filepath, exception) for failures). List-like for backward compatibility:
            len(), iteration, and indexing work on successful results.
        """
        self.results = []
        failed_files: list[tuple[str, BaseException]] = []

        if not filepaths:
            return ProcessResult(results=[], failed_files=[])

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
                        print(f"  Plugin bound std: {_result_uncertainty_std(result):.6e} Hz")
                except Exception as e:
                    failed_files.append((str(filepath), e))
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

            if failed_files:
                logger.warning(
                    "batch_processing_errors",
                    extra={
                        "event": "batch_processing_errors",
                        "n_errors": len(failed_files),
                        "n_total": len(filepaths),
                    },
                )
                if verbose:
                    print(f"\n{len(failed_files)} file(s) failed to process")

            if logger.isEnabledFor(logging.INFO):
                logger.info(
                    "batch_processing_complete",
                    extra={
                        "event": "batch_processing_complete",
                        "n_successful": len(self.results),
                        "n_failed": len(failed_files),
                        "n_total": len(filepaths),
                    },
                )
            if verbose:
                print(f"\nSuccessfully processed {len(self.results)} files")

            return ProcessResult(results=self.results, failed_files=failed_files)
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

            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # Submit all tasks
                future_to_filepath = {
                    executor.submit(_process_single_file, filepath, self.analyzer): filepath
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
                            print(f"  Plugin bound std: {_result_uncertainty_std(result):.6e} Hz")
                    except Exception as e:
                        failed_files.append((str(filepath), e))
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

            if failed_files:
                logger.warning(
                    "batch_processing_errors",
                    extra={
                        "event": "batch_processing_errors",
                        "n_errors": len(failed_files),
                        "n_total": len(filepaths),
                    },
                )
                if verbose:
                    print(f"\n{len(failed_files)} file(s) failed to process")

            if logger.isEnabledFor(logging.INFO):
                logger.info(
                    "batch_processing_complete",
                    extra={
                        "event": "batch_processing_complete",
                        "n_successful": len(self.results),
                        "n_failed": len(failed_files),
                        "n_total": len(filepaths),
                    },
                )
            if verbose:
                print(f"\nSuccessfully processed {len(self.results)} files")

            return ProcessResult(results=self.results, failed_files=failed_files)

    def process_directory(
        self,
        directory: str,
        pattern: str = "*",
        verbose: bool = True,
        n_jobs: int | None = None,
    ) -> ProcessResult:
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
        ProcessResult
            Object with `.results` (successful analyses) and `.failed_files`
            (filepath, exception) for failures. See process_files() for details.

        Raises:
        -------
        FileNotFoundError
            If directory does not exist
        ValueError
            If directory path contains path traversal (e.g., `../`) that would
            escape the intended base directory
        """
        dir_path = Path(directory).resolve()
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory does not exist: {directory}")
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {directory}")

        # Validate pattern does not contain path traversal
        if ".." in pattern or "/" in pattern or "\\" in pattern:
            raise ValueError(
                "Pattern must not contain path traversal ('..', '/', '\\'). "
                "Use only filename patterns (e.g., '*' or 'data_*')."
            )

        csv_files = sorted(glob.glob(str(dir_path / f"{pattern}.csv")))
        mat_files = sorted(glob.glob(str(dir_path / f"{pattern}.mat")))
        all_files = csv_files + mat_files

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "directory_scan_complete",
                extra={
                    "event": "directory_scan_complete",
                    "directory": str(dir_path),
                    "pattern": pattern,
                    "n_csv": len(csv_files),
                    "n_mat": len(mat_files),
                    "n_total": len(all_files),
                },
            )

        if verbose:
            print(f"Found {len(csv_files)} CSV files and {len(mat_files)} MAT files")

        return self.process_files(all_files, verbose=verbose, n_jobs=n_jobs)

    def calculate_q_factors(
        self, *, include_invalid: bool = False, q_preference: str | None = None
    ) -> list[float]:
        """
        Calculate Q factors for all processed results.

        The estimator preference comes from the constructor q_preference
        (overridable per call). With "demod" (default) a valid segmented-
        demodulation Q is used when available; with "profile" a valid profile Q
        takes precedence. If the preferred estimator's Q is invalid or
        limit-only, the result is skipped by default instead of falling back
        to NLS. Pass include_invalid=True to use raw values for
        diagnostic/debug workflows. Results without validity metadata keep the
        older Q = π * f * τ fallback behavior.

        Returns:
        --------
        List[float]
            Q factor for each result
        """
        preference = q_preference if q_preference is not None else self.q_preference
        q_factors = []
        for r in self.results:
            q, q_valid, q_status = _result_q_value(
                r, include_invalid=include_invalid, q_preference=preference
            )
            r["Q"] = q
            r["Q_valid"] = q_valid
            r["Q_status"] = q_status
            if q is not None:
                q_factors.append(q)

        return q_factors

    def get_summary_table(self) -> dict:
        """
        Create a numeric summary table with all analysis results.

        Returns:
        --------
        Dict
            Dictionary with 'data' (list of dicts) and 'columns' (list of column names)
            suitable for creating pandas DataFrame. Numeric fields are returned
            as raw numbers; use get_formatted_summary_table() for display strings.
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
                    "T (s)": r["T"],
                    "T_crop (s)": r["T_crop"],
                    "fs (Hz)": r["fs"],
                    "tau_seed (s)": r.get("tau_seed"),
                    "tau_seed_method": r.get("tau_seed_method"),
                    "tau_est (s)": r["tau_est"],
                    "tau_est_fit_success": r.get("tau_est_fit_success"),
                    "tau_nls (s)": r.get("tau_nls"),
                    "tau_dft (s)": r.get("tau_dft"),
                    "tau_est_low_confidence": r.get("tau_est_low_confidence"),
                    "tau_nls_at_lower_bound": r.get("tau_nls_at_lower_bound"),
                    "tau_nls_at_upper_bound": r.get("tau_nls_at_upper_bound"),
                    "tau_dft_at_lower_bound": r.get("tau_dft_at_lower_bound"),
                    "tau_dft_at_upper_bound": r.get("tau_dft_at_upper_bound"),
                    "f_NLS (Hz)": r["f_nls"],
                    "f_DFT (Hz)": r["f_dft"],
                    "|f_NLS - f_DFT| (Hz)": abs(r["f_nls"] - r["f_dft"]),
                    "Q_pre_crop": r.get("Q_pre_crop"),
                    "Q_NLS": r.get("Q_nls"),
                    "Q_NLS_raw": r.get("Q_nls_raw"),
                    "Q_NLS_valid": r.get("Q_nls_valid"),
                    "Q_NLS_status": r.get("Q_nls_status"),
                    "Q_NLS_reasons": ", ".join(r.get("Q_nls_reasons", [])),
                    "Q_DFT": r.get("Q_dft"),
                    "Q_DFT_raw": r.get("Q_dft_raw"),
                    "Q_DFT_valid": r.get("Q_dft_valid"),
                    "Q_DFT_status": r.get("Q_dft_status"),
                    "Q_DFT_reasons": ", ".join(r.get("Q_dft_reasons", [])),
                    "Q_profile": r.get("Q_profile"),
                    "Q_profile_raw": r.get("Q_profile_raw"),
                    "Q_profile_valid": r.get("Q_profile_valid"),
                    "Q_profile_status": r.get("Q_profile_status"),
                    "Q_profile_reasons": ", ".join(r.get("Q_profile_reasons", [])),
                    "Q_profile_ci95": r.get("Q_profile_ci95"),
                    "Q_profile_lower_limit_95": r.get("Q_profile_lower_limit_95"),
                    "Q_profile_upper_limit_95": r.get("Q_profile_upper_limit_95"),
                    "tau_profile (s)": r.get("tau_profile"),
                    "f_profile (Hz)": r.get("f_profile"),
                    "Q_demod": r.get("Q_demod"),
                    "Q_demod_valid": r.get("Q_demod_valid"),
                    "Q_demod_status": r.get("Q_demod_status"),
                    "Q_demod_reasons": ", ".join(r.get("Q_demod_reasons", [])),
                    "Q_demod_ci95": r.get("Q_demod_ci95"),
                    "tau_demod (s)": r.get("tau_demod"),
                    "f_demod (Hz)": r.get("f_demod"),
                    "coherence_ratio": r.get("coherence_ratio"),
                    "coherence_gate_fired": r.get("coherence_gate_fired"),
                    "plateau_amplitude": r.get("Q_demod_plateau_amplitude"),
                    "plateau_detected": r.get("Q_demod_plateau_detected"),
                    "NLS success": r.get("nls_success"),
                    "DFT success": r.get("dft_success"),
                    "NLS used fallback": r.get("nls_used_fallback"),
                    "DFT used fallback": r.get("dft_used_fallback"),
                    "Plugin bound std (Hz)": _result_uncertainty_std(r),
                    "uncertainty_valid": r.get("uncertainty_valid"),
                    "A0_est": r["A0_est"],
                    "sigma_est": r["sigma_est"],
                }
            )

        # Add Q factor if calculated
        if "Q" in self.results[0]:
            for i, r in enumerate(self.results):
                summary_data[i]["Q"] = r["Q"]

        columns = list(summary_data[0].keys()) if summary_data else []

        return {"data": summary_data, "columns": columns}

    def get_formatted_summary_table(self) -> dict:
        """
        Create a display-oriented summary table with formatted string values.

        The primary get_summary_table() API returns raw numeric values for
        analysis. This helper preserves the older notebook-friendly display form.
        """
        table = self.get_summary_table()
        formatted_data = []
        for row in table["data"]:
            formatted = dict(row)
            for key in (
                "T (s)",
                "T_crop (s)",
                "fs (Hz)",
                "tau_seed (s)",
                "tau_est (s)",
                "tau_nls (s)",
                "tau_dft (s)",
                "tau_profile (s)",
                "tau_demod (s)",
            ):
                if key in formatted:
                    formatted[key] = _format_optional_float(formatted[key], ".2f")
            for key in ("f_NLS (Hz)", "f_DFT (Hz)", "f_profile (Hz)", "f_demod (Hz)"):
                if key in formatted:
                    formatted[key] = _format_optional_float(formatted[key], ".6f")
            for key in ("|f_NLS - f_DFT| (Hz)", "Plugin bound std (Hz)", "sigma_est"):
                if key in formatted:
                    formatted[key] = _format_optional_float(formatted[key], ".6e")
            for key in (
                "Q_pre_crop",
                "Q_NLS",
                "Q_NLS_raw",
                "Q_DFT",
                "Q_DFT_raw",
                "Q_profile",
                "Q_profile_raw",
                "Q_profile_lower_limit_95",
                "Q_profile_upper_limit_95",
                "Q_demod",
                "Q",
            ):
                if key in formatted:
                    formatted[key] = _format_optional_float(formatted[key], ".2e")
            if "A0_est" in formatted:
                formatted["A0_est"] = _format_optional_float(formatted["A0_est"], ".4f")
            formatted_data.append(formatted)
        return {"data": formatted_data, "columns": table["columns"]}

    def consistency_analysis(self) -> dict:
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
            "nls_pairwise_indices": list(zip(i_indices, j_indices, strict=False)),
            "dft_pairwise_indices": list(zip(i_indices, j_indices, strict=False)),
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

    def crlb_comparison_analysis(self) -> dict:
        """
        Compare frequency estimation differences with plug-in uncertainty diagnostics.

        Computes:
        - Frequency differences between NLS and DFT
        - Heuristic ratio of differences to plug-in bound
        - Statistics comparing differences to the plug-in bound

        Returns:
        --------
        Dict
            Dictionary with analysis results including:
            - 'frequency_diffs': array of abs(f_NLS - f_DFT)
        - 'plugin_crlb_stds': array of plug-in CRLB standard deviations
            - 'ratios': array of abs(f_NLS - f_DFT) / plugin_bound_std
            - 'crlb_statistics': dict with mean, min, max plug-in bound
        - 'ratio_statistics': heuristic dict with mean, median, min, max ratios
        """
        if not self.results:
            return {}

        # Vectorized extraction and computation
        f_nls_all = np.array([r["f_nls"] for r in self.results], dtype=float)
        f_dft_all = np.array([r["f_dft"] for r in self.results], dtype=float)
        crlb_stds = np.array([_result_uncertainty_std(r) for r in self.results], dtype=float)

        # Compute differences vectorized
        diffs = np.abs(f_nls_all - f_dft_all)

        # Compute heuristic ratios (difference / plug-in bound) vectorized
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
            "plugin_crlb_stds": np.array(crlb_stds),
            "ratios": ratios,
            "valid_ratios": valid_ratios,
            "crlb_statistics": crlb_stats,
            "ratio_statistics": ratio_stats,
        }

    def get_q_factor_statistics(
        self, *, include_invalid: bool = False, q_preference: str | None = None
    ) -> dict:
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

        # Ensure Q factors are calculated using the requested validity policy.
        self.calculate_q_factors(include_invalid=include_invalid, q_preference=q_preference)

        q_values = np.array([r["Q"] for r in self.results if r.get("Q") is not None], dtype=float)
        skipped_count = len(self.results) - len(q_values)
        profile_statuses = [
            str(r.get("Q_profile_status"))
            for r in self.results
            if "Q_profile_status" in r and not bool(r.get("Q_profile_valid", False))
        ]
        profile_limit_count = sum(
            1
            for status in profile_statuses
            if status in {"lower_limit", "upper_limit", "unbounded"}
        )
        invalid_count = sum(
            1
            for r in self.results
            if (
                "Q_profile_valid" in r
                and not bool(r.get("Q_profile_valid", False))
                and str(r.get("Q_profile_status"))
                not in {"lower_limit", "upper_limit", "unbounded"}
            )
            or (
                "Q_profile_valid" not in r
                and "Q_nls_valid" in r
                and not bool(r.get("Q_nls_valid", False))
            )
        )

        if len(q_values) == 0:
            return {
                "values": q_values,
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "range": np.nan,
                "n_total": len(self.results),
                "n_valid": 0,
                "n_skipped": skipped_count,
                "n_invalid": invalid_count,
                "n_profile_limits": profile_limit_count,
                "include_invalid": include_invalid,
            }

        return {
            "values": q_values,
            "mean": np.mean(q_values),
            "std": np.std(q_values),
            "min": np.min(q_values),
            "max": np.max(q_values),
            "range": np.max(q_values) - np.min(q_values),
            "n_total": len(self.results),
            "n_valid": len(q_values),
            "n_skipped": skipped_count,
            "n_invalid": invalid_count,
            "n_profile_limits": profile_limit_count,
            "include_invalid": include_invalid,
        }

    def get_consistency_table(self) -> dict:
        """
        Create a numeric table showing frequency estimates and deviations from mean.

        Returns:
        --------
        Dict
            Dictionary with 'data' (list of dicts) and 'columns' (list of column names)
            Suitable for creating pandas DataFrame. Numeric fields are returned
            as raw numbers; use get_formatted_consistency_table() for display strings.
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
                    "f_NLS (Hz)": r["f_nls"],
                    "f_DFT (Hz)": r["f_dft"],
                    "Deviation from NLS mean (Hz)": r["f_nls"] - nls_mean,
                    "Deviation from DFT mean (Hz)": r["f_dft"] - dft_mean,
                    "Plugin bound std (Hz)": _result_uncertainty_std(r),
                    "NLS success": r.get("nls_success"),
                    "DFT success": r.get("dft_success"),
                    "DFT used fallback": r.get("dft_used_fallback"),
                }
            )

        columns = list(consistency_data[0].keys()) if consistency_data else []

        return {"data": consistency_data, "columns": columns}

    def get_formatted_consistency_table(self) -> dict:
        """Create a display-oriented consistency table with formatted strings."""
        table = self.get_consistency_table()
        formatted_data = []
        for row in table["data"]:
            formatted = dict(row)
            for key in ("f_NLS (Hz)", "f_DFT (Hz)"):
                formatted[key] = _format_optional_float(formatted[key], ".9f")
            for key in (
                "Deviation from NLS mean (Hz)",
                "Deviation from DFT mean (Hz)",
                "Plugin bound std (Hz)",
            ):
                formatted[key] = _format_optional_float(formatted[key], ".6e")
            formatted_data.append(formatted)
        return {"data": formatted_data, "columns": table["columns"]}
