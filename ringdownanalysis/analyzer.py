"""
Analysis pipeline for real ring-down measurement data.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .crlb import CRLBCalculator
from .data_loader import RingDownDataLoader
from .estimators import (
    DFTFrequencyEstimator,
    NLSFrequencyEstimator,
    _estimate_initial_parameters_from_dft,
    _estimate_initial_tau_from_envelope,
)

logger = logging.getLogger(__name__)


def _parse_array_input(
    t: np.ndarray | None = None,
    data: np.ndarray | None = None,
    fs: float | None = None,
    *,
    time_col: str | int = 0,
    data_col: str | int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse array-like inputs into (t, data) numpy arrays.

    Supports:
    - (t, data): two array-likes
    - (data, fs): data array and sampling frequency (t = arange(len(data))/fs)
    - (data=DataFrame): extract t and data from columns via time_col, data_col

    Returns:
    --------
    t : np.ndarray
        Time array (s), starting from 0
    data : np.ndarray
        Signal array
    """
    import pandas as pd

    if data is None:
        raise ValueError("data is required")

    # DataFrame: extract time and signal columns
    if isinstance(data, pd.DataFrame):
        t_arr = np.asarray(
            data.iloc[:, time_col] if isinstance(time_col, int) else data[time_col],
            dtype=np.float64,
        )
        data_arr = np.asarray(
            data.iloc[:, data_col] if isinstance(data_col, int) else data[data_col],
            dtype=np.float64,
        )
        t_arr = t_arr - t_arr[0]
    elif t is not None:
        t_arr = np.asarray(t, dtype=np.float64)
        data_arr = np.asarray(data, dtype=np.float64)
    elif fs is not None:
        data_arr = np.asarray(data, dtype=np.float64)
        t_arr = np.arange(len(data_arr), dtype=np.float64) / fs
    else:
        raise ValueError("Either t or fs must be provided when data is not a DataFrame")

    if len(t_arr) != len(data_arr):
        raise ValueError(f"t and data must have same length, got {len(t_arr)} and {len(data_arr)}")
    if len(t_arr) < 2:
        raise ValueError("At least 2 samples required for analysis")

    return t_arr, data_arr


class RingDownAnalyzer:
    """
    Analyzes real ring-down measurement data.

    Performs the following pipeline:
    1. Load data from file
    2. Estimate tau from full data using NLS
    3. Crop data to max_tau_multiplier*tau to avoid long noisy tail
    4. Estimate frequency using NLS and DFT methods
    5. Estimate noise parameters for CRLB calculation
    6. Calculate CRLB
    """

    def __init__(
        self,
        nls_estimator: NLSFrequencyEstimator | None = None,
        dft_estimator: DFTFrequencyEstimator | None = None,
    ):
        """
        Initialize analyzer.

        Parameters:
        -----------
        nls_estimator : NLSFrequencyEstimator, optional
            NLS frequency estimator. If None, creates default (tau unknown).
        dft_estimator : DFTFrequencyEstimator, optional
            DFT frequency estimator. If None, creates default (rectangular window).
        """
        self.nls_estimator = nls_estimator or NLSFrequencyEstimator(tau_known=None)
        self.dft_estimator = dft_estimator or DFTFrequencyEstimator(window="rect")
        self.crlb_calc = CRLBCalculator()

    def estimate_tau(
        self,
        data: np.ndarray,
        t: np.ndarray,
        fs: float,
        initial_params: tuple | None = None,
        *,
        tau_init: float | None = None,
        max_nfev: int | None = None,
        ftol: float | None = None,
        xtol: float | None = None,
        gtol: float | None = None,
    ) -> float:
        """
        Estimate tau from full data using NLS fit.

        Parameters:
        -----------
        data : np.ndarray
            Signal data
        t : np.ndarray
            Time array (s)
        fs : float
            Sampling frequency (Hz)
        initial_params : tuple, optional
            (f0_init, phi0_init, A0_init, c0) to avoid redundant DFT. If None, estimated from data.
        tau_init : float, optional
            Initial guess for tau (s). If None, estimated from envelope decay.
        max_nfev : int, optional
            Maximum number of function evaluations for the fit. Default 150.
        ftol, xtol, gtol : float, optional
            Convergence tolerances for least_squares. Defaults 1e-8.

        Returns:
        --------
        float
            Estimated tau value in seconds
        """
        N = len(data)
        t_norm = t - t[0]

        # Get initial parameter estimates (use cached if provided)
        if initial_params is not None:
            f0_init, phi0_init, A0_init, c0 = initial_params
        else:
            f0_init, phi0_init, A0_init, c0 = _estimate_initial_parameters_from_dft(data, fs)

        # Initial tau guess
        if tau_init is None:
            tau_init = _estimate_initial_tau_from_envelope(data, t_norm)

        # NLS fit to estimate tau: fit (A0, f, phi, tau, c)
        def residuals_tau(p):
            A0, f, phi, tau, c = p
            return (A0 * np.exp(-t_norm / tau) * np.cos(2.0 * np.pi * f * t_norm + phi) + c) - data

        df = fs / N
        f_low = max(0.0, f0_init - max(0.2 * f0_init, 2 * df))
        f_high = min(0.5 * fs, f0_init + max(0.2 * f0_init, 2 * df))

        lb = [0.0, f_low, -np.pi, t_norm[1], -np.inf]
        ub = [10.0 * A0_init, f_high, np.pi, 10.0 * t_norm[-1], np.inf]
        # Extend tau bounds if tau_init is provided and outside default range
        tau_ub_default = 10.0 * t_norm[-1]
        if tau_init is not None and tau_init > tau_ub_default:
            ub[3] = max(ub[3], tau_init * 1.1)  # Allow tau above default upper bound
        if tau_init is not None and tau_init < lb[3]:
            lb[3] = max(t_norm[1], min(lb[3], tau_init * 0.9))

        ls_kwargs: dict = {"method": "trf", "verbose": 0}
        if max_nfev is not None:
            ls_kwargs["max_nfev"] = max_nfev
        else:
            ls_kwargs["max_nfev"] = 150
        if ftol is not None:
            ls_kwargs["ftol"] = ftol
        else:
            ls_kwargs["ftol"] = 1e-8
        if xtol is not None:
            ls_kwargs["xtol"] = xtol
        else:
            ls_kwargs["xtol"] = 1e-8
        if gtol is not None:
            ls_kwargs["gtol"] = gtol
        else:
            ls_kwargs["gtol"] = 1e-8

        res_tau = least_squares(
            residuals_tau,
            x0=np.array([A0_init, f0_init, phi0_init, tau_init, c0]),
            bounds=(lb, ub),
            **ls_kwargs,
        )

        if res_tau.success:
            _, _, _, tau_est, _ = res_tau.x
            # Sanity check (use extended tau_ub if tau_init was large)
            tau_ub = ub[3]  # Already extended above if tau_init was outside default
            if tau_est <= 0 or not np.isfinite(tau_est) or tau_est > tau_ub or tau_est < t_norm[1]:
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning(
                        "tau_sanity_check_failed",
                        extra={
                            "event": "tau_sanity_check_failed",
                            "tau_est": float(tau_est),
                            "tau_init": float(tau_init),
                            "t_max": float(t_norm[-1]),
                        },
                    )
                return tau_init
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "tau_estimated",
                    extra={
                        "event": "tau_estimated",
                        "tau_est": float(tau_est),
                        "tau_init": float(tau_init),
                        "nfev": res_tau.nfev,
                    },
                )
            return tau_est
        else:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    "tau_estimation_failed",
                    extra={
                        "event": "tau_estimation_failed",
                        "tau_init": float(tau_init),
                        "fit_message": res_tau.message,
                        "nfev": res_tau.nfev,
                    },
                )
            return tau_init

    def crop_data_to_tau(
        self,
        t: np.ndarray,
        data: np.ndarray,
        tau_est: float,
        min_samples: int = 100,
        max_tau_multiplier: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Crop data to max_tau_multiplier*tau_est to avoid long noisy tail affecting frequency estimation.

        Parameters:
        -----------
        t : np.ndarray
            Time array
        data : np.ndarray
            Signal array
        tau_est : float
            Estimated tau value in seconds
        min_samples : int
            Minimum number of samples required. If cropped data is shorter, return original.
        max_tau_multiplier : float
            Multiplier for tau_est to determine maximum record length. Default is 1.0.

        Returns:
        --------
        (t_crop, data_cropped) : tuple
            Cropped time and signal arrays
        """
        t_crop_max = max_tau_multiplier * tau_est
        crop_idx = t <= t_crop_max
        t_crop = t[crop_idx]
        data_cropped = data[crop_idx]

        # If cropped data is too short, return original
        # Use views instead of copies when possible
        if len(t_crop) < min_samples:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    "data_crop_too_short",
                    extra={
                        "event": "data_crop_too_short",
                        "n_cropped": len(t_crop),
                        "min_samples": min_samples,
                        "tau_est": float(tau_est),
                    },
                )
            return t, data

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "data_cropped",
                extra={
                    "event": "data_cropped",
                    "n_original": len(t),
                    "n_cropped": len(t_crop),
                    "tau_est": float(tau_est),
                    "crop_time": float(t_crop[-1]) if len(t_crop) > 0 else 0.0,
                },
            )

        return t_crop, data_cropped

    def estimate_noise_parameters(
        self,
        data_cropped: np.ndarray,
        t_crop: np.ndarray,
        tau_est: float,
        fs: float,
        initial_params: tuple | None = None,
        *,
        max_nfev: int | None = None,
        ftol: float | None = None,
        xtol: float | None = None,
        gtol: float | None = None,
    ) -> tuple[float, float]:
        """
        Estimate A0 (initial amplitude) and sigma (noise std) from cropped data.

        Parameters:
        -----------
        data_cropped : np.ndarray
            Cropped signal data
        t_crop : np.ndarray
            Cropped time array
        tau_est : float
            Estimated tau value in seconds
        fs : float
            Sampling frequency (Hz)
        initial_params : tuple, optional
            (f0_init, phi0_init, A0_init, c0) to avoid redundant DFT. If None, estimated from data.
        max_nfev : int, optional
            Maximum number of function evaluations for the fit. Default 100.
        ftol, xtol, gtol : float, optional
            Convergence tolerances for least_squares. Default ftol 1e-6; xtol/gtol use scipy defaults.

        Returns:
        --------
        (A0_est, sigma_est) : tuple
            Estimated A0 and sigma
        """
        N_crop = len(data_cropped)
        t_crop_norm = t_crop - t_crop[0]

        # Initial estimate from first portion
        n_init = min(1000, N_crop // 10)
        A0_est = np.sqrt(2.0) * np.std(data_cropped[:n_init])

        # Fit model to get residuals for noise estimation
        def model_residuals(p):
            A0, f, phi, c = p
            return (
                A0 * np.exp(-t_crop_norm / tau_est) * np.cos(2.0 * np.pi * f * t_crop_norm + phi)
                + c
            ) - data_cropped

        # Get initial guesses (use cached if provided)
        if initial_params is not None:
            f0_init, phi0_init, A0_init, c0 = initial_params
        else:
            f0_init, phi0_init, A0_init, c0 = _estimate_initial_parameters_from_dft(
                data_cropped, fs
            )

        # Quick fit to get residuals
        df = fs / N_crop
        f_low = max(0.0, f0_init - max(0.2 * f0_init, 2 * df))
        f_high = min(0.5 * fs, f0_init + max(0.2 * f0_init, 2 * df))

        ls_kwargs: dict = {"method": "trf", "verbose": 0}
        if max_nfev is not None:
            ls_kwargs["max_nfev"] = max_nfev
        else:
            ls_kwargs["max_nfev"] = 100
        if ftol is not None:
            ls_kwargs["ftol"] = ftol
        else:
            ls_kwargs["ftol"] = 1e-6
        if xtol is not None:
            ls_kwargs["xtol"] = xtol
        if gtol is not None:
            ls_kwargs["gtol"] = gtol

        res_fit = least_squares(
            model_residuals,
            x0=np.array([A0_init, f0_init, phi0_init, c0]),
            bounds=([0.0, f_low, -np.pi, -np.inf], [10.0 * A0_init, f_high, np.pi, np.inf]),
            **ls_kwargs,
        )

        if res_fit.success:
            residuals = res_fit.fun
            sigma_est = np.std(residuals)
            A0_est = res_fit.x[0]
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "noise_parameters_estimated",
                    extra={
                        "event": "noise_parameters_estimated",
                        "A0_est": float(A0_est),
                        "sigma_est": float(sigma_est),
                        "nfev": res_fit.nfev,
                    },
                )
        else:
            # Fallback: estimate noise from tail
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    "noise_estimation_fallback",
                    extra={
                        "event": "noise_estimation_fallback",
                        "fit_message": res_fit.message,
                        "nfev": res_fit.nfev,
                    },
                )
            tail_start = max(int(0.8 * len(data_cropped)), len(data_cropped) - 1000)
            sigma_est = np.std(data_cropped[tail_start:])
            A0_est = np.sqrt(2.0) * np.std(data_cropped[:n_init])

        return A0_est, sigma_est

    def _run_analysis_pipeline(
        self,
        t: np.ndarray,
        data: np.ndarray,
        fs: float,
        max_tau_multiplier: float,
        *,
        initial_params: tuple | None = None,
        tau_init: float | None = None,
        max_nfev: int | None = None,
        ftol: float | None = None,
        xtol: float | None = None,
        gtol: float | None = None,
    ) -> dict:
        """Run the full analysis pipeline on (t, data) arrays."""
        if initial_params is not None:
            initial_params_full = initial_params
            initial_params_cropped = initial_params
        else:
            initial_params_full = _estimate_initial_parameters_from_dft(data, fs)
            initial_params_cropped = None  # Will compute after crop

        tau_est = self.estimate_tau(
            data,
            t,
            fs,
            initial_params=initial_params_full,
            tau_init=tau_init,
            max_nfev=max_nfev,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
        )

        t_crop, data_cropped = self.crop_data_to_tau(
            t, data, tau_est, min_samples=1000, max_tau_multiplier=max_tau_multiplier
        )

        min_samples_for_analysis = 1000
        if len(t_crop) < min_samples_for_analysis:
            t_crop = t
            data_cropped = data

        if initial_params_cropped is None:
            initial_params_cropped = _estimate_initial_parameters_from_dft(data_cropped, fs)

        fit_kwargs = {}
        if max_nfev is not None:
            fit_kwargs["max_nfev"] = max_nfev
        if ftol is not None:
            fit_kwargs["ftol"] = ftol
        if xtol is not None:
            fit_kwargs["xtol"] = xtol
        if gtol is not None:
            fit_kwargs["gtol"] = gtol
        if tau_init is not None:
            fit_kwargs["tau_init"] = tau_init

        result_nls = self.nls_estimator.estimate_full(
            data_cropped,
            fs,
            initial_params=initial_params_cropped,
            **fit_kwargs,
        )
        result_dft = self.dft_estimator.estimate_full(
            data_cropped,
            fs,
            initial_params=initial_params_cropped,
            **fit_kwargs,
        )

        f_nls = result_nls.f
        f_dft = result_dft.f
        Q_nls = result_nls.Q
        Q_dft = result_dft.Q
        tau_nls = result_nls.tau

        A0_est, sigma_est = self.estimate_noise_parameters(
            data_cropped,
            t_crop,
            tau_est,
            fs,
            initial_params=initial_params_cropped,
            max_nfev=max_nfev,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
        )

        N_crop = len(data_cropped)
        crlb_var_f = self.crlb_calc.variance(A0_est, sigma_est, fs, N_crop, tau_est)
        crlb_std_f = np.sqrt(crlb_var_f) if np.isfinite(crlb_var_f) else np.inf

        return {
            "t": t,
            "data": data,
            "V2": None,
            "t_crop": t_crop,
            "data_cropped": data_cropped,
            "fs": fs,
            "tau_est": tau_est,
            "tau_nls": tau_nls,
            "f_nls": f_nls,
            "f_dft": f_dft,
            "Q_nls": Q_nls,
            "Q_dft": Q_dft,
            "A0_est": A0_est,
            "sigma_est": sigma_est,
            "crlb_std_f": crlb_std_f,
            "N": len(t),
            "N_crop": len(t_crop),
            "T": t[-1],
            "T_crop": t_crop[-1] if len(t_crop) > 0 else 0,
        }

    def analyze_array(
        self,
        t: np.ndarray | None = None,
        data: np.ndarray | None = None,
        fs: float | None = None,
        *,
        time_col: str | int = 0,
        data_col: str | int = 1,
        max_tau_multiplier: float = 1.0,
        initial_params: tuple | None = None,
        tau_init: float | None = None,
        max_nfev: int | None = None,
        ftol: float | None = None,
        xtol: float | None = None,
        gtol: float | None = None,
    ) -> dict:
        """
        Analyze ring-down data from numpy arrays or pandas Series/DataFrame.

        Parameters:
        -----------
        t : np.ndarray or pd.Series, optional
            Time array (s). Required unless data is a DataFrame or fs is provided.
        data : np.ndarray, pd.Series, or pd.DataFrame
            Signal data. Required.
        fs : float, optional
            Sampling frequency (Hz). If provided with t=None, time is inferred as
            t = np.arange(len(data)) / fs.
        time_col : str or int, optional
            Column index or name for time when data is a DataFrame. Default 0.
        data_col : str or int, optional
            Column index or name for signal when data is a DataFrame. Default 1.
        max_tau_multiplier : float
            Multiplier for tau_est when cropping data. Default 1.0.
        initial_params : tuple, optional
            (f0_init, phi0_init, A0_init, c0) to avoid redundant DFT. If None, estimated from data.
        tau_init : float, optional
            Initial guess for tau (s). If None, estimated from envelope decay.
        max_nfev : int, optional
            Maximum function evaluations for NLS fits. Increase for noisy data (e.g. 300–500).
        ftol, xtol, gtol : float, optional
            Convergence tolerances for least_squares. Relax (e.g. 1e-6) for noisy data.

        Returns:
        --------
        dict
            Results dictionary (same structure as analyze_file, minus filename/type).

        Raises:
        -------
        ValueError
            If data is invalid, lengths mismatch, or neither t nor fs provided.

        Examples:
        ---------
        >>> # From numpy arrays
        >>> result = analyzer.analyze_array(t, data)
        >>> # From data and sampling rate
        >>> result = analyzer.analyze_array(data=data, fs=1000.0)
        >>> # From pandas DataFrame
        >>> result = analyzer.analyze_array(data=df, time_col="time", data_col="phase")
        >>> # Noisy data: relax tolerances and increase max_nfev
        >>> result = analyzer.analyze_array(t, data, max_nfev=500, ftol=1e-6)
        """
        t_arr, data_arr = _parse_array_input(
            t=t, data=data, fs=fs, time_col=time_col, data_col=data_col
        )

        fs = 1.0 / np.mean(np.diff(t_arr))

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "analysis_start",
                extra={
                    "event": "analysis_start",
                    "source": "array",
                    "n_samples": len(t_arr),
                },
            )

        result = self._run_analysis_pipeline(
            t_arr,
            data_arr,
            fs,
            max_tau_multiplier,
            initial_params=initial_params,
            tau_init=tau_init,
            max_nfev=max_nfev,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
        )

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "analysis_complete",
                extra={
                    "event": "analysis_complete",
                    "source": "array",
                    "f_nls": float(result["f_nls"]),
                    "f_dft": float(result["f_dft"]),
                    "tau_est": float(result["tau_est"]),
                    "crlb_std_f": float(result["crlb_std_f"])
                    if np.isfinite(result["crlb_std_f"])
                    else None,
                },
            )

        return result

    def analyze_file(
        self,
        filepath: str,
        max_tau_multiplier: float = 1.0,
        *,
        initial_params: tuple | None = None,
        tau_init: float | None = None,
        max_nfev: int | None = None,
        ftol: float | None = None,
        xtol: float | None = None,
        gtol: float | None = None,
    ) -> dict:
        """
        Process a single data file and return analysis results.

        Parameters:
        -----------
        filepath : str
            Path to the data file
        max_tau_multiplier : float
            Multiplier for tau_est to determine maximum record length when cropping data.
            Default is 1.0.
        initial_params : tuple, optional
            (f0_init, phi0_init, A0_init, c0) to avoid redundant DFT. If None, estimated from data.
        tau_init : float, optional
            Initial guess for tau (s). If None, estimated from envelope decay.
        max_nfev : int, optional
            Maximum function evaluations for NLS fits. Increase for noisy data (e.g. 300–500).
        ftol, xtol, gtol : float, optional
            Convergence tolerances for least_squares. Relax (e.g. 1e-6) for noisy data.

        Returns:
        --------
        dict
            Results dictionary with all analysis data

        Raises:
        -------
        FileNotFoundError
            If the file does not exist
        ValueError
            If the file format is unsupported, data is invalid (e.g., empty CSV,
            malformed MAT structure), or file exceeds size limit
        """
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "analysis_start",
                extra={
                    "event": "analysis_start",
                    "filepath": str(filepath),
                },
            )

        t, data, V2, file_type = RingDownDataLoader.load(filepath)
        fs = 1.0 / np.mean(np.diff(t))

        result = self._run_analysis_pipeline(
            t,
            data,
            fs,
            max_tau_multiplier,
            initial_params=initial_params,
            tau_init=tau_init,
            max_nfev=max_nfev,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
        )

        result["filename"] = Path(filepath).name
        result["type"] = file_type
        result["V2"] = V2

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "analysis_complete",
                extra={
                    "event": "analysis_complete",
                    "filepath": str(filepath),
                    "f_nls": float(result["f_nls"]),
                    "f_dft": float(result["f_dft"]),
                    "tau_est": float(result["tau_est"]),
                    "crlb_std_f": float(result["crlb_std_f"])
                    if np.isfinite(result["crlb_std_f"])
                    else None,
                },
            )

        return result
